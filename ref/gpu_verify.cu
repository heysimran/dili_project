/**
 * GPU-accelerated verification bottleneck: w' = A·z − c·t
 * NTT -> pointwise multiplication -> inverse NTT on device.
 * Supports Dilithium 2, 3, 5 via DILITHIUM_MODE.
 *
 * KEY FIXES over previous version:
 *   - cudaMalloc/Free and stream are persistent (init once, not per call)
 *   - cudaMemcpyToSymbol(zetas) done once at init
 *   - Contiguous device buffers, fewer memcpy calls
 *   - Proper warmup support
 */

#include "gpu_verify.h"
#include "params.h"
#include "reduce.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* ------------------------------------------------------------------ */
/* Error checking macro                                                 */
/* ------------------------------------------------------------------ */
#define CUDA_CHECK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", \
            cudaGetErrorString(e), __FILE__, __LINE__); \
    return -1; \
  } \
} while(0)

#define CUDA_CHECK_VOID(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", \
            cudaGetErrorString(e), __FILE__, __LINE__); \
    return; \
  } \
} while(0)

/* ------------------------------------------------------------------ */
/* Zetas (same as ref/ntt.c, must match exactly)                       */
/* ------------------------------------------------------------------ */
static const int32_t h_zetas[256] = {
         0,    25847, -2608894,  -518909,   237124,  -777960,  -876248,   466468,
   1826347,  2353451,  -359251, -2091905,  3119733, -2884855,  3111497,  2680103,
   2725464,  1024112, -1079900,  3585928,  -549488, -1119584,  2619752, -2108549,
  -2118186, -3859737, -1399561, -3277672,  1757237,   -19422,  4010497,   280005,
   2706023,    95776,  3077325,  3530437, -1661693, -3592148, -2537516,  3915439,
  -3861115, -3043716,  3574422, -2867647,  3539968,  -300467,  2348700,  -539299,
  -1699267, -1643818,  3505694, -3821735,  3507263, -2140649, -1600420,  3699596,
    811944,   531354,   954230,  3881043,  3900724, -2556880,  2071892, -2797779,
  -3930395, -1528703, -3677745, -3041255, -1452451,  3475950,  2176455, -1585221,
  -1257611,  1939314, -4083598, -1000202, -3190144, -3157330, -3632928,   126922,
   3412210,  -983419,  2147896,  2715295, -2967645, -3693493,  -411027, -2477047,
   -671102, -1228525,   -22981, -1308169,  -381987,  1349076,  1852771, -1430430,
  -3343383,   264944,   508951,  3097992,    44288, -1100098,   904516,  3958618,
  -3724342,    -8578,  1653064, -3249728,  2389356,  -210977,   759969, -1316856,
    189548, -3553272,  3159746, -1851402, -2409325,  -177440,  1315589,  1341330,
   1285669, -1584928,  -812732, -1439742, -3019102, -3881060, -3628969,  3839961,
   2091667,  3407706,  2316500,  3817976, -3342478,  2244091, -2446433, -3562462,
    266997,  2434439, -1235728,  3513181, -3520352, -3759364, -1197226, -3193378,
    900702,  1859098,   909542,   819034,   495491, -1613174,   -43260,  -522500,
   -655327, -3122442,  2031748,  3207046, -3556995,  -525098,  -768622, -3595838,
    342297,   286988, -2437823,  4108315,  3437287, -3342277,  1735879,   203044,
   2842341,  2691481, -2590150,  1265009,  4055324,  1247620,  2486353,  1595974,
  -3767016,  1250494,  2635921, -3548272, -2994039,  1869119,  1903435, -1050970,
  -1333058,  1237275, -3318210, -1430225,  -451100,  1312455,  3306115, -1962642,
  -1279661,  1917081, -2546312, -1374803,  1500165,   777191,  2235880,  3406031,
   -542412, -2831860, -1671176, -1846953, -2584293, -3724270,   594136, -3776993,
  -2013608,  2432395,  2454455,  -164721,  1957272,  3369112,   185531, -1207385,
  -3183426,   162844,  1616392,  3014001,   810149,  1652634, -3694233, -1799107,
  -3038916,  3523897,  3866901,   269760,  2213111,  -975884,  1717735,   472078,
   -426683,  1723600, -1803090,  1910376, -1667432, -1104333,  -260646, -3833893,
  -2939036, -2235985,  -420899, -2286327,   183443,  -976891,  1612842, -3545687,
   -554416,  3919660,   -48306, -1362209,  3937738,  1400424,  -846154,  1976782
};

__constant__ int32_t d_zetas[256];

/* ------------------------------------------------------------------ */
/* Persistent GPU state (allocated once, reused across calls)          */
/* ------------------------------------------------------------------ */
static int32_t *d_mat  = NULL;   /* K*L*N int32 */
static int32_t *d_z    = NULL;   /* L*N   int32 */
static int32_t *d_cp   = NULL;   /* N     int32 */
static int32_t *d_t1   = NULL;   /* K*N   int32 */
static int32_t *d_w1   = NULL;   /* K*N   int32 */
static int32_t *d_ntt  = NULL;   /* (L+1+K)*N int32  -- scratch for batched NTT */
static cudaStream_t g_stream     = NULL;
static int g_initialized         = 0;

/* ------------------------------------------------------------------ */
/* Benchmark accumulators                                               */
/* ------------------------------------------------------------------ */
#ifdef GPU_VERIFY_BENCHMARK
static double gpu_kernel_total_ms  = 0.0;
static double gpu_e2e_total_ms     = 0.0;
static int    gpu_n_calls          = 0;

static void gpu_benchmark_atexit(void) {
  if (gpu_n_calls == 0) return;
  fprintf(stderr,
    "\n[GPU_VERIFY_BENCHMARK] Over %d calls:\n"
    "  Kernel only (NTT+pw+iNTT):  total %.3f ms,  avg %.6f ms/call\n"
    "  End-to-end (incl. memcpy):  total %.3f ms,  avg %.6f ms/call\n",
    gpu_n_calls,
    gpu_kernel_total_ms, gpu_kernel_total_ms / gpu_n_calls,
    gpu_e2e_total_ms,    gpu_e2e_total_ms    / gpu_n_calls);
}
#endif

/* ------------------------------------------------------------------ */
/* Device helpers                                                       */
/* ------------------------------------------------------------------ */
__device__ __forceinline__ int32_t montgomery_reduce(int64_t a) {
  int32_t t = (int32_t)((int64_t)(int32_t)a * (int64_t)QINV);
  return (int32_t)((a - (int64_t)t * Q) >> 32);
}

__device__ __forceinline__ int32_t reduce32_dev(int32_t a) {
  int32_t t = (a + (1 << 22)) >> 23;
  return a - t * Q;
}

/* ------------------------------------------------------------------ */
/* Kernels                                                              */
/* ------------------------------------------------------------------ */

/* NTT: one block per polynomial, 128 threads. */
__global__ void kernel_ntt(int32_t *polys, int npolys) {
  int poly_idx = blockIdx.x;
  if (poly_idx >= npolys) return;
  int32_t *a = polys + poly_idx * N;

  __shared__ int32_t sh[N];
  int tid = threadIdx.x;
  sh[tid]       = a[tid];
  sh[tid + 128] = a[tid + 128];
  __syncthreads();

  for (int len = 128; len > 0; len >>= 1) {
    int k_start = (len == 128) ? 1 : (256 / len);
    for (int b = tid; b < 128; b += 128) {
      int start_index = b / len;
      int j = start_index * (2 * len) + (b % len);
      int32_t zeta = d_zetas[k_start + start_index];
      int32_t t = montgomery_reduce((int64_t)zeta * sh[j + len]);
      sh[j + len] = sh[j] - t;
      sh[j]       = sh[j] + t;
    }
    __syncthreads();
  }

  a[tid]       = sh[tid];
  a[tid + 128] = sh[tid + 128];
}

/* Inverse NTT: one block per polynomial, 128 threads. */
__global__ void kernel_invntt(int32_t *polys, int npolys) {
  const int32_t f = 41978;
  int poly_idx = blockIdx.x;
  if (poly_idx >= npolys) return;
  int32_t *a = polys + poly_idx * N;

  __shared__ int32_t sh[N];
  int tid = threadIdx.x;
  sh[tid]       = a[tid];
  sh[tid + 128] = a[tid + 128];
  __syncthreads();

  for (int len = 1; len < N; len <<= 1) {
    for (int b = tid; b < 128; b += 128) {
      int start_index = b / len;
      int j = start_index * (2 * len) + (b % len);
      int k = (256 / len) - 1 - start_index;
      int32_t zeta = -d_zetas[k];
      int32_t t    = sh[j];
      sh[j]        = t + sh[j + len];
      sh[j + len]  = montgomery_reduce((int64_t)zeta * (t - sh[j + len]));
    }
    __syncthreads();
  }

  sh[tid]       = montgomery_reduce((int64_t)f * sh[tid]);
  sh[tid + 128] = montgomery_reduce((int64_t)f * sh[tid + 128]);
  __syncthreads();
  a[tid]       = sh[tid];
  a[tid + 128] = sh[tid + 128];
}

/* shiftl: t1[i] <<= D. One block per poly, N threads. */
__global__ void kernel_shiftl(int32_t *polys, int npolys) {
  int poly_idx = blockIdx.x;
  if (poly_idx >= npolys) return;
  int i = threadIdx.x;
  if (i < N) polys[poly_idx * N + i] <<= D;
}

/*
 * w1 = A*z - c*t1 in NTT domain.
 * Each block handles one row i of A (0..K-1).
 * N threads per block, each handles one coefficient k.
 */
__global__ void kernel_w1_from_az_ct(const int32_t *d_mat, const int32_t *d_z,
                                     const int32_t *d_cp,  const int32_t *d_t1,
                                     int32_t *d_w1) {
  int i = blockIdx.x;
  int k = threadIdx.x;
  if (i >= K || k >= N) return;

  int64_t sum = 0;
  for (int j = 0; j < L; j++)
    sum += (int64_t)d_mat[(i * L + j) * N + k] * (int64_t)d_z[j * N + k];

  int32_t az_k = montgomery_reduce(sum);
  int32_t ct_k = montgomery_reduce((int64_t)d_cp[k] * (int64_t)d_t1[i * N + k]);
  d_w1[i * N + k] = az_k - ct_k;
}

/* Reduce w1 coefficients. K blocks, N threads. */
__global__ void kernel_reduce(int32_t *d_w1) {
  int poly_idx = blockIdx.x;
  int i        = threadIdx.x;
  if (poly_idx < K && i < N)
    d_w1[poly_idx * N + i] = reduce32_dev(d_w1[poly_idx * N + i]);
}

/* ------------------------------------------------------------------ */
/* One-time GPU initialisation                                          */
/* ------------------------------------------------------------------ */
static int gpu_init(void) {
  if (g_initialized) return 0;

  /* Upload zetas once */
  if (cudaMemcpyToSymbol(d_zetas, h_zetas, 256 * sizeof(int32_t))
      != cudaSuccess) return -1;

  /* Allocate all device buffers once */
  if (cudaMalloc(&d_mat, K * L * N * sizeof(int32_t)) != cudaSuccess) return -1;
  if (cudaMalloc(&d_z,   L     * N * sizeof(int32_t)) != cudaSuccess) return -1;
  if (cudaMalloc(&d_cp,          N * sizeof(int32_t)) != cudaSuccess) return -1;
  if (cudaMalloc(&d_t1,  K     * N * sizeof(int32_t)) != cudaSuccess) return -1;
  if (cudaMalloc(&d_w1,  K     * N * sizeof(int32_t)) != cudaSuccess) return -1;
  int n_ntt = L + 1 + K;
  if (cudaMalloc(&d_ntt, n_ntt * N * sizeof(int32_t)) != cudaSuccess) return -1;

  if (cudaStreamCreate(&g_stream) != cudaSuccess) return -1;

#ifdef GPU_VERIFY_BENCHMARK
  static int atexit_done = 0;
  if (!atexit_done) { atexit_done = 1; atexit(gpu_benchmark_atexit); }
#endif

  g_initialized = 1;
  return 0;
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */
extern "C" {

/*
 * gpu_compute_wprime()
 * Computes w1 = A*z - c*t1 on the GPU (NTT domain).
 * Call gpu_warmup() before benchmarking to eliminate first-call overhead.
 */
int gpu_compute_wprime(polyveck *w1, const polyvecl mat[K], const polyvecl *z,
                       const poly *cp, const polyveck *t1) {
  if (gpu_init() != 0) return -1;

  const int n_ntt = L + 1 + K;

  /* ---- E2E timer start ------------------------------------------- */
#ifdef GPU_VERIFY_BENCHMARK
  cudaEvent_t e2e_start, e2e_stop, kern_start, kern_stop;
  cudaEventCreate(&e2e_start);  cudaEventCreate(&e2e_stop);
  cudaEventCreate(&kern_start); cudaEventCreate(&kern_stop);
  cudaEventRecord(e2e_start, g_stream);
#endif

  /* ---- Upload inputs (contiguous copies where possible) ----------- */
  /* mat: K*L polynomials laid out as [K][L][N] */
  cudaMemcpyAsync(d_mat, mat[0].vec[0].coeffs,
                  K * L * N * sizeof(int32_t),
                  cudaMemcpyHostToDevice, g_stream);

  /* z: L polynomials [L][N] */
  cudaMemcpyAsync(d_z, z->vec[0].coeffs,
                  L * N * sizeof(int32_t),
                  cudaMemcpyHostToDevice, g_stream);

  /* cp: single polynomial [N] */
  cudaMemcpyAsync(d_cp, cp->coeffs,
                  N * sizeof(int32_t),
                  cudaMemcpyHostToDevice, g_stream);

  /* t1: K polynomials [K][N] */
  cudaMemcpyAsync(d_t1, t1->vec[0].coeffs,
                  K * N * sizeof(int32_t),
                  cudaMemcpyHostToDevice, g_stream);

  /* ---- Preprocessing --------------------------------------------- */
  kernel_shiftl<<<K, N, 0, g_stream>>>(d_t1, K);

  /* Pack z, cp, t1 into one contiguous buffer for batched NTT */
  cudaMemcpyAsync(d_ntt,
                  d_z,  L * N * sizeof(int32_t),
                  cudaMemcpyDeviceToDevice, g_stream);
  cudaMemcpyAsync(d_ntt + L * N,
                  d_cp, N * sizeof(int32_t),
                  cudaMemcpyDeviceToDevice, g_stream);
  cudaMemcpyAsync(d_ntt + (L + 1) * N,
                  d_t1, K * N * sizeof(int32_t),
                  cudaMemcpyDeviceToDevice, g_stream);

  /* ---- Kernel section (timed separately) -------------------------- */
#ifdef GPU_VERIFY_BENCHMARK
  cudaEventRecord(kern_start, g_stream);
#endif

  /* Batched NTT: all L+1+K polynomials in one launch */
  kernel_ntt<<<n_ntt, 128, 0, g_stream>>>(d_ntt, n_ntt);

  /* Unpack back */
  cudaMemcpyAsync(d_z,  d_ntt,
                  L * N * sizeof(int32_t),
                  cudaMemcpyDeviceToDevice, g_stream);
  cudaMemcpyAsync(d_cp, d_ntt + L * N,
                  N * sizeof(int32_t),
                  cudaMemcpyDeviceToDevice, g_stream);
  cudaMemcpyAsync(d_t1, d_ntt + (L + 1) * N,
                  K * N * sizeof(int32_t),
                  cudaMemcpyDeviceToDevice, g_stream);

  /* Pointwise: w1 = Az - ct */
  kernel_w1_from_az_ct<<<K, N, 0, g_stream>>>(d_mat, d_z, d_cp, d_t1, d_w1);
  kernel_reduce<<<K, N, 0, g_stream>>>(d_w1);

  /* Inverse NTT */
  kernel_invntt<<<K, 128, 0, g_stream>>>(d_w1, K);

#ifdef GPU_VERIFY_BENCHMARK
  cudaEventRecord(kern_stop, g_stream);
#endif

  /* ---- Download result ------------------------------------------- */
  cudaMemcpyAsync(w1->vec[0].coeffs, d_w1,
                  K * N * sizeof(int32_t),
                  cudaMemcpyDeviceToHost, g_stream);

  cudaStreamSynchronize(g_stream);

  /* ---- Accumulate timings ---------------------------------------- */
#ifdef GPU_VERIFY_BENCHMARK
  cudaEventRecord(e2e_stop, g_stream);
  cudaEventSynchronize(e2e_stop);

  float km = 0, em = 0;
  cudaEventElapsedTime(&km, kern_start, kern_stop);
  cudaEventElapsedTime(&em, e2e_start,  e2e_stop);

  gpu_kernel_total_ms += km;
  gpu_e2e_total_ms    += em;
  gpu_n_calls++;

  cudaEventDestroy(kern_start); cudaEventDestroy(kern_stop);
  cudaEventDestroy(e2e_start);  cudaEventDestroy(e2e_stop);
#endif

  return 0;
}

/*
 * gpu_warmup()
 * Call ONCE before benchmarking. Eliminates:
 *   - CUDA context initialisation (~100ms first call)
 *   - JIT compilation overhead
 *   - Cache cold-start effects
 * Pass dummy (real) inputs so the kernel actually executes.
 */
void gpu_warmup(polyveck *w1, const polyvecl mat[K], const polyvecl *z,
                const poly *cp, const polyveck *t1) {
  if (gpu_init() != 0) return;
  /* Run 20 warmup calls, ignore results */
  for (int i = 0; i < 20; i++)
    gpu_compute_wprime(w1, mat, z, cp, t1);
  cudaDeviceSynchronize();
  /* Reset benchmark counters so warmup doesn't pollute results */
#ifdef GPU_VERIFY_BENCHMARK
  gpu_kernel_total_ms = 0.0;
  gpu_e2e_total_ms    = 0.0;
  gpu_n_calls         = 0;
#endif
  fprintf(stderr, "[GPU] Warmup complete (20 calls).\n");
}

/*
 * gpu_cleanup()
 * Free persistent GPU resources. Call at program exit if desired.
 * (atexit handles benchmark printing separately.)
 */
void gpu_cleanup(void) {
  if (!g_initialized) return;
  cudaStreamSynchronize(g_stream);
  cudaFree(d_mat); cudaFree(d_z); cudaFree(d_cp);
  cudaFree(d_t1);  cudaFree(d_w1); cudaFree(d_ntt);
  cudaStreamDestroy(g_stream);
  d_mat = d_z = d_cp = d_t1 = d_w1 = d_ntt = NULL;
  g_initialized = 0;
}

} /* extern "C" */
