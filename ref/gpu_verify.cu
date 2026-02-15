/**
 * GPU-accelerated verification bottleneck: w' = A·z − c·t
 * NTT -> pointwise multiplication -> inverse NTT on device.
 * Dilithium2 (K=4, L=4, N=256) only. Build with -DDILITHIUM_MODE=2.
 */

#include "gpu_verify.h"
#include "params.h"
#include "reduce.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    return -1; \
  } \
} while(0)

/* Zetas table from ref/ntt.c (must match exactly) */
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

__constant__ int32_t d_zetas[N];

/* Device: Montgomery reduce */
__device__ __forceinline__ int32_t montgomery_reduce(int64_t a) {
  int32_t t = (int64_t)(int32_t)a * QINV;
  t = (int32_t)((a - (int64_t)t * Q) >> 32);
  return t;
}

/* Device: reduce32 */
__device__ __forceinline__ int32_t reduce32_dev(int32_t a) {
  int32_t t = (a + (1 << 22)) >> 23;
  return a - t * Q;
}

/* One block per polynomial (128 threads). In-place NTT matching ref loop order. */
__global__ void kernel_ntt(int32_t *polys, int npolys) {
  int poly_idx = blockIdx.x;
  if (poly_idx >= npolys) return;
  int32_t *a = polys + poly_idx * N;

  __shared__ int32_t sh[N];
  int tid = threadIdx.x;
  if (tid < 128) sh[tid] = a[tid];
  if (tid + 128 < N) sh[tid + 128] = a[tid + 128];
  __syncthreads();

  for (int len = 128; len > 0; len >>= 1) {
    /* ref: k starts 0, ++k per start; len=128 uses zetas[1], len=64 uses 2,3, etc. */
    int k_start = (len == 128) ? 1 : (256 / len);
    for (int b = tid; b < 128; b += blockDim.x) {
      int start_index = b / len;
      int j = start_index * (2 * len) + (b % len);
      int32_t zeta = d_zetas[k_start + start_index];
      int32_t t = montgomery_reduce((int64_t)zeta * (int64_t)sh[j + len]);
      sh[j + len] = sh[j] - t;
      sh[j] = sh[j] + t;
    }
    __syncthreads();
  }

  if (tid < 128) a[tid] = sh[tid];
  if (tid + 128 < N) a[tid + 128] = sh[tid + 128];
}

/* One block per polynomial. In-place invNTT matching ref. */
__global__ void kernel_invntt(int32_t *polys, int npolys) {
  const int32_t f = 41978;
  int poly_idx = blockIdx.x;
  if (poly_idx >= npolys) return;
  int32_t *a = polys + poly_idx * N;

  __shared__ int32_t sh[N];
  int tid = threadIdx.x;
  if (tid < 128) sh[tid] = a[tid];
  if (tid + 128 < N) sh[tid + 128] = a[tid + 128];
  __syncthreads();

  for (int len = 1; len < N; len <<= 1) {
    /* ref invntt: k starts 256, then --k per start; for this len, k = 256/len - 1 - start_index */
    for (int b = tid; b < 128; b += blockDim.x) {
      int start_index = b / len;
      int j = start_index * (2 * len) + (b % len);
      int k = (256 / len) - 1 - start_index;
      int32_t zeta = -d_zetas[k];
      int32_t t = sh[j];
      sh[j] = t + sh[j + len];
      sh[j + len] = montgomery_reduce((int64_t)zeta * (int64_t)(t - sh[j + len]));
    }
    __syncthreads();
  }

  if (tid < N)
    sh[tid] = montgomery_reduce((int64_t)f * (int64_t)sh[tid]);
  __syncthreads();
  if (tid < 128) a[tid] = sh[tid];
  if (tid + 128 < N) a[tid + 128] = sh[tid + 128];
}

/* shiftl: a[i] <<= D. One block per polynomial, N threads. */
__global__ void kernel_shiftl(int32_t *polys, int npolys) {
  int poly_idx = blockIdx.x;
  if (poly_idx >= npolys) return;
  int32_t *a = polys + poly_idx * N;
  int i = threadIdx.x;
  if (i < N) a[i] <<= D;
}

/* Kernel: compute w1 = A*z - c*t1 in NTT domain.
 * Layout: d_mat [K][L][N], d_z [L][N], d_cp [N], d_t1 [K][N], d_w1 [K][N].
 * Each block handles one row i: w1[i] = sum_j pointwise(mat[i][j], z[j]) - pointwise(cp, t1[i]).
 */
__global__ void kernel_w1_from_az_ct(const int32_t *d_mat, const int32_t *d_z,
                                     const int32_t *d_cp, const int32_t *d_t1,
                                     int32_t *d_w1) {
  int i = blockIdx.x;
  int k = threadIdx.x;
  if (i >= K || k >= N) return;

  int64_t sum = 0;
  for (int j = 0; j < L; j++) {
    int mat_idx = ((i * L) + j) * N + k;
    int z_idx = j * N + k;
    sum += (int64_t)d_mat[mat_idx] * (int64_t)d_z[z_idx];
  }
  int32_t az_k = montgomery_reduce(sum);
  int32_t ct_k = montgomery_reduce((int64_t)d_cp[k] * (int64_t)d_t1[i * N + k]);
  d_w1[i * N + k] = az_k - ct_k;
}

/* Reduce32 on w1 (K polynomials). One block per polynomial, N threads. */
__global__ void kernel_reduce(int32_t *d_w1) {
  int poly_idx = blockIdx.x;
  int i = threadIdx.x;
  if (poly_idx < K && i < N)
    d_w1[poly_idx * N + i] = reduce32_dev(d_w1[poly_idx * N + i]);
}

/* Flattened layout helpers: mat is row-major K*L*N, z is L*N, t1/w1 are K*N. */
static void host_to_device_verify(const polyvecl mat[K], const polyvecl *z,
                                  const poly *cp, const polyveck *t1,
                                  int32_t *d_mat, int32_t *d_z, int32_t *d_cp,
                                  int32_t *d_t1, cudaStream_t st) {
  for (int i = 0; i < K; i++)
    for (int j = 0; j < L; j++)
      cudaMemcpyAsync(d_mat + (i * L + j) * N, mat[i].vec[j].coeffs, N * sizeof(int32_t), cudaMemcpyHostToDevice, st);
  for (int j = 0; j < L; j++)
    cudaMemcpyAsync(d_z + j * N, z->vec[j].coeffs, N * sizeof(int32_t), cudaMemcpyHostToDevice, st);
  cudaMemcpyAsync(d_cp, cp->coeffs, N * sizeof(int32_t), cudaMemcpyHostToDevice, st);
  for (int i = 0; i < K; i++)
    cudaMemcpyAsync(d_t1 + i * N, t1->vec[i].coeffs, N * sizeof(int32_t), cudaMemcpyHostToDevice, st);
}

static void device_to_host_w1(int32_t *d_w1, polyveck *w1, cudaStream_t st) {
  for (int i = 0; i < K; i++)
    cudaMemcpyAsync(w1->vec[i].coeffs, d_w1 + i * N, N * sizeof(int32_t), cudaMemcpyDeviceToHost, st);
}

#ifdef GPU_VERIFY_BENCHMARK
static double gpu_benchmark_total_ms = 0;
static int gpu_benchmark_n_calls = 0;
static void gpu_benchmark_atexit(void) {
  fprintf(stderr, "[GPU_VERIFY_BENCHMARK] NTT+pointwise+invNTT kernel only: total %.3f ms over %d calls, avg %.6f ms/call\n",
          gpu_benchmark_total_ms, gpu_benchmark_n_calls,
          gpu_benchmark_n_calls ? gpu_benchmark_total_ms / gpu_benchmark_n_calls : 0);
}
#endif

extern "C" {

int gpu_compute_wprime(polyveck *w1, const polyvecl mat[K], const polyvecl *z,
                      const poly *cp, const polyveck *t1) {
#if DILITHIUM_MODE != 2
  (void)w1; (void)mat; (void)z; (void)cp; (void)t1;
  return -1; /* only Dilithium2 supported */
#else
  int32_t *d_mat = NULL, *d_z = NULL, *d_cp = NULL, *d_t1 = NULL, *d_w1 = NULL;
  cudaStream_t st;
  cudaStreamCreate(&st);

#ifdef GPU_VERIFY_BENCHMARK
  static int atexit_registered = 0;
  if (!atexit_registered) {
    atexit_registered = 1;
    atexit(gpu_benchmark_atexit);
  }
#endif

  CUDA_CHECK(cudaMemcpyToSymbol(d_zetas, h_zetas, N * sizeof(int32_t)));

  CUDA_CHECK(cudaMalloc(&d_mat, K * L * N * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_z, L * N * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_cp, N * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_t1, K * N * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_w1, K * N * sizeof(int32_t)));

  host_to_device_verify(mat, z, cp, t1, d_mat, d_z, d_cp, d_t1, st);

  /* Ref uses mat in standard form; NTT only z, cp, and t1 (after shiftl). */
  kernel_shiftl<<<K, N, 0, st>>>(d_t1, K);

  int n_ntt = L + 1 + K;  /* z (L), cp (1), t1 (K) */
  int32_t *d_all_ntt = NULL;
  CUDA_CHECK(cudaMalloc(&d_all_ntt, n_ntt * N * sizeof(int32_t)));

  size_t off = 0;
  for (int j = 0; j < L; j++) {
    cudaMemcpyAsync(d_all_ntt + off * N, d_z + j * N, N * sizeof(int32_t), cudaMemcpyDeviceToDevice, st);
    off++;
  }
  cudaMemcpyAsync(d_all_ntt + off * N, d_cp, N * sizeof(int32_t), cudaMemcpyDeviceToDevice, st);
  off++;
  for (int i = 0; i < K; i++) {
    cudaMemcpyAsync(d_all_ntt + off * N, d_t1 + i * N, N * sizeof(int32_t), cudaMemcpyDeviceToDevice, st);
    off++;
  }

  cudaEvent_t evt_start, evt_stop;
#ifdef GPU_VERIFY_BENCHMARK
  cudaEventCreate(&evt_start);
  cudaEventCreate(&evt_stop);
  cudaEventRecord(evt_start, st);
#endif

  kernel_ntt<<<n_ntt, 128, 0, st>>>(d_all_ntt, n_ntt);
  CUDA_CHECK(cudaGetLastError());

  off = 0;
  for (int j = 0; j < L; j++) {
    cudaMemcpyAsync(d_z + j * N, d_all_ntt + off * N, N * sizeof(int32_t), cudaMemcpyDeviceToDevice, st);
    off++;
  }
  cudaMemcpyAsync(d_cp, d_all_ntt + off * N, N * sizeof(int32_t), cudaMemcpyDeviceToDevice, st);
  off++;
  for (int i = 0; i < K; i++) {
    cudaMemcpyAsync(d_t1 + i * N, d_all_ntt + off * N, N * sizeof(int32_t), cudaMemcpyDeviceToDevice, st);
    off++;
  }

  /* mat stays in standard form; z, cp, t1 now in NTT (matches ref). */
  kernel_w1_from_az_ct<<<K, N, 0, st>>>(d_mat, d_z, d_cp, d_t1, d_w1);
  CUDA_CHECK(cudaGetLastError());

  kernel_reduce<<<K, N, 0, st>>>(d_w1);

  kernel_invntt<<<K, 128, 0, st>>>(d_w1, K);
  CUDA_CHECK(cudaGetLastError());

#ifdef GPU_VERIFY_BENCHMARK
  cudaEventRecord(evt_stop, st);
  cudaEventSynchronize(evt_stop);
  float kernel_ms = 0.0f;
  cudaEventElapsedTime(&kernel_ms, evt_start, evt_stop);
  cudaEventDestroy(evt_start);
  cudaEventDestroy(evt_stop);
  gpu_benchmark_total_ms += (double)kernel_ms;
  gpu_benchmark_n_calls++;
#endif

  device_to_host_w1(d_w1, w1, st);
  CUDA_CHECK(cudaStreamSynchronize(st));

  cudaFree(d_all_ntt);
  cudaFree(d_w1);
  cudaFree(d_t1);
  cudaFree(d_cp);
  cudaFree(d_z);
  cudaFree(d_mat);
  cudaStreamDestroy(st);
  return 0;
#endif
}

} /* extern "C" */
