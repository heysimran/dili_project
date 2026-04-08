#ifndef GPU_VERIFY_H
#define GPU_VERIFY_H

#include "params.h"
#include "polyvec.h"
#include "poly.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute w1 = A*z - c*t1 on GPU (NTT domain).
 * GPU buffers are allocated once on first call and reused.
 * Returns 0 on success, -1 on error.
 */
int gpu_compute_wprime(polyveck *w1,
                       const polyvecl mat[K],
                       const polyvecl *z,
                       const poly *cp,
                       const polyveck *t1);

/**
 * Run 20 warmup iterations before benchmarking.
 * Also resets benchmark counters so warmup is excluded from results.
 */
void gpu_warmup(polyveck *w1,
                const polyvecl mat[K],
                const polyvecl *z,
                const poly *cp,
                const polyveck *t1);

/**
 * Free persistent GPU allocations. Optional — OS will reclaim anyway.
 */
void gpu_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* GPU_VERIFY_H */
