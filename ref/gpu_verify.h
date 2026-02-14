#ifndef GPU_VERIFY_H
#define GPU_VERIFY_H

#include "params.h"
#include "poly.h"
#include "polyvec.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute w' = A·z − c·t (polynomial domain) on GPU.
 * Performs NTT -> pointwise mult -> invNTT for the verification bottleneck.
 * Uses ref types: mat (K×L), z (L), cp (1), t1 (K); writes result into w1 (K).
 * t1 is used as 2^D·t1 (shiftl applied inside).
 *
 * Returns 0 on success, -1 if GPU path unavailable or error.
 * On -1, caller should fall back to CPU verification.
 */
int gpu_compute_wprime(polyveck *w1,
                       const polyvecl mat[K],
                       const polyvecl *z,
                       const poly *cp,
                       const polyveck *t1);

#ifdef __cplusplus
}
#endif

#endif /* GPU_VERIFY_H */
