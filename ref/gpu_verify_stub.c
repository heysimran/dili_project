/**
 * Stub when building without CUDA: GPU path disabled, verification uses CPU only.
 */
#include "gpu_verify.h"

int gpu_compute_wprime(polyveck *w1,
                       const polyvecl mat[K],
                       const polyvecl *z,
                       const poly *cp,
                       const polyveck *t1)
{
  (void)w1;
  (void)mat;
  (void)z;
  (void)cp;
  (void)t1;
  return -1;
}
