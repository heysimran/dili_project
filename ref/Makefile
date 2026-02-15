CC ?= /usr/bin/cc
CFLAGS += -Wall -Wextra -Wpedantic -Wmissing-prototypes -Wredundant-decls \
  -Wshadow -Wvla -Wpointer-arith -O3 -fomit-frame-pointer
NISTFLAGS += -Wno-unused-result -O3 -fomit-frame-pointer
SOURCES = sign.c packing.c polyvec.c poly.c ntt.c reduce.c rounding.c
HEADERS = config.h params.h api.h sign.h packing.h polyvec.h poly.h ntt.h \
  reduce.h rounding.h symmetric.h randombytes.h gpu_verify.h
KECCAK_SOURCES = $(SOURCES) fips202.c symmetric-shake.c
KECCAK_HEADERS = $(HEADERS) fips202.h
# Stub needed by all targets using sign.c (which calls gpu_compute_wprime)
GPU_STUB = gpu_verify_stub.c

# CUDA (optional): nvcc for .cu; never use CC on .cu files
NVCC ?= nvcc
NVCCFLAGS = -O3 -I. -arch=sm_50

.PHONY: all speed speed_cuda shared clean

all: \
  test/test_dilithium2 \
  test/test_dilithium3 \
  test/test_dilithium5 \
  test/test_vectors2 \
  test/test_vectors3 \
  test/test_vectors5

nistkat: \
  nistkat/PQCgenKAT_sign2 \
  nistkat/PQCgenKAT_sign3 \
  nistkat/PQCgenKAT_sign5

speed: \
  test/test_mul \
  test/test_speed2 \
  test/test_speed3 \
  test/test_speed5

speed_cuda: speed test/test_speed2_cuda

shared: \
  libpqcrystals_dilithium2_ref.so \
  libpqcrystals_dilithium3_ref.so \
  libpqcrystals_dilithium5_ref.so \
  libpqcrystals_fips202_ref.so \

libpqcrystals_fips202_ref.so: fips202.c fips202.h
	$(CC) -shared -fPIC $(CFLAGS) -o $@ $<

libpqcrystals_dilithium2_ref.so: $(SOURCES) $(GPU_STUB) $(HEADERS) symmetric-shake.c
	$(CC) -shared -fPIC $(CFLAGS) -DDILITHIUM_MODE=2 \
	  -o $@ $(SOURCES) $(GPU_STUB) symmetric-shake.c

libpqcrystals_dilithium3_ref.so: $(SOURCES) $(GPU_STUB) $(HEADERS) symmetric-shake.c
	$(CC) -shared -fPIC $(CFLAGS) -DDILITHIUM_MODE=3 \
	  -o $@ $(SOURCES) $(GPU_STUB) symmetric-shake.c

libpqcrystals_dilithium5_ref.so: $(SOURCES) $(GPU_STUB) $(HEADERS) symmetric-shake.c
	$(CC) -shared -fPIC $(CFLAGS) -DDILITHIUM_MODE=5 \
	  -o $@ $(SOURCES) $(GPU_STUB) symmetric-shake.c

test/test_dilithium2: test/test_dilithium.c randombytes.c $(GPU_STUB) $(KECCAK_SOURCES) \
  $(KECCAK_HEADERS)
	$(CC) $(CFLAGS) -DDILITHIUM_MODE=2 \
	  -o $@ $< randombytes.c $(GPU_STUB) $(KECCAK_SOURCES)

test/test_dilithium3: test/test_dilithium.c randombytes.c $(GPU_STUB) $(KECCAK_SOURCES) \
  $(KECCAK_HEADERS)
	$(CC) $(CFLAGS) -DDILITHIUM_MODE=3 \
	  -o $@ $< randombytes.c $(GPU_STUB) $(KECCAK_SOURCES)

test/test_dilithium5: test/test_dilithium.c randombytes.c $(GPU_STUB) $(KECCAK_SOURCES) \
  $(KECCAK_HEADERS)
	$(CC) $(CFLAGS) -DDILITHIUM_MODE=5 \
	  -o $@ $< randombytes.c $(GPU_STUB) $(KECCAK_SOURCES)

test/test_vectors2: test/test_vectors.c $(GPU_STUB) $(KECCAK_SOURCES) \
  $(KECCAK_HEADERS)
	$(CC) $(CFLAGS) -DDILITHIUM_MODE=2 \
	  -o $@ $< $(GPU_STUB) $(KECCAK_SOURCES)

test/test_vectors3: test/test_vectors.c $(GPU_STUB) $(KECCAK_SOURCES) $(KECCAK_HEADERS)
	$(CC) $(CFLAGS) -DDILITHIUM_MODE=3 \
	  -o $@ $< $(GPU_STUB) $(KECCAK_SOURCES)

test/test_vectors5: test/test_vectors.c $(GPU_STUB) $(KECCAK_SOURCES) \
  $(KECCAK_HEADERS)
	$(CC) $(CFLAGS) -DDILITHIUM_MODE=5 \
	  -o $@ $< $(GPU_STUB) $(KECCAK_SOURCES)

test/test_speed2: test/test_speed.c test/speed_print.c test/speed_print.h \
  test/cpucycles.c test/cpucycles.h randombytes.c gpu_verify_stub.c \
  $(KECCAK_SOURCES) $(KECCAK_HEADERS)
	$(CC) $(CFLAGS) -DDILITHIUM_MODE=2 \
	  -o $@ $< test/speed_print.c test/cpucycles.c randombytes.c \
	  gpu_verify_stub.c $(KECCAK_SOURCES)

# CUDA-accelerated verification: .cu compiled ONLY by nvcc, never by cc
# 1) Compile gpu_verify.cu -> gpu_verify_cuda.o with nvcc
# 2) Compile C sources with gcc, link with gpu_verify_cuda.o and -lcudart
gpu_verify_cuda.o: gpu_verify.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -DDILITHIUM_MODE=2 -c -o $@ $<

test/test_speed2_cuda: test/test_speed.c test/speed_print.c test/speed_print.h \
  test/cpucycles.c test/cpucycles.h randombytes.c gpu_verify.cu gpu_verify_cuda.o \
  $(KECCAK_SOURCES) $(KECCAK_HEADERS)
	$(CC) $(CFLAGS) -DDILITHIUM_MODE=2 -o $@ test/test_speed.c test/speed_print.c \
	  test/cpucycles.c randombytes.c $(KECCAK_SOURCES) gpu_verify_cuda.o -lcudart -lstdc++

test/test_speed3: test/test_speed.c test/speed_print.c test/speed_print.h \
  test/cpucycles.c test/cpucycles.h randombytes.c $(GPU_STUB) $(KECCAK_SOURCES) \
  $(KECCAK_HEADERS)
	$(CC) $(CFLAGS) -DDILITHIUM_MODE=3 \
	  -o $@ $< test/speed_print.c test/cpucycles.c randombytes.c \
	  $(GPU_STUB) $(KECCAK_SOURCES)

test/test_speed5: test/test_speed.c test/speed_print.c test/speed_print.h \
  test/cpucycles.c test/cpucycles.h randombytes.c $(GPU_STUB) $(KECCAK_SOURCES) \
  $(KECCAK_HEADERS)
	$(CC) $(CFLAGS) -DDILITHIUM_MODE=5 \
	  -o $@ $< test/speed_print.c test/cpucycles.c randombytes.c \
	  $(GPU_STUB) $(KECCAK_SOURCES)

test/test_mul: test/test_mul.c randombytes.c $(GPU_STUB) $(KECCAK_SOURCES) $(KECCAK_HEADERS)
	$(CC) $(CFLAGS) -UDBENCH -o $@ $< randombytes.c $(GPU_STUB) $(KECCAK_SOURCES)

nistkat/PQCgenKAT_sign2: nistkat/PQCgenKAT_sign.c nistkat/rng.c nistkat/rng.h $(GPU_STUB) $(KECCAK_SOURCES) \
  $(KECCAK_HEADERS)
	$(CC) $(NISTFLAGS) -DDILITHIUM_MODE=2 \
	  -o $@ $< nistkat/rng.c $(GPU_STUB) $(KECCAK_SOURCES) $(LDFLAGS) -lcrypto

nistkat/PQCgenKAT_sign3: nistkat/PQCgenKAT_sign.c nistkat/rng.c nistkat/rng.h $(GPU_STUB) $(KECCAK_SOURCES) \
  $(KECCAK_HEADERS)
	$(CC) $(NISTFLAGS) -DDILITHIUM_MODE=3 \
	  -o $@ $< nistkat/rng.c $(GPU_STUB) $(KECCAK_SOURCES) $(LDFLAGS) -lcrypto

nistkat/PQCgenKAT_sign5: nistkat/PQCgenKAT_sign.c nistkat/rng.c nistkat/rng.h $(GPU_STUB) $(KECCAK_SOURCES) \
  $(KECCAK_HEADERS)
	$(CC) $(NISTFLAGS) -DDILITHIUM_MODE=5 \
	  -o $@ $< nistkat/rng.c $(GPU_STUB) $(KECCAK_SOURCES) $(LDFLAGS) -lcrypto

clean:
	rm -f *~ test/*~ *.gcno *.gcda *.lcov
	rm -f gpu_verify_cuda.o test/test_speed2_cuda
	rm -f libpqcrystals_dilithium2_ref.so
	rm -f libpqcrystals_dilithium3_ref.so
	rm -f libpqcrystals_dilithium5_ref.so
	rm -f libpqcrystals_fips202_ref.so
	rm -f test/test_dilithium2
	rm -f test/test_dilithium3
	rm -f test/test_dilithium5
	rm -f test/test_vectors2
	rm -f test/test_vectors3
	rm -f test/test_vectors5
	rm -f test/test_speed2
	rm -f test/test_speed3
	rm -f test/test_speed5
	rm -f test/test_mul
	rm -f nistkat/PQCgenKAT_sign2
	rm -f nistkat/PQCgenKAT_sign3
	rm -f nistkat/PQCgenKAT_sign5
