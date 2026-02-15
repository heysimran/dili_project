# GPU-Accelerated Verification (Dilithium2 Ref)

This directory contains an optional **CUDA** acceleration for the verification bottleneck in the Dilithium2 reference implementation. Only the computation **w′ = A·z − c·t** (NTT → pointwise multiplication → inverse NTT) runs on the GPU; keygen and signing are unchanged and run on the CPU.

## Files added

| File | Purpose |
|------|--------|
| `gpu_verify.h` | Declares `gpu_compute_wprime()`. |
| `gpu_verify.cu` | CUDA implementation: NTT, pointwise mult, invNTT, and wrapper. |
| `gpu_verify_stub.c` | Stub when not linking CUDA: always returns -1 (CPU path). |
| `README_GPU_VERIFY.md` | This file. |

- **`sign.c`** is the only modified existing file: `crypto_sign_verify_internal()` calls `gpu_compute_wprime()` and falls back to the original CPU block if it returns non-zero.
- **`Makefile`** was updated to link the stub into `test_speed2` and to add the `test_speed2_cuda` target.

## Build (Linux / WSL / macOS with CUDA)

**CPU-only (default):**

```bash
cd ref
make speed
./test/test_speed2
```

**With CUDA (Dilithium2 verification on GPU):**

```bash
cd ref
make speed_cuda
# or: make test/test_speed2_cuda
./test/test_speed2_cuda
```

The CUDA build compiles `gpu_verify.cu` with `nvcc` only (never `cc`), then links with the C objects and `-lcudart`.

Requires `nvcc` and CUDA runtime (e.g. CUDA Toolkit). If the GPU path is unavailable, the code falls back to CPU verification automatically.

---

## Running on Google Colab (Tesla T4)

You can run the updated repo on Colab and use a **Tesla T4** GPU to test the CUDA-accelerated verification.

### 1. Enable GPU in Colab

1. Open a new notebook: [colab.research.google.com](https://colab.research.google.com).
2. **Runtime → Change runtime type**.
3. Set **Hardware accelerator** to **GPU** (Colab often assigns a T4 when available).
4. Save and run the cells below.

### 2. Clone the repo and build in Colab

Run these in Colab cells (or as a single script):

```bash
# Clone your repo (replace with your actual repo URL if different)
!git clone https://github.com/pq-crystals/dilithium.git
%cd dilithium/ref
```

If you already have the repo elsewhere (e.g. uploaded or a fork), adjust the path after `%cd` so you are inside the **ref** directory.

### 3. Install build tools (if needed)

Colab usually has `make` and `gcc`. For the CUDA build you need `nvcc`:

```bash
# Check if nvcc is available (it usually is when GPU is enabled)
!which nvcc
!nvcc --version
```

If `nvcc` is not found, install the CUDA toolkit in the notebook (example for Ubuntu):

```bash
# Only if nvcc is missing (Colab often has it already)
# !apt-get update && apt-get install -y nvidia-cuda-toolkit
```

### 4. Build and run

**CPU-only speed test (no GPU):**

```bash
%cd dilithium/ref
!make test/test_speed2
!./test/test_speed2
```

**CUDA-accelerated speed test (uses T4):**

```bash
%cd dilithium/ref
!make speed_cuda
!./test/test_speed2_cuda
```

### 5. Confirm GPU is used

To confirm a GPU is present and used:

```python
# In a Python cell
import subprocess
print(subprocess.check_output(["nvidia-smi"], text=True))
```

You should see a Tesla T4 (or another GPU) listed. The speed test prints timings; verification will use the GPU when `test_speed2_cuda` is run and a device is available.

### 6. Full Colab notebook example

Minimal sequence you can paste into Colab:

```python
# Cell 1: clone and go to ref
import os
os.chdir("/content")
if not os.path.isdir("dilithium"):
    os.system("git clone https://github.com/pq-crystals/dilithium.git")
os.chdir("dilithium/ref")

# Cell 2: build CUDA speed test
os.system("make speed_cuda")

# Cell 3: run (uses GPU if available)
os.system("./test/test_speed2_cuda")
```

**Using your own repo (with GPU code):**  
- Push your modified `dilithium` repo to GitHub, then in Colab run:  
  `!git clone https://github.com/YOUR_USERNAME/YOUR_DILITHIUM_REPO.git dilithium`  
  and use `dilithium/ref` as above.  
- Or zip the repo, upload to Colab, and run `!unzip -q dilithium.zip && cd dilithium/ref`.  
The upstream [pq-crystals/dilithium](https://github.com/pq-crystals/dilithium) does not include this GPU code; you need a copy that has the `ref/gpu_verify.*` files and the changes in `ref/sign.c` and `ref/Makefile`.

---

## Summary

- **CPU build:** `make speed` → `./test/test_speed2` (uses stub; verification is CPU-only).
- **GPU build:** `make speed_cuda` or `make test/test_speed2_cuda` → `./test/test_speed2_cuda` (verification uses GPU when available). The `.cu` file is compiled only by `nvcc`, never by `cc`.
- **Colab:** Enable GPU (Runtime → Change runtime type → GPU), clone the repo (or your fork with the GPU code), `cd ref`, then run the same make and run commands as above.
