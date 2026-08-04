> [!IMPORTANT]
> This build documentation is specific only to NVIDIA Jetson (Jetson AGX Xavier / Orin / TX2) embedded platforms. You can find the build documentation for other architectures: [build.md](build.md).

# Build llama.cpp locally (for NVIDIA Jetson)

The main product of this project is the `llama` library. Its C-style interface can be found in [include/llama.h](../include/llama.h).

The project also includes many example programs and tools using the `llama` library. The examples range from simple, minimal code snippets to sophisticated sub-projects such as an OpenAI-compatible HTTP server.

**To get the code:**

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

## CUDA Build

Jetson platforms use NVIDIA Tegra SoCs with integrated GPU sharing unified memory. The build process is identical to regular CUDA builds but requires specifying the correct architecture.

### Minimal Build

```bash
cmake -S . -B build             \\
    -DCMAKE_BUILD_TYPE=Release  \\
    -DGGML_CUDA=ON              \\
    -DCMAKE_CUDA_ARCHITECTURES=72

cmake --build build --config Release -j $(nproc)
```

> **Note:** On some Jetson models (e.g., AGX Xavier), the CUDA compiler may not be in the default PATH. Specify it explicitly if needed:
> `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`

### Target Architecture Matrix

| Platform  | SoC              | CUDA Arch | GPU Cores | GPU Arch |
|-----------|------------------|-----------|-----------|----------|
| AGX Xavier | Tegra194         | 72        | 512       | Volta gen1 (SM72) |
| Xavier NX  | Tegra194 (binned)| 72        | 384       | Volta gen1 (SM72) |
| Orin AGX   | Tegra234         | 87        | 2048      | Ampere (SM87)     |
| Orin NX    | Tegra234 (binned)| 87        | 1024      | Ampere (SM87)     |
| Orin Nano  | Tegra234 (binned)| 87        | 512       | Ampere (SM87)     |
| TX2        | Tegra186         | 62        | 256       | Pascal (SM62)     |

### Recommended CMake Flags

Beyond the basic CUDA build, these flags are recommended for Jetson platforms:

```bash
cmake -S . -B build                                                        \\
    -DCMAKE_BUILD_TYPE=Release                                             \\
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc                         \\
    -DCMAKE_CUDA_ARCHITECTURES=72                                          \\
    -DGGML_CUDA=ON                                                         \\
    -DGGML_CUDA_FORCE_MMQ=OFF                                              \\
    -DGGML_CUDA_NO_VMM=ON                                                  \\
    -DGGML_CUDA_GRAPHS=ON                                                  \\
    -DGGML_CUDA_F16=ON                                                     \\
    -DGGML_CUDA_FA=ON                                                      \\
    -DGGML_CUDA_COMPRESSION_MODE=size                                      \\
    -DGGML_CUDA_NCCL=ON

cmake --build build --config Release -j $(nproc)
```

> Adjust `CMAKE_CUDA_ARCHITECTURES` to your platform (72 for Xavier, 87 for Orin, 62 for TX2).

### Flag Explanations

| Flag | Purpose |
|------|---------|
| `GGML_CUDA_FORCE_MMQ=OFF` | Use MMV kernel for matrix-vector ops — strongly recommended for small models (<7B) on Volta (see Performance section) |
| `GGML_CUDA_NO_VMM=ON` | Disable Virtual Memory Management — Jetson uses unified memory, VMM is unnecessary |
| `GGML_CUDA_GRAPHS=ON` | Enable CUDA Graph capture — reduces kernel launch overhead from the slower ARM CPU |
| `GGML_CUDA_FA=ON` | Enable flash attention — improves KV cache bandwidth utilization |
| `GGML_CUDA_F16=ON` | Use FP16 for intermediate matmul computations |
| `GGML_CUDA_COMPRESSION_MODE=size` | Optimize CUDA kernel binary size (reduces compilation time for limited storage) |
| `GGML_CUDA_NCCL=ON` | Enable NCCL support (for multi-GPU Jetson carriers) |

### Incremental Compilation (Time-Saving)

Full CUDA compilation on Jetson can take 30-60 minutes due to the number of template instantiations. For faster iteration:

```bash
# Only rebuild the CUDA kernel library
cmake --build build --target ggml-cuda -j2

# Only rebuild the server binary
cmake --build build --target llama-server -j2

# Only rebuild the benchmark tool
cmake --build build --target llama-bench -j2
```

## Cross-Compilation Strategy

On a multi-Jetson setup, compile on the more powerful device and scp the binaries:

1. Build on AGX Xavier (8-core): `cmake --build build -j8`
2. Stop services: `systemctl stop llama-server`
3. Backup old binaries: `cp -a build/bin build/bin.bak`
4. SCP: `scp -r build/bin nvidia@xavier-nx:/path/to/llama.cpp/build/`
5. Restart services

> Compiling directly on a lower-end Jetson (e.g., Xavier NX with 4-core Carmel) is significantly slower.

## Performance Tuning: GGML_CUDA_FORCE_MMQ for Small Models

### Background

By default, llama.cpp enables `GGML_CUDA_FORCE_MMQ=ON`, which forces all matrix operations to use the MMA (matrix-matrix) kernel path. This is optimal for large models (7B+) on desktop GPUs with high compute density.

On Jetson Volta GPUs (SM72), this default is **suboptimal for small models** (1B-3B active parameters). The root cause:

- Small models have small matrices
- MMA kernels are compute-bound on Volta's 384-512 cores
- Small matrices underutilize the MMA pipeline
- The MMV (matrix-vector) kernel path has better register utilization for these sizes

### Measured Impact

Benchmark on **Jetson Xavier NX** (384 Volta cores) with **MiniCPM5-1B Q4_K_M** (656 MB, 1.8B total params, ~1B active):

| Config | Prompt Processing | Token Generation |
|:------:|:-----------------:|:----------------:|
| FORCE_MMQ=ON (default) | ~30 t/s | 21-25 t/s |
| **FORCE_MMQ=OFF** | **46.7 t/s (+55%)** | 22.2 t/s |

**Key finding:** For small models (≤3B active) on Volta-class Jetson GPUs, **FORCE_MMQ=OFF provides a significant prompt processing speedup** (45-55%) with no degradation in token generation speed. The MMV kernel path is better suited to the small matrix dimensions these models present.

For larger models (7B+), the difference is negligible — both paths converge to similar performance.

### Memory Considerations

Jetson platforms use **unified memory** — GPU and CPU share the same pool. Key constraints:

| Platform | Unified Memory | Usable for Model |
|----------|:--------------:|:----------------:|
| AGX Xavier | ~29 GiB | Up to ~20 GiB models (with swap) |
| Xavier NX  | ~6.7 GiB      | Models up to ~4-5 GiB |
| Orin AGX  | ~56 GiB       | Up to ~40 GiB models |

- Use `-ngl N` to control GPU offload. `-ngl 99` offloads all layers.
- **Do not use `--mlock`** on Jetson — unified memory already pins GPU pages; mlock can cause regressions.
- For small memory devices (Xavier NX), 1B-3B Q4_K_M models are optimal (fits entirely on GPU at -ngl 24).

### Recommended Server Launch Parameters

For a typical small model deployment (1B-3B, Q4_K_M):

```bash
./bin/llama-server \
    -m /path/to/model.gguf \
    --port 8080 --host 127.0.0.1 \
    -ngl 99 \
    -t 4 \
    --batch-size 512 --ubatch-size 256 \
    --temp 0.5 \
    -c 4096 \
    --flash-attn on \
    -ctk q8_0 -ctv q8_0
```

## FAQ

### 1. Compilation fails with "No CMAKE_CUDA_COMPILER could be found"

Add `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` to your cmake command. This is common on Jetson where CUDA is installed but not in the default cached PATH.

### 2. Build keeps failing after cmake reconfiguration

Do not run `rm -rf build` unless you are prepared for a full 30-60 min rebuild. Instead, re-run cmake with changed flags — it will override cached values:

```bash
cmake -S . -B build <all-flags>   # overwrites cache
cmake --build build --target ggml-cuda -j2
```

### 3. GPU memory not releasing between model tests

Jetson unified memory does not always release GPU pages immediately. Either:
- Restart llama-server between model tests, or
- Use a single server instance and reload the model at the API level

### 4. Getting poor token generation speed (<5 t/s)

Check the following:
- Is the model too large for available memory? Check for swap thrashing (`vmstat 1`).
- Is `FORCE_CUBLAS=ON` in your build? This is a known performance killer on Jetson Volta. Rebuild with `FORCE_CUBLAS=OFF`.
- Are background nvcc processes running? Check with `ps aux | grep -E 'nvcc|cicc'`. Kill them and restart.

### 5. "CUDA error: invalid argument" during inference

Background compilation processes (nvcc/cicc) may be holding a CUDA context. Kill them:

```bash
pkill -9 nvcc cicc ccache
```

Then restart llama-server.

### 6. Is flash attention supported?

Yes, with `GGML_CUDA=ON` and `GGML_CUDA_FA=ON` in cmake. Note that flash attention on Volta (SM72) uses a specialized tile implementation — it still provides KV cache bandwidth benefits.

### 7. How do I benchmark inference speed?

```bash
./bin/llama-bench -m model.gguf -p 512 -n 128 -ngl 99 -t 2 -r 3
```

Parameters:
- `-p 512`: 512 token prompt processing
- `-n 128`: 128 token generation
- `-ngl 99`: full GPU offload
- `-t 2`: CPU threads (2 is optimal on Jetson Volta — GPU is bottleneck)
- `-r 3`: repeat 3 times for averaged results

### 8. Can I run this on RISC-V or s390x?

This guide is for Jetson Arm64/CUDA platforms only. See [build-riscv64-spacemit.md](build-riscv64-spacemit.md) or [build-s390x.md](build-s390x.md) for other architectures.

## Appendix A: Jetson Platform Support Matrix

| Feature              | SM72 (Volta) | SM87 (Ampere) |
|----------------------|:------------:|:-------------:|
| CUDA Graphs          | ✅           | ✅             |
| Flash Attention      | ✅ (tile)    | ✅             |
| FP16 matmul          | ✅           | ✅             |
| FORCE_MMQ=OFF (MMV)  | ✅ (optimal) | ✅             |
| Tensor Cores (gen1)  | ✅           | —             |
| Tensor Cores (gen3)  | —            | ✅             |

- ✅ = supported and verified
- 🚫 = unsupported

## Appendix B: Verified Model Configurations (SM72 Volta, 384 cores)

| Model | Quantization | Size | ngl | PP t/s | TG t/s | Notes |
|-------|:------------:|:----:|:---:|:------:|:------:|-------|
| MiniCPM5-1B | Q4_K_M | 656 MB | 24 | 46.7 | 22.2 | Optimal production model for Xavier NX |
| Granite-3.1-3B | Q4_K_M | 1.88 GB | 10 | 7.2 | 8.0 | Falls back to CPU for remaining layers |

> Results from Jetson Xavier NX (384 Volta cores, 6.7 GB unified memory). PP = prompt processing, TG = token generation.

Last Updated by **hernandez42** on Aug 4, 2026.
