# ImageProcessing

C++17 and CUDA image filtering for the **NVIDIA GeForce RTX 5060 Ti 16 GB**.
This is a separate implementation based on the CPU convolution, CUDA kernels,
memory experiments, and five filter families in
[xFilters by Tushar Gautam](https://github.com/tgautam03/xFilters).
The upstream MIT copyright notice is retained in `LICENSE`.

**Validated locally:** the CUDA 12.0 PTX compatibility build runs on the RTX 5060 Ti.
All 240 GPU/CPU correctness comparisons and the CPU suite passed. Native `sm_120`
compilation awaits CUDA Toolkit 12.8+. Full evidence is in [VALIDATION.md](VALIDATION.md).

## Implemented

- CPU reference implementation and three CUDA kernels: direct global-memory
  convolution, constant-memory filter coefficients, and shared-memory tiling
  with constant-memory coefficients.
- Pageable and pinned host buffers for each CUDA kernel: **six GPU configurations**.
- Five fixed 3x3 filters: `sharpen`, `high-pass`, `low-pass`, `gaussian`, and `d-gaussian`.
- JPEG/PNG/PNM and other stb-supported image inputs, RGB-to-grayscale conversion,
  and grayscale PNG outputs. Synthetic input makes benchmarks reproducible without
  downloading an image.
- Rectangular images, zero-valued out-of-image pixels, one-pixel dimensions,
  and image dimensions that are not multiples of the 16x16 CUDA block size.
- CPU/GPU comparisons on raw floating-point outputs before PNG clamping,
  CUDA error checks, RAII buffer/event cleanup, and CSV benchmark export.
- Native `sm_120` build configuration, an optional older-toolkit PTX build,
  and a CPU-only build.

The operation follows the upstream convention: filter coefficients are applied
in stored order (**cross-correlation**, commonly called convolution in image
processing). Coefficients are not reversed. `sharpen` uses the upstream formula
at alpha 0.8; `low-pass` uses 1/9 weights. `gaussian` uses the symmetric normalized
1-2-1 / 2-4-2 / 1-2-1 kernel, correcting the asymmetric upstream implementation.
`d-gaussian` retains the upstream `d_Gaussian` coefficient matrix; the name is
preserved for continuity and is not a claim of a newly derived Gaussian derivative.

## Hardware and CUDA setup

The RTX 5060 Ti has CUDA compute capability **12.0**, compiled as `sm_120`.
Use **CUDA Toolkit 12.8 or newer** for a native build. Download a toolkit compatible
with your Linux distribution and installed driver from
[NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads), or use the
[CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) to select 12.8.x.
Choose the toolkit packages if your existing NVIDIA driver already works; installing
a toolkit does not inherently require replacing a working display driver.

Check the compiler and GPU separately:

```bash
nvidia-smi
nvcc --version
```

The `CUDA Version` reported by `nvidia-smi` indicates driver compatibility, not
the installed `nvcc` compiler version. During initial development this machine
had driver 595.99.02 and `nvcc` 12.0.140. CUDA 12.0 cannot generate native `sm_120` code.
The default build deliberately reports this mismatch and explains the options.

After installing a newer toolkit, select its compiler explicitly if the shell
still finds `/usr/bin/nvcc`. For a CUDA 12.8 installation at its usual location:

```bash
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
cmake --build build-native -j
ctest --test-dir build-native --output-on-failure
```

Use the actual toolkit installation path. CMake defaults to architecture `120`;
no dependency download occurs during configuration. Requirements are CMake 3.24+,
a C++17 compiler, and a host compiler supported by the selected CUDA toolkit.

### Optional CUDA 12.0-12.7 compatibility build

Older CUDA toolkits can emit forward-compatible PTX, which a Blackwell-capable
driver can JIT compile for this GPU. This is an explicit alternative to native
compilation, using `86-virtual` (PTX only), following
[NVIDIA's Blackwell compatibility guidance](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/).
The driver must support both the installed GPU and the generated PTX version.
With the default warmup, initial JIT compilation is excluded from timed kernel batches.

On this machine GCC 12 is available and can be selected for the CUDA 12.0 host compiler:

```bash
cmake -S . -B build-ptx -DCMAKE_BUILD_TYPE=Release \
  -DIMAGEPROCESSING_LEGACY_PTX=ON \
  -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_CUDA_HOST_COMPILER=g++-12
cmake --build build-ptx -j
ctest --test-dir build-ptx --output-on-failure
```

The executable reports the build mode, CUDA compiler version, actual GPU, compute
capability, CUDA runtime, and driver API version. The CSV preserves this metadata.
Prefer native compilation for subsequent optimization work.

### CPU-only build

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release \
  -DIMAGEPROCESSING_ENABLE_CUDA=OFF
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
./build-cpu/image_filters --backend cpu --filter gaussian --iterations 10
```

A GPU request fails visibly when CUDA or a GPU is unavailable; it does not silently
produce CPU timings under a GPU label. CPU-only tests remain available while a
new toolkit is being installed.

## Run filtering and benchmarks

Run from this repository's root. Replace `build-ptx` with `build-native` as appropriate.

```bash
# Deterministic 2048 x 1328 image, CPU and all six GPU configurations.
./build-ptx/image_filters --backend all --filter high-pass \
  --iterations 100 --warmup 10 --csv output/high-pass.csv

# Your own image, including rectangular portraits or landscapes.
./build-ptx/image_filters --input /path/to/photo.jpg --filter gaussian \
  --backend all --output output/gaussian --csv output/gaussian.csv

# GPU variants only; a CPU reference is still computed for correctness.
./build-ptx/image_filters --backend gpu --width 1920 --height 1080 \
  --filter sharpen --iterations 100 --csv output/sharpen.csv
```

Use `--help` for all options. Each GPU configuration is compared with the CPU
reference using `abs(error) <= 1e-5 + 1e-5 * abs(reference)`. Nonfinite results fail.
PNG pixels are clamped to [0, 1] only for visualization. A failed GPU comparison
returns exit code 2, marks its CSV row as incorrect, and does not save that GPU PNG.

## Benchmark definitions

All reported durations are **milliseconds**.

| Column | Definition |
| --- | --- |
| `allocation_ms` | One wall-clock measurement of host staging and device buffer allocation, after CUDA context initialization; deallocation excluded. |
| `kernel_ms` | CUDA-event batch duration divided by iterations, after warmup and initial uploads; excludes transfers and allocation. |
| `h2d_ms` | Mean synchronized wall-clock input upload duration in the pipeline trials; excludes the preceding host staging copy. |
| `filter_upload_ms` | Mean synchronized wall-clock filter upload duration. |
| `d2h_ms` | Mean wall-clock blocking output download duration; excludes the subsequent host staging copy. |
| `pipeline_ms` | Mean wall-clock synchronous pipeline: host input staging, image/filter uploads, one kernel, synchronization, download, and host result copy. Includes timing instrumentation; excludes allocation, image I/O, grayscale conversion, validation, and PNG writing. |
| `kernel_gflops` | Nominal 18 operations per output pixel divided by kernel time; boundary work is approximated. |

CPU `kernel_ms` and `pipeline_ms` both time the CPU convolution function, including
its output-vector allocation, divided by iterations. They do not include image I/O.
The CPU row is the reference, so its correctness status is not an independent test.
GPU transfer measurements use wall-clock timing rather than interpreting CUDA
events around pageable memory copies as full host-visible transfer latency.
Kernel and pipeline measurements are separate trials; stage values should not
be summed with the kernel-only batch to reconstruct `pipeline_ms`.

The default warmup runs before the measured GPU batch; `--warmup 0` can include
first-use overhead and is not recommended for performance studies. Configurations run sequentially in a
fixed order. Small differences can reflect clock changes, display activity, or
measurement noise. Shared memory, constant memory, and pinned memory are experiments;
none is assumed faster before measurements. Pinned allocation is more expensive
and uses page-locked system RAM, so its allocation cost is reported separately.
Constant coefficients are shared at device scope; concurrent calls with different
filters are not supported in this version.

The repository does not import upstream RTX 3090 measurements as local results.
See [VALIDATION.md](VALIDATION.md) for the checks actually performed on this machine.
Use fresh, passing CSV results when writing resume performance claims.

## Layout

```text
include/imageprocessing/  Image/filter types and CPU/GPU public interfaces
src/convolution.cpp      CPU reference, filter coefficients, comparisons
src/gpu.cu               CUDA kernels, buffers, error checks, timing
src/image_io.cpp         stb decoding, grayscale conversion, PNG encoding
src/main.cpp             CLI and CSV reporting
tests/                   CPU checks and 240 GPU/CPU comparisons
third_party/             Vendored stb image I/O headers and attribution
docs/                    Development roadmap
```

The CPU tests cover rectangular indexing, known border sums, asymmetric-filter
orientation, five filters, invalid inputs, nonfinite comparisons, and PNG I/O.
GPU tests compare every filter and every GPU configuration across eight image
shapes, including thin images and partial blocks. GPU tests report `Skipped`
when no CUDA GPU is accessible; a skipped test does not establish GPU correctness.

## Future plans

The initial scope reproduces the upstream learning progression: CPU baseline,
naive CUDA, constant memory, shared memory, and pinned-memory combinations.
The next steps are tracked in [docs/ROADMAP.md](docs/ROADMAP.md): native `sm_120`
validation, repeatable performance studies, profiler-guided optimization, and
streaming experiments. These are plans, not completed features or resume claims.
