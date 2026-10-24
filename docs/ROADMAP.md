# Development roadmap

## Initial implementation

- [x] CPU reference and naive/constant/tiled CUDA kernels.
- [x] Pageable/pinned combinations for all three kernels.
- [x] Rectangular images, explicit zero borders, and partial tiles.
- [x] Five filter families, grayscale conversion, PNG output, and image CLI.
- [x] RAII memory/event cleanup and checked CUDA calls.
- [x] CPU correctness tests and GPU/CPU comparison suite.
- [x] Kernel-only and synchronous-pipeline benchmarks with CSV metadata.
- [x] Native RTX 5060 Ti build settings, PTX alternative, and CPU-only option.
- [x] Validate the PTX build on the local RTX 5060 Ti with all 240 correctness cases.

## Next: establish the native RTX 5060 Ti baseline

- [ ] Install/select CUDA Toolkit 12.8+ and compile native `sm_120` kernels.
- [ ] Run all CPU and GPU tests with the native build.
- [ ] Run NVIDIA Compute Sanitizer memcheck and synccheck on boundary cases.
- [ ] Compare native and PTX builds without including first-use JIT time.
- [ ] Record driver, toolkit, compiler, clocks, image dimensions, and filter.
- [ ] Repeat benchmark processes; report medians and variability, not a single best run.
- [ ] Publish validated local measurements with the exact source commit and commands.

## Next: explain bottlenecks

- [ ] Use Nsight Compute to inspect memory throughput, cache behavior,
      occupancy, register use, and shared-memory behavior.
- [ ] Use Nsight Systems to separate CPU staging, transfers, synchronization,
      and kernel execution across repeated images.
- [ ] Compare block sizes and tile sizes while retaining full correctness tests.
- [ ] Test larger filters and different image dimensions before generalizing results.

## Later: extend the implementation

- [ ] Reusable processing contexts and persistent buffers for repeated images.
- [ ] Optional CUDA streams and asynchronous transfers, followed by overlap measurements.
- [ ] Separable Gaussian filtering and comparison against the direct 3x3 implementation.
- [ ] Configurable filter radius and explicitly defined border policies.
- [ ] RGB channel processing and an image-batch mode.
- [ ] Automated CPU CI and a separately provisioned GPU test workflow.

No driver, Linux-kernel module, OpenGL/DirectX path, video pipeline, asynchronous
overlap, or profiler result is implemented by this initial version.
