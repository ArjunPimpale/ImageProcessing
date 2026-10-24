# Validation record

Checks performed on October 24, 2026, before the initial commit was published.

## Environment

- NVIDIA GeForce RTX 5060 Ti, 16 GB model.
- NVIDIA driver 595.99.02; `nvidia-smi` reports driver CUDA compatibility 13.2.
- Active CUDA compiler: `/usr/bin/nvcc`, CUDA Toolkit 12.0.140.
- GCC 12 is available for the CUDA 12.0 compatibility build; GCC 13 is the system default.

Native `sm_120` compilation requires a newer toolkit and is not established by
a successful PTX compatibility build. No upstream RTX 3090 timings are reused.

## Passed checks

1. Release CPU-only build with GCC 13.3.0 and the CPU correctness test.
2. Release PTX-only CUDA build with nvcc 12.0.140 and GCC 12.4.0.
   `cuobjdump --list-ptx` shows the `sm_86` PTX entry, and `--list-elf` reports
   no embedded device ELF/cubin. The driver JIT compiles this PTX for the RTX 5060 Ti.
3. Both CTest tests passed with GPU access: CPU correctness plus **240 GPU/CPU
   comparisons**, covering five filters, six configurations, and eight shapes.
   The GPU test ran on the RTX 5060 Ti; it was not skipped.
4. CLI run on seeded 2048 x 1328 input, `high-pass`, 100 measured iterations,
   and 10 warmup iterations. All six GPU rows passed validation.
5. Real JPEG run on the upstream local `data/8k.jpg` image (2048 x 1328),
   `gaussian`, five measured iterations, two warmup iterations. All six GPU
   configurations passed comparison and wrote PNGs. That external JPEG is not
   redistributed in this repository.
6. CPU PNG encoding/decoding round-trip test and visual inspection of a real-image
   grayscale Gaussian output.
7. CLI error paths rejected zero iterations, an unknown filter, a missing input
   file, and a GPU request from a CPU-only build.
8. Default native configuration correctly rejected CUDA 12.0 with an actionable
   CUDA Toolkit 12.8+ download message.

## Reproduction

Use the PTX build commands in the README, then:

```bash
ctest --test-dir build-ptx --output-on-failure -V
./build-ptx/image_filters --backend all --filter high-pass \
  --iterations 100 --warmup 10 --output output/rtx5060ti-high-pass \
  --csv output/rtx5060ti-high-pass.csv
```

The initial raw benchmark is retained in
[`benchmarks/rtx5060ti_ptx_high-pass.csv`](benchmarks/rtx5060ti_ptx_high-pass.csv).
It is one development run with a desktop display active, not a controlled
multi-run performance study. See the README for timing boundaries and caveats.

## Remaining validation

- Native `sm_120` compilation and execution with CUDA Toolkit 12.8+.
- Compute Sanitizer memcheck/synccheck using a Blackwell-capable tool version.
- Repeated processes, confidence/variability reporting, and Nsight profiling.
- GPU platforms other than the local RTX 5060 Ti.
