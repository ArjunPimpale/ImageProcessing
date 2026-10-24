# Initial local benchmark

`rtx5060ti_ptx_high-pass.csv` is a raw development measurement from the local
RTX 5060 Ti 16 GB, using the CUDA 12.0.140 PTX compatibility build with GCC 12.4.0.
Input: deterministic synthetic grayscale image, seed 42, 2048 x 1328 pixels.
Filter: `high-pass`. Measured iterations: 100; warmup iterations: 10.
All GPU rows passed CPU-reference validation.

The same executable measures a separate kernel-only batch and repeated complete
synchronous pipelines; see the root README for exact timing definitions.
The display was active. The CSV is one run and should not be presented as a
general hardware comparison or a stable optimization percentage.

Reproduce from the repository root:

```bash
./build-ptx/image_filters --backend all --filter high-pass \
  --iterations 100 --warmup 10 --output output/rtx5060ti-high-pass \
  --csv output/rtx5060ti-high-pass.csv
```

The CSV is committed alongside the source used to produce it. Find its recording
commit with `git log -1 --format=%H -- benchmarks/rtx5060ti_ptx_high-pass.csv`.
