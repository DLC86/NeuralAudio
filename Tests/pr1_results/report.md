# NeuralAudio PR #1 — independent A/B/C validation

Three builds on the same GitHub-hosted Linux x86-64 runner, same compiler and synthetic fixture audio.
Versions: baseline main; GEMM commit 7f5ffd0; full PR 41807a0.
All variants use **100 ms update cadence and 100 ms linear sample-by-sample smoothing**.

## Summary

| Variant | Comparisons | Worst residual (dB rel. signal) | Max abs sample delta | Median CPU ratio while moving | Median CPU ratio idle |
|---|---:|---:|---:|---:|---:|
| gemm | 32 | -114.1 | 1.07e-06 | 0.528x | 1.002x |
| full | 32 | -114.1 | 1.07e-06 | 0.519x | 0.991x |

## Audio caveats

Residual comparisons are against baseline, aligned by construction with the same input and event indices, not manually aligned. The metrics include worst absolute sample difference, residual energy, first-difference residual and event-adjacent peak differences. These are not perceptual listening tests.

## Moving controls: per-model/per-buffer timings

| Version | Model | Quality | Scenario | Block | Base CPU % | New CPU % | CPU ratio | P99 deadline % (base/new) | Worst residual dBc |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| full | hyper_a2_low_rank | qfull | fast | b128 | 29.6 | 8.9 | 0.30x | 31.3/10.4 | -117.3 |
| full | hyper_a2_low_rank | qlite | fast | b128 | 5.0 | 2.5 | 0.51x | 5.8/3.2 | -130.6 |
| full | hyper_a2_low_rank | qfull | fast | b32 | 29.9 | 9.9 | 0.33x | 33.9/13.5 | -117.3 |
| full | hyper_a2_low_rank | qlite | fast | b32 | 5.0 | 2.7 | 0.53x | 6.7/4.1 | -130.6 |
| full | hyper_a2_low_rank | qfull | slow | b128 | 29.4 | 8.6 | 0.29x | 31.7/10.4 | -114.1 |
| full | hyper_a2_low_rank | qlite | slow | b128 | 5.1 | 2.5 | 0.49x | 8.1/3.3 | -130.6 |
| full | hyper_a2_low_rank | qfull | slow | b32 | 29.7 | 9.9 | 0.33x | 34.8/13.1 | -114.1 |
| full | hyper_a2_low_rank | qlite | slow | b32 | 5.0 | 2.7 | 0.53x | 6.5/4.0 | -130.6 |
| full | hyper_a2_low_rank | qfull | step | b128 | 11.6 | 7.3 | 0.63x | 30.6/9.3 | -123.3 |
| full | hyper_a2_low_rank | qlite | step | b128 | 2.2 | 1.7 | 0.74x | 6.8/2.9 | -136.1 |
| full | hyper_a2_low_rank | qfull | step | b32 | 13.4 | 9.4 | 0.70x | 32.6/11.4 | -123.0 |
| full | hyper_a2_low_rank | qlite | step | b32 | 2.3 | 1.8 | 0.79x | 5.2/3.0 | -136.2 |
| full | switch_a2 | qfull | fast | b128 | 29.7 | 8.7 | 0.29x | 31.5/10.4 | -132.8 |
| full | switch_a2 | qlite | fast | b128 | 4.9 | 2.5 | 0.51x | 5.7/2.9 | -134.5 |
| full | switch_a2 | qfull | fast | b32 | 30.0 | 9.9 | 0.33x | 34.7/13.5 | -132.8 |
| full | switch_a2 | qlite | fast | b32 | 5.0 | 2.6 | 0.52x | 6.7/4.1 | -134.5 |
| full | switch_a2 | qfull | slow | b128 | 29.4 | 8.9 | 0.30x | 31.8/10.4 | -133.4 |
| full | switch_a2 | qlite | slow | b128 | 4.9 | 2.5 | 0.51x | 5.4/3.0 | -133.5 |
| full | switch_a2 | qfull | slow | b32 | 29.7 | 9.8 | 0.33x | 33.3/13.6 | -133.4 |
| full | switch_a2 | qlite | slow | b32 | 5.0 | 2.6 | 0.53x | 6.4/4.0 | -133.5 |
| full | switch_a2 | qfull | step | b128 | 11.6 | 7.6 | 0.66x | 30.3/11.1 | -138.5 |
| full | switch_a2 | qlite | step | b128 | 2.2 | 1.7 | 0.76x | 5.3/2.8 | -139.6 |
| full | switch_a2 | qfull | step | b32 | 14.2 | 9.3 | 0.66x | 43.0/11.7 | -138.2 |
| full | switch_a2 | qlite | step | b32 | 2.3 | 1.8 | 0.79x | 6.0/3.0 | -139.6 |
| gemm | hyper_a2_low_rank | qfull | fast | b128 | 29.6 | 8.7 | 0.29x | 31.3/10.0 | -117.3 |
| gemm | hyper_a2_low_rank | qlite | fast | b128 | 5.0 | 2.5 | 0.51x | 5.8/3.3 | -130.6 |
| gemm | hyper_a2_low_rank | qfull | fast | b32 | 29.9 | 10.2 | 0.34x | 33.9/13.8 | -117.3 |
| gemm | hyper_a2_low_rank | qlite | fast | b32 | 5.0 | 2.7 | 0.54x | 6.7/4.1 | -130.6 |
| gemm | hyper_a2_low_rank | qfull | slow | b128 | 29.4 | 8.6 | 0.29x | 31.7/10.6 | -114.1 |
| gemm | hyper_a2_low_rank | qlite | slow | b128 | 5.1 | 2.5 | 0.49x | 8.1/3.0 | -130.6 |
| gemm | hyper_a2_low_rank | qfull | slow | b32 | 29.7 | 10.1 | 0.34x | 34.8/13.6 | -114.1 |
| gemm | hyper_a2_low_rank | qlite | slow | b32 | 5.0 | 2.7 | 0.54x | 6.5/4.1 | -130.6 |
| gemm | hyper_a2_low_rank | qfull | step | b128 | 11.6 | 7.3 | 0.63x | 30.6/9.7 | -123.3 |
| gemm | hyper_a2_low_rank | qlite | step | b128 | 2.2 | 1.7 | 0.75x | 6.8/2.9 | -136.1 |
| gemm | hyper_a2_low_rank | qfull | step | b32 | 13.4 | 9.4 | 0.70x | 32.6/11.6 | -123.0 |
| gemm | hyper_a2_low_rank | qlite | step | b32 | 2.3 | 1.8 | 0.80x | 5.2/3.1 | -136.2 |
| gemm | switch_a2 | qfull | fast | b128 | 29.7 | 8.9 | 0.30x | 31.5/14.0 | -132.8 |
| gemm | switch_a2 | qlite | fast | b128 | 4.9 | 2.6 | 0.53x | 5.7/3.7 | -134.5 |
| gemm | switch_a2 | qfull | fast | b32 | 30.0 | 10.0 | 0.33x | 34.7/13.3 | -132.8 |
| gemm | switch_a2 | qlite | fast | b32 | 5.0 | 2.7 | 0.53x | 6.7/4.1 | -134.5 |
| gemm | switch_a2 | qfull | slow | b128 | 29.4 | 8.7 | 0.29x | 31.8/10.5 | -133.4 |
| gemm | switch_a2 | qlite | slow | b128 | 4.9 | 2.5 | 0.51x | 5.4/3.2 | -133.5 |
| gemm | switch_a2 | qfull | slow | b32 | 29.7 | 10.1 | 0.34x | 33.3/15.4 | -133.4 |
| gemm | switch_a2 | qlite | slow | b32 | 5.0 | 2.7 | 0.54x | 6.4/4.0 | -133.5 |
| gemm | switch_a2 | qfull | step | b128 | 11.6 | 7.3 | 0.63x | 30.3/9.2 | -138.5 |
| gemm | switch_a2 | qlite | step | b128 | 2.2 | 1.7 | 0.77x | 5.3/2.8 | -139.6 |
| gemm | switch_a2 | qfull | step | b32 | 14.2 | 9.3 | 0.66x | 43.0/11.5 | -138.2 |
| gemm | switch_a2 | qlite | step | b32 | 2.3 | 1.8 | 0.80x | 6.0/3.0 | -139.6 |

Prepared-path comparisons: **32**; bit-identical audio to full PR direct path: **True**.

**Important:** CPU improvements reported here are not Raspberry Pi measurements. Real-hardware validation and multi-thread stress tests would still be needed before shipping.

Test-gate failures: 0

