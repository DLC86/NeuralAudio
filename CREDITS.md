# Credits and contributors

NeuralAudio was created by [Mike Oliphant](https://github.com/mikeoliphant).
See the [upstream NeuralAudio contributors](https://github.com/mikeoliphant/NeuralAudio/graphs/contributors).

## Contributions to this fork

- **[@IllusionFpv](https://github.com/IllusionFpv)** — contributed the A2/HyperWaveNet per-sample weight-ramp optimization using block GEMMs and Raspberry Pi 5 performance benchmarks in [PR #1](https://github.com/DLC86/NeuralAudio/pull/1). The GEMM-based smoothing optimization has been integrated into this fork. The separate off-audio-thread conditioning proposal in that PR has **not** been integrated.

## Third-party dependencies

Depending on how it is compiled, this repository uses code from a number of sources. Please see the individual repositories for license information.

- https://github.com/sdatkinson/NeuralAmpModelerCore
- https://github.com/jatinchowdhury18/RTNeural
- https://gitlab.com/libeigen/eigen
- https://github.com/nlohmann/json
