# NeuralAudio — DLC86 fork (v0.1.6)

This is [DLC86's fork](https://github.com/DLC86/NeuralAudio) of
[Mike Oliphant's NeuralAudio](https://github.com/mikeoliphant/NeuralAudio).
The upstream project is a fast, portable C++ library for neural amp models,
including static NAM WaveNet/A1/A2, LSTM, and RTNeural models. **Those original
capabilities are retained**; the changes below are specific to this fork.

The main addition is a portable **parametric NAM (.pnam) inference runtime**,
designed for desktop and embedded Linux hosts that need to expose and change
knobs, switches, model quality, and playback calibration without flattening
the neural model to a fixed setting.

For a complete comparison with upstream, see the
[upstream → fork diff](https://github.com/mikeoliphant/NeuralAudio/compare/main...DLC86:NeuralAudio:main).
For implementation details, see the [PNAM integration guide](PNAM.md) and
[standalone C example](Examples/PnamRender.c).

## What this fork adds or changes compared with upstream

### 1. Parametric model loading and backwards-compatible formats

- Loads **HyperWaveNet** (full and low-rank hypernetworks),
  **ConcatWaveNet**, and the original nested **HyperA2** Full/Lite format.
- Loads the newer **`ParametricSlimmableContainer`** name while retaining
  support for existing `HyperA2` exports. Both container names use the
  existing Full/Lite neural architecture internally; there is no retraining or
  weight conversion just to change the top-level name.
- Supports legacy **`ParametricOutputGain`** wrappers and the new optional
  **`extensions.linear_volume`** in a root-level `.pnam` object. Neural
  `config.params` and the network weights remain unchanged by this extension.
- Models without a volume extension continue to behave as before. Updated
  runtimes read old formats, but an older player that knows only `HyperA2`
  will **not** recognize `ParametricSlimmableContainer` automatically.

### 2. Runtime knobs, switches and Linear Volume DSP

- Enumerates continuous knobs and any number of categorical switches, with
  name, limits, default value, number of positions and position labels.
- Allows validated parameter changes **without recreating the neural network**;
  the parametric NAM Core updates conditioned weights and retains the running
  convolution history. Rejects invalid counts, out-of-range knob values and
  invalid switch positions rather than applying a partial update.
- Hypernetwork weight updates are rate-limited by model-sample time and
  **interpolated linearly at every sample directly inside the convolution kernels**
  while the knobs move, with just one active audio network. Developers configure
  the two intervals independently using `SetWeightUpdateIntervalMs()` and
  `SetWeightSmoothingMs()`, both defaulting to **100 ms**. The C API exposes
  `PnamSetWeightUpdateIntervalMs` and `PnamSetWeightSmoothingMs`.
  When controls are stationary, the optimized steady-state kernels remain in use.
  Full and Lite maintain independent update clocks.
- A measured **Linear Volume** control may be exposed beside the neural
  controls. Its monotonic PCHIP curve is applied **after** the network as a
  separate output-gain stage, with smoothed changes to avoid clicks. It does
  not modify hypernetwork input dimensions or weights.
- This is a **mono neural inference path**. Host-side audio I/O, external
  resampling and host automation synchronization remain the host's
  responsibility.

### 3. Full/Lite selection and the parametric fast path

- Supports Full (default) and Lite variants of nested parametric models
  through the quality-scaling interface. Both use the same visible controls.
- Uses the pinned parametric NAM Core fork with A2-specialized kernels and
  small-block, multi-frame weight reuse when the model topology supports it;
  otherwise it falls back to the general parametric backend.
- Prepares and processes **only the active size**, rather than running two
  networks at once. Switching Full/Lite and calling `Prewarm()` are
  **not real-time-safe** operations; perform these outside the audio callback.
- Audio runs on arbitrary block sizes up to the prepared maximum, with
  internal splitting for larger requests and no additional block accumulation.

### 4. Metadata, calibration and output loudness

- Adds `Raw`, `Calibrated`, `Normalized` and `Auto` output modes for
  **all NeuralAudio models**, with **Auto as the default**. For parametric
  models the selected output gain is applied **inside `Process()`**; for
  static NAM/Keras models the host applies the gain returned by
  `GetRecommendedOutputDBAdjustment()` as before. Do not apply it twice.
- Calibrated uses exported interface/output dBu metadata. Normalized targets
  **−18 dB** based on exported loudness. Auto chooses the appropriate mode
  using the model's `gear_type`; missing/nonfinite metadata safely falls
  back to Raw without inventing calibration values.
- Exposes requested versus effective output modes, calibration input level,
  and the full original metadata JSON, including custom and nested fields
  such as per-size training/validation statistics.

### 5. Expanded C API for portable/embedded hosts

The existing `NeuralAudioCAPI` gains:
`SetModelOutputMode`, `GetModelOutputMode`,
`GetModelEffectiveOutputMode` for all model types, and
`PnamSetWeightUpdateIntervalMs`, `PnamSetWeightSmoothingMs` for
parametric timing control, alongside
`CreateModelFromFileUtf8`, `IsParametric`, `PnamGetControlCount`,
`PnamGetControlName`, `PnamGetControlRange`, `PnamGetPositionCount`,
`PnamGetPositionName`, `PnamSetControls`, `PnamSetOutputMode`,
`PnamGetOutputMode`, `PnamGetEffectiveOutputMode`,
`PnamSetCalibrationLevelDBu`, `PnamHasLite`, `PnamSetLite` and
`PnamGetMetadataJson`.

The C header is usable from C and C++ and is designed for hardware hosts.
The new setters report success/failure; metadata queries support a
length-first buffer protocol and never write a truncated JSON result.
`Examples/PnamRender.c` demonstrates the complete C workflow.

### 6. Build, dependencies and regression coverage

- Adds `BUILD_PNAM` and `BUILD_PNAM_TESTS` CMake options, building the PNAM
  C++ runtime, C API and test/example executables without requiring JUCE,
  CUDA or training libraries. **Python 3 is required at CMake configure time**
  to apply the idempotent pinned-core interpolation patches; Python is not
  required at runtime. PNAM requires `BUILD_NAMCORE=ON` and
  `NAM_ENABLE_A2_FAST=ON`. The C++ runtime requires **C++20**.
- Pins a parametric NAM Core submodule rather than using upstream NeuralAudio's
  original NAM Core revision. The upstream dependencies and their licenses
  remain credited; always clone **recursively**.
- Adds checked-in synthetic Full/Lite and switch-coverage fixtures, C++ and C
  integration tests, plus CI for **Windows x64, Linux x64, macOS, and Linux
  ARM64**. CI verifies output calibration, audio against golden references,
  Full/Lite selection, switch combinations, metadata, and new/legacy Linear
  Volume layouts. These tests do **not** assert a device-specific real-time
  CPU benchmark.
- Includes portable PNAM-specific workflows and an optional output-gain
  preview packaging workflow. The main build workflow continues to exercise
  the original project; the PNAM workflow runs its own cross-platform suite.

### Build and integration quick start

```sh
git clone --recursive https://github.com/DLC86/NeuralAudio.git
cd NeuralAudio
cmake -S . -B build/pnam -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_RTNEURAL=OFF -DBUILD_INTERNAL_STATIC_WAVENET=OFF \
  -DBUILD_STATIC_INTERNAL_NAMA2=OFF -DBUILD_PNAM_TESTS=ON
cmake --build build/pnam --config Release --parallel 2
ctest --test-dir build/pnam -C Release --output-on-failure
```

This fork is a **runtime library**, not a VST3 plugin, standalone amp-sim
application, model trainer or complete hardware firmware. For plugin binaries,
see the separate [NAM Parametric Plugin](https://github.com/DLC86/NamParametricPlugin).
For detailed host integration, control/threading contracts, output modes,
performance scope and C API usage, consult [PNAM.md](PNAM.md).

---

## Original NeuralAudio documentation

The following sections describe Mike Oliphant's original APIs and behavior,
which remain available in this fork. Parametric model behavior specific to
this fork is documented above and in PNAM.md.

NeuralAudio is a C++ library designed to make it easy to use neural network machine learning models (ie: guitar amplifier captures/profiles) in real-time audio applications.

Priorities are a simple, but powerful API, a clean, maintainable codebase, and highly optimized performance across platforms.

# License

This repository is licensed under the [MIT license](https://github.com/mikeoliphant/NeuralAudio/blob/main/LICENSE). It is a liberal license, but please make sure that you comply with the terms - as well as the terms of [this project's dependencies](https://github.com/mikeoliphant/NeuralAudio/blob/main/CREDITS.md). I would also appreciate it if you would let me know if you are using this library.

# Supported Models and Underlying Libraries

NeuralAudio currently supports the following model types:

- [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler) (NAM) WaveNet and LSTM models, A1 and A2 support
- [RTNeural](https://github.com/jatinchowdhury18/RTNeural) keras models (LSTM, GRU)

For WaveNet, the internal implmeentation supports optimized static implemenationas the offical NAM A1 and A2 network architectures:  A1 "Standard", "Lite", "Feather", "Nano" and A2 "Lite" and "Full".

For LSTM, the internal implementation supports optimized static models architectures for 1x8, 1x12, 1x16, 1x24, 2x8, 2x12, and 2x16 models.

All A1 NAM files with WaveNet and LSTM architectures not supported statically will fall back on a less performant dynamic implementation.

All non-standard A2 models currently use the NAM Core implementation (and consequently require building with NAM Core enabled).

All keras models not supported internally will fall back to the RTNeural implmentation.

# API overview

Models are loaded with a model loader:

```
NeuralModelLoader loader;

NeuralModel* model = loader.CreateFromFile("<path to model file>");
```

To process a model:

```
model->Process(pointerToFloatInputData, pointerToFloatOutputData, int numSamples);
```

## Setting maximum buffer size

Some models need to allocate memory based on the size of the audio buffers being used. You need to make sure that processing does not exceed the specified maximum buffer size.

The default maximum size is 128 samples. To change it, change the default size on the model loader:

```
loader.SetDefaultMaxAudioBufferSize(maxSize);
```

if you want to change the maximum buffer size of an already created model, do:

```
model->SetMaxAudioBufferSize(int maxSize);
```

***Note: this is not real-time safe, and should not be done on a real-time audio thread.***

## Input/Output calibration

Use ```model->GetRecommendedInputDBAdjustment()``` and ```model->GetRecommendedOutputDBAdjustment()``` to obtain the ideal input and output volume level adjustments in dB.

To set a known audio input level (ie: from an audio interface), use ```loader.SetAudioInputLevelDBu(float audioDBu)```. This is set at 12DBu by default.

## Model load behavior

By default, models are loaded using the internal NeuralAudio implementation (if possible). If you would like to force the use of the NAM Core or RTNeural implementations, you can use:

```
loader.SetWaveNetLoadMode(loadMode);
```

and

```
loader.SetLSTMLoadMode(loadMode);
```

where "loadMode" is one of:

```
NeuralAudio::EModelLoadMode::Internal
NeuralAudio::EModelLoadMode::NAMCore
NeuralAudio::EModelLoadMode::RTNeural  (only supported for LSTM)
```

You can check which implementation was actually used to load the model with ```model->GetLoadMode()```.

**NOTE:** Because of compile time and executable size considerations, only the internal, NAM Core and dynamic RTNeural implementations are built by default. If you want to use RTNeural for LSTM models, it is recommended that you add ```-DBUILD_STATIC_RTNEURAL=ON``` to your cmake commandline. This will create static model implmentations for the same set of LSTM models as the internal implmentation, and results in increased performance. Interal static LSTM model support is also off by default - to turn it on use ```-DBUILD_INTERNAL_STATIC_LSTM=ON```.

### Composite model load behavior

Some models (notably NAM A2 models) are comprised of multiple sub-models. By default, all sub-models will be fully loaded and initialized when the model is loaded.

If you wish to avoid the overhead of initializing unused models and only initialize the active model on load, you can do:

```
loader->SetCompositeModelLoadMode(ECompositeModelLoadMode::OnDemand);
```

Note that this means that switching to a different model for the first time via quality scaling will ***not be realtime safe***.

## Setting model quality scaling factor

Some models (notably, slimmable NAM A2 models) support quality scaling - trading off quality for performance.

Quality scaling is a floating point range from 0.0 (highest performance) to 1.0 (highest quality).

To set the default quality scaling factor, set it on the loader:

```
loader.SetDefaultQualityScaleFactor(scaleFactor);
```

To check if a model supports quality scaling, do:

```
if (model->HasQualityScaling()) ...
```

To set the quality scaling factor for a loaded model, do:

```
model->SetQualityScaleFactor(scaleFactor);
```

***Note: This operation is not real-time safe if the quality scale factor results in switching to an uninitialized model.*** If you are using the default composite model loading behavior, setting the quality scale factor is always real-time safe. If you are using "OnDemand" composite model loading, you can check whether a quality scale change is real-time safe by doing:

```
if (!model->IsQualityChangeRealtimeSafe(newScaleFactor)
{
  (call SetQualityScaleFactor(), but ensure it is not done in a real-time context)
}
```

To get the current quality scaling factor for a model, do:

```
float scaleFactor = model->GetQualityScaleFactor();
```

## Model oversampling

WaveNet models are altered (via scaling the convolution dilation sizes) to produce the correct output when the external sample rate is an even multiple of the model sample rate.

By default, the external sample rate is set to 48kHz. To change it, do:

```
loader.SetExternalSampleRate(sampleRate);
```

Note that ***models are only altered on load***. If you change the external sample rate you will need to reload any existing models where you want the change to take effect.

## Getting model metadata

To retrieve arbitrary metadata fields from models that contain them, do:

```
std::string fieldName = "this_is_a_field_name";

std::string metadataValue = model->GetMetadata(fieldName);

if (!metadataValue.empty())
{
  // do something
}
```
Results are always returned a strings. Field names are case sensitive.

To get the model version string, do:

```
std::string version = model->GetModelVersion()
```

The string will be empty if no version information exists.

## Getting the model receptive field size

WaveNet models have a fixed receptive field size (ie: size of the input that the output depends on).

To get this value, do:

```
int receptiveFieldSamples = model->GetReceptiveFieldSize();
```

Note that this can return -1, which means that the receptive field size is unknown, or not fixed (ie: LSTM models technically have an infinite tail because of their internal feedback loop).

This method is only supported for "internal" and NAM Core models. For RTNeural it will always return -1.

# Building

First clone the repository:
```bash
git clone --recurse-submodules https://github.com/DLC86/NeuralAudio.git
cd NeuralAudio/build
```

Then compile the library using:

**Linux/MacOS**
```bash
cmake .. -DCMAKE_BUILD_TYPE="Release"
make -j4
```

**Windows**
```bash
cmake.exe -G "Visual Studio 17 2022" -A x64 ..
cmake --build . --config=release -j4
```

Note - you'll have to change the Visual Studio version if you are using a different one.

## Performance considerations

Optimization for this library is highly dependent on the details of the compiler you are using and the architecture of the system you intend to run on. In particular, getting the best performance relies on critical paths in the code being vectorized efficiently by the compiler for the target CPU.

The CMake setup for the library *does not* specify any optimization compiler flags - you are responsible for doing that yourself. Specifically, that means doing the highest possible compiler optimization (ie: "-O3")
and making sure that the compiler is targeting your CPU properly (ie: "-march=native" for GCC/Clang, "/arch:AVX2" for MSVC).

One more specific note - the ```MULTIFRAME_8X8_CONVOLUTION``` option described in the next section enables very impactful performance increases when set to "4" or "8" ("0" is the default) on appropriate hardware and compilers.
It also significantly *slows* performance if the optimizations are not properly supported. In general, assuming you have a very recent compiler you should be able to set it to "4" for systems with 128bit intrinsics (ie: Raspberry Pi 4) and "8" for systems with 256bit intrinsics (ie: Raspberry Pi 5, most x64 PCs). The best way to know what works in your scenario is to test it. The CMake config for this library does its best to default to the correct setting
based on architecture and compiler.

The "ModelTest" application binaries provided in the [Releases section](https://github.com/mikeoliphant/NeuralAudio/releases) have been optimized for various specific platforms and can be used as a basis for comparison.

## CMake Options

```-DBUILD_INTERNAL_STATIC_WAVENET=ON|OFF```: Build internal static WaveNet model architectures (faster internal WaveNet, but slower compile, larger size).

```-DBUILD_INTERNAL_STATIC_LSTM=ON|OFF```: Build internal static LSTM model architectures (faster internal LSTM, but slower compile, larger size).

```-DBUILD_STATIC_INTERNAL_NAMA2=ON|OFF```: Build internal static A2 implementation.

```-DMULTIFRAME_8X8_CONVOLUTION=0|4|8```: Use optimized multiframe 8x8 convolution. Much faster on very modern compilers. Much slower on older compilers. Defaults to "0" (disabled).

```-DDEFAULT_QUALITY_SCALE="X.X"```: Default model quality scale factor (0.0 to 1.0). Be sure to use quotes around value. Defaults to "1.0".

```-DDEFAULT_INPUT_DBU="XX"```: Default dBu level for model input calibration. Be sure to use quotes around floating point values. Defaults to "12.0".

```-DWAVENET_FRAMES=XXX```: Sample buffer size for the internal WaveNet implementation. Defaults to **64**. If you know you are going to be using a fixed sample buffer smaller or larger than this, use that instead. Note that the model will still be able to process any buffer size - it is just optimized for this size.

```-DBUFFER_PADDING=XXX```: Amount of padding to convolution layer buffers. This allows ring buffer resets to be staggered accross layers to improve performance. It also uses a significant amount of memory. It is set to **24** by default. It can be set all the way down to 0 to reduce memory usage.

```-DWAVENET_MATH=XXX```
```-DLSTM_MATH=XXX```: Which math approximations (tanh and sigmoid) to use for WaveNet and LSTM models. Options are:

  - ```FastMath``` (the default): Use the same approximations as NAM Core.
  - ```EigenMath```: Use Eigen's builtin tanh approximation. Somewhat slower, but more accurate.
  - ```StdMath```: Use standard math functions. No approxmation used - much slower.

```-DBUILD_UTILS=ON|OFF```: Build performance/accuracy testing tools (located in the "Utils" folder).

```-DBUILD_NAMCORE=ON|OFF```: Support loading models using the NAM Core implemenations.

```-DNAM_USE_INLINE_GEMM=ON|OFF```: Enable use of inline matrix multiplication in NAM Core.

```-DNAM_ENABLE_A2_FAST=ON|OFF```: Enable use of A2 fast path wavenet in NAM Core.

```-DNAM_A2_RING_MODE=0|1```: Ring-buffer strategy for A2 FAST. Defaults to "1"

  - ```0```: linear memmove-rewind (variable worst-case latency, sporadic spikes)
  - ```1```(the default): pow2 + tail mirror (constant per-block work, branchless reads)

```-DBUILD_STATIC_RTNEURAL=ON|OFF```: Build static RTNeural model architectures (slower compile, larger size - only use if you plan on forcing RTNeural model loading).

# Software/Hardware Using NeuralAudio

The following applications and devices are using the NeuralAudio library for model processing:

- [neural-amp-modeler-lv2](https://github.com/mikeoliphant/neural-amp-modeler-lv2): LV2 plugin for using neural network machine learning amp models.
- [stompbox](https://github.com/mikeoliphant/stompbox): Guitar amplification and effects pedalboard simulation.
- [NeuralRack](https://github.com/brummer10/NeuralRack): Neural Model and Impulse Response File loader for Linux/Windows.
- [Darkglass Anagram](https://www.darkglass.com/creation/anagram): Bass guitar effects unit.
- [neural_tilde](https://github.com/apresta/neural_tilde): Max/MSP external for running neural amplifier captures.
- [Pi Pedal](https://github.com/rerdavies/pipedal): Guitar Effect Pedal for Raspberry Pi.
- [Casette](https://github.com/pawelKapl/Casette): Raspberry Pie guitar processor.
