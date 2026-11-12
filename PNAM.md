# PNAM runtime for hardware integrations

This fork adds `.pnam` and legacy parametric `.nam` loading to NeuralAudio.
Supported architectures: `ParametricSlimmableContainer` (new nested Full + Lite
container), legacy `HyperA2` (the same Full/Lite networks), `HyperWaveNet`
(full and low-rank hypernetworks), and `ConcatWaveNet`. Existing
`ParametricOutputGain` wrappers and newer optional
`extensions.linear_volume` curves are also supported. Knobs and any number
of independent switches are exposed with ranges, defaults, position counts
and labels. Linear Volume is an independent PCHIP attenuation stage applied
after the neural network with gain smoothing, not another hypernetwork input.

The PNAM backend uses the shared parametric NAM Core fork. Compatible HyperA2
branches select the specialized A2 kernels, including NeuralAudio-style
multi-frame weight reuse on small x64 blocks. Other topologies retain the generic
backend. Weight updates reuse allocated buffers and preserve convolution history;
the network is not recreated when controls move. This is not a conversion of a
PNAM into one frozen static model.

The default HyperWaveNet behavior in this fork uses a **single audio network**.
Target weights are updated at most once every **100 ms** of model-sample time,
and the coefficients inside each convolution are **interpolated linearly per
sample** over **100 ms**. The two intervals are independently configurable,
without retraining or changing the exported model. Full/Lite variants keep
separate update clocks. A2 fast kernels retain their original optimized
stationary processing path, using a separate single-network sample-ramped
kernel path only while a control is moving. ConcatWaveNet keeps its existing
control smoothing because it has no hypernetwork-generated weights.

## Optional control display indices

Current ParaNAM Trainer exports may set a zero-based `index` on every
`config.params[]` neural control. These positions share one sequence
with `extensions.linear_volume.index`, if present. The PNAM adapter sorts
`GetControls()` and `GetControlValues()` into this visual order and
expects `SetControlValues()` in the same order. It maps values back to
the original positional neural input order before conditioning weights.
No weights, parameter types or model calculations change. The C API
control enumeration and setter follow the same order.

Older PNAMs without neural indices keep their historical ordering. New
files must supply a complete, unique `0..N-1` display permutation
(including Linear Volume when present). Inconsistent indices are rejected
rather than silently changing the sound. The `index` field is UI/API
metadata, not the hypernetwork input position.

### Indexed switch options

New switch entries may replace the legacy `enum_names` list with
`options: [{"name":"Clean","value":0,"index_sw":2}, ...]`.
The `value` is the original zero-based *neural* switch value, and
`index_sw` is the zero-based visual position. Both fields must independently
form complete, unique permutations `0..N-1` for that switch. The `default`
field remains the original neural value. A switch may define `index` for its
own control order and `index_sw` for each option's order.

The C++ `ControlSpec.positions` and C
`PnamGetPositionName` return labels in `index_sw` display order, while
`ControlSpec.positionValues` and `PnamGetPositionValue` give each
displayed option's **original neural value**. Call `SetControlValues` /
`PnamSetControls` with that value, **not** the displayed index. Old files
with `enum_names` are still accepted and get natural `0..N-1` order.
Neural conditioning and exported weights are never reordered.

## Build

Requires a C++20 compiler and CMake 3.20 or later. No JUCE, CUDA, or training
dependencies are needed. Python 3 is needed only while configuring CMake to
apply the idempotent pinned-core patches; not on deployed systems. Initialize the pinned dependencies when cloning:

```sh
git clone --recursive https://github.com/DLC86/NeuralAudio.git
cd NeuralAudio
cmake -S . -B build/pnam -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_RTNEURAL=OFF -DBUILD_INTERNAL_STATIC_WAVENET=OFF \
  -DBUILD_STATIC_INTERNAL_NAMA2=OFF -DBUILD_PNAM_TESTS=ON
cmake --build build/pnam --config Release --parallel 2
ctest --test-dir build/pnam -C Release --output-on-failure
```

The `NeuralAudio` CMake target exposes the C++ API; `NeuralAudioCAPI` exposes a C
interface suitable for hosts that do not use C++. `pnam_render` is a minimal C
example (see [Examples/PnamRender.c](Examples/PnamRender.c)).
Link the C++ target or the C API library as appropriate for the host; do not embed
a second copy of the C++ runtime alongside the C API shared library.
Use `-DBUILD_PNAM_TESTS=OFF` in a product build. `BUILD_PNAM=OFF` removes the PNAM
adapter; enabling it requires `BUILD_NAMCORE=ON` and `NAM_ENABLE_A2_FAST=ON`.

## C++ integration

```cpp
#include <NeuralAudio/ParametricModel.h>

NeuralAudio::NeuralModelLoader loader;
loader.SetDefaultMaxAudioBufferSize(128);
std::unique_ptr<NeuralAudio::NeuralModel> model(loader.CreateFromFile("amp.pnam"));
auto* pnam = dynamic_cast<NeuralAudio::ParametricModel*>(model.get());
if (!pnam) throw std::runtime_error("Not a supported PNAM model");

// Populate the UI from GetControls(): name, minimum, maximum, defaultValue,
// and positions (empty for a knob, one label per switch position otherwise).
auto values = std::vector<float>(pnam->GetControlValues().begin(),
                                 pnam->GetControlValues().end());
pnam->SetControlValues(values);
pnam->SetAudioInputLevelDBu(12.0f); // same interface calibration reference as NAM On Steroids
// Output Mode is Auto by default for ALL model architectures; this is optional:
pnam->SetOutputMode(NeuralAudio::OutputMode::Auto);
// Independent timings; both default to 100 ms and can be set separately:
pnam->SetWeightUpdateIntervalMs(100.0f);
pnam->SetWeightSmoothingMs(100.0f);

// Process at GetSampleRate(). Input/output are mono buffers.
model->Process(input, output, numFrames);
```

The loader defaults to Full. For a nested `HyperA2` or
`ParametricSlimmableContainer`, select Lite with
`SetQualityScaleFactor(0.5f)` or Full with `SetQualityScaleFactor(1.0f)`.
All controls are retained in both variants. Size changes are **offline** operations:
stop the callback, select the size, call `Prewarm()`, then restart audio. The
inactive model is not continually processed, to avoid doubling CPU usage.
`IsQualityChangeRealtimeSafe()` reports this restriction.

Load, buffer preparation, metadata queries/serialization, and prewarm must happen
outside the audio callback. Valid `SetControlValues()` and `SetOutputMode()` calls
can happen between `Process()` calls, on the same thread. Do not call them
concurrently with processing. Switch values are zero-based integer indices;
invalid values are rejected without changing any controls. Arbitrary block lengths
are processed immediately, internally splitting at the prepared maximum; there
is no extra block-buffering latency. Hosts supply sample-rate conversion, audio
I/O. HyperWaveNet smoothing is handled internally by per-sample linear kernel
interpolation; no host-side de-click is needed for those weight changes. The
host still owns automation timing and any desired extra de-click for other
models/controls. The interpolation intervals are model-sample-clock based,
not wall-clock based. Longer ramp times make controls respond more gradually.

## Output modes and calibration

`Process()` applies the selected output gain. It never normalizes individual knob
positions or changes their relative loudness. Normalized uses the single exported
model loudness value, with the same −18 dB target as NAM On Steroids.

| Requested mode | Applied gain in dB | Missing/invalid metadata |
| --- | --- | --- |
| Raw | 0 | Raw |
| Calibrated | `output_level_dbu - interface calibration` | Raw |
| Normalized | `-18 - loudness` | Raw |
| Auto (default) | Normalized for `amp`, `pedal_amp`, `amp_cab`, `amp_pedal_cab`; Calibrated for other/unknown gear types | Raw if the selected mode lacks its metadata |

Gear type matching is case-insensitive. Auto follows NAM On Steroids; it does not
silently substitute Calibrated if an amp lacks loudness, or Normalized if a pedal
lacks output calibration.

`GetOutputMode()` returns the requested mode, `GetEffectiveOutputMode()` returns
Raw/Calibrated/Normalized after fallback, and `GetOutputGainDB()` reports the gain
already applied. `GetRecommendedOutputDBAdjustment()` returns zero for PNAM to
prevent older hosts applying normalization a second time. Set Raw if the host
wants to implement all output scaling itself.

Input calibration remains explicit in the host, following NeuralAudio's existing
convention: apply `GetRecommendedInputDBAdjustment()` when enabled. It is
`interface calibration - input_level_dbu`, or zero if input calibration metadata
is missing. Changing `SetAudioInputLevelDBu()` updates the reference used by
Calibrated output too, matching NAM On Steroids.

`GetMetadataJson()` retains all source metadata, including nested/custom fields,
ESR Full/Lite, author and calibration values. Missing, null, nonnumeric or
non-finite calibration/loudness fields cannot activate level scaling. No invented
loudness or calibration is used for old models.

## C API

Include `NeuralAudioCAPI/NeuralAudioCApi.h` (valid C and C++). Use
`CreateModelFromFileUtf8()`, then `IsParametric()`. New PNAM functions include:

- `PnamGetControlCount/Name/Range`, `PnamGetPositionCount/Name`, `PnamSetControls`.
- `PnamSetWeightUpdateIntervalMs`, `PnamSetWeightSmoothingMs` (default 100/100 ms; independent, linear).
- `PnamSetOutputMode` with `PNAM_RAW`, `PNAM_CALIBRATED`, `PNAM_NORMALIZED`, `PNAM_AUTO`.
- `PnamGetOutputMode`, `PnamGetEffectiveOutputMode`, `PnamSetCalibrationLevelDBu`.
- `PnamHasLite`, `PnamSetLite` (offline, also prewarms), and `PnamGetMetadataJson`.
- `SetModelOutputMode`, `GetModelOutputMode`,
  `GetModelEffectiveOutputMode` apply to static and parametric models.
  For static models the host applies the returned gain via
  `GetRecommendedOutputDBAdjustment()`; PNAM already applies gain in Process().

Setters return 1 on success and 0 on error. Name pointers remain valid until the
model is destroyed. Metadata retrieval returns the required buffer size including
the terminating NUL: query with NULL/0, allocate that many bytes outside the audio
thread, then retrieve it. An undersized buffer is not partially overwritten.

The callback itself is still `Process()`. Do not apply output compensation again.

## Validation and deployment scope

Tests cover PyTorch golden audio for full/low-rank HyperA2 Full and Lite,
the renamed ParametricSlimmableContainer, legacy and extension-based Linear
Volume attenuation, all 30
combinations of three switches (3 × 2 × 5) mixed with two knobs, HyperWaveNet and
ConcatWaveNet, requested/effective output modes, rendered output gain, unknown and
missing metadata, and both C and C++ APIs. Fixtures are synthetic random-weight
test models, not captures intended for playing.

CI builds Windows x64, Linux x64, macOS and Linux ARM64. A successful ARM64
build is not a real-time performance measurement of a particular board. This is a
C++/Eigen runtime for hosts with a C++20 runtime and sufficient RAM; it is not a
bare-metal microcontroller firmware or an audio-device driver.

Upstream NeuralAudio and dependency license notices are retained. No architecture
change or retraining is required for compatible PNAM files.
