#!/usr/bin/env python3
"""Add allocation-free per-sample linear weight interpolation to the A2 fast core.

This runs after the pinned generic direct-kernel patch. Existing static kernels
remain bit-identical when no interpolation is active.
"""
from pathlib import Path
import sys

core = Path(sys.argv[1] if len(sys.argv) > 1 else "deps/NeuralAmpModelerCore") / "NAM"
fast = core / "wavenet/a2_fast.cpp"
marker = "// NeuralAudio fast A2 direct kernel weight smoothing"
if marker in fast.read_text():
    print("NeuralAudio fast A2 smoothing already installed")
    sys.exit(0)

def replace(path, before, after):
    p = core / path
    content = p.read_text()
    count = content.count(before)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match for {before[:80]!r}, got {count}")
    p.write_text(content.replace(before, after))

replace("kernel_weight_smoothing.h",
        "namespace nam::kernel_weight_smoothing {\n",
        """namespace nam::kernel_weight_smoothing {
class WeightSmoothingControl {
public:
  virtual ~WeightSmoothingControl() = default;
  virtual void ConfigureKernelSmoothing(int transition_frames, int curve) noexcept = 0;
  [[nodiscard]] virtual bool HasActiveWeightSmoothing() const noexcept = 0;
};
""")
replace("wavenet/hyperwavenet.h",
        '#include "model.h"',
        '#include "model.h"\n#include "../kernel_weight_smoothing.h"')
replace("wavenet/hyperwavenet.h",
        "class HyperWaveNet : public WaveNet, public IParametricControl",
        "class HyperWaveNet : public WaveNet, public IParametricControl, public kernel_weight_smoothing::WeightSmoothingControl")

replace("wavenet/a2_fast.h",
        "  virtual void SetWeights(std::span<const float> weights) = 0;",
        """  virtual void SetWeights(std::span<const float> weights) = 0;
  virtual void SetWeightRamp(std::span<const float> start, std::span<const float> target) = 0;
  virtual void SetBlockAlphas(const float* alphas) noexcept = 0;""")

replace("wavenet/a2_fast.cpp",
        '#include "a2_fast.h"',
        '#include "a2_fast.h"\n'+marker)
replace("wavenet/a2_fast.cpp",
        """    _load_weights(weights);
  }

protected:
""",
        """    _load_weights(weights);
    _block_alphas = nullptr;
  }
  void SetWeightRamp(std::span<const float> start, std::span<const float> target) override
  {
    if (start.size() != _weight_count || target.size() != _weight_count)
      throw std::invalid_argument("A2FastModel: incorrect weight-ramp size");
    _load_weights(start);
    _load_delta(target);
    _block_alphas = nullptr;
  }
  void SetBlockAlphas(const float* alphas) noexcept override { _block_alphas = alphas; }

protected:
""")
replace("wavenet/a2_fast.cpp",
        "    std::vector<float> conv_w;\n    std::array<float, Channels> conv_b{};",
        """    std::vector<float> conv_w, conv_delta;
    std::array<float, Channels> conv_b{}, conv_b_delta{};""")
replace("wavenet/a2_fast.cpp",
        "    std::array<float, Channels> mixin_w{};",
        "    std::array<float, Channels> mixin_w{}, mixin_delta{};")
replace("wavenet/a2_fast.cpp",
        "    std::array<float, Channels * Channels> l1x1_w{};\n    std::array<float, Channels> l1x1_b{};",
        """    std::array<float, Channels * Channels> l1x1_w{}, l1x1_delta{};
    std::array<float, Channels> l1x1_b{}, l1x1_b_delta{};""")
replace("wavenet/a2_fast.cpp",
        "  std::array<float, Channels> _rechannel_w{};",
        "  std::array<float, Channels> _rechannel_w{}, _rechannel_delta{};")
replace("wavenet/a2_fast.cpp",
        "  std::array<std::array<float, Channels>, kHeadKernelSize> _head_w{};",
        """  std::array<std::array<float, Channels>, kHeadKernelSize> _head_w{};
  std::array<std::array<float, Channels>, kHeadKernelSize> _head_delta{};""")
replace("wavenet/a2_fast.cpp",
        "  float _head_b = 0.0f;",
        "  float _head_b = 0.0f, _head_b_delta = 0.0f;")
replace("wavenet/a2_fast.cpp",
        "  float _head_scale = 1.0f;",
        "  float _head_scale = 1.0f, _head_scale_delta = 0.0f;")
replace("wavenet/a2_fast.cpp",
        "  size_t _weight_count = 0;",
        """  size_t _weight_count = 0;
  const float* _block_alphas = nullptr; // only nonnull for ramped DSP blocks""")
replace("wavenet/a2_fast.cpp",
        """  void _load_weights(std::span<const float> weights);
  void _ring_write""",
        """  void _load_weights(std::span<const float> weights);
  void _load_delta(std::span<const float> target);
  void _layer_forward_smooth(Layer& layer, const float* cond, int frames);
  void _head_forward_smooth(float* output, int frames);
  void _ring_write""")
replace("wavenet/a2_fast.cpp",
        "    _layers[i].conv_w.assign(static_cast<size_t>(kKernelSizes[i]) * Channels * Channels, 0.0f);",
        """    _layers[i].conv_w.assign(static_cast<size_t>(kKernelSizes[i]) * Channels * Channels, 0.0f);
    _layers[i].conv_delta.assign(_layers[i].conv_w.size(), 0.0f);""")

# The additional A2 delta arrays are allocated at construction, never during
# Process(). Their ordering mirrors _load_weights() exactly.
replace("wavenet/a2_fast.cpp",
        """// -----------------------------------------------------------------------------
// Buffer sizing
// -----------------------------------------------------------------------------""",
        """// -----------------------------------------------------------------------------
// Delta-weight decoder: identical layout to _load_weights; no allocations.
// -----------------------------------------------------------------------------
template <int Channels>
void A2FastModel<Channels>::_load_delta(std::span<const float> target)
{
  auto it = target.begin();
  auto take = [&]() -> float { return *it++; };
  for (int i = 0; i < Channels; ++i)
    _rechannel_delta[i] = take() - _rechannel_w[i];
  for (int li = 0; li < kNumLayers; ++li) {
    auto& L = _layers[li];
    const int K = L.kernel_size;
    for (int i = 0; i < Channels; ++i)
      for (int j = 0; j < Channels; ++j)
        for (int k = 0; k < K; ++k) {
          const auto index = static_cast<size_t>(k) * Channels * Channels + j * Channels + i;
          L.conv_delta[index] = take() - L.conv_w[index];
        }
    for (int i = 0; i < Channels; ++i) L.conv_b_delta[i] = take() - L.conv_b[i];
    for (int i = 0; i < Channels; ++i) L.mixin_delta[i] = take() - L.mixin_w[i];
    for (int i = 0; i < Channels; ++i)
      for (int j = 0; j < Channels; ++j) {
        const int index = j * Channels + i;
        L.l1x1_delta[index] = take() - L.l1x1_w[index];
      }
    for (int i = 0; i < Channels; ++i) L.l1x1_b_delta[i] = take() - L.l1x1_b[i];
  }
  for (int j = 0; j < Channels; ++j)
    for (int k = 0; k < kHeadKernelSize; ++k)
      _head_delta[k][j] = take() - _head_w[k][j];
  _head_b_delta = take() - _head_b;
  _head_scale_delta = take() - _head_scale;
}

// -----------------------------------------------------------------------------
// Buffer sizing
// -----------------------------------------------------------------------------""")

# This is a separate fast A2 DSP branch while a weight ramp is active. It
# computes exactly one convolutional network; each coefficient is w+a*dw.
# Idle processing keeps the original optimized tiled/Eigen path unchanged.
replace("wavenet/a2_fast.cpp",
        """// Runtime dispatcher: selects the K-specialized kernel for this layer.""",
        """template <int Channels>
void A2FastModel<Channels>::_layer_forward_smooth(Layer& L, const float* cond, int frames)
{
  _ring_write(L, frames);
  const int K = L.kernel_size;
  const int D = L.dilation;
#if NAM_A2_RING_MODE == 1
  const int mask = L.pow2_mask;
  auto tap_base = [&](int back) { return (L.write_pos - frames - back * D) & mask; };
#else
  const int base = L.write_pos - frames;
  auto tap_base = [&](int back) { return base - back * D; };
#endif
  for (int f = 0; f < frames; ++f) {
    const float a = _block_alphas[f];
    float* z = _z.data() + static_cast<size_t>(f) * Channels;
    float* sum = _head_sum.data() + static_cast<size_t>(f) * Channels;
    float* lin = _layer_in.data() + static_cast<size_t>(f) * Channels;
    for (int o = 0; o < Channels; ++o)
      z[o] = L.conv_b[o] + a * L.conv_b_delta[o] +
             (L.mixin_w[o] + a * L.mixin_delta[o]) * cond[f];
    for (int k = 0; k < K; ++k) {
      const float* w = L.conv_w.data() + static_cast<size_t>(k) * Channels * Channels;
      const float* dw = L.conv_delta.data() + static_cast<size_t>(k) * Channels * Channels;
      const float* h = L.history.data() + static_cast<size_t>(tap_base(K - 1 - k) + f) * Channels;
      for (int c = 0; c < Channels; ++c)
        for (int o = 0; o < Channels; ++o) {
          const int idx = c * Channels + o;
          z[o] += (w[idx] + a * dw[idx]) * h[c];
        }
    }
    for (int o = 0; o < Channels; ++o) {
      z[o] = z[o] < 0 ? z[o] * kLeakySlope : z[o];
      sum[o] += z[o];
      lin[o] += L.l1x1_b[o] + a * L.l1x1_b_delta[o];
    }
    for (int c = 0; c < Channels; ++c)
      for (int o = 0; o < Channels; ++o) {
        const int idx = c * Channels + o;
        lin[o] += (L.l1x1_w[idx] + a * L.l1x1_delta[idx]) * z[c];
      }
  }
}

// Runtime dispatcher: selects the K-specialized kernel for this layer.""")

replace("wavenet/a2_fast.cpp",
        """// DSP::process override
// -----------------------------------------------------------------------------""",
        """template <int Channels>
void A2FastModel<Channels>::_head_forward_smooth(float* output, int frames)
{
  _head_ring_write(frames);
#if NAM_A2_RING_MODE == 1
  const int mask = _head_pow2_mask;
  auto col = [&](int f, int k) {
    return (_head_write_pos - frames + f - (kHeadKernelSize - 1 - k)) & mask;
  };
#else
  const int base = _head_write_pos - frames;
  auto col = [&](int f, int k) { return base + f - (kHeadKernelSize - 1 - k); };
#endif
  for (int f = 0; f < frames; ++f) {
    const float a = _block_alphas[f];
    float y = _head_b + a * _head_b_delta;
    for (int k = 0; k < kHeadKernelSize; ++k) {
      const float* h = _head_history.data() + static_cast<size_t>(col(f, k)) * Channels;
      for (int c = 0; c < Channels; ++c)
        y += (_head_w[k][c] + a * _head_delta[k][c]) * h[c];
    }
    output[f] = y * (_head_scale + a * _head_scale_delta);
  }
}

// DSP::process override
// -----------------------------------------------------------------------------""")
replace("wavenet/a2_fast.cpp",
        """    for (int c = 0; c < Channels; c++)
      lin[c] = _rechannel_w[c] * x;""",
        """    if (_block_alphas) {
      const float a = _block_alphas[f];
      for (int c = 0; c < Channels; c++)
        lin[c] = (_rechannel_w[c] + a * _rechannel_delta[c]) * x;
    } else {
      for (int c = 0; c < Channels; c++)
        lin[c] = _rechannel_w[c] * x;
    }""")
replace("wavenet/a2_fast.cpp",
        """  for (int li = 0; li < kNumLayers; li++)
    _layer_forward(li, cond, num_frames);""",
        """  for (int li = 0; li < kNumLayers; li++) {
    if (_block_alphas)
      _layer_forward_smooth(_layers[li], cond, num_frames);
    else
      _layer_forward(li, cond, num_frames);
  }""")
replace("wavenet/a2_fast.cpp",
        """  _head_forward(head_out, num_frames);""",
        """  if (_block_alphas)
    _head_forward_smooth(head_out, num_frames);
  else
    _head_forward(head_out, num_frames);""")

# The A2 wrapper is the owner's pending/active weight-interpolation state.
replace("wavenet/hyperwavenet.cpp",
        '#include "a2_fast.h"',
        '#include "a2_fast.h"\n#include "../kernel_weight_smoothing.h"')
replace("wavenet/hyperwavenet.cpp",
        "class FastHyperWaveNet final : public nam::DSP, public nam::IParametricControl",
        """class FastHyperWaveNet final : public nam::DSP, public nam::IParametricControl,
                               public nam::kernel_weight_smoothing::WeightSmoothingControl""")
replace("wavenet/hyperwavenet.cpp",
        """  int GetPrewarmSamples() override { return _audio->GetPrewarmSamples(); }
  void process""",
        """  int GetPrewarmSamples() override { return _audio->GetPrewarmSamples(); }
  void ConfigureKernelSmoothing(int frames, int curve) noexcept override
  {
    (void)curve; // NeuralAudio always uses linear time interpolation.
    if (HasActiveWeightSmoothing()) return;
    _smooth_frames = std::max(0, frames);
    _smooth_position = _smooth_frames;
  }
  bool HasActiveWeightSmoothing() const noexcept override
  {
    return _smooth_frames > 0 && _has_applied && _smooth_position < _smooth_frames;
  }
  void process""")
replace("wavenet/hyperwavenet.cpp",
        """    if (_dirty)
    {
      _hypernet.ApplyConditioning(_base, _params, _conditioned);
      _weights->SetWeights(_conditioned);
      _dirty = false;
    }
    _audio->process(input, output, frames);""",
        """    if (_dirty)
    {
      if (HasActiveWeightSmoothing()) {
        const int last = std::max(0, _smooth_position - 1);
        const float a = static_cast<float>(last) / static_cast<float>(_smooth_frames);
        for (size_t i = 0; i < _current.size(); ++i)
          _current[i] = _start[i] + a * (_conditioned[i] - _start[i]);
      }
      _hypernet.ApplyConditioning(_base, _params, _conditioned);
      if (!_has_applied || _smooth_frames == 0) {
        _weights->SetWeights(_conditioned);
        _current = _conditioned;
        _has_applied = true;
        _smooth_position = _smooth_frames;
      } else if (_current == _conditioned) {
        _weights->SetWeights(_conditioned);
        _smooth_position = _smooth_frames;
      } else {
        _start = _current;
        _weights->SetWeightRamp(_start, _conditioned);
        _smooth_position = 0;
      }
      _dirty = false;
    }
    if (!HasActiveWeightSmoothing()) {
      _audio->process(input, output, frames);
      return;
    }
    for (int f = 0; f < frames; ++f) {
      const int position = std::min(_smooth_frames, _smooth_position + f);
      _kernel_alphas[static_cast<size_t>(f)] =
        static_cast<float>(position) / static_cast<float>(_smooth_frames);
    }
    _weights->SetBlockAlphas(_kernel_alphas.data());
    _audio->process(input, output, frames);
    _weights->SetBlockAlphas(nullptr);
    _smooth_position = std::min(_smooth_frames, _smooth_position + frames);
    if (_smooth_position == _smooth_frames) {
      _current = _conditioned;
      _weights->SetWeights(_conditioned);
    }""")
replace("wavenet/hyperwavenet.cpp",
        """    _audio->Reset(mExternalSampleRate, size);
  }

private:
  std::unique_ptr<nam::DSP> _audio;""",
        """    _audio->Reset(mExternalSampleRate, size);
    _kernel_alphas.resize(static_cast<size_t>(size));
    _has_applied = false;
    _dirty = true;
    _smooth_position = _smooth_frames;
  }

private:
  std::unique_ptr<nam::DSP> _audio;""")
replace("wavenet/hyperwavenet.cpp",
        """  std::vector<float> _params, _base, _conditioned;
  nam::Hypernetwork _hypernet;""",
        """  std::vector<float> _params, _base, _conditioned;
  std::vector<float> _current = _base, _start = _base;
  std::vector<float> _kernel_alphas;
  nam::Hypernetwork _hypernet;
  int _smooth_frames = 0, _smooth_position = 0;
  bool _has_applied = false;""")

print("NeuralAudio fast A2 direct-kernel interpolation installed", core)
