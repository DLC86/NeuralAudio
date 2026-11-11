#!/usr/bin/env python3
"""Apply direct, sample-accurate kernel interpolation to the pinned NAM core.

Experiment-only: run AFTER core-hyperwavenet-single-weight-smoothing.patch.
Edits are deliberately anchored and abort on unexpected core revisions.
No model weights are shipped. No JUCE/plugin files are modified here.
"""
from pathlib import Path
import sys

core = Path(sys.argv[1] if len(sys.argv) > 1 else "external/nam_core_parametric") / "NAM"
if (core / "kernel_weight_smoothing.h").exists():
    if "ScopedBlockAlphas scope" in (core / "wavenet/hyperwavenet.cpp").read_text():
        print("Direct-kernel interpolation already installed")
        sys.exit(0)
    raise RuntimeError("Partially applied kernel interpolation; reset pinned submodule")
def replace(path, before, after):
    file = core / path
    data = file.read_text()
    count = data.count(before)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match for {before[:65]!r}, found {count}")
    file.write_text(data.replace(before, after))

(core / "kernel_weight_smoothing.h").write_text(r"""#pragma once
namespace nam::kernel_weight_smoothing {
inline thread_local bool capture_target = false;
inline thread_local const float* block_alphas = nullptr;
struct ScopedTargetCapture {
  bool old = capture_target;
  ScopedTargetCapture() noexcept { capture_target = true; }
  ~ScopedTargetCapture() noexcept { capture_target = old; }
};
// Polynomial curves: no trigonometric functions in the audio callback.
inline float EvaluateAlpha(const float t, const int curve) noexcept {
  switch (curve) {
    case 1: return t * t * t * (t * (t * 6.f - 15.f) + 10.f); // quintic C2
    case 2: return t;                                        // linear control
    case 3: {                                                // septic C3
      const float t2 = t * t;
      return t2 * t2 * (35.f + t * (-84.f + t * (70.f - 20.f * t)));
    }
    default: return t * t * (3.f - 2.f * t);                 // cubic C1
  }
}
struct ScopedBlockAlphas {
  const float* old = block_alphas;
  explicit ScopedBlockAlphas(const float* a) noexcept { block_alphas = a; }
  ~ScopedBlockAlphas() noexcept { block_alphas = old; }
};
}
""")

# Allocate delta matrices at model construction, NEVER in the audio callback.
replace("dsp.h", "  Eigen::MatrixXf _weight;\n  // For depthwise convolution",
        "  Eigen::MatrixXf _weight;\n  Eigen::MatrixXf _ramp_weight;\n  bool _ramp_weight_changed = false;\n  // For depthwise convolution")
replace("dsp.h", "  Eigen::VectorXf _depthwise_weight;\n  bool _is_depthwise = false;",
        "  Eigen::VectorXf _depthwise_weight;\n  Eigen::VectorXf _ramp_depthwise_weight;\n  bool _is_depthwise = false;")
replace("dsp.h", "  Eigen::VectorXf _bias;\n  int _num_groups;\n\nprivate:\n  Eigen::MatrixXf _output;",
        "  Eigen::VectorXf _bias;\n  Eigen::VectorXf _ramp_bias;\n  bool _ramp_bias_changed = false;\n  int _num_groups;\n\nprivate:\n  Eigen::MatrixXf _output;")
replace("dsp.cpp", '#include "dsp.h"', '#include "dsp.h"\n#include "kernel_weight_smoothing.h"')
replace("dsp.cpp", "    this->_depthwise_weight.resize(in_channels);\n    this->_depthwise_weight.setZero();",
        "    this->_depthwise_weight.resize(in_channels);\n    this->_depthwise_weight.setZero();\n    this->_ramp_depthwise_weight.setZero(in_channels);")
replace("dsp.cpp", "    this->_weight.resize(out_channels, in_channels);\n    this->_weight.setZero();",
        "    this->_weight.resize(out_channels, in_channels);\n    this->_weight.setZero();\n    this->_ramp_weight.setZero(out_channels, in_channels);")
replace("dsp.cpp", "    this->_bias.resize(out_channels);\n    this->_bias.setZero();\n  }\n}",
        "    this->_bias.resize(out_channels);\n    this->_bias.setZero();\n    this->_ramp_bias.setZero(out_channels);\n  }\n}")
replace("dsp.cpp", "void nam::Conv1x1::set_weights_(std::vector<float>::iterator& weights)\n{\n  if (this->_is_depthwise)",
        "void nam::Conv1x1::set_weights_(std::vector<float>::iterator& weights)\n{\n  const bool capture = kernel_weight_smoothing::capture_target;\n  _ramp_weight_changed = false;\n  _ramp_bias_changed = false;\n  if (this->_is_depthwise)")
replace("dsp.cpp", "      this->_depthwise_weight(c) = *(weights++);",
        """      const float v = *(weights++);
      if (capture) {
        const float d = v - this->_depthwise_weight(c);
        this->_ramp_depthwise_weight(c) = d;
        _ramp_weight_changed |= (d != 0.f);
      } else this->_depthwise_weight(c) = v;""")
replace("dsp.cpp", "          this->_weight(g * out_per_group + i, g * in_per_group + j) = *(weights++);",
        """          const auto row = g * out_per_group + i;
          const auto col = g * in_per_group + j;
          const float v = *(weights++);
          if (capture) {
            const float d = v - this->_weight(row, col);
            this->_ramp_weight(row, col) = d;
            _ramp_weight_changed |= (d != 0.f);
          } else this->_weight(row, col) = v;""")
replace("dsp.cpp", "  if (this->_do_bias)\n    for (int i = 0; i < this->_bias.size(); i++)\n      this->_bias(i) = *(weights++);\n}",
        """  if (this->_do_bias)
    for (int i = 0; i < this->_bias.size(); i++) {
      const float v = *(weights++);
      if (capture) {
        const float d = v - this->_bias(i);
        this->_ramp_bias(i) = d;
        _ramp_bias_changed |= (d != 0.f);
      } else this->_bias(i) = v;
    }
}""")
file = core / "dsp.cpp"
data = file.read_text()
assert data.count("void nam::Conv1x1::process_(") == 1
assert data.rstrip().endswith("}")
position = data.rfind("}\n")
assert position > data.index("void nam::Conv1x1::process_(")
extra = r"""  // Existing fast/fused base convolution stays intact. Add only the delta.
  const float* alpha = kernel_weight_smoothing::block_alphas;
  if (alpha && (_ramp_weight_changed || _ramp_bias_changed))
  {
    const int out_ch = static_cast<int>(get_out_channels());
    const int in_ch = static_cast<int>(get_in_channels());
    for (int f = 0; f < num_frames; ++f)
      for (int o = 0; o < out_ch; ++o)
      {
        float d = _ramp_bias_changed ? _ramp_bias(o) : 0.f;
        if (_ramp_weight_changed) {
          if (_is_depthwise) d += _ramp_depthwise_weight(o) * input(o, f);
          else for (int i = 0; i < in_ch; ++i)
            d += _ramp_weight(o, i) * input(i, f);
        }
        _output(o, f) += alpha[f] * d;
      }
  }
"""
file.write_text(data[:position] + extra + data[position:])

replace("conv1d.h", "  std::vector<Eigen::MatrixXf> _weight;\n  // For depthwise convolution",
        "  std::vector<Eigen::MatrixXf> _weight;\n  std::vector<Eigen::MatrixXf> _ramp_weight;\n  bool _ramp_weight_changed = false;\n  // For depthwise convolution")
replace("conv1d.h", "  std::vector<Eigen::VectorXf> _depthwise_weight;\n  bool _is_depthwise = false;",
        "  std::vector<Eigen::VectorXf> _depthwise_weight;\n  std::vector<Eigen::VectorXf> _ramp_depthwise_weight;\n  bool _is_depthwise = false;")
replace("conv1d.h", "  Eigen::VectorXf _bias;\n  int _dilation;",
        "  Eigen::VectorXf _bias;\n  Eigen::VectorXf _ramp_bias;\n  bool _ramp_bias_changed = false;\n  int _dilation;")
replace("conv1d.cpp", '#include "conv1d.h"', '#include "conv1d.h"\n#include "kernel_weight_smoothing.h"')
replace("conv1d.cpp", "void Conv1D::set_weights_(std::vector<float>::iterator& weights)\n{\n  if (this->_is_depthwise)",
        "void Conv1D::set_weights_(std::vector<float>::iterator& weights)\n{\n  const bool capture = kernel_weight_smoothing::capture_target;\n  _ramp_weight_changed = false;\n  _ramp_bias_changed = false;\n  if (this->_is_depthwise)")
replace("conv1d.cpp", "        this->_depthwise_weight[k](c) = *(weights++);",
        """        const float v = *(weights++);
        if (capture) {
          const float d = v - this->_depthwise_weight[k](c);
          this->_ramp_depthwise_weight[k](c) = d;
          _ramp_weight_changed |= (d != 0.f);
        } else this->_depthwise_weight[k](c) = v;""")
replace("conv1d.cpp", "            this->_weight[k](g * out_per_group + i, g * in_per_group + j) = *(weights++);",
        """            const auto row = g * out_per_group + i;
            const auto col = g * in_per_group + j;
            const float v = *(weights++);
            if (capture) {
              const float d = v - this->_weight[k](row, col);
              this->_ramp_weight[k](row, col) = d;
              _ramp_weight_changed |= (d != 0.f);
            } else this->_weight[k](row, col) = v;""")
replace("conv1d.cpp", "  for (long i = 0; i < this->_bias.size(); i++)\n    this->_bias(i) = *(weights++);",
        """  for (long i = 0; i < this->_bias.size(); i++) {
    const float v = *(weights++);
    if (capture) {
      const float d = v - this->_bias(i);
      this->_ramp_bias(i) = d;
      _ramp_bias_changed |= (d != 0.f);
    } else this->_bias(i) = v;
  }""")
replace("conv1d.cpp", "    this->_depthwise_weight.resize(kernel_size);\n    for (int i = 0; i < kernel_size; i++)",
        "    this->_depthwise_weight.resize(kernel_size);\n    this->_ramp_depthwise_weight.resize(kernel_size);\n    for (int i = 0; i < kernel_size; i++)")
replace("conv1d.cpp", "      this->_depthwise_weight[i].resize(in_channels);\n      this->_depthwise_weight[i].setZero();",
        "      this->_depthwise_weight[i].resize(in_channels);\n      this->_depthwise_weight[i].setZero();\n      this->_ramp_depthwise_weight[i].setZero(in_channels);")
replace("conv1d.cpp", "    this->_weight.resize(kernel_size);\n    for (int i = 0; i < kernel_size; i++)",
        "    this->_weight.resize(kernel_size);\n    this->_ramp_weight.resize(kernel_size);\n    for (int i = 0; i < kernel_size; i++)")
replace("conv1d.cpp", "      this->_weight[i].setZero();\n    }\n    this->_depthwise_weight.clear();",
        "      this->_weight[i].setZero();\n      this->_ramp_weight[i].setZero(out_channels, in_channels);\n    }\n    this->_depthwise_weight.clear();")
replace("conv1d.cpp", "    this->_bias.resize(out_channels);\n    this->_bias.setZero();",
        "    this->_bias.resize(out_channels);\n    this->_bias.setZero();\n    this->_ramp_bias.setZero(out_channels);")
replace("conv1d.cpp", "  // Advance ring buffer write pointer after processing\n  _input_buffer.Advance(num_frames);",
        """  // Evaluate a second *convolution delta* only for conditioned kernels;
  // unconditioned dilated kernels still use the original optimized path.
  const float* alpha = kernel_weight_smoothing::block_alphas;
  if (alpha && _ramp_weight_changed)
  {
    const int out_ch = static_cast<int>(get_out_channels());
    const int in_ch = static_cast<int>(get_in_channels());
    const int taps = static_cast<int>(get_kernel_size());
    for (int k = 0; k < taps; ++k)
    {
      const long lookback = this->_dilation * (taps - 1 - k);
      const auto history = _input_buffer.Read(num_frames, lookback);
      for (int f = 0; f < num_frames; ++f)
        for (int o = 0; o < out_ch; ++o)
        {
          float d = 0.f;
          if (_is_depthwise) d = _ramp_depthwise_weight[k](o) * history(o, f);
          else for (int i = 0; i < in_ch; ++i)
            d += _ramp_weight[k](o, i) * history(i, f);
          _output(o, f) += alpha[f] * d;
        }
    }
  }
  if (alpha && _ramp_bias_changed)
  {
    const int out_ch = static_cast<int>(get_out_channels());
    for (int f = 0; f < num_frames; ++f)
      for (int o = 0; o < out_ch; ++o)
        _output(o, f) += alpha[f] * _ramp_bias(o);
  }

  // Advance ring buffer write pointer after processing
  _input_buffer.Advance(num_frames);""")

replace("wavenet/model.h", "  float _head_scale;\n\n  std::unique_ptr<detail::Head>",
        "  float _head_scale;\n  float _ramp_head_scale_delta = 0.f;\n\n  std::unique_ptr<detail::Head>")
replace("wavenet/model.cpp", '#include "model.h"', '#include "model.h"\n#include "../kernel_weight_smoothing.h"')
replace("wavenet/model.cpp", "  this->_head_scale = *(it++); // TODO " + chr(96) + "LayerArray.absorb_head_scale()" + chr(96),
        """  const float scale = *(it++);
  if (kernel_weight_smoothing::capture_target)
    _ramp_head_scale_delta = scale - _head_scale;
  else { _head_scale = scale; _ramp_head_scale_delta = 0.f; }""")
replace("wavenet/model.cpp", "      for (int s = 0; s < num_frames; s++)\n        this->_scaled_head_scratch(ch, s) = this->_head_scale * final_head_outputs(ch, s);",
        """      for (int s = 0; s < num_frames; s++)
      {
        const float* alphas = kernel_weight_smoothing::block_alphas;
        const float scale = _head_scale + (alphas ? alphas[s] * _ramp_head_scale_delta : 0.f);
        this->_scaled_head_scratch(ch, s) = scale * final_head_outputs(ch, s);
      }""")
replace("wavenet/model.cpp", "      dst[s] = scale * src[s];",
        """      const float* alphas = kernel_weight_smoothing::block_alphas;
      dst[s] = (scale + (alphas ? alphas[s] * _ramp_head_scale_delta : 0.f)) * src[s];""")
replace("wavenet/model.cpp", "        output[ch][s] = this->_head_scale * final_head_outputs(ch, s);",
        """        const float* alphas = kernel_weight_smoothing::block_alphas;
        const float scale = _head_scale + (alphas ? alphas[s] * _ramp_head_scale_delta : 0.f);
        output[ch][s] = scale * final_head_outputs(ch, s);""")

replace("wavenet/hyperwavenet.h", "  [[nodiscard]] bool HasActiveWeightSmoothing() const noexcept;\n\nprivate:",
        "  [[nodiscard]] bool HasActiveWeightSmoothing() const noexcept;\n  void ConfigureKernelSmoothing(int transition_frames, int curve) noexcept;\n\nprotected:\n  void SetMaxBufferSize(int maxBufferSize) override;\n\nprivate:")
replace("wavenet/hyperwavenet.h", "  std::vector<float> _start_weights;\n  nam::Hypernetwork",
        "  std::vector<float> _start_weights;\n  std::vector<float> _kernel_alphas;\n  int _kernel_curve = 0;\n  nam::Hypernetwork")
replace("wavenet/hyperwavenet.cpp", '#include "hyperwavenet.h"',
        '#include "hyperwavenet.h"\n#include "../kernel_weight_smoothing.h"')
replace("wavenet/hyperwavenet.cpp", "  _smooth_step_frames = std::max(1, step_frames);",
        "  _smooth_step_frames = std::max(0, step_frames); // 0 means per-sample kernel interpolation")
replace("wavenet/hyperwavenet.cpp",
        "bool HyperWaveNet::HasActiveWeightSmoothing() const noexcept\n{\n  return _smooth_frames > 0 && _has_applied_weights && _smooth_position < _smooth_frames;\n}\n\nvoid HyperWaveNet::process",
        """bool HyperWaveNet::HasActiveWeightSmoothing() const noexcept
{
  return _smooth_frames > 0 && _has_applied_weights && _smooth_position < _smooth_frames;
}

void HyperWaveNet::ConfigureKernelSmoothing(const int transition_frames, const int curve) noexcept
{
  if (HasActiveWeightSmoothing()) return; // Preserve the current ramp.
  _kernel_curve = std::clamp(curve, 0, 3);
  ConfigureWeightSmoothing(transition_frames, 0);
  _smooth_position = _smooth_frames; // Avoid a phantom ramp on duration increase.
}

void HyperWaveNet::SetMaxBufferSize(const int frames)
{
  WaveNet::SetMaxBufferSize(frames);
  _kernel_alphas.resize(static_cast<size_t>(frames));
}

void HyperWaveNet::process""")
replace("wavenet/hyperwavenet.cpp",
        "    // Only ONE expensive hypernetwork evaluation for the new destination.\n    _hypernet.ApplyConditioning(_base_weights, _params, _conditioned);",
        """    // Restart interrupted automation from the last audible coefficients.
    if (_smooth_step_frames == 0 && _has_applied_weights &&
        _smooth_position < _smooth_frames)
    {
      const int last = std::max(0, _smooth_position - 1);
      const float x = static_cast<float>(last) / static_cast<float>(_smooth_frames);
      const float a = kernel_weight_smoothing::EvaluateAlpha(x, _kernel_curve);
      for (size_t i = 0; i < _smoothed_weights.size(); ++i)
        _smoothed_weights[i] = _start_weights[i] + a * (_conditioned[i] - _start_weights[i]);
    }
    // Only ONE expensive hypernetwork evaluation for the new destination.
    _hypernet.ApplyConditioning(_base_weights, _params, _conditioned);""")
replace("wavenet/hyperwavenet.cpp",
        "      _smooth_position = 0;\n      _smooth_step_remaining = 0;\n    }\n    _dirty = false;",
        """      _smooth_position = 0;
      _smooth_step_remaining = 0;
      if (_smooth_step_frames == 0)
      {
        WaveNet::set_weights_(_start_weights);
        kernel_weight_smoothing::ScopedTargetCapture capture;
        WaveNet::set_weights_(_conditioned);
      }
    }
    _dirty = false;""")
replace("wavenet/hyperwavenet.cpp",
        "  if (!HasActiveWeightSmoothing())\n  {\n    WaveNet::process(input, output, num_frames);\n    return;\n  }\n\n  int offset = 0;",
        """  if (!HasActiveWeightSmoothing())
  {
    WaveNet::process(input, output, num_frames);
    return;
  }

  if (_smooth_step_frames == 0)
  {
    // One normal-size block; all kernel deltas receive an alpha per sample.
    for (int i = 0; i < num_frames; ++i)
    {
      const int pos = std::min(_smooth_frames, _smooth_position + i);
      const float x = static_cast<float>(pos) / static_cast<float>(_smooth_frames);
      _kernel_alphas[static_cast<size_t>(i)] = kernel_weight_smoothing::EvaluateAlpha(x, _kernel_curve);
    }
    {
      kernel_weight_smoothing::ScopedBlockAlphas scope(_kernel_alphas.data());
      WaveNet::process(input, output, num_frames);
    }
    _smooth_position = std::min(_smooth_frames, _smooth_position + num_frames);
    if (_smooth_position == _smooth_frames)
    {
      std::copy(_conditioned.begin(), _conditioned.end(), _smoothed_weights.begin());
      WaveNet::set_weights_(_smoothed_weights);
    }
    return;
  }

  int offset = 0;""")
print("Direct-kernel core experiment applied", core)
