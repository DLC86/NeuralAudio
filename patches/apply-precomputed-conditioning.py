#!/usr/bin/env python3
"""Let a host run the A2 hypernetwork off the audio thread.

Runs after apply-neuralaudio-fast-smoothing.py. Adds nam::IPrecomputedConditioning
(NAM/parametric_control.h) and implements it in FastHyperWaveNet: ComputeConditioned()
evaluates the hypernetwork on any thread, with a hypernetwork instance of its own (the
original keeps scratch buffers), and SetParamsConditioned() hands the result over, so
process() only installs it. ParametricModel::PrepareControlValues/CommitPrepared use it.
Without these calls nothing changes: SetParams() still evaluates in process().
"""
from pathlib import Path
import sys

core = Path(sys.argv[1] if len(sys.argv) > 1 else "deps/NeuralAmpModelerCore") / "NAM"
marker = "class IPrecomputedConditioning"
if marker in (core / "parametric_control.h").read_text():
    print("Precomputed hypernetwork conditioning already installed")
    sys.exit(0)

def replace(path, before, after):
    p = core / path
    content = p.read_text()
    count = content.count(before)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match for {before[:80]!r}, got {count}")
    p.write_text(content.replace(before, after))

replace("parametric_control.h",
        """  virtual const std::vector<ParamSpec>& GetParamSpecs() const = 0;
};
""",
        """  virtual const std::vector<ParamSpec>& GetParamSpecs() const = 0;
};

/// Optional, for models whose controls reach the audio through a hypernetwork: lets the host
/// run the hypernetwork on another thread (a control thread) and hand the result to the audio
/// thread, which then only installs it. Query via dynamic_cast<IPrecomputedConditioning*>.
///
/// Contract:
/// - ComputeConditioned() may run on any thread, concurrently with process()/SetParams() on the
///   audio thread: it uses a hypernetwork instance of its own. Calls to ComputeConditioned()
///   on one object must not overlap each other.
/// - SetParamsConditioned() follows SetParams()'s contract (serialized with process()) and is
///   allocation-free; `conditioned` must come from ComputeConditioned() with the same params.
class IPrecomputedConditioning
{
public:
  virtual ~IPrecomputedConditioning() = default;

  /// Number of floats ComputeConditioned() writes.
  virtual size_t ConditionedSize() const = 0;

  /// Runs the hypernetwork for `params` into `out` (resized to ConditionedSize(); no
  /// allocation once it has that size). \\throws std::invalid_argument on invalid params.
  virtual void ComputeConditioned(std::span<const float> params, std::vector<float>& out) const = 0;

  /// SetParams(params) with the hypernetwork output already computed: the next process()
  /// installs it (ramping as SetParams would) without evaluating the hypernetwork.
  virtual void SetParamsConditioned(std::span<const float> params, std::span<const float> conditioned) = 0;
};
""")
replace("wavenet/hyperwavenet.cpp",
        """class FastHyperWaveNet final : public nam::DSP, public nam::IParametricControl,
                               public nam::kernel_weight_smoothing::WeightSmoothingControl""",
        """class FastHyperWaveNet final : public nam::DSP, public nam::IParametricControl,
                               public nam::kernel_weight_smoothing::WeightSmoothingControl,
                               public nam::IPrecomputedConditioning""")
replace("wavenet/hyperwavenet.cpp",
        """  , _conditioned(_base)
  , _hypernet(spec, _specs, state)
  {""",
        """  , _conditioned(_base)
  , _hypernet(spec, _specs, state)
  , _external_hypernet(_hypernet)
  , _precomputed(_base.size())
  {""")
replace("wavenet/hyperwavenet.cpp",
        """  std::span<const float> GetParams() const override
  {
    nam::DebugParamApiGuard guard(_debug_param_api_active);
    return _params;
  }
  int ParamDim() const override { return static_cast<int>(_params.size()); }""",
        """  std::span<const float> GetParams() const override
  {
    nam::DebugParamApiGuard guard(_debug_param_api_active);
    return _params;
  }
  size_t ConditionedSize() const override { return _base.size(); }
  void ComputeConditioned(std::span<const float> values, std::vector<float>& out) const override
  {
    // The hypernetwork keeps scratch buffers: this thread uses a copy of its own.
    _external_hypernet.ValidateParams(values);
    _external_hypernet.ApplyConditioning(_base, values, out);
  }
  void SetParamsConditioned(std::span<const float> values, std::span<const float> conditioned) override
  {
    nam::DebugParamApiGuard guard(_debug_param_api_active);
    if (values.size() != _params.size() || conditioned.size() != _precomputed.size())
      throw std::invalid_argument("FastHyperWaveNet: precomputed conditioning of the wrong size");
    if (std::equal(values.begin(), values.end(), _params.begin()))
      return;
    std::copy(values.begin(), values.end(), _params.begin());
    std::copy(conditioned.begin(), conditioned.end(), _precomputed.begin());
    _have_precomputed = true;
    _dirty = true;
  }
  int ParamDim() const override { return static_cast<int>(_params.size()); }""")
replace("wavenet/hyperwavenet.cpp",
        """    _hypernet.ValidateParams(values);
    if (!std::equal(values.begin(), values.end(), _params.begin()))
    {
      std::copy(values.begin(), values.end(), _params.begin());
      _dirty = true;
    }
  }""",
        """    _hypernet.ValidateParams(values);
    if (!std::equal(values.begin(), values.end(), _params.begin()))
    {
      std::copy(values.begin(), values.end(), _params.begin());
      _have_precomputed = false;
      _dirty = true;
    }
  }""")
replace("wavenet/hyperwavenet.cpp",
        """      _hypernet.ApplyConditioning(_base, _params, _conditioned);
      if (!_has_applied || _smooth_frames == 0) {""",
        """      if (_have_precomputed)
        std::copy(_precomputed.begin(), _precomputed.end(), _conditioned.begin());
      else
        _hypernet.ApplyConditioning(_base, _params, _conditioned);
      _have_precomputed = false;
      if (!_has_applied || _smooth_frames == 0) {""")
replace("wavenet/hyperwavenet.cpp",
        """  nam::Hypernetwork _hypernet;
  int _smooth_frames = 0, _smooth_position = 0;
  bool _has_applied = false;""",
        """  nam::Hypernetwork _hypernet;
  nam::Hypernetwork _external_hypernet; // ComputeConditioned() only (another thread)
  std::vector<float> _precomputed; // from SetParamsConditioned(), installed by process()
  bool _have_precomputed = false;
  int _smooth_frames = 0, _smooth_position = 0;
  bool _has_applied = false;""")

print("Precomputed hypernetwork conditioning installed", core)
