#include "ParametricModel.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <NAM/dsp.h>
#include <NAM/get_dsp.h>
#include <NAM/parametric_control.h>
#include <NAM/wavenet/hyper_a2.h>
#include <NAM/kernel_weight_smoothing.h>
#include "OutputGainCurve.h"

namespace NeuralAudio
{
struct ParametricModel::Impl
{
    std::vector<std::unique_ptr<nam::DSP>> models;
    std::vector<nam::IParametricControl*> controls;
    std::vector<nam::kernel_weight_smoothing::WeightSmoothingControl*> smoothers;
    std::vector<bool> controlPending, smoothingConfigPending;
    std::vector<int> updateCooldown;
    float weightUpdateIntervalMs = 100.0f;
    float weightSmoothingMs = 100.0f;

    int SamplesForMs(size_t branch, float ms) const
    {
        const double rate = models.at(branch)->GetExpectedSampleRate();
        return std::max(0, static_cast<int>(std::lround(rate * ms / 1000.0)));
    }

    void ConfigureSmoothing(size_t branch)
    {
        auto* smoothing = smoothers.at(branch);
        if (!smoothing || !smoothingConfigPending[branch] ||
            smoothing->HasActiveWeightSmoothing()) return;
        smoothing->ConfigureKernelSmoothing(SamplesForMs(branch, weightSmoothingMs), 2);
        smoothingConfigPending[branch] = false;
    }
    std::vector<ControlSpec> specs;
    pnam::OutputGainCurve volume;
    std::vector<float> values, neuralValues;
    // Optional display-order mapping. Neural weights always use original indices.
    std::vector<size_t> displayToNative;
    std::vector<float> displayValues;
    double volumeTarget=1,volumeCurrent=1,volumeAlpha=1;
    nlohmann::json metadata = nlohmann::json::object();
    size_t active = 0;
    int blockSize = 128;
    float outputLevel = 12, outputGain = 1;
    OutputMode requested = OutputMode::Auto;
    OutputMode effective = OutputMode::Raw;

    bool HasNumber(const char* key) const
    {
        auto it = metadata.find(key);
        return it != metadata.end() && it->is_number() &&
               std::isfinite(it->get<double>()) && std::isfinite(it->get<float>());
    }
    float GainDB() const
    {
        if (effective == OutputMode::Calibrated)
            return metadata.at("output_level_dbu").get<float>() - outputLevel;
        if (effective == OutputMode::Normalized)
            return -18.0f - metadata.at("loudness").get<float>();
        return 0;
    }
    void UpdateGain()
    {
        effective = requested;
        if (requested == OutputMode::Auto)
        {
            std::string gear;
            auto it = metadata.find("gear_type");
            if (it != metadata.end() && it->is_string()) gear = it->get<std::string>();
            std::transform(gear.begin(), gear.end(), gear.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            effective = (gear == "amp" || gear == "pedal_amp" || gear == "amp_cab" ||
                         gear == "amp_pedal_cab") ? OutputMode::Normalized : OutputMode::Calibrated;
        }
        if ((effective == OutputMode::Calibrated && !HasNumber("output_level_dbu")) ||
            (effective == OutputMode::Normalized && !HasNumber("loudness")))
            effective = OutputMode::Raw;
        outputGain = std::pow(10.0f, GainDB() / 20.0f);
        if (!std::isfinite(outputGain)) { effective = OutputMode::Raw; outputGain = 1; }
    }
};

ParametricModel::ParametricModel() : impl(std::make_unique<Impl>()) { loader = nullptr; }
ParametricModel::~ParametricModel() = default;

void ParametricModel::LoadFromJson(const nlohmann::json& source)
{
    auto next = std::make_unique<Impl>();
    next->requested = impl->requested;
    next->outputLevel = impl->outputLevel;
    next->weightUpdateIntervalMs = impl->weightUpdateIntervalMs;
    next->weightSmoothingMs = impl->weightSmoothingMs;
    const auto* gainConfig = pnam::OutputGainCurve::FindConfig(source);
    const auto base=pnam::OutputGainCurve::NeuralSource(source);
    const auto architecture = base.at("architecture").get<std::string>();
    const bool a2 = architecture == "HyperA2" ||
                    architecture == "ParametricSlimmableContainer";
    nam::ScopedPrewarmOnResetDefault noPrewarm(false);
    for (int i = 0; i < (a2 ? 2 : 1); ++i)
    {
        auto child = a2 ? nam::wavenet::SelectHyperA2Branch(base, i == 0) : base;
        // Preserve the source metadata verbatim for the host, but do not pass
        // invalid optional level fields to NAM Core's numeric metadata parser.
        if (child.contains("metadata") && !child["metadata"].is_object())
            child["metadata"] = nlohmann::json::object();
        for (const char* key : {"input_level_dbu", "output_level_dbu", "loudness"})
            if (child.contains("metadata") && child["metadata"].contains(key)) {
                const auto& v = child["metadata"][key];
                if (!v.is_number() || !std::isfinite(v.get<double>()) || !std::isfinite(v.get<float>()))
                    child["metadata"].erase(key);
            }
        auto model = nam::get_dsp(child);
        auto* control = dynamic_cast<nam::IParametricControl*>(model.get());
        if (!control || model->NumInputChannels() != 1 || model->NumOutputChannels() != 1)
            throw std::runtime_error("PNAM requires a mono parametric model");
        next->controls.push_back(control);
        next->models.push_back(std::move(model));
    }
    next->active = next->models.size() - 1; // Full is the default, independent of export preference.
    next->smoothers.resize(next->models.size(), nullptr);
    next->controlPending.resize(next->models.size(), false);
    next->smoothingConfigPending.resize(next->models.size(), true);
    next->updateCooldown.resize(next->models.size(), 0);
    for (size_t i = 0; i < next->models.size(); ++i) {
        next->smoothers[i] =
            dynamic_cast<nam::kernel_weight_smoothing::WeightSmoothingControl*>(next->models[i].get());
        next->ConfigureSmoothing(i);
    }
    for (const auto& spec : next->controls.front()->GetParamSpecs())
        next->specs.push_back({spec.name, spec.min, spec.max, spec.defaultValue, spec.enum_names});
    const auto initial = next->controls.front()->GetParams();
    next->neuralValues.assign(initial.begin(), initial.end());
    next->values = next->neuralValues;
    if (gainConfig) {
        next->volume.Load(*gainConfig,next->specs.size());
        for (const auto& s:next->specs) if (s.name==next->volume.name)
            throw std::runtime_error("Output volume is already a neural control");
        next->values.insert(next->values.begin()+static_cast<std::ptrdiff_t>(next->volume.index),next->volume.defaultValue);
        next->specs.insert(next->specs.begin()+static_cast<std::ptrdiff_t>(next->volume.index),ControlSpec{
            next->volume.name,static_cast<float>(next->volume.minimum),static_cast<float>(next->volume.maximum),static_cast<float>(next->volume.defaultValue),{}});
        next->volumeTarget=next->volumeCurrent=next->volume.Evaluate(next->volume.defaultValue);
        const double rate=next->models.front()->GetExpectedSampleRate();
        next->volumeAlpha=1-std::exp(-1/(.005*(rate>0 ? rate : 48000)));
    }
    // New files can assign a single presentation index across neural controls
    // and Linear Volume. Older files without neural indices remain unchanged.
    const auto& stored = base.at("config").at("params");
    const bool indexed = std::any_of(stored.begin(), stored.end(),
        [](const nlohmann::json& entry) { return entry.contains("index"); });
    if (indexed) {
        const size_t count = next->specs.size();
        if (stored.size() + (next->volume.enabled ? 1u : 0u) != count)
            throw std::runtime_error("PNAM display control count mismatch");
        next->displayToNative.resize(count);
        std::vector<bool> seen(count, false);
        size_t neural = 0;
        for (size_t native = 0; native < count; ++native) {
            size_t display = native;
            if (!next->volume.enabled || native != next->volume.index) {
                const auto& entry = stored.at(neural++);
                if (!entry.contains("index") || !entry.at("index").is_number_integer())
                    throw std::runtime_error("PNAM requires all neural display indices");
                const int pos = entry.at("index").get<int>();
                if (pos < 0 || static_cast<size_t>(pos) >= count)
                    throw std::runtime_error("PNAM display index outside range");
                display = static_cast<size_t>(pos);
            }
            if (seen[display]) throw std::runtime_error("Duplicate PNAM display index");
            seen[display] = true;
            next->displayToNative[display] = native;
        }
        const auto nativeSpecs = next->specs;
        const auto rawValues = next->volume.enabled
            ? std::span<const float>(next->values)
            : next->controls.front()->GetParams();
        next->displayValues.resize(count);
        for (size_t display = 0; display < count; ++display) {
            const size_t native = next->displayToNative[display];
            next->specs[display] = nativeSpecs[native];
            next->displayValues[display] = rawValues[native];
        }
    }
    next->metadata = source.value("metadata", nlohmann::json::object());
    if (!next->metadata.is_object()) next->metadata = nlohmann::json::object();
    next->UpdateGain();
    impl = std::move(next);
    metadata.clear();
    modelInputLevelDBu = 12;
    modelOutputLevelDBu = 12;
    modelLoudnessDB = -18;
    ReadNAMConfig(source);
    SetMaxAudioBufferSize(loader ? loader->GetDefaultMaxAudioBufferSize() : 128);
    SetQualityScaleFactor(loader ? loader->GetDefaultQualityScaleFactor() : 1.0f);
}

const std::vector<ControlSpec>& ParametricModel::GetControls() const { return impl->specs; }
std::span<const float> ParametricModel::GetControlValues() const
{
    if (!impl->displayToNative.empty()) return impl->displayValues;
    return impl->values;
}
void ParametricModel::SetControlValues(std::span<const float> values)
{
    // Validate once, before committing either variant. Both branches share identical specs.
    if (values.size() != impl->specs.size()) throw std::invalid_argument("PNAM control count");
    for (size_t i = 0; i < values.size(); ++i)
    {
        const auto& s = impl->specs[i];
        const float v = values[i];
        if (!std::isfinite(v) || v < s.minimum || v > s.maximum ||
            (s.IsSwitch() && (v != std::floor(v) || v < 0 || v >= s.positions.size())))
            throw std::invalid_argument("PNAM control value outside its range");
    }
    std::vector<float> reordered;
    std::span<const float> nativeValues = values;
    if (!impl->displayToNative.empty()) {
        reordered.resize(values.size());
        for (size_t display = 0; display < values.size(); ++display)
            reordered[impl->displayToNative[display]] = values[display];
        nativeValues = reordered;
    }
    if (impl->volume.enabled) {
        bool changed=false;
        size_t n=0;
        for (size_t i=0;i<nativeValues.size();++i) if (i!=impl->volume.index) {
            changed |= impl->neuralValues[n]!=nativeValues[i];
            impl->neuralValues[n++]=nativeValues[i];
        }
        if (changed)
            for (size_t i=0;i<impl->controls.size();++i) {
                if (impl->smoothers[i]) impl->controlPending[i] = true;
                else impl->controls[i]->SetParams(impl->neuralValues);
            }
        std::copy(nativeValues.begin(),nativeValues.end(),impl->values.begin());
        impl->volumeTarget=impl->volume.Evaluate(nativeValues[impl->volume.index]);
    } else {
        const bool changed = !std::equal(nativeValues.begin(), nativeValues.end(),
                                         impl->neuralValues.begin());
        if (changed) {
            impl->neuralValues.assign(nativeValues.begin(), nativeValues.end());
            impl->values = impl->neuralValues;
            for (size_t i=0;i<impl->controls.size();++i) {
                if (impl->smoothers[i]) impl->controlPending[i] = true;
                else impl->controls[i]->SetParams(nativeValues);
            }
        }
    }
    if (!impl->displayToNative.empty())
        impl->displayValues.assign(values.begin(), values.end());
}
void ParametricModel::SetWeightUpdateIntervalMs(const float ms)
{
    if (!std::isfinite(ms) || ms < 0 || ms > 10000)
        throw std::invalid_argument("Weight-update interval must be 0..10000 ms");
    impl->weightUpdateIntervalMs = ms;
    for (size_t i=0;i<impl->updateCooldown.size();++i)
        impl->updateCooldown[i] = std::min(impl->updateCooldown[i], impl->SamplesForMs(i, ms));
}
void ParametricModel::SetWeightSmoothingMs(const float ms)
{
    if (!std::isfinite(ms) || ms < 0 || ms > 10000)
        throw std::invalid_argument("Weight smoothing must be 0..10000 ms");
    impl->weightSmoothingMs = ms;
    for (size_t i=0;i<impl->smoothers.size();++i) {
        impl->smoothingConfigPending[i] = true;
        impl->ConfigureSmoothing(i);
    }
}
float ParametricModel::GetWeightUpdateIntervalMs() const { return impl->weightUpdateIntervalMs; }
float ParametricModel::GetWeightSmoothingMs() const { return impl->weightSmoothingMs; }

const nlohmann::json& ParametricModel::GetMetadataJson() const { return impl->metadata; }
bool ParametricModel::HasInputCalibration() const { return impl->HasNumber("input_level_dbu"); }
bool ParametricModel::HasOutputCalibration() const { return impl->HasNumber("output_level_dbu"); }
bool ParametricModel::HasLoudness() const { return impl->HasNumber("loudness"); }
void ParametricModel::SetOutputMode(OutputMode mode)
{
    if (mode != OutputMode::Raw && mode != OutputMode::Calibrated &&
        mode != OutputMode::Normalized && mode != OutputMode::Auto)
        throw std::invalid_argument("Invalid output mode");
    impl->requested = mode;
    impl->UpdateGain();
}
OutputMode ParametricModel::GetOutputMode() const { return impl->requested; }
OutputMode ParametricModel::GetEffectiveOutputMode() const { return impl->effective; }
void ParametricModel::SetAudioInputLevelDBu(float value)
{
    if (!std::isfinite(value)) throw std::invalid_argument("Invalid interface calibration");
    NeuralModel::SetAudioInputLevelDBu(value);
    impl->outputLevel = value;
    impl->UpdateGain();
}
float ParametricModel::GetOutputGainDB() const { return impl->GainDB(); }
float ParametricModel::GetRecommendedInputDBAdjustment()
{
    return HasInputCalibration() ? NeuralModel::GetRecommendedInputDBAdjustment() : 0;
}
bool ParametricModel::HasQualityScaling() { return impl->models.size() == 2; }
float ParametricModel::GetQualityScaleFactor() { return HasQualityScaling() && impl->active == 0 ? 0.5f : 1; }
bool ParametricModel::IsQualityChangeRealtimeSafe(float factor)
{
    return !HasQualityScaling() || (factor <= 0.5f ? 0u : 1u) == impl->active;
}
void ParametricModel::SetQualityScaleFactor(float factor)
{
    if (!std::isfinite(factor)) throw std::invalid_argument("Invalid quality factor");
    impl->active = HasQualityScaling() && factor <= 0.5f ? 0 : impl->models.size() - 1;
}
int ParametricModel::GetReceptiveFieldSize() { return impl->models.at(impl->active)->GetPrewarmSamples(); }
void ParametricModel::SetMaxAudioBufferSize(int size)
{
    if (size <= 0) throw std::invalid_argument("PNAM buffer size must be positive");
    impl->blockSize = size;
    for (auto& model : impl->models) model->Reset(model->GetExpectedSampleRate(), size);
}
void ParametricModel::Process(const float* input, float* output, size_t samples)
{
    auto& model = impl->models.at(impl->active);
    while (samples)
    {
        impl->ConfigureSmoothing(impl->active);
        // One hypernetwork evaluation per configured update window. Pending
        // host moves collapse to the latest destination, independently of
        // the kernel ramp. This uses model-sample time, not wall-clock time.
        const size_t branch = impl->active;
        if (impl->smoothers[branch] && impl->controlPending[branch] &&
            impl->updateCooldown[branch] == 0) {
            impl->controls[branch]->SetParams(impl->neuralValues);
            impl->controlPending[branch] = false;
            impl->updateCooldown[branch] =
                impl->SamplesForMs(branch, impl->weightUpdateIntervalMs);
        }
        const auto count = static_cast<int>(std::min(samples, static_cast<size_t>(impl->blockSize)));
        auto* in = const_cast<float*>(input);
        auto* out = output;
        model->process(&in, &out, count);
        if (impl->volume.enabled) {
            for (int i=0;i<count;++i) {
                impl->volumeCurrent += impl->volumeAlpha*(impl->volumeTarget-impl->volumeCurrent);
                if (std::abs(impl->volumeTarget-impl->volumeCurrent)<1e-9) impl->volumeCurrent=impl->volumeTarget;
                output[i] *= static_cast<float>(impl->volumeCurrent)*impl->outputGain;
            }
        } else if (impl->outputGain != 1)
            for (int i = 0; i < count; ++i) output[i] *= impl->outputGain;
        if (impl->smoothers[branch])
            impl->updateCooldown[branch] =
                std::max(0, impl->updateCooldown[branch] - count);
        input += count; output += count; samples -= count;
    }
}
void ParametricModel::Prewarm()
{
    for (size_t i=0;i<impl->models.size();++i)
    {
        auto& model = impl->models[i];
        // Prewarm is explicitly offline; commit all pending controls and
        // reset the ramp before generating history for reproducible fixtures.
        impl->controls[i]->SetParams(impl->neuralValues);
        impl->controlPending[i] = false;
        impl->updateCooldown[i] = 0;
        // Reset commits ConcatWaveNet's control ramps as well as audio history.
        // Prewarm is offline, including after a Full/Lite switch.
        model->Reset(model->GetExpectedSampleRate(), impl->blockSize);
        model->prewarm();
    }
}
} // namespace NeuralAudio
