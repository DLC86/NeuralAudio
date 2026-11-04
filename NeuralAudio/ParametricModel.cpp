#include "ParametricModel.h"

#include <array>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <NAM/dsp.h>
#include <NAM/get_dsp.h>
#include <NAM/parametric_control.h>
#include <NAM/wavenet/hyper_a2.h>

namespace NeuralAudio
{
struct ParametricModel::Impl
{
    std::vector<std::unique_ptr<nam::DSP>> models;
    std::vector<nam::IParametricControl*> controls;
    std::vector<ControlSpec> specs;
    nlohmann::json metadata = nlohmann::json::object();
    size_t active = 0;
    int blockSize = 128;
    float outputLevel = 12, outputGain = 1;
    OutputMode requested = OutputMode::Calibrated;
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
    const bool a2 = source.at("architecture") == "HyperA2";
    nam::ScopedPrewarmOnResetDefault noPrewarm(false);
    for (int i = 0; i < (a2 ? 2 : 1); ++i)
    {
        auto child = a2 ? nam::wavenet::SelectHyperA2Branch(source, i == 0) : source;
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
    for (const auto& spec : next->controls.front()->GetParamSpecs())
        next->specs.push_back({spec.name, spec.min, spec.max, spec.defaultValue, spec.enum_names});
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
    return impl->controls.at(impl->active)->GetParams();
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
    for (auto* control : impl->controls) control->SetParams(values);
}
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
        const auto count = static_cast<int>(std::min(samples, static_cast<size_t>(impl->blockSize)));
        auto* in = const_cast<float*>(input);
        auto* out = output;
        model->process(&in, &out, count);
        if (impl->outputGain != 1)
            for (int i = 0; i < count; ++i) output[i] *= impl->outputGain;
        input += count; output += count; samples -= count;
    }
}
void ParametricModel::Prewarm()
{
    for (auto& model : impl->models)
    {
        // Reset commits ConcatWaveNet's control ramps as well as audio history.
        // Prewarm is offline, including after a Full/Lite switch.
        model->Reset(model->GetExpectedSampleRate(), impl->blockSize);
        model->prewarm();
    }
}
} // namespace NeuralAudio
