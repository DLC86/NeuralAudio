#pragma once

#include <memory>
#include <span>
#include "NeuralModelImpl.h"

namespace NeuralAudio
{
struct ControlSpec
{
    std::string name;
    float minimum, maximum, defaultValue;
    std::vector<std::string> positions;
    bool IsSwitch() const { return !positions.empty(); }
};

// Mono PNAM runtime, independent of JUCE. Load/prepare/quality changes are
// non-realtime operations. Serialize control updates with Process on the audio thread.
class ParametricModel final : public NeuralModelImpl
{
public:
    ParametricModel();
    ~ParametricModel() override;
    void LoadFromJson(const nlohmann::json& source);
    const std::vector<ControlSpec>& GetControls() const;
    std::span<const float> GetControlValues() const;
    void SetControlValues(std::span<const float> values);
    const nlohmann::json& GetMetadataJson() const;
    bool HasInputCalibration() const;
    bool HasOutputCalibration() const;
    bool HasLoudness() const;

    void SetOutputMode(OutputMode mode) override;
    OutputMode GetOutputMode() const override;
    OutputMode GetEffectiveOutputMode() const override;
    void SetAudioInputLevelDBu(float level) override;
    float GetOutputGainDB() const;
    float GetRecommendedInputDBAdjustment() override;
    // Output gain is already applied by Process; prevents legacy hosts applying it twice.
    float GetRecommendedOutputDBAdjustment() override { return 0; }

    EModelLoadMode GetLoadMode() override { return EModelLoadMode::NAMCore; }
    bool HasQualityScaling() override;
    float GetQualityScaleFactor() override;
    bool IsQualityChangeRealtimeSafe(float factor) override;
    void SetQualityScaleFactor(float factor) override;
    int GetReceptiveFieldSize() override;
    void SetMaxAudioBufferSize(int size) override;
    void Process(const float* input, float* output, size_t samples) override;
    void Prewarm() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace NeuralAudio
