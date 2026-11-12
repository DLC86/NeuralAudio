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
    std::vector<std::string> positions; // display order (index_sw)
    std::vector<int> positionValues; // associated original neural 0..N-1 value
    bool IsSwitch() const { return !positions.empty(); }
};

// Control values prepared off the audio thread (ParametricModel::PrepareControlValues): the
// network values, the Linear Volume target and, for the active network when it has a
// hypernetwork, its output. Sized once by MakePreparedControls; reusable.
struct PreparedControls
{
    std::vector<float> display, native, neural;
    double volumeTarget = 1;
    size_t branch = 0;
    bool conditioned = false; // `weights` holds the hypernetwork output for `neural`
    std::vector<float> weights;
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
    // SetControlValues in two halves. PrepareControlValues validates and runs the hypernetwork
    // of the active network on the calling thread (a control thread; it may overlap Process,
    // not another PrepareControlValues on this model). CommitPrepared hands the result over
    // between Process() calls like SetControlValues, without allocating or running the
    // hypernetwork; a model with the same file (2x oversampling's second copy) may take it too.
    PreparedControls MakePreparedControls() const;
    void PrepareControlValues(std::span<const float> values, PreparedControls& out) const;
    void CommitPrepared(const PreparedControls& prepared);
    // Independent, model-sample-time cadence and ramp (milliseconds). Both
    // default to 100 ms. HyperWaveNet uses linear per-sample coefficient ramps.
    // Call these setters between Process() calls, not concurrently with audio.
    void SetWeightUpdateIntervalMs(float ms);
    void SetWeightSmoothingMs(float ms);
    [[nodiscard]] float GetWeightUpdateIntervalMs() const;
    [[nodiscard]] float GetWeightSmoothingMs() const;
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
