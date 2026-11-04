#include "NeuralAudioCApi.h"
#include "NeuralModel.h"
#ifdef BUILD_PNAM
#include "NeuralAudio/ParametricModel.h"
#endif
#include <cstring>
#include <memory>
#include <stdexcept>

struct NeuralModel
{
    NeuralAudio::NeuralModel* model;
};

struct NeuralModelLoader
{
	NeuralAudio::NeuralModelLoader* loader;
};

NeuralModelLoader* CreateLoader()
{
	NeuralModelLoader* loader = new NeuralModelLoader();

	loader->loader = new NeuralAudio::NeuralModelLoader();

	return loader;
}

void DeleteLoader(NeuralModelLoader* loader)
{
	delete loader->loader;
	delete loader;
}

NeuralModel* CreateModelFromFile(NeuralModelLoader* loader, const wchar_t* modelPath)
{
    try {
        if (!loader || !modelPath) return nullptr;
        std::unique_ptr<NeuralAudio::NeuralModel> result(loader->loader->CreateFromFile(modelPath));
        return result ? new NeuralModel{result.release()} : nullptr;
    } catch (...) { return nullptr; }
}

NeuralModel* CreateModelFromFileUtf8(NeuralModelLoader* loader, const char* modelPath)
{
    try {
        if (!loader || !modelPath) return nullptr;
        const std::u8string utf8(reinterpret_cast<const char8_t*>(modelPath));
        std::unique_ptr<NeuralAudio::NeuralModel> result(loader->loader->CreateFromFile(std::filesystem::path(utf8)));
        return result ? new NeuralModel{result.release()} : nullptr;
    } catch (...) { return nullptr; }
}

void DeleteModel(NeuralModel* model)
{
	if (!model) return;
    delete model->model;
    delete model;
}

void SetLSTMLoadMode(NeuralModelLoader* loader, int loadMode)
{
	loader->loader->SetLSTMLoadMode((NeuralAudio::EModelLoadMode)loadMode);
}

void SetWaveNetLoadMode(NeuralModelLoader* loader, int loadMode)
{
	loader->loader->SetWaveNetLoadMode((NeuralAudio::EModelLoadMode)loadMode);
}

void SetAudioInputLevelDBu(NeuralModelLoader* loader, float audioDBu)
{
	loader->loader->SetAudioInputLevelDBu(audioDBu);
}

void SetDefaultMaxAudioBufferSize(NeuralModelLoader* loader, int maxSize)
{
	loader->loader->SetDefaultMaxAudioBufferSize(maxSize);
}

int GetLoadMode(NeuralModel* model)
{
	return model->model->GetLoadMode();
}

bool IsStatic(NeuralModel* model)
{
	return model->model->IsStatic();
}

void SetMaxAudioBufferSize(NeuralModel* model, int maxSize)
{
	model->model->SetMaxAudioBufferSize(maxSize);
}

float GetRecommendedInputDBAdjustment(NeuralModel* model)
{
	return model->model->GetRecommendedInputDBAdjustment();
}

float GetRecommendedOutputDBAdjustment(NeuralModel* model)
{
	return model->model->GetRecommendedOutputDBAdjustment();
}

float GetSampleRate(NeuralModel* model)
{
	return model->model->GetSampleRate();
}

void Process(NeuralModel* model, float* input, float* output, size_t numSamples)
{
    model->model->Process(input, output, numSamples);
}

namespace {
#ifdef BUILD_PNAM
NeuralAudio::ParametricModel* Pnam(NeuralModel* model) {
    return model ? dynamic_cast<NeuralAudio::ParametricModel*>(model->model) : nullptr;
}
template <typename F> int WithPnam(NeuralModel* model, F action) {
    try {
        auto* p = Pnam(model);
        if (!p) return 0;
        action(*p);
        return 1;
    } catch (...) { return 0; }
}
#endif
}

bool IsParametric(NeuralModel* model) {
#ifdef BUILD_PNAM
    return Pnam(model) != nullptr;
#else
    return false;
#endif
}
size_t PnamGetControlCount(NeuralModel* model) {
#ifdef BUILD_PNAM
    if (auto* p = Pnam(model)) return p->GetControls().size();
#endif
    return 0;
}
const char* PnamGetControlName(NeuralModel* model, size_t index) {
#ifdef BUILD_PNAM
    if (auto* p = Pnam(model); p && index < p->GetControls().size())
        return p->GetControls()[index].name.c_str();
#endif
    return nullptr;
}
int PnamGetControlRange(NeuralModel* model, size_t index, float* low, float* high, float* nominal) {
#ifdef BUILD_PNAM
    return WithPnam(model, [&](auto& p) {
        if (!low || !high || !nominal) throw std::invalid_argument("NULL range");
        const auto& s = p.GetControls().at(index);
        *low = s.minimum; *high = s.maximum; *nominal = s.defaultValue;
    });
#else
    return 0;
#endif
}
size_t PnamGetPositionCount(NeuralModel* model, size_t index) {
#ifdef BUILD_PNAM
    if (auto* p = Pnam(model); p && index < p->GetControls().size())
        return p->GetControls()[index].positions.size();
#endif
    return 0;
}
const char* PnamGetPositionName(NeuralModel* model, size_t index, size_t position) {
#ifdef BUILD_PNAM
    if (auto* p = Pnam(model); p && index < p->GetControls().size()) {
        const auto& positions = p->GetControls()[index].positions;
        if (position < positions.size()) return positions[position].c_str();
    }
#endif
    return nullptr;
}
int PnamSetControls(NeuralModel* model, const float* values, size_t count) {
#ifdef BUILD_PNAM
    return WithPnam(model, [&](auto& p) {
        if (!values) throw std::invalid_argument("NULL controls");
        p.SetControlValues(std::span<const float>(values, count));
    });
#else
    return 0;
#endif
}
int PnamSetOutputMode(NeuralModel* model, int mode) {
#ifdef BUILD_PNAM
    return WithPnam(model, [&](auto& p) { p.SetOutputMode(static_cast<NeuralAudio::OutputMode>(mode)); });
#else
    return 0;
#endif
}
int PnamGetOutputMode(NeuralModel* model) {
#ifdef BUILD_PNAM
    if (auto* p = Pnam(model)) return static_cast<int>(p->GetOutputMode());
#endif
    return -1;
}
int PnamGetEffectiveOutputMode(NeuralModel* model) {
#ifdef BUILD_PNAM
    if (auto* p = Pnam(model)) return static_cast<int>(p->GetEffectiveOutputMode());
#endif
    return -1;
}
int PnamSetCalibrationLevelDBu(NeuralModel* model, float level) {
#ifdef BUILD_PNAM
    return WithPnam(model, [&](auto& p) { p.SetAudioInputLevelDBu(level); });
#else
    return 0;
#endif
}
int PnamSetLite(NeuralModel* model, bool lite) {
#ifdef BUILD_PNAM
    return WithPnam(model, [&](auto& p) {
        if (lite && !p.HasQualityScaling()) throw std::invalid_argument("No Lite variant");
        p.SetQualityScaleFactor(lite ? 0.5f : 1.0f);
        p.Prewarm();
    });
#else
    return 0;
#endif
}
bool PnamHasLite(NeuralModel* model) {
#ifdef BUILD_PNAM
    if (auto* p = Pnam(model)) return p->HasQualityScaling();
#endif
    return false;
}
size_t PnamGetMetadataJson(NeuralModel* model, char* buffer, size_t capacity) {
#ifdef BUILD_PNAM
    try {
        if (auto* p = Pnam(model)) {
            auto text = p->GetMetadataJson().dump();
            if (buffer && capacity >= text.size() + 1) std::memcpy(buffer, text.c_str(), text.size() + 1);
            return text.size() + 1;
        }
    } catch (...) {}
#endif
    return 0;
}


