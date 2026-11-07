#include "NeuralAudio/ParametricModel.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

using NeuralAudio::OutputMode;
using NeuralAudio::ParametricModel;
void Check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
nlohmann::json Read(const std::filesystem::path& path) {
    std::ifstream file(path);
    return nlohmann::json::parse(file);
}
std::unique_ptr<ParametricModel> Load(nlohmann::json source, NeuralAudio::NeuralModelLoader& loader) {
    std::unique_ptr<NeuralAudio::NeuralModel> model(loader.CreateFromJson(source, ".pnam"));
    auto* p = dynamic_cast<ParametricModel*>(model.get());
    Check(p != nullptr, "PNAM loader did not create a parametric model");
    model.release();
    return std::unique_ptr<ParametricModel>(p);
}

void TestOutputVolume(const nlohmann::json& base, NeuralAudio::NeuralModelLoader& loader) {
    auto source=base;
    const nlohmann::json curve={{"schema_version",1},{"name","Volume"},{"index",1},
        {"min",0.},{"max",10.},{"default",5.},{"interpolation","pchip"},
        {"points",{{0.,0.},{10.,1.}}},
        {"segments",{{{"left",0.},{"right",10.},{"coefficients",{0.,0.,.1,0.}}}}}};
    source["architecture"]="ParametricOutputGain";
    source["config"]={{"base_architecture",base["architecture"]},
        {"base_config",base["config"]},{"output_gain",curve}};
    for (float quality : {1.f,.5f}) {
        auto model=Load(source,loader), reference=Load(base,loader);
        model->SetQualityScaleFactor(quality); reference->SetQualityScaleFactor(quality);
        model->SetOutputMode(OutputMode::Raw); reference->SetOutputMode(OutputMode::Raw);
        Check(model->GetControls().size()==3 && model->GetControls()[1].name=="Volume","Volume UI order");
        model->SetControlValues(std::vector<float>{2,5,0});
        reference->SetControlValues(std::vector<float>{2,0});
        model->Prewarm(); reference->Prewarm();
        std::vector<float> input(4096,.1f), actual(4096), expected(4096);
        model->Process(input.data(),actual.data(),4096);
        reference->Process(input.data(),expected.data(),4096);
        for (size_t i=0;i<actual.size();++i)
            Check(std::abs(actual[i]-.5f*expected[i])<2e-5,"Volume must scale output after neural processing");
        model->SetControlValues(std::vector<float>{2,0,0});
        model->Process(input.data(),actual.data(),4096);
        Check(std::abs(actual.back())<1e-7,"Smoothed volume must reach mute");
        bool rejected=false;
        try { model->SetControlValues(std::vector<float>{2,11,0}); } catch (...) {rejected=true;}
        Check(rejected && model->GetControlValues()[1]==0,"Invalid volume changed controls");
    }
}

int main(int argc, char** argv) {
    try {
        Check(argc == 2, "Usage: pnam_test <fixtures>");
        const std::filesystem::path dir(argv[1]);
        const auto manifest = Read(dir / "manifest.json");
        const auto input = manifest.at("input").get<std::vector<float>>();
        NeuralAudio::NeuralModelLoader loader;
        loader.SetDefaultMaxAudioBufferSize(257);
        TestOutputVolume(Read(dir/"hyper_a2_low_rank.nam"),loader);
        double maximum = 0;
        for (const auto& test : manifest.at("cases")) {
            const auto source = Read(dir / test.at("model").get<std::string>());
            auto model = Load(source, loader);
            Check(model->GetQualityScaleFactor() == 1, "Full must be the default");
            Check(model->GetControls().size() == 2 && model->GetControls()[1].IsSwitch(), "Control specs");
            model->SetOutputMode(OutputMode::Raw);
            for (bool lite : {false, true}) {
                model->SetQualityScaleFactor(lite ? 0.5f : 1.0f);
                for (int index = 0; index < 2; ++index) {
                    const auto values = manifest.at("params")[index].get<std::vector<float>>();
                    model->SetControlValues(values);
                    model->Prewarm();
                    const auto expected = test.at("expected").at(lite ? "lite" : "full")[index]
                                              .get<std::vector<float>>();
                    std::vector<float> output(input.size());
                    model->Process(input.data(), output.data(), output.size()); // internally chunks safely
                    for (size_t i = 0; i < output.size(); ++i) {
                        Check(std::isfinite(output[i]), "Nonfinite audio");
                        maximum = std::max(maximum, std::abs(static_cast<double>(output[i]) - expected[i]));
                    }
                }
            }
            const auto before = std::vector<float>(model->GetControlValues().begin(), model->GetControlValues().end());
            auto invalid = before; invalid[1] = 0.5f;
            bool rejected = false;
            try { model->SetControlValues(invalid); } catch (const std::invalid_argument&) { rejected = true; }
            Check(rejected && std::equal(before.begin(), before.end(), model->GetControlValues().begin()),
                  "Invalid switch changed control values");
        }
        Check(maximum < 2e-5, "PNAM disagrees with PyTorch");

        const auto combinations = Read(dir / "switch_combinations.json");
        const auto switchInput = combinations.at("input").get<std::vector<float>>();
        size_t renders = 0;
        for (const auto& test : combinations.at("cases")) {
            auto model = Load(Read(dir / test.at("model").get<std::string>()), loader);
            model->SetOutputMode(OutputMode::Raw);
            Check(model->GetControls().size() == 5 && model->GetControls()[0].positions.size() == 3 &&
                  model->GetControls()[2].positions.size() == 2 && model->GetControls()[4].positions.size() == 5,
                  "Mixed control definitions");
            for (bool lite : {false, true}) {
                if (lite && !test.at("dual_size").get<bool>()) continue;
                model->SetQualityScaleFactor(lite ? 0.5f : 1.0f);
                for (size_t i = 0; i < combinations.at("settings").size(); ++i) {
                    const auto values = combinations.at("settings")[i].get<std::vector<float>>();
                    model->SetControlValues(values);
                    model->Prewarm();
                    std::vector<float> output(switchInput.size());
                    model->Process(switchInput.data(), output.data(), output.size());
                    const auto expected = test.at("expected").at(lite ? "lite" : "full")[i].get<std::vector<float>>();
                    double difference = 0;
                    for (size_t j = 0; j < output.size(); ++j) {
                        Check(std::isfinite(output[j]), "Nonfinite switch audio");
                        difference = std::max(difference, std::abs(static_cast<double>(output[j]) - expected[j]));
                    }
                    if (difference >= 2e-5)
                        throw std::runtime_error("Switch parity " + test.at("model").get<std::string>() +
                            " lite=" + std::to_string(lite) + " row=" + std::to_string(i) +
                            " max_error=" + std::to_string(difference));
                    ++renders;
                }
            }
        }
        Check(renders == 120, "Not all 3 x 2 x 5 switch combinations rendered");

        auto source = Read(dir / "hyper_a2_low_rank.nam");
        for (const char* gear : {"amp", "PEDAL_AMP", "amp_cab", "amp_pedal_cab", "pedal", "preamp", "studio", "unknown", ""}) {
            source["metadata"] = {{"input_level_dbu", 18}, {"output_level_dbu", 6}, {"loudness", -24},
                                  {"gear_type", gear}, {"custom", {{"esr", {0.01, 0.02}}}}};
            auto model = Load(source, loader);
            Check(model->GetMetadataJson() == source.at("metadata"), "Metadata was lost");
            model->SetAudioInputLevelDBu(12);
            model->SetOutputMode(OutputMode::Calibrated);
            Check(model->GetOutputGainDB() == -6, "Calibration sign");
            model->SetOutputMode(OutputMode::Normalized);
            Check(model->GetOutputGainDB() == 6, "Normalization gain");
            // Check rendered audio, not just the gain calculation.
            model->SetOutputMode(OutputMode::Raw);
            model->Prewarm();
            std::vector<float> raw(input.size()), normalized(input.size());
            model->Process(input.data(), raw.data(), raw.size());
            model->SetOutputMode(OutputMode::Normalized);
            model->Prewarm();
            model->Process(input.data(), normalized.data(), normalized.size());
            const float gain = std::pow(10.0f, 6.0f / 20);
            for (size_t i = 0; i < raw.size(); ++i)
                Check(std::abs(normalized[i] - raw[i] * gain) < 2e-5, "Output mode not applied to audio");
            model->SetOutputMode(OutputMode::Auto);
            const bool amp = std::string(gear) == "amp" || std::string(gear) == "PEDAL_AMP" ||
                std::string(gear) == "amp_cab" || std::string(gear) == "amp_pedal_cab";
            Check(model->GetEffectiveOutputMode() == (amp ? OutputMode::Normalized : OutputMode::Calibrated), "Auto gear rule");
            for (const char* missing : {"output_level_dbu", "loudness"}) {
                auto legacy = source;
                legacy["metadata"].erase(missing);
                auto old = Load(legacy, loader);
                const auto mode = std::string(missing) == "loudness" ? OutputMode::Normalized : OutputMode::Calibrated;
                old->SetOutputMode(mode);
                Check(old->GetEffectiveOutputMode() == OutputMode::Raw && old->GetOutputGainDB() == 0,
                      "Legacy model did not fall back to Raw");
                Check(old->GetOutputMode() == mode, "Requested mode was lost");
                old->SetOutputMode(OutputMode::Auto);
                if ((amp && std::string(missing) == "loudness") || (!amp && std::string(missing) == "output_level_dbu"))
                    Check(old->GetEffectiveOutputMode() == OutputMode::Raw, "Auto missing metadata fallback");
            }
        }
        for (auto invalid : {nlohmann::json(), nlohmann::json("bad"), nlohmann::json(1e300)}) {
            source["metadata"] = {{"output_level_dbu", invalid}, {"loudness", invalid}};
            auto model = Load(source, loader);
            for (auto mode : {OutputMode::Calibrated, OutputMode::Normalized}) {
                model->SetOutputMode(mode);
                Check(model->GetEffectiveOutputMode() == OutputMode::Raw, "Invalid level fallback");
            }
        }

        std::cout << "PNAM C++ controls, output modes, legacy fallback and PyTorch parity passed; max_error=" << maximum << '\n';
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
