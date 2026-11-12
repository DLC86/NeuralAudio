// Deterministic, identical-host A/B/C benchmark of the NeuralAudio PR #1.
// Built against the chosen source revision; it does NOT change the DSP.
// Output: raw float32 audio plus per-block processing and preparation timings.
#include "NeuralAudio/ParametricModel.h"
#include "NeuralAudio/NeuralModelImpl.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

static std::unique_ptr<NeuralAudio::ParametricModel> Load(const fs::path& file,
                                                           NeuralAudio::NeuralModelLoader& loader) {
  std::ifstream input(file);
  if (!input) throw std::runtime_error("Missing fixture: " + file.string());
  auto json = nlohmann::json::parse(input);
  std::unique_ptr<NeuralAudio::NeuralModel> model(loader.CreateFromJson(json, ".pnam"));
  auto* p = dynamic_cast<NeuralAudio::ParametricModel*>(model.get());
  if (!p) throw std::runtime_error("Fixture is not parametric");
  model.release();
  return std::unique_ptr<NeuralAudio::ParametricModel>(p);
}
static double Microseconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::micro>(b-a).count();
}
static double Percentile(std::vector<double> a, double q) {
  if (a.empty()) return 0.;
  std::sort(a.begin(), a.end());
  return a[static_cast<size_t>(std::min<double>(a.size()-1, std::floor(q * (a.size()-1))))];
}
int main(int argc, char** argv) try {
  // fixture, scenario, block samples, quality, transfer mode, samples, output-prefix
  if (argc != 8) {
    std::cerr << "usage: pnam_pr_benchmark fixture scenario block quality direct|prepared samples prefix\n";
    return 2;
  }
  const fs::path fixture(argv[1]);
  const std::string scenario(argv[2]), transfer(argv[5]);
  const int block = std::stoi(argv[3]), total = std::stoi(argv[6]);
  const float quality = std::stof(argv[4]);
  const fs::path prefix(argv[7]);
  if (block < 16 || block > 8192 || total < 48000 || (quality!=1.f && quality!=.5f))
    throw std::runtime_error("Invalid benchmark parameters");
  if (scenario!="idle" && scenario!="slow" && scenario!="fast" && scenario!="step")
    throw std::runtime_error("Unknown scenario");
  if (transfer!="direct" && transfer!="prepared")
    throw std::runtime_error("Unknown transfer method");
#ifndef PR1_HAS_PREPARED
  if (transfer=="prepared") throw std::runtime_error("Prepared API is not compiled in this variant");
#endif

  NeuralAudio::NeuralModelLoader loader;
  auto model = Load(fixture, loader);
  model->SetOutputMode(NeuralAudio::OutputMode::Raw);
  model->SetQualityScaleFactor(quality);
  model->SetWeightUpdateIntervalMs(100.f);
  model->SetWeightSmoothingMs(100.f);
  model->SetMaxAudioBufferSize(block);
  model->Prewarm();
  const auto& specs = model->GetControls();
  std::vector<float> values(model->GetControlValues().begin(), model->GetControlValues().end());
  int knob = -1, sw = -1;
  for (size_t i=0;i<specs.size();++i) {
    if (!specs[i].IsSwitch() && knob < 0) knob = static_cast<int>(i);
    if (specs[i].IsSwitch() && sw < 0) sw = static_cast<int>(i);
  }
  if (knob < 0 && sw < 0) throw std::runtime_error("Fixture has no movable controls");

#ifdef PR1_HAS_PREPARED
  auto staging = model->MakePreparedControls();
#endif

  std::vector<float> input(static_cast<size_t>(total)), output(static_cast<size_t>(total));
  uint32_t seed=0x8f359b77u;
  constexpr double pi=3.14159265358979323846;
  for(int i=0;i<total;++i) {
    seed=seed*1664525u+1013904223u;
    const float noise=(static_cast<float>((seed>>12)&0xfffff)/524288.f-1.f)*0.009f;
    const double sec=static_cast<double>(i)/48000.0;
    input[static_cast<size_t>(i)] = 0.12f*std::sin(2.*pi*110.*sec)
                                   +0.055f*std::sin(2.*pi*1047.*sec)
                                   +0.018f*std::sin(2.*pi*5207.*sec)+noise;
  }
  int nextEvent = scenario=="idle" ? total+1 : scenario=="fast" ? 960
                  : scenario=="slow" ? 1920 : 12000;
  int eventInterval = scenario=="fast" ? 960 : scenario=="slow" ? 1920 : 24000;
  double processTotalUs = 0., prepareTotalUs = 0., eventTotalUs=0.;
  double maxBlockRatio = 0.;
  std::vector<double> processUs, blockRatio;
  size_t events=0;
  int pos=0;
  for (;pos<total;) {
    if(pos==nextEvent) {
      auto t0=Clock::now();
      ++events;
      if(knob>=0) {
        const auto& spec=specs[static_cast<size_t>(knob)];
        if(scenario=="slow")
          values[static_cast<size_t>(knob)] =
             spec.minimum+(spec.maximum-spec.minimum) * (0.5f+0.46f*std::sin(events*0.17f));
        else if(scenario=="fast")
          values[static_cast<size_t>(knob)] = (events&1)?spec.maximum:spec.minimum;
        else if(scenario=="step")
          values[static_cast<size_t>(knob)] = (events&1)?spec.maximum:spec.minimum;
      }
      if(sw>=0 && (scenario=="fast" ? events%7==0 : scenario=="step" ? true : events%6==0)) {
        const auto& spec=specs[static_cast<size_t>(sw)];
        const size_t index=events % spec.positionValues.size();
        values[static_cast<size_t>(sw)] = static_cast<float>(spec.positionValues[index]);
      }
      if(transfer=="direct") {
        model->SetControlValues(values);
      } else {
#ifdef PR1_HAS_PREPARED
        const auto p0=Clock::now();
        model->PrepareControlValues(values, staging);
        const auto p1=Clock::now();
        model->CommitPrepared(staging);
        prepareTotalUs += Microseconds(p0,p1);
#endif
      }
      auto t1=Clock::now();
      eventTotalUs += Microseconds(t0,t1);
      nextEvent+=eventInterval;
    }
    int count=std::min(block,total-pos);
    if(nextEvent>pos && nextEvent<pos+count) count=nextEvent-pos;
    if(count<=0) throw std::runtime_error("Benchmark block scheduling error");
    auto t0=Clock::now();
    model->Process(input.data()+pos,output.data()+pos,static_cast<size_t>(count));
    auto t1=Clock::now();
    const double elapsed=Microseconds(t0,t1);
    processTotalUs+=elapsed;
    processUs.push_back(elapsed);
    const double ratio=elapsed/(static_cast<double>(count)/48000.*1000000.);
    blockRatio.push_back(ratio*100.);
    maxBlockRatio=std::max(maxBlockRatio,ratio);
    pos+=count;
  }
  if(!std::all_of(output.begin(),output.end(),[](float s){return std::isfinite(s);}))
    throw std::runtime_error("Non-finite output");

  fs::create_directories(prefix.parent_path());
  {
    std::ofstream file(prefix.string()+".f32",std::ios::binary);
    file.write(reinterpret_cast<const char*>(output.data()),
               static_cast<std::streamsize>(output.size()*sizeof(float)));
    if(!file) throw std::runtime_error("Unable to write audio output");
  }
  nlohmann::json summary = {
    {"fixture",fixture.filename().string()},
    {"scenario",scenario},
    {"transfer",transfer},
    {"quality",quality},
    {"block",block},
    {"sample_rate",48000},
    {"samples",total},
    {"events",events},
    {"process_total_us",processTotalUs},
    {"event_total_us",eventTotalUs},
    {"prepare_total_us",prepareTotalUs},
    {"process_realtime_cpu_pct",processTotalUs/((double)total/48000.*1.e6)*100.},
    {"mean_process_block_us",processTotalUs/processUs.size()},
    {"p99_process_block_us",Percentile(processUs,.99)},
    {"max_process_block_us",Percentile(processUs,1.)},
    {"p99_deadline_pct",Percentile(blockRatio,.99)},
    {"worst_deadline_pct",maxBlockRatio*100.},
    {"compiler",std::string(__VERSION__)}
  };
  {
    std::ofstream file(prefix.string()+".json");
    file<<summary.dump(2)<<"\n";
    if(!file) throw std::runtime_error("Unable to write summary");
  }
  std::cout<<summary.dump()<<"\n";
  return 0;
} catch(const std::exception& e) {
  std::cerr<<"PR1 BENCH ERROR: "<<e.what()<<"\n";
  return 1;
}
