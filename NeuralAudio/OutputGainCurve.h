#pragma once

// Measured output attenuator, shared by ParaNAM Compare, the plugin and NeuralAudio.
// Copyright (c) 2026 DLC86. Distributed under the containing project's license.
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <NAM/dsp.h>

namespace pnam {
struct OutputGainCurve {
  struct Segment { double left, right; std::array<double,4> c; };
  bool enabled = false;
  size_t index = 0;
  std::string name;
  double minimum = 0, maximum = 1, defaultValue = 1;
  std::vector<Segment> segments;

  static nlohmann::json Unwrap(const nlohmann::json& source) {
    if (source.at("architecture") != "ParametricOutputGain") return source;
    auto result = source;
    result["architecture"] = source.at("config").at("base_architecture");
    result["config"] = source.at("config").at("base_config");
    if (result["architecture"] != "HyperA2" && result["architecture"] != "HyperWaveNet")
      throw std::runtime_error("Output gain requires HyperA2 or HyperWaveNet");
    return result;
  }

  void Load(const nlohmann::json& config, size_t neuralCount) {
    if (config.at("schema_version") != 1 || config.at("interpolation") != "pchip")
      throw std::runtime_error("Unsupported output-gain curve version");
    if (!config.at("index").is_number_integer() || config.at("index") < 0)
      throw std::runtime_error("Invalid output-gain control index");
    index = config.at("index").get<size_t>();
    name = config.at("name").get<std::string>();
    minimum = config.at("min").get<double>(); maximum = config.at("max").get<double>();
    defaultValue = config.at("default").get<double>();
    if (index > neuralCount || name.empty() || !std::isfinite(minimum) ||
        !std::isfinite(maximum) || !std::isfinite(defaultValue) || minimum >= maximum ||
        defaultValue < minimum || defaultValue > maximum)
      throw std::runtime_error("Invalid output-gain control metadata");
    const auto& points = config.at("points");
    const auto& raw = config.at("segments");
    if (!points.is_array() || points.size() < 2 || points.size() > 65 ||
        !raw.is_array() || raw.size()+1 != points.size())
      throw std::runtime_error("Invalid output-gain point count");
    double previousX = minimum, previousY = -1;
    for (size_t i=0; i<points.size(); ++i) {
      const double x = points[i].at(0).get<double>(), y = points[i].at(1).get<double>();
      if (!std::isfinite(x) || !std::isfinite(y) || y < 0 || y > 1 || y < previousY ||
          (i && x <= previousX) || (!i && x != minimum) ||
          (i+1 == points.size() && (x != maximum || std::abs(y-1)>1e-6)))
        throw std::runtime_error("Invalid output-gain measured points");
      previousX=x; previousY=y;
    }
    segments.clear();
    for (size_t i=0; i<raw.size(); ++i) {
      Segment s{raw[i].at("left").get<double>(),raw[i].at("right").get<double>(),{}};
      const auto& c = raw[i].at("coefficients");
      if (!c.is_array() || c.size()!=4 || s.left!=points[i][0].get<double>() || s.right!=points[i+1][0].get<double>())
        throw std::runtime_error("Invalid output-gain interpolation interval");
      for (size_t k=0;k<4;++k) {
        s.c[k]=c[k].get<double>();
        if (!std::isfinite(s.c[k])) throw std::runtime_error("Nonfinite output-gain coefficient");
      }
      const double width=s.right-s.left;
      const auto eval=[&](double t){return ((s.c[0]*t+s.c[1])*t+s.c[2])*t+s.c[3];};
      const auto derivative=[&](double t){return (3*s.c[0]*t+2*s.c[1])*t+s.c[2];};
      const double end=eval(width);
      if (!std::isfinite(end) || std::abs(eval(0)-points[i][1].get<double>())>1e-6 ||
          std::abs(end-points[i+1][1].get<double>())>1e-6 || derivative(0)<-1e-7 || derivative(width)<-1e-7)
        throw std::runtime_error("Output-gain interpolation disagrees with measurements");
      if (s.c[0]!=0) {
        const double critical=-s.c[1]/(3*s.c[0]);
        if (critical>0 && critical<width && derivative(critical)<-1e-7)
          throw std::runtime_error("Output-gain curve is not monotonic");
      }
      segments.push_back(s);
    }
    enabled=true;
  }

  double Evaluate(double value) const noexcept {
    if (!enabled) return 1;
    value=std::clamp(value,static_cast<double>(minimum),static_cast<double>(maximum));
    auto it=std::upper_bound(segments.begin(),segments.end(),value,
      [](double x,const Segment& s){return x<s.left;});
    if (it!=segments.begin()) --it;
    const double t=value-it->left;
    return std::clamp(((it->c[0]*t+it->c[1])*t+it->c[2])*t+it->c[3],0.,1.);
  }
};
} // namespace pnam
