#!/usr/bin/env python3
"""Compare deterministic float32 audio and measured callback times for PR #1.

No weighted LUFS metric is used: amplitude, residual relative to program,
sample-level differences, residual derivative and event-adjacent peaks matter.
"""
import array
import csv
import json
import math
import pathlib
import statistics
import sys

OUT = pathlib.Path(sys.argv[1] if len(sys.argv)>1 else "out")
records = {}
for item in sorted(OUT.glob("*.json")):
    if item.name.startswith("report"): continue
    doc = json.loads(item.read_text())
    if not isinstance(doc,dict) or "process_total_us" not in doc: continue
    name = item.stem
    v = name.split("__")
    if len(v) != 6:
        raise RuntimeError("Unexpected output name: "+name)
    variant, fixture, scenario, block, quality, transfer = v
    key=(fixture,scenario,block,quality,transfer)
    records[(variant,*key)]=(doc,item.with_suffix(".f32"))

def get_audio(file):
    a=array.array("f")
    with file.open("rb") as f: a.frombytes(f.read())
    if sys.byteorder!="little": a.byteswap()
    return a

def db(x):
    return 20*math.log10(max(float(x),1e-20))

def compare(a,b,scenario):
    if len(a)!=len(b): raise ValueError("Different lengths")
    total=len(a)
    energy_ref = energy_err = peak_ref = peak_err = 0.
    energy_d_err=0.
    last_err=0.
    event_windows = []
    interval = {"idle":total+1,"slow":1920,"fast":960,"step":24000}[scenario]
    event = {"idle":total+1,"slow":1920,"fast":960,"step":12000}[scenario]
    # A 192-sample window centered on each automation event.
    while event < total:
        start=max(0,event-96)
        end=min(total,event+96)
        event_windows.append((start,end))
        event+=interval
    max_event_err=0.
    for i, (x,y) in enumerate(zip(a,b)):
        err=float(y)-float(x)
        peak_ref=max(peak_ref,abs(x))
        peak_err=max(peak_err,abs(err))
        energy_ref+=x*x
        energy_err+=err*err
        if i: energy_d_err+=(err-last_err)**2
        last_err=err
    for start,end in event_windows:
        max_event_err=max(max_event_err,max((abs(float(b[i])-float(a[i])) for i in range(start,end)),default=0.))
    ref_rms=math.sqrt(energy_ref/total)
    err_rms=math.sqrt(energy_err/total)
    return dict(
        reference_rms_dbfs=db(ref_rms),
        residual_rms_dbfs=db(err_rms),
        residual_relative_db=db(err_rms)-db(ref_rms),
        max_sample_delta=peak_err,
        max_sample_relative_to_ref_peak_db=db(peak_err)-db(peak_ref),
        rms_first_difference_dbfs=db(math.sqrt(energy_d_err/max(1,total-1))),
        worst_around_events=max_event_err,
        identical=peak_err==0.,
    )

audios={}
def audio(key):
    if key not in audios: audios[key]=get_audio(records[key][1])
    return audios[key]

results=[]
missing=[]
for k,(source_info,path) in records.items():
    v,fixture,scenario,block,quality,transfer=k
    if v=="base" and transfer=="direct": continue
    reference = ("base",fixture,scenario,block,quality,"direct")
    if transfer=="prepared":
        reference=("full",fixture,scenario,block,quality,"direct")
    if reference not in records:
        missing.append((k,reference))
        continue
    baseline=records[reference][0]
    result=compare(audio(reference),audio(k),scenario)
    result.update(dict(
        variant=v,fixture=fixture,scenario=scenario,block=block,quality=quality,
        transfer=transfer,
        reference_variant=reference[0],
        reference_cpu_pct=baseline["process_realtime_cpu_pct"],
        candidate_cpu_pct=source_info["process_realtime_cpu_pct"],
        candidate_cpu_ratio=source_info["process_total_us"]/baseline["process_total_us"],
        reference_p99_deadline_pct=baseline["p99_deadline_pct"],
        candidate_p99_deadline_pct=source_info["p99_deadline_pct"],
        reference_worst_deadline_pct=baseline["worst_deadline_pct"],
        candidate_worst_deadline_pct=source_info["worst_deadline_pct"],
        prepare_total_us=source_info["prepare_total_us"],
        control_events=source_info["events"],
    ))
    results.append(result)

if missing:
    raise RuntimeError("Missing reference results: "+str(missing[:5]))
if not results:
    raise RuntimeError("No comparisons generated")

# Hard fail for pronounced numerical differences; legitimate floating-point
# reordering is expected, so do not demand bit identity for GEMM optimization.
failures=[]
for row in results:
    if row["transfer"]=="prepared":
        if row["max_sample_delta"]>1e-6:
            failures.append(f"prepare-vs-direct difference: {row}")
    elif row["residual_relative_db"] > -80. and row["max_sample_delta"] > 1e-5:
        failures.append(f"significant audio deviation: {row}")

summary=[]
for variant in ["gemm","full"]:
    direct=[r for r in results if r["variant"]==variant and r["transfer"]=="direct"]
    if not direct: continue
    moving=[r for r in direct if r["scenario"]!="idle"]
    summary.append({
        "variant":variant,
        "number_of_audio_comparisons":len(direct),
        "worst_relative_residual_db":max(x["residual_relative_db"] for x in direct),
        "max_absolute_difference":max(x["max_sample_delta"] for x in direct),
        "median_moving_cpu_ratio":statistics.median(x["candidate_cpu_ratio"] for x in moving),
        "mean_moving_cpu_ratio":statistics.mean(x["candidate_cpu_ratio"] for x in moving),
        "median_idle_cpu_ratio":statistics.median(x["candidate_cpu_ratio"] for x in direct if x["scenario"]=="idle"),
        "highest_candidate_deadline_pct":max(x["candidate_worst_deadline_pct"] for x in moving),
    })
prepared=[r for r in results if r["transfer"]=="prepared"]
report={
    "measurements":"GitHub Actions ubuntu-24.04 runner, single runner, same compiler/host, deterministic input",
    "note":"Timing is measured wall-time spent inside Process; prep time is separate (prepared path). Benchmarks are synthetic fixtures, not real guitar amps.",
    "records":len(records),
    "comparisons":results,
    "summary":summary,
    "prepared_comparisons":len(prepared),
    "prepared_identical":all(x["identical"] for x in prepared),
    "failures":failures,
}
(OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
with (OUT/"measurements.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(results[0]))
    w.writeheader()
    w.writerows(results)

lines=[
"# NeuralAudio PR #1 — independent A/B/C validation",
"",
"Three builds on the same GitHub-hosted Linux x86-64 runner, same compiler and synthetic fixture audio.",
"Versions: baseline main; GEMM commit 7f5ffd0; full PR 41807a0.",
"All variants use **100 ms update cadence and 100 ms linear sample-by-sample smoothing**.",
"",
"## Summary",
"",
"| Variant | Comparisons | Worst residual (dB rel. signal) | Max abs sample delta | Median CPU ratio while moving | Median CPU ratio idle |",
"|---|---:|---:|---:|---:|---:|",
]
for s in summary:
    lines.append(f"| {s['variant']} | {s['number_of_audio_comparisons']} | {s['worst_relative_residual_db']:.1f} | {s['max_absolute_difference']:.3g} | {s['median_moving_cpu_ratio']:.3f}x | {s['median_idle_cpu_ratio']:.3f}x |")
lines+=["",
"## Audio caveats",
"",
"Residual comparisons are against baseline, aligned by construction with the same input and event indices, not manually aligned. The metrics include worst absolute sample difference, residual energy, first-difference residual and event-adjacent peak differences. These are not perceptual listening tests.",
"",
"## Moving controls: per-model/per-buffer timings",
"",
"| Version | Model | Quality | Scenario | Block | Base CPU % | New CPU % | CPU ratio | P99 deadline % (base/new) | Worst residual dBc |",
"|---|---|---|---|---:|---:|---:|---:|---:|---:|",
]
for r in results:
    if r["transfer"]!="direct" or r["scenario"]=="idle": continue
    lines.append(f"| {r['variant']} | {r['fixture']} | {r['quality']} | {r['scenario']} | {r['block']} | {r['reference_cpu_pct']:.1f} | {r['candidate_cpu_pct']:.1f} | {r['candidate_cpu_ratio']:.2f}x | {r['reference_p99_deadline_pct']:.1f}/{r['candidate_p99_deadline_pct']:.1f} | {r['residual_relative_db']:.1f} |")
lines+=["",
f"Prepared-path comparisons: **{len(prepared)}**; bit-identical audio to full PR direct path: **{all(x['identical'] for x in prepared)}**.",
"",
"**Important:** CPU improvements reported here are not Raspberry Pi measurements. Real-hardware validation and multi-thread stress tests would still be needed before shipping.",
"",
"Test-gate failures: "+str(len(failures)), "",
]
for fail in failures[:20]:lines.append("- "+fail[:300])
(OUT/"report.md").write_text("\n".join(lines)+"\n")
print("\n".join(lines[:18]))
print("FAILURES:",len(failures))
sys.exit(1 if failures else 0)
