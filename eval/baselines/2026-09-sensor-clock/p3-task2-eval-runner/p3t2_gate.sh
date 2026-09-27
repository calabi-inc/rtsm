#!/usr/bin/env bash
# P3 task 2 gate (G3-2, GPU): `rtsm eval` reproduces the anchor and isolates its runs.
#   1. `rtsm eval recordings/session1 --mode as_deployed --repeats 3 --set segmentation.backend=dual` (the replay source):
#      every run = 124/65 @53 processed, fingerprint ad6f71a5b89c8506, dequeue (86) + receiver (240) sequences identical to B1
#      ((frame_seq, t_sensor_ns, outcome, reason) / (decision, reason, frame_seq, t_sensor_ns, is_keyframe)); repeats.json
#      identical_fingerprints == true; pose ledger 240 lines, obs 695, view 53.
#   2. The same through recordings/session1_bag (the bag source): identical to 1.
#   3. The harness anchor through the refactored run.py (`benchmark_datasheet.py dual`, packaged defaults + a diagnostics profile):
#      124/65 @53 ad6f71a5b89c8506 with the same sequences -- the engine factory moved nothing.
#   4. `--mode dense` on session1: keyframes >= 1.0 s apart on the sensor clock, throttle 0.2 s, 0 gate_rejected outcomes,
#      >= 1 dequeue line with gate_shadow, every enqueued frame processed or frame_rejected (never gate_rejected); counts reported.
#   5. Isolation: rtsm/cfg/rtsm.yaml byte-identical before/after; model_store/faiss untouched (listing + mtimes); the three
#      replay runs' FAISS files exist in their own run dirs.
#   6. The CPU suite is run by the PR, not here.
set -u
cd /c/Users/konam/Desktop/calabi-repo/rtsm || exit 1
SP="C:/Users/konam/AppData/Local/Temp/claude/C--Users-konam-OneDrive-Desktop-calabi-repo-rtsm/949c0384-41c4-4c7e-9f67-2ca869f46d96/scratchpad"
OUT="$SP/p3t2_gate"; rm -rf "$OUT"; mkdir -p "$OUT"
sha256sum rtsm/cfg/rtsm.yaml > "$OUT/yaml.before"
ls -l --time-style=full-iso model_store/faiss > "$OUT/faiss.before" 2>/dev/null
run_eval() {  # label, then args
  local label="$1"; shift
  python -X utf8 -m rtsm eval "$@" --out "$OUT/$label" --set segmentation.backend=dual > "$OUT/$label.stdout" 2>&1; local rc=$?
  echo "== eval $label rc=$rc =="; tail -3 "$OUT/$label.stdout" | cut -c1-200
}
run_eval s1_replay recordings/session1 --mode as_deployed --repeats 3
run_eval s1_bag    recordings/session1_bag --mode as_deployed --repeats 1
run_eval s1_dense  recordings/session1 --mode dense --repeats 1
# 3. the harness through run.py (patches the yaml in place and restores it)
printf 'diagnostics:\n  enabled: true\n  ledgers: true\n  event_log_path: "%s/harness.events.jsonl"\n' "$OUT" > "$OUT/harness.profile.yaml"
moved=0
if [ -d model_store/faiss ] && [ ! -d model_store/faiss.pre-p3t2 ]; then mv model_store/faiss model_store/faiss.pre-p3t2; moved=1; fi
rm -rf model_store/faiss reports/datasheet_raw_dual*.json
python -X utf8 scripts/benchmark_datasheet.py dual --profile "$OUT/harness.profile.yaml" > "$OUT/harness.stdout" 2>&1; echo "== harness rc=$? =="
raw=$(ls reports/datasheet_raw_dual.*.json 2>/dev/null | head -1); cp -f "$raw" "$OUT/harness.json" 2>/dev/null
rm -rf model_store/faiss
if [ $moved = 1 ]; then mv model_store/faiss.pre-p3t2 model_store/faiss; fi
sha256sum rtsm/cfg/rtsm.yaml > "$OUT/yaml.after"
ls -l --time-style=full-iso model_store/faiss > "$OUT/faiss.after" 2>/dev/null

python -X utf8 - "$OUT" <<'PY'
import glob, json, os, sys
from collections import Counter
from rtsm.evaluation.ledger import by_kind, read_events
from rtsm.evaluation.runner import fingerprint
out = sys.argv[1]; ref = 'eval/baselines/2026-09-sensor-clock'
ANCHOR = 'ad6f71a5b89c8506'

def seqs(rows):
    k = by_kind(rows)
    dq = [(r['frame_seq'], r['t_sensor_ns'], r['outcome'], r['reason']) for r in k.get('dequeue', [])]
    rx = [(r['decision'], r['reason'], r['frame_seq'], r.get('t_sensor_ns'), r.get('is_keyframe')) for r in k.get('receiver', [])]
    return k, dq, rx

def same(x, y):
    if x == y: return f'identical ({len(x)})'
    i = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
    return f'DIFFER at {i}: {x[i] if i < len(x) else None} vs {y[i] if i < len(y) else None} (lens {len(x)} vs {len(y)})'

b1_rows = read_events(f'{ref}/B1.events.jsonl'); _k, b1_dq, b1_rx = seqs(b1_rows)
b1 = json.load(open(f'{ref}/B1.json', encoding='utf-8'))
b1_fp = fingerprint((b1.get('objects_full') or {}).get('objects') or [])
print(f'B1: fingerprint {b1_fp} (anchor {ANCHOR}) dequeue {len(b1_dq)} receiver {len(b1_rx)}')

def check_run(run_dir):
    s = json.load(open(os.path.join(run_dir, 'summary.json'), encoding='utf-8'))
    rows = read_events(os.path.join(run_dir, 'events.jsonl')); k, dq, rx = seqs(rows)
    m = s['memory']; fp = m['fingerprint']
    ok = (fp == ANCHOR and m['objects_count'] == 124 and m['confirmed_count'] == 65 and s['frames']['processed'] == 53
          and dq == b1_dq and rx == b1_rx and len(k.get('pose', [])) == 240 and len(k.get('obs', [])) == 695 and len(k.get('view', [])) == 53
          and s['aborted'] is None)
    print(f'   {os.path.basename(os.path.dirname(run_dir))}/{os.path.basename(run_dir)}: {m["objects_count"]}/{m["confirmed_count"]} @{s["frames"]["processed"]} fp {fp} | dequeue {same(dq, b1_dq)} | receiver {same(rx, b1_rx)} | pose {len(k.get("pose", []))} obs {len(k.get("obs", []))} view {len(k.get("view", []))} | wall {s["wall_s"]} s | faiss {os.path.isdir(os.path.join(run_dir, "faiss"))} -> {ok}')
    return ok

ok1 = all(check_run(d) for d in sorted(glob.glob(f'{out}/s1_replay/run_*')))
rep = json.load(open(f'{out}/s1_replay/repeats.json', encoding='utf-8'))
ok1 = ok1 and rep['runs'] == 3 and rep['identical_fingerprints'] is True
print(f'1. replay source x3 -> {ok1} | repeats {rep["fingerprints"]} identical {rep["identical_fingerprints"]}')
ok2 = all(check_run(d) for d in sorted(glob.glob(f'{out}/s1_bag/run_*')))
print(f'2. bag source -> {ok2}')

try:
    h = json.load(open(f'{out}/harness.json', encoding='utf-8'))
    h_rows = read_events(f'{out}/harness.events.jsonl'); _k, h_dq, h_rx = seqs(h_rows)
    h_fp = fingerprint((h.get('objects_full') or {}).get('objects') or [])
    wm = h.get('working_memory') or {}
    ok3 = h_fp == ANCHOR and h_dq == b1_dq and h_rx == b1_rx
    print(f'3. harness through run.py: {wm.get("objects")}/{wm.get("confirmed")} fp {h_fp} | dequeue {same(h_dq, b1_dq)} | receiver {same(h_rx, b1_rx)} -> {ok3}')
except Exception as e:
    ok3 = False; print(f'3. harness through run.py: FAILED to evaluate ({type(e).__name__}: {e})')

d = json.load(open(glob.glob(f'{out}/s1_dense/run_1/summary.json')[0], encoding='utf-8'))
d_rows = read_events(f'{out}/s1_dense/run_1/events.jsonl'); dk = by_kind(d_rows)
kf_ts = sorted(r['t_sensor_ns'] for r in dk.get('receiver', []) if r['decision'] == 'enqueued' and r['is_keyframe'])
kf_gaps = [(b - a) / 1e9 for a, b in zip(kf_ts, kf_ts[1:])]
enq = [r for r in dk.get('receiver', []) if r['decision'] == 'enqueued']
outcomes = Counter(r['outcome'] for r in dk.get('dequeue', []))
shadow = sum(1 for r in dk.get('dequeue', []) if r.get('gate_shadow'))
ok4 = (d['resolved']['gate_mode'] == 'shadow' and d['resolved']['keyframe_rule']['kind'] == 'interval' and abs(d['resolved']['nonkf_min_interval_s'] - 0.2) < 1e-9
       and (not kf_gaps or min(kf_gaps) >= 1.0 - 1e-6) and outcomes.get('gate_rejected', 0) == 0 and shadow >= 1
       and outcomes.get('processed', 0) + outcomes.get('frame_rejected', 0) == len(enq) and d['aborted'] is None)
print(f'4. dense: gate {d["resolved"]["gate_mode"]} kf {d["resolved"]["keyframe_rule"]} throttle {d["resolved"]["nonkf_min_interval_s"]} | keyframes {len(kf_ts)} min gap {min(kf_gaps) if kf_gaps else None:.3f} s | enqueued {len(enq)} outcomes {dict(outcomes)} gate_shadow lines {shadow} | memory {d["memory"]["objects_count"]}/{d["memory"]["confirmed_count"]} fp {d["memory"]["fingerprint"]} | wall {d["wall_s"]} s -> {ok4}')

yb = open(f'{out}/yaml.before').read().split()[0]; ya = open(f'{out}/yaml.after').read().split()[0]
fb = open(f'{out}/faiss.before').read() if os.path.exists(f'{out}/faiss.before') else ''; fa = open(f'{out}/faiss.after').read() if os.path.exists(f'{out}/faiss.after') else ''
faiss_dirs = [os.path.isdir(os.path.join(d, 'faiss')) and bool(os.listdir(os.path.join(d, 'faiss'))) for d in sorted(glob.glob(f'{out}/s1_replay/run_*'))]
ok5 = yb == ya and fb == fa and all(faiss_dirs) and len(faiss_dirs) == 3
print(f'5. isolation: rtsm.yaml unchanged {yb == ya} | model_store/faiss unchanged {fb == fa} | per-run faiss stores {faiss_dirs} -> {ok5}')
res = {'p1_replay_x3': ok1, 'p2_bag': ok2, 'p3_harness_run_py': ok3, 'p4_dense': ok4, 'p5_isolation': ok5}
print(f'HARD GATE: {"PASS" if all(res.values()) else "FAIL"} | {res}')
PY
echo P3T2_GATE_DONE
