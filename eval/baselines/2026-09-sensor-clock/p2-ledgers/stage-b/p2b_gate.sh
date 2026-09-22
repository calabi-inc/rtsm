#!/usr/bin/env bash
# P2 stage B gate (G2-B): the observation ledger. Two headless dual replays of session1 on the packaged replay defaults
# (ingest.policy auto -> lossless, ingest.clock auto -> sensor), diagnostics ON for both; ledgers ON for B_on, OFF for B_off.
#   1. B_on, B_off: object multiset == sensor anchor 124/65@53 ad6f71a5b89c8506; dequeue (86) and receiver (240) sequences
#      identical to B1 including reasons (ledger kinds excluded by kind) -> the hook changes no decision, on or off.
#   2. B_off: no `obs` kind; B_on: `obs` present, `pose` still 240 (stage A intact); meta ledgers {enabled, schema 1, jsonl}.
#   3. COUNTS: sum obs[matched] == sum frame.n_matched; sum obs[created] == sum frame.n_created; every obs t_sensor_ns has a
#      frame line; PER FRAME #obs == frame.scoring.n_selected (the associator sees exactly the post-dedup top-K; a missed
#      exit shows here); no obs line on a frame without a frame line.
#   4. LOGIC (the record reflects the decision): every SCORED matched line has cos_sim >= assoc.cos_min, dist_m <= assoc.gate_dist_base_m
#      and (px_err is null or px_err <= assoc.gate_reproj_px); every created line with n_gate_survivors > 0 has max_cos < cos_min;
#      matched_without_scoring lines (the fallback's stale best_id) carry no residuals and are COUNTED (the flaw is recorded, not hidden).
#   5. LOGIC (raw point vs memory, exact EMA property): for every object in the final /objects list, xyz_world lies inside the
#      axis-aligned bbox of that object's created + matched p_world values expanded by 1e-4 m; objects with hits == 1 have
#      xyz_world == their created p_world to 1e-5. Valid only without pose corrections: the replay log must report 0 text
#      messages (asserted). Objects evicted before the end (not in /objects) are skipped and counted.
#   6. LOGIC (view bin): for every object, the set of view_bin values over its created + matched obs == the WM's view_bins_keys.
#   Recorded, not gated: |p_cam.z - mask.depth_p50| < 0.5 m fraction; range_m p50/p95; obs bytes per frame; t_total on/off;
#   ledger_ms p50/p95 from the frame lines.
set -u
cd /c/Users/konam/Desktop/calabi-repo/rtsm || exit 1
SP="C:/Users/konam/AppData/Local/Temp/claude/C--Users-konam-OneDrive-Desktop-calabi-repo-rtsm/949c0384-41c4-4c7e-9f67-2ca869f46d96/scratchpad"
OUT="$SP/p2b_gate"; mkdir -p "$OUT"
moved=0
if [ -d model_store/faiss ] && [ ! -d model_store/faiss.pre-p2b ]; then mv model_store/faiss model_store/faiss.pre-p2b; moved=1; fi
run() {
  local label="$1"; local ledgers="$2"; shift 2
  printf 'diagnostics:\n  enabled: true\n  ledgers: %s\n  event_log_path: "%s/%s.events.jsonl"\n' "$ledgers" "$OUT" "$label" > "$OUT/$label.profile.yaml"
  rm -rf model_store/faiss reports/datasheet_raw_dual*.json
  python -X utf8 scripts/benchmark_datasheet.py dual --profile "$OUT/$label.profile.yaml" "$@" > "$OUT/$label.harness.stdout" 2>&1; local rc=$?
  local raw; raw=$(ls reports/datasheet_raw_dual.*.json 2>/dev/null | head -1)
  cp -f "$raw" "$OUT/$label.json" 2>/dev/null; cp -f reports/run_dual.log "$OUT/$label.log" 2>/dev/null
  ls rtsm/cfg/rtsm.yaml.*bak* 2>/dev/null && echo "WARN: stray harness backup present"
  echo "== $label rc=$rc =="
}
run B_on true
run B_off false
rm -rf model_store/faiss
if [ $moved = 1 ]; then mv model_store/faiss.pre-p2b model_store/faiss; fi
python -X utf8 - "$OUT" <<'PY'
import json, sys, hashlib, os, re
from collections import Counter, defaultdict
import numpy as np
from rtsm.cfg import load_config
from rtsm.evaluation.ledger import by_kind, ledger_meta, observation_summary, read_events
out = sys.argv[1]; ref = 'eval/baselines/2026-09-sensor-clock'
acfg = load_config("rtsm.yaml")["assoc"]
COS_MIN = float(acfg["cos_min"]); GATE_D = float(acfg["gate_dist_base_m"]); GATE_PX = float(acfg["gate_reproj_px"])

def load(jp, ep):
    d = json.load(open(jp, encoding='utf-8')); rows = read_events(ep)
    full = (d.get('objects_full') or {}).get('objects') or []
    ms = Counter((o.get('label_primary'), tuple(round(float(v), 3) for v in o.get('xyz_world') or []), int(o.get('hits') or 0), bool(o.get('confirmed'))) for o in full)
    sha = hashlib.sha256(json.dumps(sorted(map(list, ms.elements())), default=str).encode()).hexdigest()[:16]
    wm = d.get('working_memory') or {}; lat = d.get('latency') or {}; k = by_kind(rows)
    dq = [(r['frame_seq'], r['t_sensor_ns'], r['outcome'], r['reason']) for r in k.get('dequeue', [])]
    rx = [(r['decision'], r['reason'], r['frame_seq'], r.get('t_sensor_ns'), r.get('is_keyframe')) for r in k.get('receiver', [])]
    return dict(d=d, rows=rows, k=k, sha=sha, objs=(wm.get('objects'), wm.get('confirmed')), frames=lat.get('frame_count'),
                dq=dq, rx=rx, meta=rows[0], lat=lat, full=full, path=ep)

def same(x, y):
    if x == y: return f'identical ({len(x)})'
    i = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
    return f'DIFFER at {i}: {x[i] if i < len(x) else None} vs {y[i] if i < len(y) else None} (lens {len(x)} vs {len(y)})'

B1 = load(f'{ref}/B1.json', f'{ref}/B1.events.jsonl')
A = load(f'{out}/B_on.json', f'{out}/B_on.events.jsonl'); N = load(f'{out}/B_off.json', f'{out}/B_off.events.jsonl')
for lbl, R in (('B_on', A), ('B_off', N)):
    print(f'{lbl}: {R["objs"]} frames {R["frames"]} sha {R["sha"]} | kinds {dict(Counter(r["kind"] for r in R["rows"]))} | ledgers {ledger_meta(R["rows"])}')
    print('   vs B1: multiset', 'IDENTICAL' if R['sha'] == B1['sha'] else 'DIFFER', '| dequeue', same(R['dq'], B1['dq']), '| receiver', same(R['rx'], B1['rx']))
ok1 = all(R['sha'] == B1['sha'] and R['dq'] == B1['dq'] and R['rx'] == B1['rx'] for R in (A, N))
ok2 = ('obs' not in N['k']) and ('obs' in A['k']) and len(A['k'].get('pose', [])) == 240 and ledger_meta(A['rows']) == {'enabled': True, 'schema': 1, 'format': 'jsonl'}
print(f'2. B_off has no obs: {"obs" not in N["k"]} | B_on obs {len(A["k"].get("obs", []))} pose {len(A["k"].get("pose", []))} -> {ok2}')

obs = A['k'].get('obs', []); frames = A['k'].get('frame', [])
fr_by_ts = {f['t_sensor_ns']: f for f in frames}
obs_by_ts = defaultdict(list)
for o in obs: obs_by_ts[o['t_sensor_ns']].append(o)
m_obs = sum(1 for o in obs if o['outcome'] == 'matched'); c_obs = sum(1 for o in obs if o['outcome'] == 'created')
m_fr = sum(f['n_matched'] for f in frames); c_fr = sum(f['n_created'] for f in frames)
orphan = [ts for ts in obs_by_ts if ts not in fr_by_ts]
per_frame_bad = [(ts, len(obs_by_ts.get(ts, [])), (f.get('scoring') or {}).get('n_selected')) for ts, f in fr_by_ts.items()
                 if len(obs_by_ts.get(ts, [])) != (f.get('scoring') or {}).get('n_selected')]
ok3 = (m_obs == m_fr and c_obs == c_fr and not orphan and not per_frame_bad and len(frames) == 53)
print(f'3. matched obs {m_obs} == frame sum {m_fr} | created obs {c_obs} == {c_fr} | orphan obs frames {len(orphan)} | per-frame #obs != n_selected: {len(per_frame_bad)} {per_frame_bad[:3]} | frames {len(frames)} | outcomes {dict(Counter(o["outcome"] for o in obs))} -> {ok3}')

stale = [o for o in obs if o['outcome'] == 'matched' and o.get('matched_without_scoring')]
scored_m = [o for o in obs if o['outcome'] == 'matched' and not o.get('matched_without_scoring')]
bad4 = [o for o in scored_m if not (o['cos_sim'] is not None and o['cos_sim'] >= COS_MIN - 1e-9 and o['dist_m'] is not None and o['dist_m'] <= GATE_D + 1e-9 and (o['px_err'] is None or o['px_err'] <= GATE_PX + 1e-9))]
bad4b = [o for o in obs if o['outcome'] == 'created' and (o.get('n_gate_survivors') or 0) > 0 and not (o.get('max_cos') is not None and o['max_cos'] < COS_MIN)]
bad4c = [o for o in stale if o.get('cos_sim') is not None or o.get('dist_m') is not None or (o.get('n_gate_survivors') or 0) != 0]
ok4 = not bad4 and not bad4b and not bad4c
print(f'4. gates cos_min {COS_MIN} gate_dist {GATE_D} gate_px {GATE_PX}: scored matches violating {len(bad4)} of {len(scored_m)} | created-with-survivors violating {len(bad4b)} of {sum(1 for o in obs if o["outcome"] == "created" and (o.get("n_gate_survivors") or 0) > 0)} | matched_without_scoring (fallback stale best_id, the associator flaw the ledger exposes) {len(stale)} of {m_obs}, carrying residuals {len(bad4c)} -> {ok4}')

txt = re.search(r'Timeline: (\d+) binary frames, (\d+) text messages', open(f'{out}/B_on.log', encoding='utf-8', errors='replace').read())
n_text = int(txt.group(2)) if txt else None
pts = defaultdict(list)
for o in obs:
    if o['outcome'] in ('matched', 'created') and o.get('object_id') and o.get('p_world') is not None:
        pts[o['object_id']].append(o['p_world'])
present = 0; skipped = 0; bad5 = []; bad5_h1 = []
for ob in A['full']:
    oid = ob.get('id'); xyz = ob.get('xyz_world')
    if oid not in pts or xyz is None:
        skipped += 1; continue
    present += 1
    P = np.asarray(pts[oid], dtype=float); x = np.asarray(xyz, dtype=float)
    lo, hi = P.min(axis=0) - 1e-4, P.max(axis=0) + 1e-4
    if not (np.all(x >= lo) and np.all(x <= hi)):
        bad5.append((oid, xyz, P.min(axis=0).round(4).tolist(), P.max(axis=0).round(4).tolist(), len(P)))
    if int(ob.get('hits') or 0) == 1:
        created = [o['p_world'] for o in obs if o.get('object_id') == oid and o['outcome'] == 'created']
        if not created or np.max(np.abs(np.asarray(created[0]) - x)) > 1e-5:
            bad5_h1.append((oid, xyz, created[:1]))
ok5 = (n_text == 0) and not bad5 and not bad5_h1 and present > 0
print(f'5. text messages in replay {n_text} | objects checked {present} (skipped {skipped}: no obs / evicted) | outside bbox {len(bad5)} {bad5[:2]} | hits==1 mismatches {len(bad5_h1)} {bad5_h1[:2]} -> {ok5}')

bins_obs = defaultdict(set)
for o in obs:
    if o['outcome'] in ('matched', 'created') and o.get('object_id') and o.get('view_bin') is not None:
        bins_obs[o['object_id']].add(int(o['view_bin']))
bad6 = []; checked6 = 0; exact6 = 0
for ob in A['full']:
    oid = ob.get('id'); keys = ob.get('view_bins_keys'); n_bins = ob.get('view_bins')
    if oid not in bins_obs: continue
    if keys is not None:                                  # /objects?include_vectors=true carries the keys
        checked6 += 1; exact6 += 1
        if set(int(k) for k in keys) != bins_obs[oid]:
            bad6.append((oid, sorted(bins_obs[oid]), sorted(int(k) for k in keys)))
    elif n_bins is not None:                              # the harness's /objects list carries the COUNT only
        checked6 += 1
        if int(n_bins) != len(bins_obs[oid]):
            bad6.append((oid, sorted(bins_obs[oid]), f'count {n_bins}'))
ok6 = checked6 > 0 and not bad6
print(f'6. view-bin sets: checked {checked6} objects ({exact6} by key set, {checked6 - exact6} by count), mismatches {len(bad6)} {bad6[:3]} -> {ok6}')

s = observation_summary(A['rows'])
zs = [(o['p_cam'][2], (o.get('mask') or {}).get('depth_p50')) for o in obs if o.get('p_cam') and (o.get('mask') or {}).get('depth_p50') is not None]
close = sum(1 for z, p in zs if abs(z - p) < 0.5) / max(1, len(zs))
led = [f['timing_ms'].get('ledger') for f in frames if f.get('timing_ms', {}).get('ledger') is not None]
t_on = ((A['lat'].get('t_total') or {}).get('mean') or 0) * 1000; t_off = ((N['lat'].get('t_total') or {}).get('mean') or 0) * 1000
print(f'info: summary n_obs {s["n_obs"]} objects_seen {s["n_objects_seen"]} created {s["n_objects_created"]} matched/object p50 {s["matched_per_object"]["p50"]} max {s["matched_per_object"]["max"]} | cos p50 {s["cos_sim"]["p50"]} p95 {s["cos_sim"]["p95"]} | dist p50 {s["dist_m"]["p50"]} p95 {s["dist_m"]["p95"]} | px_err p50 {s["px_err"]["p50"]} | range p50 {s["range_m"]["p50"]} p95 {s["range_m"]["p95"]} | bins {s["view_bins"]} | |p_cam.z - depth_p50| < 0.5 m: {close:.2%} of {len(zs)}')
print(f'info: ledger_ms p50 {np.median(led) if led else None} p95 {np.percentile(led, 95) if led else None} | t_total mean on {t_on:.1f} / off {t_off:.1f} ms (ratio {t_on / t_off if t_off else float("nan"):.3f}) | events.jsonl on {os.path.getsize(A["path"])} B / off {os.path.getsize(N["path"])} B -> {(os.path.getsize(A["path"]) - os.path.getsize(N["path"]) - 240 * 497) / max(1, len(obs)):.0f} B per obs line (net of the 240 pose lines at 497 B)')
print('HARD GATE:', 'PASS' if (ok1 and ok2 and ok3 and ok4 and ok5 and ok6) else 'FAIL', '|', dict(p1=ok1, p2=ok2, p3=ok3, p4=ok4, p5=ok5, p6=ok6))
PY
echo P2B_GATE_DONE
