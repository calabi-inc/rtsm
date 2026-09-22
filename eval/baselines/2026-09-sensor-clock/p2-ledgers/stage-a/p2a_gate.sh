#!/usr/bin/env bash
# P2 stage A gate (G2-A): the pose ledger. Two headless dual replays of session1 on the packaged replay defaults
# (ingest.policy auto -> lossless, ingest.clock auto -> sensor), diagnostics ON for both; ledgers ON for A_on, OFF for A_off.
#   1. A_on, A_off: object multiset == sensor anchor 124/65@53 ad6f71a5b89c8506; dequeue (86) and receiver (240)
#      sequences identical to B1 INCLUDING reasons (ledger kinds excluded by kind) -> no behaviour change, on or off.
#   2. A_off: no `pose` kind in the file; meta ledgers == {enabled: false}.
#   3. A_on counts: #pose == 240 == #receiver(source=replay) == /stats.robot_pose.writes_accepted; rx_seq join 240/240
#      (unique); every line tracking_state normal + mailbox_write true; meta schema_version 3, ledgers {enabled, schema 1, jsonl}.
#   4. LOGIC (same statistic, same point): pose.depth_valid_frac == receiver.depth_valid_frac on EVERY joined pair, none null.
#   5. LOGIC (same pose the mailbox holds): the LAST pose line's t_wc / q_wc_xyzw == robot_pose.xyz / quaternion_xyzw
#      to 1e-6; its t_sensor_ns == robot_pose.sensor_ts_ns; its epoch == robot_pose.frame_epoch.
#   6. LOGIC (clock + raw map): the pose lines' t_sensor_ns sequence == the receiver lines' t_sensor_ns sequence; every
#      line carries a conf_hist and all histograms sum to ONE value (the raw confidence map size; 256x192 = 49152 expected).
#   7. pose_health(A_on): n_limited_episodes == 0 and writes_expected == writes_accepted (gated); sensor_hz, dt, jitter,
#      gaps, discontinuities, depth_valid / conf2 statistics RECORDED as the session1 reference (README).
#   Informational: t_total mean on vs off (the <= 5 % predicate is G2-C's, over 3 x 3 runs), file sizes, bytes per pose line.
set -u
cd /c/Users/konam/Desktop/calabi-repo/rtsm || exit 1
SP="C:/Users/konam/AppData/Local/Temp/claude/C--Users-konam-OneDrive-Desktop-calabi-repo-rtsm/949c0384-41c4-4c7e-9f67-2ca869f46d96/scratchpad"
OUT="$SP/p2a_gate"; mkdir -p "$OUT"
moved=0
if [ -d model_store/faiss ] && [ ! -d model_store/faiss.pre-p2a ]; then mv model_store/faiss model_store/faiss.pre-p2a; moved=1; fi
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
run A_on true
run A_off false
rm -rf model_store/faiss
if [ $moved = 1 ]; then mv model_store/faiss.pre-p2a model_store/faiss; fi
python -X utf8 - "$OUT" <<'PY'
import json, sys, hashlib, os
from collections import Counter
from rtsm.evaluation.ledger import by_kind, ledger_meta, pose_health, read_events
out = sys.argv[1]; ref = 'eval/baselines/2026-09-sensor-clock'

def load(jp, ep):
    d = json.load(open(jp, encoding='utf-8')); rows = read_events(ep)
    full = (d.get('objects_full') or {}).get('objects') or []
    ms = Counter((o.get('label_primary'), tuple(round(float(v), 3) for v in o.get('xyz_world') or []), int(o.get('hits') or 0), bool(o.get('confirmed'))) for o in full)
    sha = hashlib.sha256(json.dumps(sorted(map(list, ms.elements())), default=str).encode()).hexdigest()[:16]
    wm = d.get('working_memory') or {}; lat = d.get('latency') or {}; k = by_kind(rows)
    rx_rows = k.get('receiver', []); dq_rows = k.get('dequeue', [])
    dq = [(r['frame_seq'], r['t_sensor_ns'], r['outcome'], r['reason']) for r in dq_rows]
    rx = [(r['decision'], r['reason'], r['frame_seq'], r.get('t_sensor_ns'), r.get('is_keyframe')) for r in rx_rows]
    return dict(d=d, rows=rows, k=k, sha=sha, objs=(wm.get('objects'), wm.get('confirmed')), frames=lat.get('frame_count'),
                dq=dq, rx=rx, rx_rows=rx_rows, meta=rows[0], pose=wm.get('robot_pose'), lat=lat, path=ep)

def same(x, y):
    if x == y: return f'identical ({len(x)})'
    i = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
    return f'DIFFER at {i}: {x[i] if i < len(x) else None} vs {y[i] if i < len(y) else None} (lens {len(x)} vs {len(y)})'

B1 = load(f'{ref}/B1.json', f'{ref}/B1.events.jsonl')
A = load(f'{out}/A_on.json', f'{out}/A_on.events.jsonl'); N = load(f'{out}/A_off.json', f'{out}/A_off.events.jsonl')
for lbl, R in (('A_on', A), ('A_off', N)):
    print(f'{lbl}: {R["objs"]} frames {R["frames"]} sha {R["sha"]} | meta policy={R["meta"].get("ingest_policy")} clock={R["meta"].get("ingest_clock")} schema {R["meta"].get("schema_version")} ledgers {R["meta"].get("ledgers")}')
    print('   vs B1: multiset', 'IDENTICAL' if R['sha'] == B1['sha'] else 'DIFFER', '| dequeue', same(R['dq'], B1['dq']), '| receiver', same(R['rx'], B1['rx']))
ok1 = all(R['sha'] == B1['sha'] and R['dq'] == B1['dq'] and R['rx'] == B1['rx'] for R in (A, N))
ok2 = ('pose' not in N['k']) and ledger_meta(N['rows']) == {'enabled': False}
print(f'2. A_off kinds {sorted(N["k"])} ledgers meta {ledger_meta(N["rows"])} -> {ok2}')

pose = A['k'].get('pose', []); rxr = [r for r in A['rx_rows'] if r['source'] == 'replay']; rp = A['pose'] or {}
by_rx = {r['rx_seq']: r for r in rxr}
joined = sum(1 for p in pose if p['rx_seq'] in by_rx)
ok3 = (len(pose) == 240 == len(rxr) == rp.get('writes_accepted') and joined == 240 and len({p['rx_seq'] for p in pose}) == 240
       and all(p['tracking_state'] == 'normal' and p['mailbox_write'] is True for p in pose)
       and A['meta'].get('schema_version') == 3 and ledger_meta(A['rows']) == {'enabled': True, 'schema': 1, 'format': 'jsonl'})
print(f'3. #pose {len(pose)} #receiver(replay) {len(rxr)} writes_accepted {rp.get("writes_accepted")} | rx_seq join {joined}/240 unique {len({p["rx_seq"] for p in pose})} | states {dict(Counter(p["tracking_state"] for p in pose))} mailbox_write {sum(1 for p in pose if p["mailbox_write"])} | regressions {rp.get("sensor_ts_regressions")} rejects {rp.get("rejected_writes")} -> {ok3}')

pairs = [(p['depth_valid_frac'], by_rx[p['rx_seq']].get('depth_valid_frac')) for p in pose if p['rx_seq'] in by_rx]
mism = [(p['rx_seq'], a, b) for p, (a, b) in zip(pose, pairs) if a != b]
ok4 = bool(pairs) and not mism and all(a is not None for a, _ in pairs)
print(f'4. depth_valid_frac pose == receiver on {len(pairs) - len(mism)}/{len(pairs)} pairs, nulls {sum(1 for a, _ in pairs if a is None)}; first mismatches {mism[:3]} -> {ok4}')

last = pose[-1] if pose else {}
dt = max((abs(a - b) for a, b in zip(last.get('t_wc') or [], rp.get('xyz') or [])), default=float('inf'))
dq_ = max((abs(a - b) for a, b in zip(last.get('q_wc_xyzw') or [], rp.get('quaternion_xyzw') or [])), default=float('inf'))
ok5 = (dt < 1e-6 and dq_ < 1e-6 and last.get('t_sensor_ns') == rp.get('sensor_ts_ns') and last.get('epoch') == rp.get('frame_epoch'))
print(f'5. last pose line t_wc {last.get("t_wc")} vs mailbox xyz {rp.get("xyz")} (max |d| {dt:.2e}) | quat max |d| {dq_:.2e} | stamp {last.get("t_sensor_ns")} == {rp.get("sensor_ts_ns")} | epoch {last.get("epoch")} == {rp.get("frame_epoch")} -> {ok5}')

seq_ok = [p['t_sensor_ns'] for p in pose] == [r['t_sensor_ns'] for r in rxr]
sums = Counter(sum(p['conf_hist']) if p.get('conf_hist') else None for p in pose)
ok6 = seq_ok and (None not in sums) and len(sums) == 1
print(f'6. t_sensor_ns sequence pose == receiver: {seq_ok} | conf_hist sums {dict(sums)} (expect one value, 49152 on session1) -> {ok6}')

h = pose_health(A['rows']); g = h['groups'].get('replay/1') or next(iter(h['groups'].values()), {})
ok7 = (g.get('n_limited_episodes') == 0 and h['total']['writes_expected'] == rp.get('writes_accepted') and h['total']['n_groups'] == 1)
print(f'7. pose_health replay/1: n_frames {g.get("n_frames")} stream {g.get("n_stream")} span {g.get("span_s")} s sensor_hz {g.get("sensor_hz")} | dt_ms p50 {g.get("dt_ms", {}).get("p50")} p95 {g.get("dt_ms", {}).get("p95")} max {g.get("dt_ms", {}).get("max")} jitter {g.get("jitter_ms")} ms | gaps {g.get("n_gaps")} {g.get("gaps")[:5] if g.get("gaps") else []} | limited episodes {g.get("n_limited_episodes")} | discontinuities {g.get("n_discontinuities")} {g.get("discontinuities")[:5] if g.get("discontinuities") else []} | depth_valid mean {g.get("depth_valid_frac", {}).get("mean")} p10 {g.get("depth_valid_frac", {}).get("p10")} | conf2 mean {g.get("conf2_frac", {}).get("mean")} p10 {g.get("conf2_frac", {}).get("p10")} | pose_errors {g.get("pose_errors")} | writes_expected {h["total"]["writes_expected"]} groups {h["total"]["n_groups"]} -> {ok7}')

t_on = ((A['lat'].get('t_total') or {}).get('mean') or 0) * 1000; t_off = ((N['lat'].get('t_total') or {}).get('mean') or 0) * 1000
sz_on = os.path.getsize(A['path']); sz_off = os.path.getsize(N['path'])
print(f'info: t_total mean on {t_on:.1f} ms / off {t_off:.1f} ms (ratio {t_on / t_off if t_off else float("nan"):.3f}; the <= 1.05 predicate is G2-C over 3x3 runs) | events.jsonl on {sz_on} B / off {sz_off} B -> {(sz_on - sz_off) / max(1, len(pose)):.0f} B per pose line')
print('HARD GATE:', 'PASS' if (ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and ok7) else 'FAIL', '|', dict(p1=ok1, p2=ok2, p3=ok3, p4=ok4, p5=ok5, p6=ok6, p7=ok7))
PY
echo P2A_GATE_DONE
