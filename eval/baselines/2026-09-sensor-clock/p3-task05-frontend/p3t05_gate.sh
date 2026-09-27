#!/usr/bin/env bash
# P3 task 0.5 gate (G3-0.5): the ingest front-end extraction is behaviour-preserving.
# Two headless dual replays of session1 on the packaged replay defaults (policy auto -> lossless, clock auto -> sensor),
# diagnostics ON for both; ledgers ON for F_on, OFF for F_off. Both go through rtsm/run.py's NEW wiring
# (SourceContext -> sources.make_source("replay") -> ReplayReceiver on IngestFrontEnd(WEBSOCKET_POLICY)).
#   1. Both: object multiset == sensor anchor 124/65@53 ad6f71a5b89c8506; dequeue (86) + receiver (240) sequences identical
#      to B1 (P1 task 1 record) as (decision, reason, frame_seq, t_sensor_ns, is_keyframe) / (frame_seq, ts, outcome, reason).
#   2. F_on vs the stage-C record C1_on (P2, the last run of the pre-extraction receiver, kept in the session scratchpad):
#      every receiver line EQUAL on every key except timestamp / queue_depth (i.e. frame_count, lane, rx_seq,
#      depth_valid_frac, source all preserved); every pose line EQUAL on every key except timestamp; #obs and #view equal.
#      Recorded as SKIPPED (not failed) when the C1_on file is gone -- predicate 1 + 3 + 4 still bind.
#   3. Source tagging: every receiver + pose line has source "replay"; meta ingest_clock sensor / ingest_policy lossless;
#      F_off has no ledger kinds; F_on: pose 240 / obs 695 / view 53 == #frame; /stats.robot_pose 240 writes / 0 regressions /
#      0 rejects, pose_clock sender, epoch 0 (the pose mailbox through the seam).
#   4. pose_health(F_on): 240 frames, span 40.519 s, 0 gaps, 0 discontinuities, 0 regressions, writes_expected 240
#      (stage A's numbers).
#   5. CPU predicates: the two golden traces (recorded from the pre-extraction receivers) + the front-end unit tests +
#      the ingest suites pass.
#   info: t_total means (single runs; no overhead predicate -- the front-end adds no per-frame work).
set -u
cd /c/Users/konam/Desktop/calabi-repo/rtsm || exit 1
SP="C:/Users/konam/AppData/Local/Temp/claude/C--Users-konam-OneDrive-Desktop-calabi-repo-rtsm/949c0384-41c4-4c7e-9f67-2ca869f46d96/scratchpad"
OUT="$SP/p3t05_gate"; mkdir -p "$OUT"
REF_C1="$SP/p2c_gate/C1_on.events.jsonl"
moved=0
if [ -d model_store/faiss ] && [ ! -d model_store/faiss.pre-p3t05 ]; then mv model_store/faiss model_store/faiss.pre-p3t05; moved=1; fi
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
run F_on true
run F_off false
rm -rf model_store/faiss
if [ $moved = 1 ]; then mv model_store/faiss.pre-p3t05 model_store/faiss; fi

echo "== 5. CPU predicates =="
python -X utf8 -m pytest tests/test_ingest_golden.py tests/test_ingest_frontend.py tests/test_admit_before_decode.py tests/test_ingest_lanes.py tests/test_pose_mailbox.py tests/test_pose_receive_time.py tests/test_sensor_clock.py tests/test_frame_flow_trace.py tests/test_websocket.py tests/test_record_replay.py tests/test_run_entrypoint.py tests/test_demo_entrypoint.py tests/evaluation -q -p no:cacheprovider > "$OUT/cpu_predicates.out" 2>&1; cpu_rc=$?
tail -2 "$OUT/cpu_predicates.out"; echo "cpu_rc=$cpu_rc"

python -X utf8 - "$OUT" "$REF_C1" "$cpu_rc" <<'PY'
import json, sys, hashlib, os
from collections import Counter
from rtsm.evaluation.ledger import by_kind, ledger_meta, pose_health, read_events
out, ref_c1, cpu_rc = sys.argv[1], sys.argv[2], int(sys.argv[3]); ref = 'eval/baselines/2026-09-sensor-clock'

def load(jp, ep):
    d = json.load(open(jp, encoding='utf-8')); rows = read_events(ep)
    full = (d.get('objects_full') or {}).get('objects') or []
    ms = Counter((o.get('label_primary'), tuple(round(float(v), 3) for v in o.get('xyz_world') or []), int(o.get('hits') or 0), bool(o.get('confirmed'))) for o in full)
    sha = hashlib.sha256(json.dumps(sorted(map(list, ms.elements())), default=str).encode()).hexdigest()[:16]
    wm = d.get('working_memory') or {}; lat = d.get('latency') or {}; k = by_kind(rows)
    dq = [(r['frame_seq'], r['t_sensor_ns'], r['outcome'], r['reason']) for r in k.get('dequeue', [])]
    rx = [(r['decision'], r['reason'], r['frame_seq'], r.get('t_sensor_ns'), r.get('is_keyframe')) for r in k.get('receiver', [])]
    return dict(d=d, rows=rows, k=k, sha=sha, objs=(wm.get('objects'), wm.get('confirmed')), frames=lat.get('frame_count'),
                dq=dq, rx=rx, meta=rows[0], t_total=((lat.get('t_total') or {}).get('mean') or 0) * 1000, path=ep)

def same(x, y):
    if x == y: return f'identical ({len(x)})'
    i = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
    return f'DIFFER at {i}: {x[i] if i < len(x) else None} vs {y[i] if i < len(y) else None} (lens {len(x)} vs {len(y)})'

B1 = load(f'{ref}/B1.json', f'{ref}/B1.events.jsonl')
R = {lbl: load(f'{out}/{lbl}.json', f'{out}/{lbl}.events.jsonl') for lbl in ('F_on', 'F_off')}
ok1 = True
for lbl, r in R.items():
    good = r['sha'] == B1['sha'] and r['dq'] == B1['dq'] and r['rx'] == B1['rx']; ok1 &= good
    print(f'{lbl}: {r["objs"]} frames {r["frames"]} sha {r["sha"]} t_total {r["t_total"]:.1f} ms | meta clock={r["meta"].get("ingest_clock")} policy={r["meta"].get("ingest_policy")} ledgers {ledger_meta(r["rows"])} | kinds {dict(Counter(x["kind"] for x in r["rows"]))}')
    print(f'   vs B1: multiset {"IDENTICAL" if r["sha"] == B1["sha"] else "DIFFER"} | dequeue {same(r["dq"], B1["dq"])} | receiver {same(r["rx"], B1["rx"])}')
print(f'1. anchor + sequences on both -> {ok1}')

F = R['F_on']
def strip(line, drop):
    return {k: v for k, v in line.items() if k not in drop}
if os.path.isfile(ref_c1):
    C = by_kind(read_events(ref_c1))
    rx_new = [strip(l, ('timestamp', 'queue_depth')) for l in F['k']['receiver']]
    rx_ref = [strip(l, ('timestamp', 'queue_depth')) for l in C['receiver']]
    po_new = [strip(l, ('timestamp',)) for l in F['k']['pose']]
    po_ref = [strip(l, ('timestamp',)) for l in C['pose']]
    def first_diff(a, b):
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return i, {k: (x.get(k), y.get(k)) for k in set(x) | set(y) if x.get(k) != y.get(k)}
        return None
    d_rx, d_po = first_diff(rx_new, rx_ref), first_diff(po_new, po_ref)
    keys_rx = sorted(set().union(*(l.keys() for l in rx_new))) if rx_new else []
    ok2 = (rx_new == rx_ref and po_new == po_ref and len(F['k']['obs']) == len(C['obs']) and len(F['k']['view']) == len(C['view']))
    print(f'2. F_on vs stage-C C1_on (full lines): receiver {len(rx_new)} vs {len(rx_ref)} equal={rx_new == rx_ref} first diff {d_rx} | keys compared {keys_rx} | pose {len(po_new)} vs {len(po_ref)} equal={po_new == po_ref} first diff {d_po} | obs {len(F["k"]["obs"])} vs {len(C["obs"])} | view {len(F["k"]["view"])} vs {len(C["view"])} -> {ok2}')
else:
    ok2 = True
    print(f'2. F_on vs stage-C C1_on: SKIPPED ({ref_c1} not present) -> recorded, not failed')

src_rx = Counter(l.get('source') for l in F['k']['receiver']); src_po = Counter(l.get('source') for l in F['k']['pose'])
off_kinds = [x for x in ('pose', 'obs', 'view') if x in R['F_off']['k']]
counts = {k: len(F['k'].get(k, [])) for k in ('receiver', 'dequeue', 'frame', 'pose', 'obs', 'view')}
rp = (F['d'].get('working_memory') or {}).get('robot_pose') or {}
mailbox = (rp.get('writes_accepted'), rp.get('sensor_ts_regressions'), rp.get('rejected_writes'), rp.get('pose_clock'), rp.get('frame_epoch'))
ok3 = (src_rx == Counter({'replay': 240}) and src_po == Counter({'replay': 240}) and not off_kinds
       and F['meta'].get('ingest_clock') == 'sensor' and F['meta'].get('ingest_policy') == 'lossless'
       and counts['pose'] == 240 and counts['obs'] == 695 and counts['view'] == counts['frame'] == 53
       and mailbox == (240, 0, 0, 'sender', 0))
print(f'3. sources receiver {dict(src_rx)} pose {dict(src_po)} | F_off ledger kinds {off_kinds} | F_on counts {counts} | /stats.robot_pose writes/regressions/rejects/clock/epoch {mailbox} (240/0/0/sender/0) -> {ok3}')

ph = pose_health(F['rows'])
g = ph['groups'].get('replay/1') or next(iter(ph['groups'].values()), {})
n = g.get('n_frames'); span = g.get('span_s'); gaps = g.get('gaps') or []; disc = g.get('discontinuities') or []
ok4 = (n == 240 and span is not None and abs(float(span) - 40.5194) < 0.01 and len(gaps) == 0 and len(disc) == 0
       and int(g.get('n_limited_episodes') or 0) == 0 and int(ph['total'].get('writes_expected') or 0) == 240
       and int(g.get('pose_errors') or 0) == 0)
print(f'4. pose_health replay/1: n_frames {n} span {span} s sensor_hz {g.get("sensor_hz")} | gaps {len(gaps)} discontinuities {len(disc)} limited episodes {g.get("n_limited_episodes")} pose_errors {g.get("pose_errors")} | writes_expected {ph["total"].get("writes_expected")} groups {ph["total"].get("n_groups")} -> {ok4}')

ok5 = cpu_rc == 0
print(f'5. CPU predicates (golden traces + front-end unit tests + ingest suites): rc {cpu_rc} -> {ok5}')
print(f'info: t_total mean F_on {F["t_total"]:.1f} ms / F_off {R["F_off"]["t_total"]:.1f} ms (single runs, no predicate) | events.jsonl on {os.path.getsize(F["path"])} B / off {os.path.getsize(R["F_off"]["path"])} B')
res = {'p1': ok1, 'p2': ok2, 'p3': ok3, 'p4': ok4, 'p5': ok5}
print(f'HARD GATE: {"PASS" if all(res.values()) else "FAIL"} | {res}')
PY
echo P3T05_GATE_DONE
