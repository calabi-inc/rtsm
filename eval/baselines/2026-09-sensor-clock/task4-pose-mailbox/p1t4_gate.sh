#!/usr/bin/env bash
# P1 task 4 gate: headless dual replay of session1 (default policy = lossless, clock = sensor).
#   HARD: multiset == sensor anchor 124/65@53 ad6f71a5b89c8506; dequeue + receiver sequences identical to B1;
#         per-line (decision, reason, frame_seq, depth_valid_frac) identical to task-2 S;
#         final /stats.robot_pose xyz / quaternion_xyzw / timestamp / frame_epoch == task-2 S (last received frame, epoch 0);
#         robot_pose payload carries the new keys (sensor_ts_ns == the last received frame's stamp, pose_clock=sender, age_s, stale);
#         single writer: writes_accepted == tracking-normal receiver lines (240), sensor_ts_regressions == 0, rejected_writes == 0.
set -u
cd /c/Users/konam/Desktop/calabi-repo/rtsm || exit 1
SP="C:/Users/konam/AppData/Local/Temp/claude/C--Users-konam-Desktop-calabi-repo-rtsm/c99f81f1-d743-4906-85ee-b894ab355ce0/scratchpad"
OUT="$SP/p1t4_gate"; mkdir -p "$OUT"
moved=0
if [ -d model_store/faiss ] && [ ! -d model_store/faiss.pre-p1t4 ]; then mv model_store/faiss model_store/faiss.pre-p1t4; moved=1; fi
run() {
  local label="$1"; shift
  printf 'diagnostics:\n  enabled: true\n  event_log_path: "%s/%s.events.jsonl"\n' "$OUT" "$label" > "$OUT/$label.profile.yaml"
  rm -rf model_store/faiss reports/datasheet_raw_dual*.json
  python -X utf8 scripts/benchmark_datasheet.py dual --profile "$OUT/$label.profile.yaml" "$@" > "$OUT/$label.harness.stdout" 2>&1; local rc=$?
  local raw; raw=$(ls reports/datasheet_raw_dual.*.json 2>/dev/null | head -1)
  cp -f "$raw" "$OUT/$label.json" 2>/dev/null; cp -f reports/run_dual.log "$OUT/$label.log" 2>/dev/null
  ls rtsm/cfg/rtsm.yaml.*bak* 2>/dev/null && echo "WARN: stray harness backup present"
  echo "== $label rc=$rc =="
}
run M
rm -rf model_store/faiss
if [ $moved = 1 ]; then mv model_store/faiss.pre-p1t4 model_store/faiss; fi
python -X utf8 - "$OUT" <<'PY'
import json,sys,hashlib
from collections import Counter
out=sys.argv[1]; ref='eval/baselines/2026-09-sensor-clock'
def load(jp, ep):
    d=json.load(open(jp,encoding='utf-8')); rows=[json.loads(l) for l in open(ep,encoding='utf-8') if l.strip()]
    full=(d.get('objects_full') or {}).get('objects') or []
    ms=Counter((o.get('label_primary'),tuple(round(float(v),3) for v in o.get('xyz_world') or []),int(o.get('hits') or 0),bool(o.get('confirmed'))) for o in full)
    sha=hashlib.sha256(json.dumps(sorted(map(list,ms.elements())),default=str).encode()).hexdigest()[:16]
    wm=d.get('working_memory') or {}; lat=d.get('latency') or {}
    rx_rows=[r for r in rows if r['kind']=='receiver']
    dq=[(r['frame_seq'],r['t_sensor_ns'],r['outcome'],r['reason']) for r in rows if r['kind']=='dequeue']
    rx=[(r['decision'],r['reason'],r['frame_seq'],r.get('t_sensor_ns'),r.get('is_keyframe')) for r in rx_rows]
    rxd=[(r['decision'],r['reason'],r['frame_seq'],None if r.get('depth_valid_frac') is None else round(r['depth_valid_frac'],6)) for r in rx_rows]
    last_rx=[r for r in rx_rows if r['reason'] not in ('tracking_state','malformed')][-1]   # the sink fires for throttled frames too
    n_parse_err=sum(1 for r in rx_rows if r['reason']=='parse_error')
    return dict(sha=sha,objs=(wm.get('objects'),wm.get('confirmed')),frames=lat.get('frame_count'),dq=dq,rx=rx,rxd=rxd,
                src=Counter(r.get('source') for r in rx_rows), meta=rows[0], pose=wm.get('robot_pose'), last_rx=last_rx, n_parse_err=n_parse_err)
def same(x,y):
    if x==y: return f'identical ({len(x)})'
    i=next((i for i,(p,q) in enumerate(zip(x,y)) if p!=q), min(len(x),len(y)))
    return f'DIFFER at {i}: {x[i] if i<len(x) else None} vs {y[i] if i<len(y) else None} (lens {len(x)} vs {len(y)})'
B1=load(f'{ref}/B1.json',f'{ref}/B1.events.jsonl'); S2=load(f'{ref}/task2-admit-before-decode/S.json',f'{ref}/task2-admit-before-decode/S.events.jsonl')
M=load(f'{out}/M.json',f'{out}/M.events.jsonl')
print(f'M: {M["objs"]} frames {M["frames"]} sha {M["sha"]} | meta policy={M["meta"].get("ingest_policy")} clock={M["meta"].get("ingest_clock")} | sources {dict(M["src"])}')
print('   vs B1: multiset', 'IDENTICAL' if M['sha']==B1['sha'] else 'DIFFER', '| dequeue', same(M['dq'],B1['dq']), '| receiver', same(M['rx'],B1['rx']))
print('   vs task-2 S per-line incl. depth_valid_frac:', same(M['rxd'],S2['rxd']))
p=M['pose'] or {}; sp=S2['pose'] or {}
core=lambda d: {k:d.get(k) for k in ('xyz','quaternion_xyzw','timestamp','frame_epoch')}
pose_ok = core(p)==core(sp)
print('   robot_pose core == task-2 S:', pose_ok, '|', core(p) if not pose_ok else '(xyz/quat/timestamp/frame_epoch match)')
keys_ok = all(k in p for k in ('sensor_ts_ns','pose_clock','age_s','stale','stale_after_s','rejected_writes','rejected_by_reason','writes_accepted','sensor_ts_regressions'))
stamp_ok = p.get('sensor_ts_ns')==M['last_rx']['t_sensor_ns']
print('   new keys present:', keys_ok, '| sensor_ts_ns == last received frame stamp:', stamp_ok, f'({p.get("sensor_ts_ns")} vs {M["last_rx"]["t_sensor_ns"]})', '| pose_clock', p.get('pose_clock'), '| stale', p.get('stale'), 'age_s', p.get('age_s'))
n_normal = sum(1 for r in M['rx'] if r[1] not in ('tracking_state','malformed'))
writer_ok = (M['n_parse_err']==0 and p.get('writes_accepted')==n_normal and p.get('sensor_ts_regressions')==0 and p.get('rejected_writes')==0)   # identity exact when no parse_error (sink fires at 5c, before the decodes)
print('   single writer: writes_accepted', p.get('writes_accepted'), '== tracking-normal receiver lines', n_normal, '| regressions', p.get('sensor_ts_regressions'), '| rejected', p.get('rejected_writes'), p.get('rejected_by_reason'), '->', 'OK' if writer_ok else 'VIOLATED')
ok = (M['sha']==B1['sha'] and M['dq']==B1['dq'] and M['rx']==B1['rx'] and M['rxd']==S2['rxd'] and dict(M['src'])=={'replay':240}
      and pose_ok and keys_ok and stamp_ok and p.get('pose_clock')=='sender' and writer_ok)
print('HARD GATE:', 'PASS' if ok else 'FAIL')
PY
echo P1T4_GATE_DONE
