#!/usr/bin/env bash
# P1 task 5 gate: headless dual replay of session1 (default policy = lossless, clock = sensor), NO viz client.
#   HARD (unchanged from task 4): multiset == sensor anchor 124/65@53 ad6f71a5b89c8506; dequeue + receiver sequences
#         identical to B1; per-line (decision, reason, frame_seq, depth_valid_frac) identical to task-2 S;
#         final /stats.robot_pose core == task-2 S; writes_accepted == tracking-normal receiver lines, 0 regressions, 0 rejects.
#   HARD (new, headless metrics; predicates as amended by the design + code reviews): rollup (from the raw JSON's `rollup`)
#         alive, still ticking at read time (stalled false, last_tick_age_s <= 2), late_ticks == 0, stale_rollups == 0,
#         ticks >= 40 and len(latency_hourly) == ticks == len(segmentation_hourly) (one bucket per tick per buffer);
#         sum(frames_in_bucket) == counters.processed; sum(gate_rejections) == counters.gate_rejections;
#         sum(throttle_skips) == counters.throttle_skips; sum(queue_drops+superseded+age_drops) == 0; no bucket flagged stale;
#         aggregate.input_hz > 0 and 0 < effective_ratio < 10; LAST bucket wm_total/wm_confirmed == /stats objects/confirmed;
#         sum(segmentation_hourly.frames_in_bucket) == counters.processed;
#         healthz.ingest: policy lossless, lane_full false, admitted_kf+admitted_nonkf == len(dequeue lines), no lane drops,
#         closed_puts 0 and depth {fifo: 0} (idle, drained; `closed` is printed, not gated: today a completed replay
#         leaves the lane open — nothing closes it before the harness reads — and a completion-time close is a wanted,
#         pending change that must not fail this gate); healthz has NO frame_flow key (replay: watchdog off).
set -u
cd /c/Users/konam/Desktop/calabi-repo/rtsm || exit 1
SP="C:/Users/konam/AppData/Local/Temp/claude/C--Users-konam-Desktop-calabi-repo-rtsm/c99f81f1-d743-4906-85ee-b894ab355ce0/scratchpad"
OUT="$SP/p1t5_gate"; mkdir -p "$OUT"
moved=0
if [ -d model_store/faiss ] && [ ! -d model_store/faiss.pre-p1t5 ]; then mv model_store/faiss model_store/faiss.pre-p1t5; moved=1; fi
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
if [ $moved = 1 ]; then mv model_store/faiss.pre-p1t5 model_store/faiss; fi
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
    last_rx=[r for r in rx_rows if r['reason'] not in ('tracking_state','malformed')][-1]
    n_parse_err=sum(1 for r in rx_rows if r['reason']=='parse_error')
    return dict(d=d,sha=sha,objs=(wm.get('objects'),wm.get('confirmed')),frames=lat.get('frame_count'),dq=dq,rx=rx,rxd=rxd,
                src=Counter(r.get('source') for r in rx_rows), meta=rows[0], pose=wm.get('robot_pose'), last_rx=last_rx, n_parse_err=n_parse_err, wm=wm, lat=lat)
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
n_normal = sum(1 for r in M['rx'] if r[1] not in ('tracking_state','malformed'))
writer_ok = (M['n_parse_err']==0 and p.get('writes_accepted')==n_normal and p.get('sensor_ts_regressions')==0 and p.get('rejected_writes')==0)
print('   robot_pose core == task-2 S:', pose_ok, '| single writer', p.get('writes_accepted'), '==', n_normal, '| regressions', p.get('sensor_ts_regressions'), '| rejected', p.get('rejected_writes'), '->', 'OK' if writer_ok else 'VIOLATED')
anchor_ok = (M['sha']==B1['sha'] and M['dq']==B1['dq'] and M['rx']==B1['rx'] and M['rxd']==S2['rxd'] and dict(M['src'])=={'replay':240} and pose_ok and p.get('pose_clock')=='sender' and writer_ok)
# ---- headless metrics ----
d=M['d']; lat=M['lat']; c=lat.get('counters') or {}
hl=d.get('latency_hourly') or []; sh=d.get('segmentation_hourly') or []
roll=d.get('rollup'); hz=d.get('healthz') or {}; ing=hz.get('ingest') or {}
S=lambda k,rows: sum(int(b.get(k,0) or 0) for b in rows)
nonempty=[b for b in hl if b.get('frames_in_bucket',0)>0]
print(f'   rollup: {roll}')
print(f'   latency_hourly: {len(hl)} buckets ({len(nonempty)} non-empty) | sum frames {S("frames_in_bucket",hl)} vs processed {c.get("processed")} | received {S("frames_received",hl)} vs {c.get("received")} | gate_rej {S("gate_rejections",hl)} vs {c.get("gate_rejections")} | throttle {S("throttle_skips",hl)} vs {c.get("throttle_skips")} | drops {S("queue_drops",hl)+S("superseded",hl)+S("age_drops",hl)} vs {c.get("queue_drops",0)+c.get("superseded",0)+c.get("age_drops",0)}')
print(f'   aggregate: input_hz {lat.get("input_hz")} processing_hz {lat.get("processing_hz")} effective_ratio {lat.get("effective_ratio")} | seg_hourly {len(sh)} buckets sum frames {S("frames_in_bucket",sh)}')
last=hl[-1] if hl else {}
print(f'   last bucket wm_total/confirmed {last.get("wm_total")}/{last.get("wm_confirmed")} vs /stats {M["wm"].get("objects")}/{M["wm"].get("confirmed")} | queue_depth_max over run {max((b.get("queue_depth_max",0) for b in hl), default=None)}')
print(f'   healthz.ingest: {ing} | frame_flow present: {"frame_flow" in hz} | status {hz.get("status")}')
n_stale = sum(1 for b in hl if b.get('stale_interval')) + sum(1 for b in sh if b.get('stale_interval'))
print(f'   stale-flagged buckets (latency+seg): {n_stale} | elapsed_s max {max((b.get("elapsed_s",0) for b in hl), default=None)}')
roll_ok = (bool(roll) and roll.get('late_ticks')==0 and roll.get('stale_rollups')==0 and (roll.get('ticks') or 0)>=40
           and roll.get('alive') is True and n_stale==0
           and roll.get('stalled') is False and (roll.get('last_tick_age_s') if roll.get('last_tick_age_s') is not None else 99) <= 2.0   # still ticking at read time (a wedged ticker reads late_ticks 0 forever)
           and len(hl)==roll.get('ticks')==len(sh))                                                                                        # one bucket per tick per buffer: no second owner, no stall
hist_ok = (len(hl)>=40 and S('frames_in_bucket',hl)==c.get('processed') and S('frames_received',hl)==c.get('received')
           and S('gate_rejections',hl)==c.get('gate_rejections')
           and S('throttle_skips',hl)==c.get('throttle_skips') and S('queue_drops',hl)+S('superseded',hl)+S('age_drops',hl)==0
           and S('frames_in_bucket',sh)==c.get('processed'))
agg_ok = (lat.get('input_hz') or 0)>0 and 0<(lat.get('effective_ratio') or 0)<10
wm_ok = last.get('wm_total')==M['wm'].get('objects') and last.get('wm_confirmed')==M['wm'].get('confirmed')
print(f'   healthz.ingest.closed = {ing.get("closed")} (informational: today nothing closes the lane at replay completion; a completion-time close is a wanted, pending change)')
ing_ok = (ing.get('policy')=='lossless' and ing.get('lane_full') is False
          and ing.get('closed_puts')==0 and ing.get('depth')=={'fifo':0}   # idle, drained (open today; closed acceptable)
          and (ing.get('admitted_kf',0)+ing.get('admitted_nonkf',0))==len(M['dq'])
          and ing.get('kf_dropped')==0 and ing.get('age_dropped')==0 and ing.get('nonkf_superseded')==0 and 'frame_flow' not in hz)
print('   headless predicates: rollup', roll_ok, '| history sums', hist_ok, '| aggregate', agg_ok, '| wm snapshot', wm_ok, '| ingest', ing_ok)
print('HARD GATE:', 'PASS' if (anchor_ok and roll_ok and hist_ok and agg_ok and wm_ok and ing_ok) else 'FAIL', '| anchor', anchor_ok)
PY
echo P1T5_GATE_DONE
