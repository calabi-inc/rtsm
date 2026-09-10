#!/usr/bin/env bash
# P1 task 3 gate (G1-A): headless dual replay of session1 through the lanes.
#   L  = default (auto -> lossless) : HARD  -> sensor anchor 124/65@53 ad6f71a5b89c8506,
#        dequeue + receiver sequences identical to B1, receiver lines all source=replay,
#        per-line (decision, reason, frame_seq, depth_valid_frac) identical to task-2 S.
#   G  = --set ingest.policy=legacy  : HARD  -> same.
#   T1..T3 = --set ingest.policy=latest : INFO ONLY (not reproducible by construction);
#        report min/max counts and the lane counters; assert invariants only.
set -u
cd /c/Users/konam/Desktop/calabi-repo/rtsm || exit 1
SP="C:/Users/konam/AppData/Local/Temp/claude/C--Users-konam-Desktop-calabi-repo-rtsm/c99f81f1-d743-4906-85ee-b894ab355ce0/scratchpad"
OUT="$SP/p1t3_gate"; mkdir -p "$OUT"
moved=0
if [ -d model_store/faiss ] && [ ! -d model_store/faiss.pre-p1t3 ]; then mv model_store/faiss model_store/faiss.pre-p1t3; moved=1; fi
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
run L
run G --set ingest.policy=legacy
run T1 --set ingest.policy=latest
run T2 --set ingest.policy=latest
run T3 --set ingest.policy=latest
rm -rf model_store/faiss
if [ $moved = 1 ]; then mv model_store/faiss.pre-p1t3 model_store/faiss; fi
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
    return dict(sha=sha,objs=(wm.get('objects'),wm.get('confirmed')),frames=lat.get('frame_count'),dq=dq,rx=rx,rxd=rxd,
                src=Counter(r.get('source') for r in rx_rows), lanes=Counter(r.get('lane') for r in rx_rows if r['decision']=='enqueued'),
                rxc=Counter((a,b) for a,b,*_ in rx), meta=rows[0], stats=wm.get('ingest_lanes'), counters=(lat.get('counters') or {}))
def same(x,y):
    if x==y: return f'identical ({len(x)})'
    i=next((i for i,(p,q) in enumerate(zip(x,y)) if p!=q), min(len(x),len(y)))
    return f'DIFFER at {i}: {x[i] if i<len(x) else None} vs {y[i] if i<len(y) else None} (lens {len(x)} vs {len(y)})'
B1=load(f'{ref}/B1.json',f'{ref}/B1.events.jsonl'); S2=load(f'{ref}/task2-admit-before-decode/S.json',f'{ref}/task2-admit-before-decode/S.events.jsonl')
ok=True
for lab in ('L','G'):
    R=load(f'{out}/{lab}.json',f'{out}/{lab}.events.jsonl')
    print(f'{lab}: {R["objs"]} frames {R["frames"]} sha {R["sha"]} | meta policy={R["meta"].get("ingest_policy")} clock={R["meta"].get("ingest_clock")} | rx {dict(R["rxc"])} | sources {dict(R["src"])} | lanes {dict(R["lanes"])} | stats {R["stats"]}')
    print('   vs B1: multiset', 'IDENTICAL' if R['sha']==B1['sha'] else 'DIFFER', '| dequeue', same(R['dq'],B1['dq']), '| receiver', same(R['rx'],B1['rx']))
    print('   vs task-2 S per-line incl. depth_valid_frac:', same(R['rxd'],S2['rxd']))
    want_policy={'L':'lossless','G':'legacy'}[lab]; want_lanes={'L':{'fifo':86},'G':{None:86}}[lab]
    print('   meta policy ==', want_policy, ':', R['meta'].get('ingest_policy')==want_policy, '| enqueued lanes ==', want_lanes, ':', dict(R['lanes'])==want_lanes)
    ok &= (R['sha']==B1['sha'] and R['dq']==B1['dq'] and R['rx']==B1['rx'] and R['rxd']==S2['rxd'] and dict(R['src'])=={'replay':240}
           and R['meta'].get('ingest_policy')==want_policy and dict(R['lanes'])==want_lanes)
print('HARD GATE:', 'PASS' if ok else 'FAIL')
print('--- policy=latest (info only; pacing-dependent, never compared to the anchor)')
for lab in ('T1','T2','T3'):
    try: R=load(f'{out}/{lab}.json',f'{out}/{lab}.events.jsonl')
    except Exception as ex: print(lab,'missing',ex); continue
    st=R['stats'] or {}
    lane_rows=[r for r in [json.loads(l) for l in open(f'{out}/{lab}.events.jsonl',encoding='utf-8') if l.strip()] if r['kind']=='receiver' and r.get('source')=='lanes']
    enq=sum(1 for r in R['rx'] if r[0]=='enqueued'); deq=len(R['dq'])
    print(f'{lab}: {R["objs"]} frames {R["frames"]} sha {R["sha"]} | rx {dict(R["rxc"])} | sources {dict(R["src"])} | enqueued {enq} - lanes-dropped {len(lane_rows)} = {enq-len(lane_rows)} vs dequeued {deq} | superseded {st.get("nonkf_superseded")} kf_dropped {st.get("kf_dropped")} age_dropped {st.get("age_dropped")} max_depth {st.get("max_depth_seen")} | counters queue_drops {R["counters"].get("queue_drops")} superseded {R["counters"].get("superseded")} age_drops {R["counters"].get("age_drops")}')
    inv = (st.get('max_depth_seen',99) <= 2 and st.get('kf_dropped',0)==0 and enq-len(lane_rows)==deq and all(r['lane'] in ('keyframe','latest') for r in lane_rows)
           and R['counters'].get('queue_drops')==0 and R['counters'].get('superseded')==st.get('nonkf_superseded'))
    print('   invariants (session1: depth<=2, no KF dropped, enqueued-lanes_dropped==dequeued, lane ids, queue_drops==0, counters.superseded==lanes):', 'OK' if inv else 'VIOLATED')
PY
echo P1T3_GATE_DONE
