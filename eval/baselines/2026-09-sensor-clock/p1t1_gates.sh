#!/usr/bin/env bash
# P1 task 1 gates on headless dual / session1, trace ON for every run:
#   A  : ingest.clock=wall            -> must reproduce the headless anchor 121/66@53 full sha 92e0a8f1
#   B1 : default (auto -> sensor) 1x
#   B2 : default (auto -> sensor) 1x
#   B3 : default (auto -> sensor) 5x  -> B1 == B2 == B3 dequeued sequence + gate reasons + object multiset
set -u
cd /c/Users/konam/Desktop/calabi-repo/rtsm || exit 1
SP="C:/Users/konam/AppData/Local/Temp/claude/C--Users-konam-Desktop-calabi-repo-rtsm/c99f81f1-d743-4906-85ee-b894ab355ce0/scratchpad"
OUT="$SP/p1t1_gates"; mkdir -p "$OUT"
moved=0
if [ -d model_store/faiss ] && [ ! -d model_store/faiss.pre-p1t1 ]; then mv model_store/faiss model_store/faiss.pre-p1t1; moved=1; fi
run() {  # $1 label, rest = extra harness args
  local label="$1"; shift
  printf 'diagnostics:\n  enabled: true\n  event_log_path: "%s/%s.events.jsonl"\n' "$OUT" "$label" > "$OUT/$label.profile.yaml"
  rm -rf model_store/faiss reports/datasheet_raw_dual*.json
  python -X utf8 scripts/benchmark_datasheet.py dual --profile "$OUT/$label.profile.yaml" "$@" > "$OUT/$label.harness.stdout" 2>&1; local rc=$?
  local raw; raw=$(ls reports/datasheet_raw_dual.*.json 2>/dev/null | head -1)
  cp -f "$raw" "$OUT/$label.json" 2>/dev/null; cp -f reports/run_dual.log "$OUT/$label.log" 2>/dev/null
  ls rtsm/cfg/rtsm.yaml.*bak* 2>/dev/null && echo "WARN: stray harness backup present"
  echo "== $label rc=$rc =="
  python -X utf8 - "$OUT/$label.json" "$OUT/$label.events.jsonl" <<'PY'
import json,sys,hashlib
from collections import Counter
d=json.load(open(sys.argv[1],encoding='utf-8'))
wm=d.get('working_memory') or {}; lat=d.get('latency') or {}
def ms(objs): return Counter((o.get('label_primary'),tuple(round(float(v),3) for v in o.get('xyz_world') or []),int(o.get('hits') or 0),bool(o.get('confirmed'))) for o in objs)
def sha(c): return hashlib.sha256(json.dumps(sorted(map(list,c.elements())),default=str).encode()).hexdigest()[:16]
full=(d.get('objects_full') or {}).get('objects') or []
known={'92e0a8f1206d77da':'HEADLESS anchor 121/66@53','fc6d9aea5df23786':'HEADLESS 122/68@54'}
fs=sha(ms(full)); cs=sha(ms([o for o in full if o.get('confirmed')]))
rows=[json.loads(l) for l in open(sys.argv[2],encoding='utf-8') if l.strip()]
meta=rows[0]; dq=[r for r in rows if r['kind']=='dequeue']; rx=[r for r in rows if r['kind']=='receiver']
print('objects/confirmed',wm.get('objects'),wm.get('confirmed'),'frames',lat.get('frame_count'),'full sha',fs,'->',known.get(fs,'NEW'),'| confirmed-only',cs,'| upserts',wm.get('upserts_total'))
print('meta.ingest_clock',meta.get('ingest_clock'),'| rtsm args',d.get('rtsm_extra_args'))
print('rx',Counter((r['decision'],r['reason']) for r in rx),'| dq',Counter((r['outcome'],r['reason']) for r in dq))
print('t_total mean',(lat.get('t_total') or {}).get('mean'),'| gate_acceptance_rate',lat.get('gate_acceptance_rate'))
PY
}
run A --set ingest.clock=wall
run B1
run B2
run B3 --replay-speed 5
rm -rf model_store/faiss
if [ $moved = 1 ]; then mv model_store/faiss.pre-p1t1 model_store/faiss; fi
echo "== G1-B comparison =="
python -X utf8 - "$OUT" <<'PY'
import json,sys,hashlib,os
from collections import Counter
out=sys.argv[1]
def load(label):
    rows=[json.loads(l) for l in open(f"{out}/{label}.events.jsonl",encoding='utf-8') if l.strip()]
    d=json.load(open(f"{out}/{label}.json",encoding='utf-8'))
    full=(d.get('objects_full') or {}).get('objects') or []
    ms=Counter((o.get('label_primary'),tuple(round(float(v),3) for v in o.get('xyz_world') or []),int(o.get('hits') or 0),bool(o.get('confirmed'))) for o in full)
    sha=hashlib.sha256(json.dumps(sorted(map(list,ms.elements())),default=str).encode()).hexdigest()[:16]
    enq=[(r['frame_seq'],r['t_sensor_ns']) for r in rows if r['kind']=='receiver' and r['decision']=='enqueued']
    thr=[(r['frame_seq'],r['t_sensor_ns']) for r in rows if r['kind']=='receiver' and r['decision']=='dropped']
    dq=[(r['frame_seq'],r['t_sensor_ns'],r['outcome'],r['reason']) for r in rows if r['kind']=='dequeue']
    wm=d.get('working_memory') or {}
    return dict(enq=enq,thr=thr,dq=dq,sha=sha,objs=(wm.get('objects'),wm.get('confirmed')))
R={k:load(k) for k in ('A','B1','B2','B3')}
def same(a,b,key):
    x,y=R[a][key],R[b][key]
    if x==y: return f'identical ({len(x)})'
    i=next((i for i,(p,q) in enumerate(zip(x,y)) if p!=q), min(len(x),len(y)))
    return f'DIFFER at index {i}: {x[i] if i<len(x) else None} vs {y[i] if i<len(y) else None} (lens {len(x)} vs {len(y)})'
for a,b in (('B1','B2'),('B1','B3'),('B2','B3')):
    print(f'{a} vs {b}: enqueued {same(a,b,"enq")} | throttled {same(a,b,"thr")} | dequeue(seq,ts,outcome,reason) {same(a,b,"dq")} | multiset {"identical" if R[a]["sha"]==R[b]["sha"] else "DIFFER"} {R[a]["sha"]} {R[b]["sha"]} | counts {R[a]["objs"]} {R[b]["objs"]}')
print(f'A (wall) vs B1 (sensor): enqueued {same("A","B1","enq")} | dequeue {same("A","B1","dq")} | multiset {R["A"]["sha"]} vs {R["B1"]["sha"]} | counts {R["A"]["objs"]} vs {R["B1"]["objs"]}')
ca=Counter((o,r) for *_,o,r in R['A']['dq']); cb=Counter((o,r) for *_,o,r in R['B1']['dq'])
print('A outcomes',dict(ca)); print('B1 outcomes',dict(cb))
PY
echo P1T1_GATES_DONE
