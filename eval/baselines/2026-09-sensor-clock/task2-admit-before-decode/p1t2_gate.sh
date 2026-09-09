#!/usr/bin/env bash
# P1 task 2 gate: sensor-mode headless dual replay must reproduce the sensor anchor (124/65@53, ad6f71a5)
# with an identical dequeue sequence vs eval/baselines/2026-09-sensor-clock/B1.events.jsonl. Wall run = info only.
set -u
cd /c/Users/konam/Desktop/calabi-repo/rtsm || exit 1
SP="C:/Users/konam/AppData/Local/Temp/claude/C--Users-konam-Desktop-calabi-repo-rtsm/c99f81f1-d743-4906-85ee-b894ab355ce0/scratchpad"
OUT="$SP/p1t2_gate"; mkdir -p "$OUT"
moved=0
if [ -d model_store/faiss ] && [ ! -d model_store/faiss.pre-p1t2 ]; then mv model_store/faiss model_store/faiss.pre-p1t2; moved=1; fi
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
run S
run W --set ingest.clock=wall
rm -rf model_store/faiss
if [ $moved = 1 ]; then mv model_store/faiss.pre-p1t2 model_store/faiss; fi
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
    dq=[(r['frame_seq'],r['t_sensor_ns'],r['outcome'],r['reason']) for r in rows if r['kind']=='dequeue']
    rx=[(r['decision'],r['reason'],r['frame_seq']) for r in rows if r['kind']=='receiver']
    dvf=[r.get('depth_valid_frac') for r in rows if r['kind']=='receiver']
    return dict(sha=sha,objs=(wm.get('objects'),wm.get('confirmed')),frames=lat.get('frame_count'),dq=dq,rx=rx,
                rxc=Counter((a,b) for a,b,_ in rx), dvf_present=sum(1 for v in dvf if v is not None), meta=rows[0])
S=load(f'{out}/S.json',f'{out}/S.events.jsonl'); W=load(f'{out}/W.json',f'{out}/W.events.jsonl')
B1=load(f'{ref}/B1.json',f'{ref}/B1.events.jsonl'); A=load(f'{ref}/A.json',f'{ref}/A.events.jsonl')
def same(x,y):
    if x==y: return f'identical ({len(x)})'
    i=next((i for i,(p,q) in enumerate(zip(x,y)) if p!=q), min(len(x),len(y)))
    return f'DIFFER at {i}: {x[i] if i<len(x) else None} vs {y[i] if i<len(y) else None} (lens {len(x)} vs {len(y)})'
print('S (sensor):', S['objs'], 'frames', S['frames'], 'sha', S['sha'], '| meta', S['meta'].get('ingest_clock'), '| rx', dict(S['rxc']), '| depth_valid_frac on', S['dvf_present'], 'of', len(S['rx']), 'receiver lines')
print('  vs sensor anchor B1:', 'multiset', 'IDENTICAL' if S['sha']==B1['sha'] else 'DIFFER', S['sha'], B1['sha'], '| dequeue', same(S['dq'],B1['dq']), '| receiver', same(S['rx'],B1['rx']))
print('W (wall):  ', W['objs'], 'frames', W['frames'], 'sha', W['sha'], '| rx', dict(W['rxc']))
print('  vs wall anchor A (info only):', 'multiset', 'IDENTICAL' if W['sha']==A['sha'] else 'DIFFER', W['sha'], A['sha'], '| dequeue', same(W['dq'],A['dq']))
PY
echo P1T2_GATE_DONE
