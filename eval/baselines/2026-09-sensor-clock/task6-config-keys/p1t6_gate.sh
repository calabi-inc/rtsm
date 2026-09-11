#!/usr/bin/env bash
# P1 task 6 gate: config key moves (io.websocket.* -> ingest.*) with the alias shim. Three headless dual replays of session1.
#   M  packaged defaults, 1x  -> every task-5 predicate unchanged: multiset == sensor anchor B1 124/65@53 ad6f71a5b89c8506,
#      dequeue (86) + receiver (240) sequences identical INCLUDING reasons (non_kf_grace_s 0.0 and the moves change nothing),
#      robot_pose 240/0/0, rollup alive/not stalled, sum(frames_in_bucket) == processed, /healthz.ingest idle + drained.
#      The log's SHA-256 must equal the CPU-side fingerprint of the same profiles on the harness-patched base (proves the
#      fingerprint replication used for L and T).
#   L  legacy-values profile (io.websocket.keyframe_every_n: 30, io.websocket.nonkf_min_interval_s: 0.5 = packaged values), 1x
#      -> identical to M (multiset, dequeue, receiver incl. reasons); the run log carries BOTH rtsm.cfg deprecation lines;
#      SHA(L.log) == fp([diag_L]) == fp([diag_L, legacy_same]) (old path -> same resolved dict).
#   T  FALSIFYING run: io.websocket.nonkf_min_interval_s: 1.0 through the OLD path, 5x (sensor clock -> speed-independent)
#      -> receiver enqueued 48 (9 keyframes + 39 non-KF) / throttled 192 (derived offline from B1's receiver lines with the
#      attempt-based throttle rule, which reproduces 86/154 at 0.5 s line for line); dequeue keyframes 9; the deprecation
#      line in the log; SHA(T.log) == fp([diag_T, legacy_throttle]) == fp([diag_T] + ingest.nonkf_min_interval_s=1.0).
#      If a runner site still read ws_cfg (or dropped the kwargs), T would reproduce 86/154 and FAIL here.
set -u
cd /c/Users/konam/Desktop/calabi-repo/rtsm || exit 1
SP="C:/Users/konam/AppData/Local/Temp/claude/C--Users-konam-Desktop-calabi-repo-rtsm/c99f81f1-d743-4906-85ee-b894ab355ce0/scratchpad"
OUT="$SP/p1t6_gate"; mkdir -p "$OUT"
printf 'io:\n  websocket:\n    keyframe_every_n: 30\n    nonkf_min_interval_s: 0.5\n' > "$OUT/legacy_same.yaml"
printf 'io:\n  websocket:\n    nonkf_min_interval_s: 1.0\n' > "$OUT/legacy_throttle.yaml"
moved=0
if [ -d model_store/faiss ] && [ ! -d model_store/faiss.pre-p1t6 ]; then mv model_store/faiss model_store/faiss.pre-p1t6; moved=1; fi
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
run L --profile "$OUT/legacy_same.yaml"
run T --profile "$OUT/legacy_throttle.yaml" --replay-speed 5
rm -rf model_store/faiss
if [ $moved = 1 ]; then mv model_store/faiss.pre-p1t6 model_store/faiss; fi
python -X utf8 - "$OUT" <<'PY'
import json,sys,hashlib,re
from collections import Counter
from rtsm.cfg import load_config, config_fingerprint
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
    return dict(d=d,sha=sha,objs=(wm.get('objects'),wm.get('confirmed')),frames=lat.get('frame_count'),dq=dq,rx=rx,
                rxc=Counter((r['decision'],r['reason']) for r in rx_rows), dqc=Counter(r['reason'] for r in rows if r['kind']=='dequeue'),
                meta=rows[0], pose=wm.get('robot_pose'), wm=wm, lat=lat)
def same(x,y):
    if x==y: return f'identical ({len(x)})'
    i=next((i for i,(p,q) in enumerate(zip(x,y)) if p!=q), min(len(x),len(y)))
    return f'DIFFER at {i}: {x[i] if i<len(x) else None} vs {y[i] if i<len(y) else None} (lens {len(x)} vs {len(y)})'
def log_sha(label):
    m=re.search(r'SHA-256 ([0-9a-f]{64})', open(f'{out}/{label}.log',encoding='utf-8',errors='replace').read()); return m.group(1) if m else None
def log_deprecations(label):
    t=open(f'{out}/{label}.log',encoding='utf-8',errors='replace').read()
    return (t.count('io.websocket.keyframe_every_n'), t.count('io.websocket.nonkf_min_interval_s'))
def cpu_fp(profiles, sets=()):
    # replicate the harness: it patches the packaged base (backend dual, visualization.enable false, analytics.buffer_frames 500)
    cfg=load_config(profiles=profiles, set_values=list(sets))
    cfg['segmentation']['backend']='dual'; cfg['visualization']['enable']=False; cfg['analytics']['buffer_frames']=500
    return config_fingerprint(cfg)
B1=load(f'{ref}/B1.json',f'{ref}/B1.events.jsonl')
M=load(f'{out}/M.json',f'{out}/M.events.jsonl'); L=load(f'{out}/L.json',f'{out}/L.events.jsonl'); T=load(f'{out}/T.json',f'{out}/T.events.jsonl')
print(f'M: {M["objs"]} frames {M["frames"]} sha {M["sha"]} | meta policy={M["meta"].get("ingest_policy")} clock={M["meta"].get("ingest_clock")}')
print('   vs B1: multiset', 'IDENTICAL' if M['sha']==B1['sha'] else 'DIFFER', '| dequeue', same(M['dq'],B1['dq']), '| receiver', same(M['rx'],B1['rx']))
p=M['pose'] or {}; d=M['d']; roll=d.get('rollup') or {}; hl=d.get('latency_hourly') or []; c=(M['lat'].get('counters') or {}); ing=(d.get('healthz') or {}).get('ingest') or {}
S=lambda k,rows: sum(int(b.get(k,0) or 0) for b in rows)
sh=d.get('segmentation_hourly') or []; last=hl[-1] if hl else {}
n_stale = sum(1 for b in hl if b.get('stale_interval')) + sum(1 for b in sh if b.get('stale_interval'))
m_task5 = (p.get('writes_accepted')==240 and p.get('sensor_ts_regressions')==0 and p.get('rejected_writes')==0
           and roll.get('alive') is True and roll.get('stalled') is False and roll.get('late_ticks')==0 and roll.get('stale_rollups')==0
           and len(hl)==roll.get('ticks')==len(sh) and n_stale==0
           and S('frames_in_bucket',hl)==c.get('processed') and S('frames_received',hl)==c.get('received')
           and S('gate_rejections',hl)==c.get('gate_rejections') and S('throttle_skips',hl)==c.get('throttle_skips')
           and S('queue_drops',hl)+S('superseded',hl)+S('age_drops',hl)==0 and S('frames_in_bucket',sh)==c.get('processed')
           and last.get('wm_total')==M['wm'].get('objects') and last.get('wm_confirmed')==M['wm'].get('confirmed')
           and (M['lat'].get('input_hz') or 0)>0 and 0<(M['lat'].get('effective_ratio') or 0)<10
           and ing.get('policy')=='lossless' and ing.get('depth')=={'fifo':0} and ing.get('closed_puts')==0 and ing.get('lane_full') is False
           and ing.get('admitted_kf',0)+ing.get('admitted_nonkf',0)==len(M['dq']) and ing.get('kf_dropped')==0 and ing.get('age_dropped')==0 and ing.get('nonkf_superseded')==0
           and 'frame_flow' not in (d.get('healthz') or {}))
print(f'   task-5 predicates (all of them): pose {p.get("writes_accepted")}/{p.get("sensor_ts_regressions")}/{p.get("rejected_writes")} | rollup ticks {roll.get("ticks")} late {roll.get("late_ticks")} stale {roll.get("stale_rollups")} stalled {roll.get("stalled")} | buckets {len(hl)}/{len(sh)} | sums frames {S("frames_in_bucket",hl)}=={c.get("processed")} received {S("frames_received",hl)}=={c.get("received")} gate_rej {S("gate_rejections",hl)}=={c.get("gate_rejections")} throttle {S("throttle_skips",hl)}=={c.get("throttle_skips")} drops {S("queue_drops",hl)+S("superseded",hl)+S("age_drops",hl)} seg {S("frames_in_bucket",sh)} | last wm {last.get("wm_total")}/{last.get("wm_confirmed")} | input_hz {M["lat"].get("input_hz")} ratio {M["lat"].get("effective_ratio")} | ingest {ing.get("policy")} depth {ing.get("depth")} admitted {ing.get("admitted_kf")}+{ing.get("admitted_nonkf")} -> {m_task5}')
m_fp = log_sha('M'); m_cpu = cpu_fp([f'{out}/M.profile.yaml'])
print(f'   fingerprint: log {m_fp} cpu {m_cpu} -> {"MATCH" if m_fp==m_cpu else "MISMATCH"}')
m_ok = M['sha']==B1['sha'] and M['dq']==B1['dq'] and M['rx']==B1['rx'] and m_task5 and m_fp==m_cpu
print(f'L: {L["objs"]} sha {L["sha"]} | vs M: multiset', 'IDENTICAL' if L['sha']==M['sha'] else 'DIFFER', '| dequeue', same(L['dq'],M['dq']), '| receiver', same(L['rx'],M['rx']))
l_dep = log_deprecations('L'); l_fp = log_sha('L'); l_cpu_diag = cpu_fp([f'{out}/L.profile.yaml']); l_cpu_both = cpu_fp([f'{out}/L.profile.yaml', f'{out}/legacy_same.yaml'])
print(f'   deprecation lines (kf, throttle): {l_dep} | fingerprint log {l_fp} == cpu[diag] {l_cpu_diag == l_fp} == cpu[diag+legacy] {l_cpu_both == l_fp}')
l_ok = L['sha']==M['sha'] and L['dq']==M['dq'] and L['rx']==M['rx'] and l_dep[0]>=1 and l_dep[1]>=1 and l_fp==l_cpu_diag==l_cpu_both
print(f'T: {T["objs"]} sha {T["sha"]} | receiver {dict(T["rxc"])} | dequeue reasons {dict(T["dqc"])}')
t_dep = log_deprecations('T'); t_fp = log_sha('T'); t_cpu_legacy = cpu_fp([f'{out}/T.profile.yaml', f'{out}/legacy_throttle.yaml']); t_cpu_new = cpu_fp([f'{out}/T.profile.yaml'], ['ingest.nonkf_min_interval_s=1.0'])
enq = T['rxc'][('enqueued','')]; thr = T['rxc'][('dropped','throttle')]; kfs = T['dqc'].get('keyframe')
print(f'   expected enqueued 48 / throttled 192 / keyframes 9 -> got {enq} / {thr} / {kfs} | deprecation (throttle) {t_dep[1]} | fingerprint log {t_fp} == cpu[diag+legacy] {t_cpu_legacy == t_fp} == cpu[diag+new] {t_cpu_new == t_fp}')
t_ok = (enq==48 and thr==192 and kfs==9 and t_dep[1]>=1 and t_fp==t_cpu_legacy==t_cpu_new)
print(f'   T multiset differs from M (expected, fewer admitted frames): {T["sha"]!=M["sha"]}')
print('HARD GATE:', 'PASS' if (m_ok and l_ok and t_ok) else 'FAIL', '| M', m_ok, '| L', l_ok, '| T', t_ok)
PY
echo P1T6_GATE_DONE
