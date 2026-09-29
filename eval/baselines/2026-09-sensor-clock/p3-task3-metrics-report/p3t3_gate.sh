#!/usr/bin/env bash
# P3 task 3 gate (G3-3, GPU): the metrics + the free report over the ledgers; the runner change moves nothing.
#   1. `rtsm eval recordings/session1 --mode as_deployed --repeats 3` (dual): every run = 124/65 @53 processed, fingerprint
#      ad6f71a5b89c8506, dequeue (86) + receiver (240) sequences identical to B1; metrics.json + report.md written; every
#      scalar carries n_runs == 3; every scalar has spread 0 (identical ledgers -> identical metrics; the report changed nothing).
#   2. Logic cross-checks on run_1: sum of cluster members == ids seen in the obs ledger; duplicates_all == ids - clusters;
#      the id-level detection numbers == observation_summary's view/obs join; all.views == sum over frames of
#      (reidentified + duplicated + missed + reidentified_outside_frustum); every worst moment's stamp is a dequeue / pose line;
#      admission counts == summary.frames; every cluster's leader median lies inside the bbox of its members' raw observations;
#      label disagreement 0 for every single-observation track; scatter n_tracks == tracks with >= 3 observations;
#      a cluster with a survivor-label conflict has >= 2 survivors.
#   3. `--mode dense`: masked + unmasked detection present; masked.views == the frame sum over unshadowed frames; cadence
#      "representative"; report names the masked rows.
#   4. `--mode every_frame`: 240 receiver lines all enqueued (0 throttled), 0 gate_rejected, processed + frame_rejected == 240,
#      keyframes >= 1.0 s apart; cadence "exhaustive"; wall time recorded.
#   5. `rtsm report` on the predicate-1 directory reproduces metrics.json + report.md byte for byte.
#   6. Informational: TUM fr1/desk (ROS 1 bag, float depth, no confidence map, no tracking topic) through `rtsm eval --max-frames 300`
#      writes a report without error.
#   7. The CPU suite is run by the PR, not here.
set -u
cd /c/Users/konam/Desktop/calabi-repo/rtsm || exit 1
SP="C:/Users/konam/AppData/Local/Temp/claude/C--Users-konam-OneDrive-Desktop-calabi-repo-rtsm/949c0384-41c4-4c7e-9f67-2ca869f46d96/scratchpad"
OUT="$SP/p3t3_gate"; rm -rf "$OUT"; mkdir -p "$OUT"
run_eval() {  # label, then args
  local label="$1"; shift
  local t0=$SECONDS
  python -X utf8 -m rtsm eval "$@" --out "$OUT/$label" --set segmentation.backend=dual > "$OUT/$label.stdout" 2>&1; local rc=$?
  echo "== eval $label rc=$rc wall=$((SECONDS - t0)) s =="; tail -4 "$OUT/$label.stdout" | cut -c1-200
}
run_eval s1_replay recordings/session1 --mode as_deployed --repeats 3
run_eval s1_dense  recordings/session1 --mode dense --repeats 1
run_eval s1_every  recordings/session1 --mode every_frame --repeats 1
run_eval fr1_desk  recordings/external/rgbd_dataset_freiburg1_desk.bag --mode as_deployed --repeats 1 --max-frames 300
# 5. regeneration
cp "$OUT/s1_replay/metrics.json" "$OUT/s1_replay/metrics.first.json"; cp "$OUT/s1_replay/report.md" "$OUT/s1_replay/report.first.md"
python -X utf8 -m rtsm report "$OUT/s1_replay" --set segmentation.backend=dual > "$OUT/report_regen.stdout" 2>&1; echo "== report regen rc=$? =="

python -X utf8 - "$OUT" <<'PY'
import glob, json, os, sys
from collections import Counter
from rtsm.evaluation.ledger import by_kind, observation_summary, read_events
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
res = {}

# ---- 1. anchor x3 + the report's floors ----
ok1 = True
for d in sorted(glob.glob(f'{out}/s1_replay/run_*')):
    s = json.load(open(os.path.join(d, 'summary.json'), encoding='utf-8'))
    rows = read_events(os.path.join(d, 'events.jsonl')); k, dq, rx = seqs(rows)
    m = s['memory']
    ok = (m['fingerprint'] == ANCHOR and m['objects_count'] == 124 and m['confirmed_count'] == 65 and s['frames']['processed'] == 53 and dq == b1_dq and rx == b1_rx)
    print(f'   {os.path.basename(d)}: {m["objects_count"]}/{m["confirmed_count"]} @{s["frames"]["processed"]} fp {m["fingerprint"]} | dequeue {same(dq, b1_dq)} | receiver {same(rx, b1_rx)} | wall {s["wall_s"]} s -> {ok}')
    ok1 = ok1 and ok
mp = f'{out}/s1_replay/metrics.json'; rp = f'{out}/s1_replay/report.md'
have = os.path.isfile(mp) and os.path.isfile(rp)
M = json.load(open(mp, encoding='utf-8')) if have else {}
agg = M.get('aggregate', {}); sc = agg.get('scalars', {})
n3 = all(v['n_runs'] == 3 for v in sc.values()); nz = {k: v['values'] for k, v in sc.items() if v['spread'] not in (0, 0.0)}
ok1 = ok1 and have and M.get('n_runs') == 3 and agg.get('floor_established') is True and agg.get('identical_fingerprints') is True and n3 and not nz
print(f'1. anchor x3 + report: files {have} n_runs {M.get("n_runs")} floor {agg.get("floor_established")} identical fp {agg.get("identical_fingerprints")} | scalars {len(sc)} all n_runs==3 {n3} | nonzero spreads {nz} -> {ok1}')
res['p1_anchor_x3_report'] = ok1

# ---- 2. logic cross-checks on run_1 ----
r1 = M['runs'][0] if M else {}
rows1 = read_events(f'{out}/s1_replay/run_1/events.jsonl'); k1 = by_kind(rows1)
s1 = json.load(open(f'{out}/s1_replay/run_1/summary.json', encoding='utf-8'))
obs_ids = {r['object_id'] for r in k1.get('obs', []) if r.get('object_id') and r.get('outcome') in ('matched', 'created')}
members = [oid for c in r1.get('clusters', []) for oid in c['members']]
c_a = (len(members) == len(set(members)) == len(obs_ids) == r1['scalars']['memory.ids_seen'])
c_b = (r1['scalars']['clusters.duplicates_all'] == len(obs_ids) - len(r1['clusters']))
vj = observation_summary(rows1)['view']
c_c = (r1['detection']['id_level']['reidentified'] == vj['in_frustum_and_matched'] and r1['detection']['id_level']['missed'] == vj['in_frustum_and_missed'])
fr_sum = sum(f['n_reidentified'] + f['n_duplicated'] + f['n_missed'] + f['n_reidentified_outside_frustum'] for f in r1['frames'])
c_d = (r1['detection']['all']['views'] == fr_sum)
stamps = {r.get('t_sensor_ns') for r in k1.get('dequeue', [])} | {r.get('t_sensor_ns') for r in k1.get('pose', [])}
c_e = all(m['t_sensor_ns'] in stamps for m in r1['worst_moments']) and len(r1['worst_moments']) == 10
adm = r1['admission']
c_f = (adm['receiver'] == s1['frames']['receiver'] and adm['dequeue'] == s1['frames']['dequeue'] and adm['outcomes'] == s1['frames']['outcomes'])
# leader median inside the bbox of the members' raw observations
import numpy as np
pts = {}
for r in k1.get('obs', []):
    if r.get('object_id') and r.get('outcome') in ('matched', 'created') and r.get('p_world'):
        pts.setdefault(r['object_id'], []).append(r['p_world'])
inside = 0
for c in r1['clusters']:
    P = np.asarray([p for oid in c['members'] for p in pts.get(oid, [])], dtype=float)
    pos = np.asarray(c['position'], dtype=float)
    if P.size and np.all(pos >= P.min(0) - 1e-3) and np.all(pos <= P.max(0) + 1e-3): inside += 1
c_g = (inside == len(r1['clusters']))
single = [o for o in r1['objects'] if o['n_obs'] == 1]
c_h = all(o['label_disagreement'] in (0.0, None) for o in single) and len(single) > 0
c_i = (r1['scatter']['n_tracks'] == sum(1 for o in r1['objects'] if o['n_obs'] >= 3))
conf = [c for c in r1['clusters'] if len(c['labels']['survivor_label_primary']) > 1]
c_j = all(c['n_survivors'] >= 2 for c in conf) and len(conf) == r1['scalars']['labels.clusters_with_survivor_label_conflict']
ok2 = all([c_a, c_b, c_c, c_d, c_e, c_f, c_g, c_h, c_i, c_j])
print(f'2. cross-checks run_1: members==ids {c_a} ({len(obs_ids)} ids, {len(r1.get("clusters", []))} clusters) | dups==ids-clusters {c_b} | id-level det == view join {c_c} ({vj["in_frustum_and_matched"]}/{vj["in_frustum_and_missed"]}) | views == frame sum {c_d} ({r1["detection"]["all"]["views"]}) | moments exist {c_e} | admission == summary {c_f} | leader inside bbox {c_g} ({inside}) | single-obs disagreement 0 {c_h} ({len(single)}) | scatter tracks {c_i} ({r1["scatter"]["n_tracks"]}) | label conflicts {c_j} ({len(conf)}) -> {ok2}')
res['p2_cross_checks'] = ok2

# ---- 3. dense: masked vs unmasked ----
D = json.load(open(f'{out}/s1_dense/metrics.json', encoding='utf-8')); d1 = D['runs'][0]
rs = json.load(open(f'{out}/s1_dense/resolved.json', encoding='utf-8'))
det = d1['detection']
masked_sum = sum(f['n_reidentified'] + f['n_duplicated'] + f['n_missed'] + f['n_reidentified_outside_frustum'] for f in d1['frames'] if not f['gate_shadow'])
all_sum = sum(f['n_reidentified'] + f['n_duplicated'] + f['n_missed'] + f['n_reidentified_outside_frustum'] for f in d1['frames'])
md = open(f'{out}/s1_dense/report.md', encoding='utf-8').read()
ok3 = (d1['shadow_mode'] is True and 'masked' in det and det['masked']['views'] == masked_sum and det['all']['views'] == all_sum
       and rs['cadence'] == 'representative' and 'masked: re-identification rate' in md and 'representative (dense)' in md)
print(f'3. dense: shadow {d1["shadow_mode"]} | all views {det["all"]["views"]} (frame sum {all_sum}) reid {det["all"]["reid_rate"]} | masked views {det["masked"]["views"]} (frame sum {masked_sum}) reid {det["masked"]["reid_rate"]} | shadowed {d1["admission"]["shadowed"]}/{d1["admission"]["processed"]} | cadence {rs["cadence"]} -> {ok3}')
res['p3_dense_masked'] = ok3

# ---- 4. every_frame ----
E = json.load(open(f'{out}/s1_every/metrics.json', encoding='utf-8')); e1 = E['runs'][0]
es = json.load(open(f'{out}/s1_every/resolved.json', encoding='utf-8'))
esum = json.load(open(f'{out}/s1_every/run_1/summary.json', encoding='utf-8'))
erows = read_events(f'{out}/s1_every/run_1/events.jsonl'); ek = by_kind(erows)
kf_ts = sorted(r['t_sensor_ns'] for r in ek.get('receiver', []) if r['decision'] == 'enqueued' and r['is_keyframe'])
kf_gaps = [(b - a) / 1e9 for a, b in zip(kf_ts, kf_ts[1:])]
outcomes = Counter(r['outcome'] for r in ek.get('dequeue', []))
emd = open(f'{out}/s1_every/report.md', encoding='utf-8').read()
ok4 = (esum['frames']['receiver'] == {'enqueued': 240} and outcomes.get('gate_rejected', 0) == 0 and outcomes.get('processed', 0) + outcomes.get('frame_rejected', 0) == 240
       and es['cadence'] == 'exhaustive' and es['nonkf_min_interval_s'] == 0.0 and (not kf_gaps or min(kf_gaps) >= 1.0 - 1e-6) and 'exhaustive (every_frame)' in emd and esum['aborted'] is None)
print(f'4. every_frame: receiver {esum["frames"]["receiver"]} | dequeue {dict(outcomes)} | keyframes {len(kf_ts)} min gap {min(kf_gaps) if kf_gaps else None} | cadence {es["cadence"]} throttle {es["nonkf_min_interval_s"]} | memory {esum["memory"]["objects_count"]}/{esum["memory"]["confirmed_count"]} | reid {e1["scalars"]["detection.reid_rate"]} masked {e1["scalars"].get("detection.masked.reid_rate")} | wall {esum["wall_s"]} s -> {ok4}')
res['p4_every_frame'] = ok4

# ---- 5. regeneration ----
import filecmp
ok5 = filecmp.cmp(f'{out}/s1_replay/metrics.first.json', f'{out}/s1_replay/metrics.json', shallow=False) and filecmp.cmp(f'{out}/s1_replay/report.first.md', f'{out}/s1_replay/report.md', shallow=False)
print(f'5. rtsm report regeneration byte-identical: {ok5}')
res['p5_regeneration'] = ok5

# ---- 6. informational: TUM fr1/desk ----
try:
    F = json.load(open(f'{out}/fr1_desk/metrics.json', encoding='utf-8')); f1 = F['runs'][0]
    fsum = json.load(open(f'{out}/fr1_desk/run_1/summary.json', encoding='utf-8'))
    print(f'6. (info) fr1_desk: processed {fsum["frames"]["processed"]} memory {fsum["memory"]["objects_count"]}/{fsum["memory"]["confirmed_count"]} | clusters {f1["scalars"]["clusters.n"]} reid {f1["scalars"]["detection.reid_rate"]} dups {f1["scalars"]["duplicates.n"]} | pose hz {f1["scalars"]["pose.sensor_hz"]} conf2 {f1["scalars"]["pose.conf2_frac_p50"]} | wall {fsum["wall_s"]} s | report {os.path.isfile(f"{out}/fr1_desk/report.md")}')
except Exception as e:
    print(f'6. (info) fr1_desk: FAILED ({type(e).__name__}: {e})')
print(f'HARD GATE: {"PASS" if all(res.values()) else "FAIL"} | {res}')
PY
echo P3T3_GATE_DONE
