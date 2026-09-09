"""Render eval/baselines/2026-09-post-reconcile-headless/README.md: headless anchor + comparison
against the viz-on anchor (full multiset and CONFIRMED-only multiset)."""
import json, os, sys, glob, hashlib
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = 'C:/Users/konam/Desktop/calabi-repo/rtsm'
os.chdir(ROOT)
H = 'eval/baselines/2026-09-post-reconcile-headless'
V = 'eval/baselines/2026-09-post-reconcile'
commit = sys.argv[1] if len(sys.argv) > 1 else '?'


def load(p):
    d = json.load(open(p, encoding='utf-8'))
    page = (d.get('objects') or {}).get('objects') or []
    full = (d.get('objects_full') or {}).get('objects') or page
    return d, page, full


def ms(objs, confirmed_only=False):
    it = [o for o in objs if (o.get('confirmed') or not confirmed_only)]
    return Counter((o.get('label_primary'), tuple(round(float(v), 3) for v in o.get('xyz_world') or []),
                    int(o.get('hits') or 0), bool(o.get('confirmed'))) for o in it)


def sha(c):
    return hashlib.sha256(json.dumps(sorted(map(list, c.elements())), default=str).encode()).hexdigest()[:16]


recs = [json.loads(l) for l in open(os.path.join(H, 'results.jsonl'), encoding='utf-8') if l.strip()]
runs = [r for r in recs if not r['name'].startswith('_')]
L = [f'# Post-reconcile anchor, HEADLESS (`--no-viz`) — tree `{commit}` (2026-09-08)', '',
     'Same recipe as `../2026-09-post-reconcile/` except the benchmark harness now runs headless by default '
     '(`visualization.enable=false` + `--no-viz`; `--viz` restores the old behaviour). This directory is the '
     '**forward reference** for P1\'s G1-A wall-parity gate; `../2026-09-post-reconcile/` remains the record of the '
     'G0 gate as passed (viz on, comparable to the pre-reconcile and demo2 baselines, which were all measured with viz on).', '',
     '## Why the numbers differ from the viz-on anchor', '',
     'With no viz server the pipeline step is faster (no per-frame JPEG encode / broadcast), so the wall-clock '
     'non-keyframe throttle and ingest-gate TTL admit a slightly different set of non-keyframes. The effect on session1: '
     'the **confirmed objects are identical** to the viz-on anchor; the difference is a handful of extra unconfirmed '
     'protos with 1–2 hits. This is the same wall-clock coupling behind the 53/54-frame flip, and it is what P1\'s '
     'sensor-time clock (execution plan P1 task 1) removes. Tier-2 `latency_hourly` columns and `input_hz` are empty/0 '
     'headless (the rollup lives in the viz push loop until P1 task 5).', '',
     '| job | run | objects/confirmed | frames | full sha | confirmed-only sha | confirmed identical to a same-frame-count viz-on run? | objects only here / only there | t_total mean | pcf | gate lines |',
     '|---|---|---|---|---|---|---|---|---|---|---|']
summary = {}
for r in runs:
    if r.get('error'):
        L.append(f"| {r['name']} | {r['run']} | ERROR {r['error']} | | | | | | | | |"); continue
    d, page, full = load(os.path.join(H, r['name'], f"run{r['run']}.json"))
    # compare only against a viz-on run of the SAME job with the SAME frame_count (the 53/54 flip changes the confirmed set)
    cmp_txt, extra = 'no same-frame-count viz-on sample', ''
    for cand in sorted(glob.glob(os.path.join(V, r['name'], 'run*.json'))):
        dv, pv, fv = load(cand)
        if ((dv.get('latency') or {}).get('frame_count')) != r.get('frame_count'):
            continue
        same_conf = sha(ms(full, True)) == sha(ms(fv, True))
        only_h = ms(full) - ms(fv); only_v = ms(fv) - ms(full)
        cmp_txt = ('YES' if same_conf else 'NO') + f' (vs viz-on {os.path.basename(cand)})'
        extra = f"+{sum(only_h.values())} / -{sum(only_v.values())} (unconf: {sum(v for k, v in only_h.items() if not k[3])})"
        break
    fs, cs = sha(ms(full)), sha(ms(full, True))
    summary.setdefault(r['name'], []).append((r.get('objects'), r.get('confirmed'), r.get('frame_count'), fs, cs))
    L.append(f"| {r['name']} | {r['run']} | {r.get('objects')}/{r.get('confirmed')} | {r.get('frame_count')} | `{fs}` | `{cs}` | {cmp_txt} | {extra} | {r.get('t_total_mean')} | {r.get('pose_conversion_failures')} | {r.get('gate_skip_log_lines')} |")
chk = os.path.join(H, 'dual', 'run0-check54.json')
if os.path.exists(chk):
    dc, pc, fc = load(chk)
    row = 'no 54-frame viz-on dual run'
    for cand in sorted(glob.glob(os.path.join(V, 'dual', 'run*.json'))):
        dv, pv, fv = load(cand)
        if ((dv.get('latency') or {}).get('frame_count')) == 54:
            only_h = ms(fc) - ms(fv); only_v = ms(fv) - ms(fc)
            row = (f"headless check run (the first `--no-viz` verification, 54 frames): {len(fc)} objects / "
                   f"{sum(1 for o in fc if o.get('confirmed'))} confirmed, full sha `{sha(ms(fc))}`; vs viz-on {os.path.basename(cand)} "
                   f"({len(fv)}/{sum(1 for o in fv if o.get('confirmed'))}): confirmed-only multiset identical = {sha(ms(fc, True)) == sha(ms(fv, True))}; "
                   f"objects only headless +{sum(only_h.values())} (unconfirmed: {sum(v for k, v in only_h.items() if not k[3])}), only viz-on -{sum(only_v.values())}")
            break
    L += ['', '## Same-frame-count check (dual, 54 frames)', '', '- ' + row]
L += ['', '## Headless floor (use these for G1-A)', '']
for name, rows in summary.items():
    objs = [o for o, *_ in rows]; conf = [c for _, c, *_ in rows]
    fshas = {f for *_, f, _c in rows}; cshas = {c for *_, c in rows}
    L.append(f"- **{name}:** objects {min(objs)}–{max(objs)}, confirmed {min(conf)}–{max(conf)}; full multiset identical across runs: {len(fshas) == 1}; confirmed-only multiset identical across runs: {len(cshas) == 1}; per run: " + ', '.join(f"{o}/{c}@{f}" for o, c, f, *_ in rows))
L += ['', '```'] + [json.dumps(m) for m in recs if m['name'].startswith('_')] + ['```', '']
open(os.path.join(H, 'README.md'), 'w', encoding='utf-8').write('\n'.join(L))
print('\n'.join(L[12:]))
