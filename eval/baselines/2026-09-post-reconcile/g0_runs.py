"""G0 measurements on the merged tree: dual x3, grounded_sam2 x3, grounded_sam2 --profile e1-demo2 x3.
Interleaved (d, g, e) x3, empty FAISS per run, user's index moved aside and restored.
Writes eval/baselines/2026-09-post-reconcile/<name>/runN.json + runN.log and results.jsonl.
"""
import json, os, sys, time, shutil, subprocess, hashlib, glob
from collections import Counter

ROOT = 'C:/Users/konam/Desktop/calabi-repo/rtsm'
os.chdir(ROOT)
OUT = 'eval/baselines/2026-09-post-reconcile'
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, 'results.jsonl')
DONE = os.path.join(OUT, 'G0_DONE')
FAISS = 'model_store/faiss'
ASIDE = 'model_store/faiss.pre-g0-20260908'
PROFILE = 'examples/rc_car_agent/e1-demo2.profile.yaml'
ANCHORS = {'b71b98ca1fc2bf0d': 'PRE-FLIP main 107/70@53', '935bc8eb254c4dfe': 'PRE-FLIP base 111/74@54',
           '1994e0fe5dd6167c': 'POST-FLIP demo2 115/66@53', '72fd5c7f0475da90': 'POST-FLIP demo2 116/68@54',
           'f12c8dc2024f0c53': 'demo2 e1 gs2 25/18@53 run1', 'a556dbdcad9fef0a': 'demo2 e1 gs2 25/18@54',
           'eab7e618f54b2f13': 'main gs2 139/81', '4a29c1ed37d2b052': 'main gs2 133/81'}
JOBS = [('dual', 'dual', []), ('grounded_sam2', 'grounded_sam2', []), ('grounded_sam2-e1', 'grounded_sam2', ['--profile', PROFILE])]


def sh(*a, check=True):
    r = subprocess.run(list(a), capture_output=True, text=True, encoding='utf-8', errors='replace')
    if check and r.returncode != 0:
        raise RuntimeError(f'{a} rc={r.returncode}\n{r.stdout}\n{r.stderr}')
    return r


def multiset(objs):
    return Counter((o.get('label_primary'), tuple(round(float(v), 3) for v in o.get('xyz_world') or []),
                    int(o.get('hits') or 0), bool(o.get('confirmed'))) for o in objs)


def ms_sha(ms):
    return hashlib.sha256(json.dumps(sorted(map(list, ms.elements())), default=str).encode()).hexdigest()[:16]


def kill_stale():
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*-m rtsm --replay*' } "
          "| ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Output ('killed ' + $_.ProcessId) }")
    return sh('powershell', '-NoProfile', '-Command', ps, check=False).stdout.strip()


def log(rec):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, default=str) + '\n')
    print(json.dumps(rec, default=str), flush=True)


def run(name, backend, extra, i):
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    shutil.rmtree(FAISS, ignore_errors=True)
    suffix = '.e1-demo2.profile' if extra else ''
    raw = f'reports/datasheet_raw_{backend}{suffix}.json'
    if os.path.exists(raw):
        os.remove(raw)
    t0 = time.time()
    with open(os.path.join(d, f'run{i}.harness.stdout'), 'w', encoding='utf-8', errors='replace') as f:
        try:
            rc = subprocess.call([sys.executable, '-X', 'utf8', 'scripts/benchmark_datasheet.py', backend, *extra],
                                 stdout=f, stderr=subprocess.STDOUT, timeout=900)
        except subprocess.TimeoutExpired:
            rc = 'TIMEOUT'
    killed = kill_stale()
    rec = {'name': name, 'run': i, 'backend': backend, 'extra': extra, 'rc': rc, 'seconds': round(time.time() - t0, 1), 'killed_stale': killed}
    if os.path.exists(raw):
        shutil.copy(raw, os.path.join(d, f'run{i}.json'))
        logp = f'reports/run_{backend}.log'
        if os.path.exists(logp):
            shutil.copy(logp, os.path.join(d, f'run{i}.log'))
            txt = open(logp, encoding='utf-8', errors='replace').read()
            rec['gate_skip_log_lines'] = txt.count('frame-quality gate')
            rec['dropping_frame_pose_lines'] = txt.count('dropping frame: pose present')
            rec['clearance_warning'] = 'io.clearance.enable=true but no receive-time clearance source' in txt
        data = json.load(open(raw, encoding='utf-8'))
        if 'error' in data:
            rec['error'] = data['error']
        else:
            wm = data.get('working_memory') or {}
            lat = data.get('latency') or {}
            page = (data.get('objects') or {}).get('objects') or []
            full = (data.get('objects_full') or {}).get('objects') or []
            extra_stats = (data.get('detailed') or {}).get('extra') or {}
            hourly = data.get('latency_hourly') or []
            fr = sum(b.get('frame_rejections', 0) for b in hourly if isinstance(b, dict) and isinstance(b.get('frame_rejections'), (int, float)))
            rec.update({'objects': wm.get('objects'), 'confirmed': wm.get('confirmed'), 'upserts_total': wm.get('upserts_total'),
                        'ltm_never_upserted': wm.get('ltm_never_upserted'), 'has_forward_clearance_key': 'forward_clearance' in wm,
                        'frame_count': lat.get('frame_count'),
                        't_total_mean': (lat.get('t_total') or {}).get('mean') if isinstance(lat.get('t_total'), dict) else lat.get('t_total'),
                        'page_sha': ms_sha(multiset(page)), 'page_count': len(page), 'full_sha': ms_sha(multiset(full)), 'full_count': len(full),
                        'pose_conversion_failures': extra_stats.get('pose_conversion_failures'),
                        'frame_rejections_hourly_sum': fr, 'hourly_buckets': len(hourly)})
            rec['anchor'] = ANCHORS.get(rec['page_sha'], 'NEW')
    else:
        rec['error'] = 'no datasheet json'
    # harness restores the yaml; be defensive
    for p in glob.glob('rtsm/cfg/rtsm.yaml.datasheet_bak.*'):
        os.remove(p)
    dirty = sh('git', 'status', '--porcelain', '--untracked-files=no').stdout.strip()
    rec['tree_dirty_after'] = dirty
    if dirty:
        sh('git', 'checkout', '--', 'rtsm/cfg/rtsm.yaml', check=False)
    log(rec)
    return rec


def main():
    log({'name': '_start', 'head': sh('git', 'rev-parse', '--short', 'HEAD').stdout.strip(), 'branch': sh('git', 'branch', '--show-current').stdout.strip(), 'time': time.strftime('%Y-%m-%dT%H:%M:%S')})
    moved = False
    if os.path.isdir(FAISS) and not os.path.isdir(ASIDE):
        shutil.move(FAISS, ASIDE); moved = True
    try:
        for i in (1, 2, 3):
            for name, backend, extra in JOBS:
                try:
                    run(name, backend, extra, i)
                except Exception as e:
                    log({'name': name, 'run': i, 'error': f'driver: {e}'})
    finally:
        kill_stale()
        shutil.rmtree(FAISS, ignore_errors=True)
        if moved and os.path.isdir(ASIDE):
            shutil.move(ASIDE, FAISS)
        log({'name': '_end', 'time': time.strftime('%Y-%m-%dT%H:%M:%S')})
        open(DONE, 'w').write('done\n')


if __name__ == '__main__':
    main()
