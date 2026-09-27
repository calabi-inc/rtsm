#!/usr/bin/env bash
# P3 task 1 gate (G3-1, CPU only): the bag reader as an ingest source on the task-0.5 seam.
#   1. REPLAY PARITY: recordings/session1_bag through BagSource (replay settings: keyframe_every_n 30, non-KF 0.5 s on the
#      sensor clock, confidence threshold 2, tracking filter on) reproduces the B1 record's 240 receiver decisions
#      (decision, reason, frame_seq, t_sensor_ns, is_keyframe, frame_count) and the replay path's 86 packets BIT FOR BIT
#      (rgb, depth after the confidence filter, confidence, intrinsics, pose, stamps, seq, keyframe origin, rx_seq, depth_valid_frac)
#      -- tests/test_bag_source.py::test_session1_bag_reproduces_the_replay_path.
#   2. EXTERNAL BAGS READ END TO END through discovery alone (no overrides), as-deployed ingest settings, a draining consumer:
#      TUM fr1/desk (rgb8, 613 RGB / 595 depth): pairs >= 560, 0 frames without pose, the 4-hop chain from /world;
#      TUM fr3/long_office_household (bgr8, 2585 / 2509): pairs >= 2450, 0 without pose; both: pair dt <= 20 ms, poses composed
#      for every pair, keyframes minted every 30th admitted frame, throttle thinning the rest.
#   3. REFUSALS with the reason, before any image is read: r2b_cafe -> {no_pose_source, unaligned_depth}; with assume_aligned ->
#      {no_pose_source}; r2b_hope -> {no_depth_topic, no_camera_info}.
#   4. `rtsm --bag`: a refused bag exits through parser.error with the reasons ABOVE the GPU check; a readable bag reaches the
#      check -- tests/test_run_entrypoint.py::test_bag_is_probed_above_the_gpu_check.
#   5. Unit suites green: tf_buffer (analytic poses), ros codecs, bag reader (synthetic bags: every encoding, pairing, TF/odom,
#      relative names, zero stamps, bare MCAP, sqlite3, ros2idl, refusals), bag source, ingest front-end, golden traces, entrypoint.
set -u
cd /c/Users/konam/Desktop/calabi-repo/rtsm || exit 1
SP="C:/Users/konam/AppData/Local/Temp/claude/C--Users-konam-OneDrive-Desktop-calabi-repo-rtsm/949c0384-41c4-4c7e-9f67-2ca869f46d96/scratchpad"
OUT="$SP/p3t1_gate"; mkdir -p "$OUT"
echo "== 1 + 4 + 5. pytest =="
python -X utf8 -m pytest tests/test_tf_buffer.py tests/test_ros_codecs.py tests/test_bag_reader.py tests/test_bag_source.py \
  tests/test_ingest_frontend.py tests/test_ingest_golden.py tests/test_run_entrypoint.py tests/test_runner_analytics_wiring.py \
  tests/evaluation/test_recording_mcap.py -q -p no:cacheprovider > "$OUT/pytest.out" 2>&1; py_rc=$?
tail -2 "$OUT/pytest.out"; echo "pytest rc=$py_rc"
grep -c "PASSED\|passed" "$OUT/pytest.out" > /dev/null
python -X utf8 -m pytest tests/test_bag_source.py::test_session1_bag_reproduces_the_replay_path tests/test_run_entrypoint.py::test_bag_is_probed_above_the_gpu_check -q -p no:cacheprovider -rA 2>&1 | grep -E "PASSED|FAILED|SKIPPED|passed|failed|skipped" | tail -3

echo "== 2 + 3. external bags =="
python -X utf8 - "$py_rc" <<'PY'
import sys, threading, time
from collections import Counter
from rtsm.io.bag_reader import probe_bag
from rtsm.io.contracts import SourceContext
from rtsm.io.ingest_queue import IngestQueue
from rtsm.io.sources import make_source
py_rc = int(sys.argv[1])

def ingest(path, **ctx_kw):
    events, packets = [], Counter()
    q = IngestQueue(64)
    stop = threading.Event()
    def drain():
        while not stop.is_set() or q.qsize():
            p = q.get(timeout=0.05)
            if p is not None:
                packets["kf" if p.is_keyframe else "nonkf"] += 1
    t = threading.Thread(target=drain, daemon=True); t.start()
    ctx = SourceContext(ingest_queue=q, clock_mode="sensor", keyframe_every_n=30, nonkf_min_interval_s=0.5,
                        require_tracking_normal=True, confidence_threshold=2, event_sink=events.append, **ctx_kw)
    src = make_source("bag", {"io": {"bag": {}}}, ctx, path=path)
    t0 = time.perf_counter(); src.start(); src.wait(3600); stop.set(); t.join(10)
    st = src.stats(); st["seconds"] = round(time.perf_counter() - t0, 1)
    st["decisions"] = dict(Counter((e.decision, e.reason) for e in events)); st["packets"] = dict(packets)
    return st

res = {}
for label, path, min_pairs in (("fr1", "recordings/external/rgbd_dataset_freiburg1_desk.bag", 560),
                               ("fr3", "recordings/external/rgbd_dataset_freiburg3_long_office_household.bag", 2450)):
    st = ingest(path)
    kf = st["packets"].get("kf", 0); nonkf = st["packets"].get("nonkf", 0)
    ok = (st["error"] is None and st["paired"] >= min_pairs and st["pose_missing"] == 0 and st["pair_dt_ms_max"] <= 20.0
          and st["tf_chain"] == ["world -> kinect", "kinect -> openni_camera", "openni_camera -> openni_rgb_frame", "openni_rgb_frame -> openni_rgb_optical_frame"]
          and st["yielded"] == st["paired"] and st["enqueued"] == kf + nonkf and kf >= 1)
    res[label] = ok
    print(f"2. {label}: {'OK' if ok else 'FAIL'} in {st['seconds']} s | rgb seen {st['frames_seen']} paired {st['paired']} unpaired rgb {st['unpaired_rgb']} depth {st['unpaired_depth']} "
          f"pose_missing {st['pose_missing']} zero_stamp {st['skipped_zero_stamp']} pair dt max {st['pair_dt_ms_max']:.2f} ms | chain {st['tf_chain']} | "
          f"registration: {st['registration'][:70]} | decisions {st['decisions']} | packets kf {kf} nonkf {nonkf} | tracking filter off: {not st['topics'].get('tracking')} | {st['bag']}")

cafe = probe_bag("recordings/external/r2b_cafe"); cafe_a = probe_bag("recordings/external/r2b_cafe", assume_aligned=True); hope = probe_bag("recordings/external/r2b_hope")
codes = lambda st: sorted(c for c, _ in (st.refusal or []))
ok3 = codes(cafe) == ["no_pose_source", "unaligned_depth"] and codes(cafe_a) == ["no_pose_source"] and codes(hope) == ["no_camera_info", "no_depth_topic"]
print(f"3. r2b_cafe -> {codes(cafe)} | assume_aligned -> {codes(cafe_a)} | r2b_hope -> {codes(hope)} -> {ok3}")
for c, d in cafe.refusal: print(f"   cafe {c}: {d[:150]}")
ok15 = py_rc == 0
print(f"1/4/5. pytest rc {py_rc} -> {ok15}")
verdict = {"p1_parity+p4_startup+p5_units": ok15, "p2_fr1": res["fr1"], "p2_fr3": res["fr3"], "p3_refusals": ok3}
print(f"HARD GATE: {'PASS' if all(verdict.values()) else 'FAIL'} | {verdict}")
PY
echo P3T1_GATE_DONE
