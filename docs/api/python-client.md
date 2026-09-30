# Python Client

`rtsm.client` is a thin client for the [REST API](rest-api.md): it needs `requests` only and imports without the perception stack, so a coordinator, a planner or a robot's agent process can use it on a machine that never loads a model.

```python
from rtsm.client import RtsmClient

rtsm = RtsmClient("http://localhost:8002", timeout_s=3.0)

rtsm.healthz()                          # True when the server answers /healthz (never raises)
rtsm.stats()                            # the /stats document: objects, confirmed, robot_pose, forward_clearance, ...
rtsm.object_count()

pose = rtsm.get_robot_pose()            # PoseSample or None before the first frame
pose, clearance = rtsm.get_pose_and_clearance()   # one /stats round-trip for a 10 Hz control loop

result = rtsm.label_query("tissue box", top_k=5)          # detector-label match over the memory
result = rtsm.semantic_query("something to drink", top_k=5)   # CLIP text query
for hit in result.results:
    print(hit.id, hit.score, hit.confirmed, hit.stability, hit.xyz_world, hit.last_seen_wall_utc)

rtsm.get_object_label(result.results[0].id)          # the object's primary label, or None
rtsm.get_object_snapshot_b64(result.results[0].id)   # base64 JPEG of its latest crop, or None
```

`PoseSample` carries `xyz`, `quaternion_xyzw`, the sender's `timestamp`, `fetched_at_mono` (the client's own monotonic clock at the response, so staleness is measured on the caller's clock) and `frame_epoch` (the receiver's world-frame discontinuity counter: poses across a bump must not be compared). `SemanticResult` carries the `query`, the `robot_pose` at query time and the `results` list of `SemanticHit`.

| method | endpoint | on failure |
|---|---|---|
| `healthz()` | `GET /healthz` | returns `False` |
| `stats()`, `object_count()`, `get_robot_pose()` | `GET /stats` | raises `requests.RequestException` |
| `get_forward_clearance()` | `GET /stats` (`forward_clearance`, opt-in via `io.clearance.enable`) | returns `None` |
| `get_pose_and_clearance()` | `GET /stats` | raises |
| `get_object_label(id)` | `GET /objects/{id}` | returns `None` |
| `get_object_snapshot_b64(id)` | `GET /objects/{id}/snapshots/0/image` | returns `None` |
| `label_query(text, top_k)` | `GET /search/label` | raises |
| `semantic_query(text, top_k)` | `GET /search/semantic` | raises |

The calls a control loop polls (health, clearance, labels, snapshots) fail soft; the calls whose absence is a real error (stats, pose, search) raise, so the caller decides. The RC-car reference agent ([rtsm-rc-car-agent](https://github.com/Vipipi/rtsm-rc-car-agent)) drives entirely through this interface, with a vendored copy of the module.
