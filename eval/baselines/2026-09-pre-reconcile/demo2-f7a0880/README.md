# Baseline floor — demo2-f7a0880

```
commit: f7a0880ffe926bfddb0e95b8fe43b22931e62fce
branch: feature/demo2-rc-car-agent (expected feature/demo2-rc-car-agent)
started: 2026-09-07T03:46:26-07:00
rtsm.yaml sha256: 59f3851da262fbf90629ee20a9aa717863418c67bc99a7993d4b8d39a605532a
python 3.12.10
torch 2.11.0+cu128 cuda True 12.8
NVIDIA GeForce RTX 5090, 616.64
finished: 2026-09-07T03:57:33-07:00
```

## dual

| run | objects | confirmed | frames | proc Hz | t_total mean | qdepth max | gate rej | frame rej | buckets | multiset |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 115 | 66 | 53 | 1.09 | 0.3184 | 0 | 15 | None | 12 | 1994e0fe5dd6167c |
| 2 | 116 | 68 | 54 | 1.12 | 0.2921 | 0 | 14 | None | 12 | 72fd5c7f0475da90 |
| 3 | 115 | 66 | 53 | 1.1 | 0.2873 | 0 | 10 | None | 6 | 1994e0fe5dd6167c |

**Floor:** objects 115–116 (spread 1), confirmed 66–68 (spread 2); multiset identical across runs: False (page-limited to the first 100 objects of /objects).

## grounded_sam2

| run | objects | confirmed | frames | proc Hz | t_total mean | qdepth max | gate rej | frame rej | buckets | multiset |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 25 | 18 | 53 | 1.08 | 0.3766 | 0 | 15 | None | 12 | f12c8dc2024f0c53 |
| 2 | 25 | 18 | 54 | 1.09 | 0.3774 | 0 | 7 | None | 6 | a556dbdcad9fef0a |
| 3 | 25 | 18 | 54 | 1.1 | 0.3574 | 0 | 23 | None | 34 | a556dbdcad9fef0a |

**Floor:** objects 25–25 (spread 0), confirmed 18–18 (spread 0); multiset identical across runs: False (page-limited to the first 100 objects of /objects).

