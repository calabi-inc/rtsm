# Baseline floor — main-5528abf

```
commit: 5528abf130799b855a64a9b3d2d22c47d6cd18cb
branch: main
started: 2026-09-07T03:33:39-07:00
rtsm.yaml sha256: 038a37465bbac7160527d10b595b78ec5d0b6abe5fec0861cb7561d03fdcdbe4
python 3.12.10
torch 2.11.0+cu128 cuda True 12.8
NVIDIA GeForce RTX 5090, 616.64
finished: 2026-09-07T03:45:36-07:00
```

## dual

| run | objects | confirmed | frames | proc Hz | t_total mean | qdepth max | gate rej | frame rej | buckets | multiset |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 107 | 70 | 53 | 1.1 | 0.2743 | 0 | 26 | 0 | 39 | b71b98ca1fc2bf0d |
| 2 | 107 | 70 | 53 | 1.08 | 0.269 | 0 | 25 | 0 | 29 | b71b98ca1fc2bf0d |
| 3 | 107 | 70 | 53 | 1.08 | 0.2939 | 0 | 15 | 0 | 12 | b71b98ca1fc2bf0d |

**Floor:** objects 107–107 (spread 0), confirmed 70–70 (spread 0); multiset identical across runs: True (page-limited to the first 100 objects of /objects).

## grounded_sam2

| run | objects | confirmed | frames | proc Hz | t_total mean | qdepth max | gate rej | frame rej | buckets | multiset |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 139 | 81 | 53 | 0.88 | 1.0544 | 11 | 24 | 0 | 35 | eab7e618f54b2f13 |
| 2 | 133 | 81 | 53 | 0.88 | 1.0499 | 4 | 16 | 0 | 17 | 4a29c1ed37d2b052 |
| 3 | 133 | 81 | 53 | 0.87 | 1.0774 | 4 | 15 | 0 | 12 | 4a29c1ed37d2b052 |

**Floor:** objects 133–139 (spread 6), confirmed 81–81 (spread 0); multiset identical across runs: False (page-limited to the first 100 objects of /objects).

