# E1 Related Work — prior-art sweep (2026-08-12)

Four-angle literature search (sim ObjectNav map ablations; real-robot
LLM/VLM memory systems; 2024–26 workshop papers; direct phrasings of our
claim), synthesized 2026-08-12. Verdict: **the exact claim is NOT
pre-empted**, but the space is populated — framing rules below.

## Framing rules (bind the prose)

1. **NEVER claim "first real-hardware memory ablation."** GOAT (RSS
   2024) owns that: Spot across 9 homes, 675 goals, "GOAT w/o Memory"
   resets the semantic map per goal — 83% SR with memory vs 60.3%
   without. We cite it as the anchor and differentiate on metrics and
   cost accounting, not on priority.
2. **Our defensible combination** (no prior work has all of it):
   pre-registered protocol · identical-agent persistence ablation on
   physical hardware · wall-clock TTA distributions with censoring at a
   fixed budget · tape-measured ground-truth success · explicit
   one-time-scan amortization ("at what cost") · commodity hardware
   (RC car + phone, ~$200).
3. **The "at what cost" framing is confirmed open**: even OK-Robot
   (iPhone-scan → robot, 10 real homes) treats the scan as free sunk
   infrastructure; Memory Over Maps (2026) prices index CONSTRUCTION
   but never computes break-even over navigation queries.

## Gap statement (drop-in for §2)

> Prior work has established THAT persistent spatial memory helps
> object-goal navigation — most directly GOAT's on-robot memory-reset
> ablation and its sim counterparts (GOAT-Bench, 3D-Mem, OVAL) — while
> the pre-built-map line (OK-Robot, VLMaps, HOV-SG) treats the scan
> that creates the memory as free sunk infrastructure. No existing
> study combines a pre-registered protocol, an identical-agent
> persistence ablation on physical hardware, wall-clock time-to-arrival
> distributions with censoring at a fixed budget, tape-measured
> success, and explicit amortization of the one-time scan cost. We
> therefore ask not whether memory helps but how much and at what cost,
> on commodity hardware where the answer bears directly on
> deployability.

## Ranked closest prior work

| # | Paper | Venue/Year | What they did | Close | Our differentiator |
|---|-------|-----------|---------------|-------|--------------------|
| 1 | GOAT: GO to Any Thing | RSS 2024 (2311.06430) | Real-Spot lifelong nav, memory-reset ablation: 83% vs 60.3% SR; per-goal SR/SPL curve rises with experience | 8 | Step-budgeted SR/SPL only — no wall-clock TTA, censoring, tape GT, scan-cost amortization, pre-registration; $75k Spot vs our $200 rig |
| 2 | Memory Over Maps | arXiv 2603.20530 (2026) | iPad pre-scan keyframe memory; memory vs frontier (VLFM) in sim (76.7% vs 52.5% SR); prices index build | 7 | Ablation sim-only; real run is a demo; costs = build pipeline, not amortized break-even; no TTA/censoring |
| 3 | GOAT-Bench | CVPR 2024 (2404.06609) | Sim benchmark quantifying SR/SPL drop without cross-task memory | 6 | Sim-only, step metrics, no hardware/timing/cost |
| 4 | OVAL | arXiv 2604.12872 (2026) | Lifelong ObjectNav, object memory + frontier; memory dominates SR/SPL gains in sim | 6 | Sim-only analogue of our physical curve |
| 5 | 3D-Mem | CVPR 2025 (2411.17735) | Explicit "w/o memory" arm cleared per subtask, sim exploration/QA | 5 | Sim-only; not nav-time |
| 6 | OneMap | ICRA 2025 (2409.11764) | Reusable open-vocab map; SPL rises across sequential goals; Spot demo | 5 | No map-reset twin; no timing/cost |
| 7 | DynaMem | ICRA 2025 (2411.04999) | Dynamic vs static spatio-semantic memory on real Stretch (70% vs 30% pick-drop) | 5 | Both arms HAVE memory; manipulation, not nav-time |
| 8 | RoboMemory | arXiv 2508.01415 (2025) | Multi-memory LLM agent; spatial-memory ablation (67%→47% SR) in sim | 5 | Sim ablation; physical part uncontrolled |
| 9 | SSMG-Nav | arXiv 2603.01813 (2026) | Persistent skeleton memory graph, lifelong ObjectNav sim ablation | 5 | Memory-variant comparison, not memory-vs-none |
| 10 | OK-Robot | arXiv 2401.12202 (2024) | iPhone scan → CLIP voxel memory → real nav+pick, 10 homes | 5 | Memory never ablated; scan never priced — our exact gap |
| 11 | IGV-RRT | arXiv 2603.21887 (2026) | Stale-prior + real-time VLM fusion for search under rearrangement | 4 | Fusion planner, not quantification; cite for freshness-gate precedent |
| 12 | Is Mapping Necessary? | CVPR 2022 (2206.00997) | Map-free agents hit 94% SR on realistic PointNav — "maps unnecessary" | 4 | THE adversarial citation: their null is what our physical ObjectNav study tests |

## Must-cite list (reviewer-proofing)

- GOAT (RSS 2024) — anchor; forbids any "first" claim
- GOAT-Bench (CVPR 2024) — sim no-memory ablation
- Memory Over Maps (2026) — freshest threat; pre-scan + cost numbers
- OK-Robot (2024) — iPhone-scan plumbing; expect "why not compared?"
- OneMap (ICRA 2025) — sequential map-reuse gains
- Is Mapping Necessary? (CVPR 2022) — the adversarial null
- Navigating to Objects in the Real World (Sci. Robotics 2023, 2212.00922) — canonical real-hardware ObjectNav methodology
- SemExp (NeurIPS 2020) — canonical semantic-map ObjectNav
- 3D-Mem (CVPR 2025) — sim w/o-memory arm
- ReMEmbR (ICRA 2025, 2409.13682) — NVIDIA queryable memory on real robot
- DynaMem (ICRA 2025) — dynamic vs static memory, real hardware
- IGV-RRT (2026) — freshness/stale-prior precedent
- Survey of Spatial Memory Representations (2604.16482, 2026) — scaffolding; taxonomy contains no cost-accounted physical ablation
- VLMaps (ICRA 2023) + HOV-SG (RSS 2024) — pre-built-map camp, scan cost uncounted

*Note: OVAL and IGV-RRT were scored partly from abstracts — pull full
texts before camera-ready.*
