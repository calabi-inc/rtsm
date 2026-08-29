# AGENTS.md — Project Instructions for Codex

## Project Context

RTSM (Real-Time Spatio-Semantic Memory) is a persistent, queryable spatial memory layer for robotics and embodied AI. It processes RGB-D + pose frames, segments objects, embeds with CLIP, associates across frames, and maintains a working memory of 3D object positions queryable via MCP and REST.

- **License:** Apache 2.0. Default `[gpu]` deps are permissive (SAM2, GDINO, open-clip). Ultralytics (AGPL) is opt-in via `[gpu-ultralytics]`.
- **Patent:** 67-page PPA filed Nov 2025. Core novel couplings: compute-aware soft selection, sweep policy, association cascade, view-binned embeddings, ProximityIndex with WM-aware eviction.
- **Solo developer project** — every change matters, be careful.

## Environment

- **OS:** Windows 11, developing directly (not WSL)
- **Shell:** Git Bash (use Unix syntax, forward slashes)
- **GPU:** NVIDIA RTX 5090, CUDA 13.0 nightly (cu130)
- **Python:** 3.12+
- **Package manager:** pip (NOT uv for published packages)
- **Git remote:** HTTPS (not SSH)
- **Avoid git worktrees** — cause IDE conflicts on this setup
- **WSL editable install quirk:** `pip install -e ".[gpu]"` on `/mnt/c/` NTFS mount may build as `UNKNOWN 0.0.0`. Fix: use `--no-build-isolation` flag, or install deps manually after `pip install -e .`

## Critical Rules

### Dependencies
- **NEVER replace CUDA torch with CPU-only torch.** After any dependency change, verify: `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`
- **NEVER add `[tool.uv]` sections to pyproject.toml** — those are dev-only and break standard pip installs. The CUDA 13.0 nightly index stays in a separate `requirements-dev-5090.txt`.
- Default `[gpu]` extras must NOT pull `ultralytics`. AGPL deps only in `[gpu-ultralytics]`.

### Before Implementing
- **Validate architectural fit BEFORE writing code.** Check: does the library/model support zero-shot? Does it exist on PyPI? What's the latency? Will it break existing pipeline? Don't repeat the MobileSAM mistake.
- **Create/update the plan BEFORE starting implementation.** Do not jump to code changes without confirming the approach.
- **Changes to image orientation or pose math must be tested against 3D map building before committing.** These have broken the pipeline multiple times.

### Code Patterns
- All segmentation backends implement `SegmentationAdapter` ABC (`rtsm/models/segmentation/base.py`)
- Factory pattern in `rtsm/models/segmentation/__init__.py` — `get_segmenter(cfg)` maps config `backend:` string to concrete adapter
- Pipeline calls `segmenter.segment(image)` and gets `SegmentationResult` — never call models directly
- Config drives everything — thresholds, backends, gates are all in `config/rtsm.yaml`
- CLIP embedding computed for ALL objects regardless of backend (drives semantic retrieval)

### Git Workflow
- One feature per branch, one PR per feature
- Branch naming: `feature/X`, `fix/X`, `docs/X`, `chore/X`
- Verify auth with `gh auth status` before git operations
- After merging: `git checkout main && git pull && git remote prune origin` then delete local branch
- Do NOT amend published commits

## Key Files

| File | Purpose |
|------|---------|
| `rtsm/core/pipeline.py` | Main processing loop, scoring, top-K selection |
| `rtsm/core/association.py` | Cascaded association (dist → reproj → cosine) |
| `rtsm/stores/working_memory.py` | Object lifecycle: create → update → promote → upsert |
| `rtsm/stores/proximity_index.py` | Spatial index with WM-aware eviction |
| `rtsm/stores/sweep_policy.py` | TTL + parallax + look-cell novelty gating |
| `rtsm/models/segmentation/` | All segmenter backends (SAM2, GDINO, FastSAM, YOLOE, dual) |
| `rtsm/io/websocket.py` | iPhone ARKit WebSocket receiver |
| `rtsm/io/mcp_embedded.py` | MCP server (SSE transport, 6 tools) |
| `rtsm/api/server.py` | FastAPI REST API + Prometheus metrics |
| `config/rtsm.yaml` | All pipeline configuration |

## Current State (updated 2026-04-08)

- **Pipeline:** Working, dual mode 210ms / grounded_sam2 510ms on RTX 5090
- **Backends:** 5 registered (grounded_sam2, sam2, fastsam, yoloe, dual)
- **MCP:** 6 tools via SSE, verified
- **Analytics:** Real-time dashboard shipped
- **Next:** Gate 1 #11 — `rtsm demo` entry point, then verify on non-5090, then PyPI publish
- **Master plan:** `.Codex/plans/lazy-conjuring-snowflake.md`
