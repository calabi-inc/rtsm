# Configuration tuning and the next product milestones

## Problem and scope

Threshold tuning is difficult because the configuration exposes many controls
without making their effects clear. The current pipeline also mixes
per-frame configuration reads with values cached by WorkingMemory, IngestGate,
and model constructors. Applying a dictionary patch live is not an atomic
pipeline update.

The first implementation is a reproducible, explained tuning workflow:

1. Preserve the existing main and demo defaults and their numerical behavior.
2. Accept small YAML override profiles and explicit command-line overrides in
   both runners. Show a resolved YAML snapshot that can be reused as a base.
3. Provide a GPU-independent configuration command with a short, symptom-based
   guide to active controls, validation of those controls, and warnings about
   ineffective or conflicting settings. Include a configuration fingerprint.
4. Test profile precedence, validation, inactive settings, snapshot round trips,
   and dispatch without GPU dependencies. Do not change model dependencies.

Architectural fit: this remains a plain dictionary at the component boundary,
requires only the existing PyYAML core dependency, and leaves model adapters and
pose math unchanged. Profiles are configuration files, not new model backends.
No new numerical preset will be advertised as calibrated without replay data.

## Implementation status

Implemented on `feature/config-tuning`, based on demo2 commit
`f7a0880ffe926bfddb0e95b8fe43b22931e62fce`:

- Shared profile and command-line override loading for the main and demo runners.
- GPU-independent `rtsm config explain/show/validate` commands, with a short
  symptom guide and advisories about ineffective thresholds.
- Resolved YAML snapshots, deterministic settings fingerprints, typo detection,
  and validation of the documented tuning controls.
- 36 focused tests passed, including unchanged packaged values, profile
  precedence, snapshot round trips, and dispatch with runtime/model imports blocked.
- Modified Python modules compiled successfully; Git whitespace checks passed.

The existing `TestConfigLoader` tests in `tests/test_demo_components.py` could
not be collected in the separate test environment because that module imports
OpenCV. GPU/model execution and a full replay were not run. Focused tests cover
the configuration behavior; model-output equivalence still needs replay validation.
No model dependency declarations or CUDA/PyTorch installation were changed.

## Follow-up: safe runtime changes

After the useful tuning controls have been tested on representative recordings,
introduce validated configuration revisions applied at a pipeline frame boundary.
Each component must explicitly implement the supported update contract, or mark
the setting as requiring restart. Record each revision with evaluation events;
reject an entire invalid revision and retain the previous one.

## Follow-up: cross-session memory

First milestone: recover complete object records after process restart in an
explicitly identified, unchanged map frame. Store authoritative metadata and
embeddings transactionally; rebuild derived indexes and test interrupted writes.
Cross-session means more than restarting FAISS. A new SLAM session can establish
a different origin: spatial queries and object merging need verified alignment,
or separate map namespaces. Do not silently combine coordinates from unrelated
sessions. Test restart recovery, index rebuild, stale observations, and frame
mismatch before claiming persistent spatial memory.

## Follow-up: ONNX export/runtime

Choose the target hardware and measure stage latency first. Select one adapter
for an ONNX proof of concept, preserve its preprocessing/output contract, and
compare its detections or embeddings against the current implementation. Measure
warm and cold latency, peak memory, and end-to-end object retrieval quality on
the same recordings. Provider fallback and unsupported operations must be
observable. Export success alone is not a deployment acceptance criterion.

## Acceptance limits

This configuration change does not prove better object accuracy, provide live
threshold updates, implement cross-session recovery, or add ONNX inference.
Those are separate features with separate validation gates.
