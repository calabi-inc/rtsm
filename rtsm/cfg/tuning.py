"""A small, code-traced tuning surface; no model imports or calibrated presets."""

from __future__ import annotations

from dataclasses import dataclass
import math

from . import ConfigError, config_fingerprint


@dataclass(frozen=True)
class Control:
    path: str
    default: float | int | None
    symptoms: tuple[str, ...]
    explanation: str
    minimum: float = 0
    maximum: float | None = None
    integer: bool = False
    backends: tuple[str, ...] = ()


# Defaults trace to segmentation/__init__.py, utils/mask_staging.py,
# core/pipeline.py, core/association.py, stores/working_memory.py and run.py.
# None denotes a required setting whose consumer has no fallback.
CONTROLS = (
    Control("segmentation.grounded_sam2.box_threshold", .25, ("missing", "pollution"),
            "Raise to reject weaker detections; lower may recover objects and add false detections.",
            maximum=1, backends=("grounded_sam2",)),
    Control("segmentation.grounded_sam2.text_threshold", .2, ("missing",),
            "Controls text-token matching. Check the prompted vocabulary before lowering this.",
            maximum=1, backends=("grounded_sam2",)),
    Control("segmentation.fastsam.conf", .4, ("missing", "pollution"),
            "Raise to reject weaker mask proposals; inspect missed objects on the same replay.",
            maximum=1, backends=("fastsam", "dual")),
    Control("segmentation.yoloe.conf", .25, ("missing", "pollution"),
            "Raise to reject weaker detections; lower trades precision for recall.",
            maximum=1, backends=("yoloe", "dual")),
    Control("segmentation.sam2.pred_iou_thresh", .7, ("missing", "pollution"),
            "Auto-mask quality cutoff; this is not used by the grounded_sam2 factory.",
            maximum=1, backends=("sam2",)),
    Control("filters.min_area_px", None, ("missing",),
            "Actual mask area cutoff in pixels. Lower for small objects; fragments may increase.",
            minimum=1, integer=True),
    Control("staging.depth_valid_min", .15, ("pollution",),
            "Rejects masks with too little valid depth after erosion. Higher can reject reflective objects.",
            maximum=1),
    Control("staging.centroid_min_valid", .25, ("pollution",),
            "Minimum valid-depth fraction before a 3D centroid is computed.",
            maximum=1),
    Control("filters.depth.sigma_max_m", .35, ("pollution",),
            "Despite the name, compares depth quantile spread in metres, not a calibrated standard deviation.",
            minimum=.000001),
    Control("staging.topk_preclip", 5, ("missing", "latency"),
            "Caps candidates encoded by CLIP after scoring/dedup. Lower saves that work, not segmentation work.",
            minimum=1, integer=True),
    Control("assoc.gate_dist_base_m", .20, ("duplicates",),
            "Cross-frame distance gate in metres. Wider may reduce splits and merge neighbouring objects.",
            minimum=.000001),
    Control("assoc.gate_reproj_px", 30., ("duplicates",),
            "Cross-frame reprojection gate in pixels. Inspect pose/depth errors before widening.",
            minimum=.000001),
    Control("assoc.cos_min", .95, ("duplicates",),
            "Cross-frame embedding similarity floor. Higher may separate objects and split identities.",
            maximum=1),
    Control("object.promote_hits", 2, ("missing", "pollution"),
            "Observations required for confirmation; repeated false detections can still pass.",
            minimum=1, integer=True),
    Control("object.stability_promote", .50, ("missing", "pollution"),
            "Confirmation score cutoff. This score is not a probability of correctness.",
            maximum=1),
    Control("object.require_view_bins", 2, ("search",),
            "View diversity required for confirmation. A stationary camera may only produce one bin.",
            minimum=1, integer=True),
    Control("ltm.ltm_min_view_bins", 2, ("search",),
            "View diversity required for vector upsert; defaults to object.require_view_bins.",
            minimum=1, integer=True),
    Control("io.websocket.nonkf_min_interval_s", .5, ("latency",),
            "Receiver throttle for non-keyframes. Higher reduces observations; keyframes follow a separate path."),
)

SYMPTOMS = {
    "missing": "Missing objects: check model vocabulary and filter drop reasons first.",
    "duplicates": "Duplicate identities: distinguish same-frame mask duplicates from cross-frame splits.",
    "pollution": "Wrong objects or positions: inspect masks, depth and pose before loosening thresholds.",
    "latency": "Slow processing: measure stage latency before sacrificing recall.",
    "search": "Confirmed but absent from search: inspect vector upsert eligibility.",
}


def _get(cfg: dict, path: str, default=None):
    node = cfg
    for part in path.split("."):
        if not isinstance(node, dict):
            raise ConfigError(f"{path}: parent section must be a mapping")
        if part not in node:
            return default
        node = node[part]
    return node


def _value(cfg: dict, control: Control):
    default = control.default
    if control.path == "ltm.ltm_min_view_bins":
        default = _get(cfg, "object.require_view_bins", 2)
    return _get(cfg, control.path, default)


def active_controls(cfg: dict):
    backend = _get(cfg, "segmentation.backend", "fastsam")
    for control in CONTROLS:
        if control.backends and backend not in control.backends:
            continue
        if control.path == "assoc.cos_min" and not _get(cfg, "assoc.use_embeddings", True):
            continue
        if control.path.startswith("io.websocket.") and _get(cfg, "io.receiver", "zeromq") != "websocket":
            continue
        yield control


def validate_tuning(cfg: dict) -> list[str]:
    """Check the documented controls and return advisories, not accuracy claims.

    This is deliberately not a complete schema for every expert setting.
    """
    backend = _get(cfg, "segmentation.backend", "fastsam")
    if backend not in ("grounded_sam2", "sam2", "fastsam", "yoloe", "dual"):
        raise ConfigError(f"Unknown segmentation.backend: {backend!r}")
    for control in active_controls(cfg):
        value = _value(cfg, control)
        valid = type(value) in (int, float) and math.isfinite(value)
        if control.integer:
            valid = valid and type(value) is int
        if not valid or value < control.minimum or (
            control.maximum is not None and value > control.maximum
        ):
            kind = "integer" if control.integer else "number"
            upper = f" and <= {control.maximum}" if control.maximum is not None else ""
            raise ConfigError(
                f"{control.path}: expected a finite {kind} >= {control.minimum}{upper}; got {value!r}"
            )

    warnings = []
    ineffective = (
        ("gates", "The gates section is not consumed by the current pipeline."),
        ("masks", "The masks section is not consumed by the current pipeline."),
        ("staging.min_area_px", "staging.min_area_px has no effect; use filters.min_area_px."),
        ("filters.depth.valid_min_pct", "filters.depth.valid_min_pct has no effect; use staging.depth_valid_min."),
    )
    for path, message in ineffective:
        if _get(cfg, path) is not None:
            warnings.append(message)
    if _get(cfg, "staging.centroid_min_valid", .25) > _get(cfg, "staging.depth_valid_min", .15):
        warnings.append("Some masks can pass the depth filter but lack enough valid depth for a 3D centroid.")
    required_bins = _get(cfg, "object.require_view_bins", 2)
    if _get(cfg, "ltm.ltm_min_view_bins", required_bins) > required_bins:
        warnings.append("LTM requires more view bins than confirmation; some confirmed objects may be absent from semantic search.")
    if not _get(cfg, "vectors.enable", True):
        warnings.append("Vector storage is disabled; semantic search will be unavailable.")
    if backend == "grounded_sam2":
        for key in ("pred_iou_thresh", "stability_score_thresh", "points_per_side"):
            if _get(cfg, f"segmentation.sam2.{key}") is not None:
                warnings.append("SAM2 automatic-mask thresholds/grid settings do not configure grounded_sam2; only sam2.model_id supplies its fallback model.")
                break
    return warnings


def explain_tuning(cfg: dict, symptom: str | None = None) -> str:
    warnings = validate_tuning(cfg)
    lines = [
        f"Config SHA-256: {config_fingerprint(cfg)}",
        f"Segmentation backend: {_get(cfg, 'segmentation.backend', 'fastsam')}",
        "Restart after applying changes. This command does not update a running process.",
        "Values are existing settings or component defaults, not calibrated recommendations.",
    ]
    if _get(cfg, "segmentation.backend") == "grounded_sam2":
        vocab = _get(cfg, "segmentation.grounded_sam2.vocab")
        lines.append(f"Detection vocabulary: {vocab if vocab is not None else 'backend default'}")
    controls = list(active_controls(cfg))
    for name in ([symptom] if symptom else SYMPTOMS):
        lines.extend(("", SYMPTOMS[name]))
        for control in controls:
            if name in control.symptoms:
                source = " [component default]" if _get(cfg, control.path) is None else ""
                lines.append(f"  {control.path} = {_value(cfg, control)}{source}")
                lines.append(f"    {control.explanation}")
    if warnings:
        lines.extend(("", "Configuration advisories:"))
        lines.extend(f"  - {warning}" for warning in warnings)
    lines.extend(("", "Change one cause at a time; replay the same data and compare wrong identities,",
                  "missing objects, position errors, and stage latency. Object count alone is not quality."))
    return "\n".join(lines)
