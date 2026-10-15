"""Validated model weight files, by content.

The AGPL backends' weights (FastSAM, YOLOE) are fetched by file *name* through ultralytics, and the file behind a
name changes upstream in place: the ``v8.4.0`` release of github.com/ultralytics/assets replaced
``yoloe-26s-seg-pf.pt`` on 2026-09-17 (40.0 MB, checkpoint 8.3.222, sha256 ``01605da3…``). On TUM fr3 that file
labelled an office ``heat``, ``razor blade``, ``array``; the file the dev box validated in March (35.2 MB, checkpoint
8.3.252, sha256 ``9d0390d8…``) labels the same scene ``bottle``, ``cup``, ``keyboard`` and is no longer served under
that name. The config fingerprint covers settings, not files, so two runs with one fingerprint and one commit ran
different models before anyone noticed (rtsm-lab record 2026-10-dual-vocab-labels).

``VALIDATED`` names the files every documented number was produced with. ``check_weight_file`` compares a file with
it; ``enforce`` applies ``segmentation.weights_policy`` (``warn``, the default; ``strict``; ``off``). The
segmentation factory calls both before a backend loads, the eval runner records the status in ``resolved.json``,
and ``scripts/fetch_models.py`` verifies after a download and on ``--verify``.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

VALIDATED: Dict[str, Dict[str, Any]] = {
    "FastSAM-x.pt": {
        "sha256": "c0be4e7ddbe4c15333d15a859c676d053c486d0a746a3be6a7a9790d52a9b6d7",
        "bytes": 144943063,
        "note": "dev box since 2025-08; every published FastSAM / dual number",
    },
    "yoloe-26s-seg.pt": {
        "sha256": "6f62bc7ed9f86056112c383e9b85023291a3929086af26b1a8762335fe39a17d",
        "bytes": 30759531,
        "note": "dev box since 2026-03 (prompted variant)",
    },
    "yoloe-26s-seg-pf.pt": {
        "sha256": "9d0390d82c41156639e6d9a04be444ace01049d524fcc695315e1539cf21519e",
        "bytes": 35163686,
        "note": "dev box since 2026-03, checkpoint 8.3.252; upstream replaced the asset of this name on 2026-09-17",
    },
}

POLICIES = ("warn", "strict", "off")
_READ_CHUNK = 1 << 20


class WeightsError(RuntimeError):
    """Raised under ``weights_policy: strict`` for a missing or non-validated weight file."""


@dataclass
class WeightStatus:
    path: str
    name: str
    exists: bool
    state: str                                  # validated | mismatch | unknown | missing
    sha256: Optional[str] = None
    bytes: Optional[int] = None
    expected_sha256: Optional[str] = None
    expected_bytes: Optional[int] = None

    @property
    def ok(self) -> bool:
        return self.state in ("validated", "unknown")

    def line(self) -> str:
        """One human-readable line for logs and the fetch script."""
        if self.state == "missing":
            return f"{self.name}: missing at {self.path}"
        if self.state == "unknown":
            return f"{self.name}: {self.bytes} bytes, sha256 {self.sha256[:16]} (not a file rtsm validates)"
        if self.state == "validated":
            return f"{self.name}: validated (sha256 {self.sha256[:16]}, {self.bytes} bytes)"
        return (f"{self.name}: NOT the validated file: sha256 {self.sha256[:16]} ({self.bytes} bytes) vs validated "
                f"{self.expected_sha256[:16]} ({self.expected_bytes} bytes)")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_READ_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def check_weight_file(path: str | Path, root: Optional[Path] = None) -> WeightStatus:
    """Compare a weight file with ``VALIDATED`` by its file name. A relative path resolves against ``root`` (default:
    the working directory, as the backends resolve it)."""
    p = Path(path)
    if not p.is_absolute():
        p = (root or Path.cwd()) / p
    name = p.name
    expected = VALIDATED.get(name)
    if not p.is_file():
        return WeightStatus(path=str(path), name=name, exists=False, state="missing",
                            expected_sha256=(expected or {}).get("sha256"), expected_bytes=(expected or {}).get("bytes"))
    digest = sha256_file(p)
    size = p.stat().st_size
    if expected is None:
        return WeightStatus(path=str(path), name=name, exists=True, state="unknown", sha256=digest, bytes=size)
    state = "validated" if digest == expected["sha256"] else "mismatch"
    return WeightStatus(path=str(path), name=name, exists=True, state=state, sha256=digest, bytes=size,
                        expected_sha256=expected["sha256"], expected_bytes=expected["bytes"])


def validation_state(path: str | Path, sha256: str) -> Optional[str]:
    """``validated`` | ``mismatch`` for a file name in ``VALIDATED``, ``None`` for any other name (used by the eval
    runner's provenance, which already has the digest)."""
    expected = VALIDATED.get(Path(path).name)
    if expected is None:
        return None
    return "validated" if sha256 == expected["sha256"] else "mismatch"


def policy_from(seg_cfg: Dict[str, Any]) -> str:
    """YAML 1.1 (PyYAML) reads a bare ``off`` as the boolean False, so ``weights_policy: off`` in a file or
    ``--set segmentation.weights_policy=off`` arrives here as False: that is the documented value, not an error."""
    raw = (seg_cfg or {}).get("weights_policy", "warn")
    if raw is False or raw is None:
        return "off"
    if raw is True:
        raise ValueError("segmentation.weights_policy must be one of ('warn', 'strict', 'off'); YAML read the value as the "
                         "boolean True (write it as a word, or quote it)")
    policy = str(raw).lower()
    if policy not in POLICIES:
        raise ValueError(f"segmentation.weights_policy must be one of {POLICIES}, got {policy!r}")
    return policy


def enforce(status: WeightStatus, policy: str, logger: Optional[logging.Logger] = None) -> WeightStatus:
    """Apply the policy to a status: ``off`` does nothing; ``warn`` logs a WARNING for a missing or mismatching file
    (an unknown name is logged at INFO, since custom weights are legitimate); ``strict`` raises for both."""
    log = logger or logging.getLogger(__name__)
    if policy == "off":
        return status
    if status.state == "validated":
        log.info("weights: " + status.line())
        return status
    if status.state == "unknown":
        log.info("weights: " + status.line())
        return status
    if status.state == "missing":
        msg = (status.line() + "; the backend's library will download whatever it serves today under that name, "
               "unverified. Run scripts/fetch_models.py, then scripts/fetch_models.py --verify.")
    else:
        msg = (status.line() + ". Detections and labels will differ from the documented runs (the file ultralytics "
               "serves under this name changed upstream on 2026-09-17). Replace it with the validated file, or set "
               "segmentation.weights_policy: off if the difference is intended.")
    if policy == "strict":
        raise WeightsError(msg)
    log.warning("weights: " + msg)
    return status


def check_and_enforce(path: str | Path, seg_cfg: Dict[str, Any], logger: Optional[logging.Logger] = None) -> WeightStatus:
    """The factory's one-liner: status of ``path`` under the config's policy."""
    return enforce(check_weight_file(path), policy_from(seg_cfg), logger)
