"""Bundled ROS message definitions the bag reader registers when a bag does
not embed them (P3, the detections adapter).

``vision_msgs/`` holds the upstream ``.msg`` texts of ros-perception/vision_msgs
(``ros2/``: branch ``ros2``, the Humble+ layout; ``ros1/``: branch
``noetic-devel``), Apache License 2.0 -- see ``vision_msgs/NOTICE.md``. rosbags'
stock typestores contain no vision_msgs type at all (verified 2026-10-02), so a
Detection2DArray topic in a sqlite3 rosbag2 or a ROS 1 bag is unreadable
without these. An MCAP that embeds its schemas wins: the reader registers ours
only for the names the typestore still lacks.
"""
from __future__ import annotations

from importlib import resources
from typing import Any, Dict, List

DETECTION_2D = "vision_msgs/msg/Detection2DArray"
DETECTION_3D = "vision_msgs/msg/Detection3DArray"
DETECTION_MSGTYPES = (DETECTION_2D, DETECTION_3D)


def vision_msgs_types(flavour: str) -> Dict[str, Any]:
    """The rosbags type definitions of the bundled vision_msgs files.
    ``flavour``: ``ros2`` (names ``vision_msgs/msg/X``) or ``ros1`` (names
    ``vision_msgs/X``; rosbags normalises both to ``vision_msgs/msg/X``)."""
    from rosbags.typesys import get_types_from_msg
    if flavour not in ("ros2", "ros1"):
        raise ValueError(f"vision_msgs flavour must be ros2 or ros1, got {flavour!r}")
    prefix = "vision_msgs/msg/" if flavour == "ros2" else "vision_msgs/"
    folder = resources.files(__package__) / "vision_msgs" / flavour
    out: Dict[str, Any] = {}
    for entry in sorted(folder.iterdir(), key=lambda e: e.name):
        if entry.name.endswith(".msg"):
            out.update(get_types_from_msg(entry.read_text(encoding="utf-8"), prefix + entry.name[:-4]))
    return out


def _closure(names: Any, defs: Dict[str, Any]) -> set:
    """Every message type the given ones depend on (through rosbags field definitions)."""
    seen: set = set()

    def walk(node: Any) -> None:
        if isinstance(node, tuple):
            for x in node:
                walk(x)
        elif isinstance(node, str) and "/" in node:
            visit(node)

    def visit(name: str) -> None:
        if name in seen or name not in defs:
            return
        seen.add(name)
        _consts, fields = defs[name]
        for _fname, ftype in fields:
            walk(ftype)

    for n in names:
        visit(n)
    return seen


def register_vision_msgs(typestore: Any, flavour: str) -> List[str]:
    """Register the bundled vision_msgs types the typestore lacks, together
    with the standard types they depend on that the typestore also lacks
    (a rosbag2 built from its own embedded definitions knows only the types
    its topics use: ``geometry_msgs/PoseWithCovariance`` is usually absent).
    Those come from the stock typestore of the same flavour. Returns the
    names added (empty when the bag already provided everything)."""
    from rosbags.typesys import Stores, get_typestore
    types = vision_msgs_types(flavour)
    stock = get_typestore(Stores.ROS2_HUMBLE if flavour == "ros2" else Stores.ROS1_NOETIC)
    defs: Dict[str, Any] = dict(stock.fielddefs)
    defs.update(types)
    needed = _closure(types, defs)
    missing = {n: defs[n] for n in sorted(needed) if n not in typestore.types}
    if missing:
        typestore.register(missing)
    return sorted(missing)
