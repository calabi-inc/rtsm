"""
RTSM configuration loader.

Config YAML files live in this package directory (rtsm/cfg/) and ship with
pip install. For dev checkouts, a `config/` symlink at repo root points here.

Usage:
    from rtsm.cfg import load_config, cfg_path

    cfg = load_config()                     # loads rtsm.yaml
    cfg = load_config("demo_config.yaml")   # loads demo_config.yaml
    path = cfg_path("clip/vocab.yaml")      # resolves to file path
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, Dict, Sequence
from copy import deepcopy
import difflib
import hashlib
import json
import math


class ConfigError(ValueError):
    """A configuration cannot be applied as written."""


def cfg_path(name: str | Path = "rtsm.yaml") -> Path:
    """Resolve a config file path.

    Lookup order:
      0. A Path object is an explicit file, with no fallback.
      1. importlib.resources (works for pip-installed package)
      2. CWD-relative ``config/<name>`` (legacy dev path via symlink)

    Args:
        name: Relative path within the cfg package, e.g. ``"rtsm.yaml"``
              or ``"clip/vocab.yaml"``.

    Returns:
        Resolved :class:`Path` to the config file.

    Raises:
        FileNotFoundError: If the config file cannot be found.
    """
    # Path objects are explicit user files; never fall back to package defaults.
    if isinstance(name, Path):
        if name.is_file():
            return name.resolve()
        raise FileNotFoundError(f"Config file not found: {name.resolve()}")

    # 1. Package resource (canonical — ships in wheel)
    pkg_path = resources.files("rtsm.cfg").joinpath(name)
    resolved = Path(str(pkg_path))
    if resolved.is_file():
        return resolved

    # 2. CWD-relative fallback (symlink or old layout)
    cwd_path = Path("config") / name
    if cwd_path.is_file():
        return cwd_path

    raise FileNotFoundError(
        f"Config file not found: {name!r}. "
        f"Searched: {resolved}, {cwd_path.resolve()}"
    )


def load_config(
    name: str | Path = "rtsm.yaml",
    *,
    profiles: Sequence[str | Path] = (),
    set_values: Sequence[str] = (),
) -> Dict[str, Any]:
    """Load and parse a YAML config file.

    Args:
        name: Packaged filename, or a Path for an explicit full base file.
        profiles: Sparse YAML files, merged over the base in order.
        set_values: Dotted path=YAML assignments, applied after all profiles.

    Returns:
        Parsed config dict.
    """
    import yaml

    cfg = _read_yaml(cfg_path(name))
    if not profiles and not set_values:
        return cfg

    # Profiles may use any shipped setting, a documented tuning control, or an
    # expert setting already declared in the selected base config.
    from .tuning import CONTROLS
    known = _leaf_paths(_read_yaml(cfg_path("rtsm.yaml")))
    known |= _leaf_paths(_read_yaml(cfg_path("demo_config.yaml")))
    known |= _leaf_paths(cfg) | {control.path for control in CONTROLS}

    for profile in profiles:
        patch = _read_yaml(cfg_path(Path(profile)))
        _check_override_keys(patch, known)
        _merge(cfg, patch)
    for assignment in set_values:
        key, sep, raw = assignment.partition("=")
        if not sep or not key or any(not part for part in key.split(".")):
            raise ConfigError(f"Expected --set path=value, got {assignment!r}")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML value for {key}: {exc}") from exc
        _check_tree(value, key)
        patch: dict = {}
        node = patch
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
        _check_override_keys(patch, known)
        _merge(cfg, patch)
    return cfg


def _read_yaml(path: Path) -> dict:
    import yaml

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConfigError(f"{path}: configuration keys must be strings")
            if key in result:
                raise ConfigError(f"{path}: duplicate key {key!r}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    try:
        cfg = yaml.load(path.read_text(encoding="utf-8-sig"), Loader=UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: expected a YAML mapping")
    _check_tree(cfg, str(path))
    return cfg


def _check_tree(value: Any, path: str, ancestors: frozenset = frozenset()) -> None:
    if isinstance(value, (dict, list)):
        if id(value) in ancestors:
            raise ConfigError(f"{path}: recursive YAML aliases are unsupported")
        ancestors = ancestors | {id(value)}
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for key, child in items:
            if isinstance(value, dict) and not isinstance(key, str):
                raise ConfigError(f"{path}: configuration keys must be strings")
            _check_tree(child, f"{path}.{key}", ancestors)
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise ConfigError(f"{path}: unsupported value type {type(value).__name__}")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ConfigError(f"{path}: value must be finite")


def _leaf_paths(cfg: dict, prefix: str = "") -> set[str]:
    paths = set()
    for key, value in cfg.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            paths.update(_leaf_paths(value, path))
        else:
            paths.add(path)
    return paths


def _check_override_keys(patch: dict, known: set[str], prefix: str = "") -> None:
    for key, value in patch.items():
        path = f"{prefix}.{key}" if prefix else key
        children = any(item.startswith(path + ".") for item in known)
        if isinstance(value, dict) and children:
            _check_override_keys(value, known, path)
        elif path not in known or children:
            matches = difflib.get_close_matches(path, known, n=1)
            hint = f" Did you mean {matches[0]!r}?" if matches else ""
            raise ConfigError(
                f"Unknown or malformed override {path!r}.{hint} "
                "Additional expert settings must be declared in the base --config file."
            )


def _merge(cfg: dict, patch: dict) -> None:
    for key, value in patch.items():
        if isinstance(value, dict):
            if key in cfg and not isinstance(cfg[key], dict):
                raise ConfigError(f"Cannot merge mapping into scalar setting {key!r}")
            _merge(cfg.setdefault(key, {}), value)
        else:
            cfg[key] = deepcopy(value)


def config_fingerprint(cfg: dict) -> str:
    """Stable identity of the resolved configuration, not of model/data files."""
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
