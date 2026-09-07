"""Configuration commands that require no GPU or model downloads."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from . import ConfigError, config_fingerprint, load_config
from .tuning import SYMPTOMS, explain_tuning, validate_tuning


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="Explicit base YAML file (otherwise use the runner's packaged default)")
    parser.add_argument("--profile", action="append", default=[], type=Path,
                        help="Sparse YAML overrides; repeat to layer profiles in order")
    parser.add_argument("--set", dest="config_sets", action="append", default=[], metavar="PATH=VALUE",
                        help="Override a setting with a YAML value; applied after profiles")


def config_from_args(args, default: str = "rtsm.yaml") -> dict:
    cfg = load_config(args.config if args.config is not None else default,
                      profiles=args.profile, set_values=args.config_sets)
    validate_tuning(cfg)
    return cfg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rtsm config",
                                     description="Explain and reproduce threshold tuning without loading models")
    parser.add_argument("action", choices=("explain", "show", "validate"), nargs="?", default="explain")
    parser.add_argument("--demo", action="store_true", help="Use packaged demo defaults as the base")
    parser.add_argument("--symptom", choices=tuple(SYMPTOMS), help="Focus the tuning explanation")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    if args.demo and args.config is not None:
        parser.error("--demo and --config select different bases; choose one")
    try:
        cfg = config_from_args(args, "demo_config.yaml" if args.demo else "rtsm.yaml")
        if args.action == "explain":
            print(explain_tuning(cfg, args.symptom))
        elif args.action == "show":
            import yaml
            # stdout remains valid YAML for redirection and round-trip loading.
            print(f"# Config SHA-256: {config_fingerprint(cfg)}")
            print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), end="")
        else:
            print(f"Tuning controls valid. Config SHA-256: {config_fingerprint(cfg)}")
            print("Validation covers documented tuning controls, not every expert setting.")
        if args.action != "explain":
            for warning in validate_tuning(cfg):
                print(f"Advisory: {warning}", file=sys.stderr)
    except (ConfigError, OSError) as exc:
        parser.error(str(exc))
