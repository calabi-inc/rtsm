"""Dispatch lightweight commands before importing the runtime and GPU stack."""

import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "config":
        from rtsm.cfg.cli import main as config_main
        config_main(sys.argv[2:])
    else:
        from rtsm.run import main as run_main
        run_main()
