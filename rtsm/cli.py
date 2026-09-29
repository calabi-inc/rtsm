"""Dispatch lightweight commands before importing the runtime and GPU stack."""

import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "config":
        from rtsm.cfg.cli import main as config_main
        config_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "eval":
        from rtsm.evaluation.runner import main as eval_main
        sys.exit(eval_main(sys.argv[2:]))
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        from rtsm.evaluation.report import main as report_main
        sys.exit(report_main(sys.argv[2:]))
    else:
        from rtsm.run import main as run_main
        run_main()
