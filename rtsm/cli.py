"""Dispatch lightweight commands before importing the runtime and GPU stack."""

import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("version", "--version", "-V"):
        from rtsm import __version__
        print(f"rtsm {__version__}")
    elif len(sys.argv) > 1 and sys.argv[1] == "config":
        from rtsm.cfg.cli import main as config_main
        config_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "eval":
        from rtsm.evaluation.runner import main as eval_main
        sys.exit(eval_main(sys.argv[2:]))
    elif len(sys.argv) > 1 and sys.argv[1] == "ros2":
        # `rtsm ros2 probe [...]`: what the ros2 source would subscribe to, no models loaded
        sub = sys.argv[2:3]
        if sub != ["probe"]:
            print("usage: rtsm ros2 probe [--seconds N] [--qos auto|reliable|best_effort] [--topic ROLE=TOPIC ...] [--json]", file=sys.stderr)
            sys.exit(2)
        from rtsm.io.ros2_source import probe_main
        sys.exit(probe_main(sys.argv[3:]))
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        from rtsm.evaluation.report import main as report_main
        sys.exit(report_main(sys.argv[2:]))
    else:
        from rtsm.run import main as run_main
        run_main()


if __name__ == "__main__":      # `python -m rtsm.cli ...` behaves like the `rtsm` console script
    main()
