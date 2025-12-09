import logging
from .run import main

# Configure process-wide logging once at entrypoint
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
# Optionally ensure our package logs at INFO (can tighten further elsewhere)
logging.getLogger("rtsm").setLevel(logging.INFO)

if __name__ == "__main__":
    main()