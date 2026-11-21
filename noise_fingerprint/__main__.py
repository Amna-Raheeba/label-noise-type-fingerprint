"""Entry point: python -m noise_fingerprint [demo|evaluate]."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
