"""Convenience entry for LongBench subset evaluation.

Equivalent to:
    python scripts/bclass_plus_experiments.py longbench
"""

import sys

from bclass_plus_experiments import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("longbench")
    main()
