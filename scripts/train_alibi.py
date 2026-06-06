"""Convenience entry for training the ALiBi baseline.

Equivalent to:
    python scripts/bclass_extra_experiments.py --mode train_alibi
"""

from bclass_extra_experiments import train_alibi


if __name__ == "__main__":
    train_alibi()
