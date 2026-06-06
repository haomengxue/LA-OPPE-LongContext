"""Convenience entry for PPL evaluation.

Equivalent to:
    python scripts/bclass_extra_experiments.py --mode eval_ppl
"""

from bclass_extra_experiments import eval_ppl_all


if __name__ == "__main__":
    eval_ppl_all()
