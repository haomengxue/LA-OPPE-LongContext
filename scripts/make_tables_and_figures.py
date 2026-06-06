"""Build Markdown summary tables from generated result CSV files.

This is a lightweight wrapper around the summary mode used in the extended
experiments. It does not regenerate plots; the released figure PNGs are kept in
figures/.
"""

from bclass_plus_experiments import run_summary


class _Args:
    pass


if __name__ == "__main__":
    run_summary(_Args())
