"""Deprecated compatibility entry point for :mod:`inference`."""

import runpy
import warnings


if __name__ == "__main__":
    warnings.warn(
        "infference.py is deprecated; use inference.py instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    runpy.run_module("inference", run_name="__main__")
