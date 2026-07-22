"""Deprecated compatibility entry point for :mod:`train`."""

import runpy
import warnings


if __name__ == "__main__":
    warnings.warn(
        "train_network3.py is deprecated; use train.py instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    runpy.run_module("train", run_name="__main__")
