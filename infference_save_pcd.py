"""Deprecated compatibility entry point for :mod:`inference_save_pcd`."""

import runpy
import warnings


if __name__ == "__main__":
    warnings.warn(
        "infference_save_pcd.py is deprecated; use inference_save_pcd.py instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    runpy.run_module("inference_save_pcd", run_name="__main__")
