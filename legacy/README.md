# Legacy training experiments

These scripts are historical training variants retained for reproducibility.
They are not the maintained training entry point.

- `train_fabric.py`: full Lightning Fabric training implementation.
- `train_fabric_single_gpu.py`: earlier single-GPU Fabric experiment.
- `train_fabric_ddp.py`: compact multi-GPU Fabric experiment.

Use `../train.py` through `../run_train.sh` for current training. The legacy
scripts use `scene/gaussian_predictor.py`, while the maintained entry point uses
`scene/gaussian_predictor_multichannel.py`. Checkpoint compatibility is not
guaranteed across these architectures.
