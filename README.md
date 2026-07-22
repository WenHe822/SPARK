# SPARK

SPARK is research code for sparse-view cone-beam CT (CBCT) reconstruction with
3D Gaussian representations. A U-Net predicts Gaussian parameters from one or
more projections, and the R²-Gaussian renderer projects the representation back
to the detector domain for supervised training.

This repository accompanies a research project and is being prepared for
publication. The current canonical entry points are `train.py`, `inference.py`,
and `inference_save_pcd.py`.

## Repository layout

```text
configs/          Hydra training configurations
data_process/     Volume preprocessing and projection generation
datasets/         CBCT datasets and camera readers
scene/            Gaussian prediction networks
r2_gaussian/      CT Gaussian rendering and voxelization
TIGRE-2.3/        Vendored TIGRE projection toolkit
utils/            Losses, geometry, camera, and visualization utilities
legacy/           Historical Lightning Fabric training experiments
train.py          Main distributed training entry point
inference.py      Volume reconstruction and projection comparison
inference_save_pcd.py
                  Gaussian parameter export
eval.py           Evaluation utilities
```

Historical training variants are documented in `legacy/README.md`. The
misspelled `infference*.py` files are compatibility entry points and may be
removed in a future release.

## Environment

The CUDA extensions were developed with Python 3.8, PyTorch 1.12.1, and CUDA
11.3. A CUDA compiler compatible with the installed PyTorch build is required.

```bash
conda create -n spark python=3.8
conda activate spark
pip install -r requirements.txt

pip install ./r2_gaussian/submodules/simple-knn
pip install ./r2_gaussian/submodules/xray-gaussian-rasterization-voxelization
```

`env_a.yml` and `env_b.yml` are retained as full historical environment
snapshots. `requirements.txt` is the shorter environment definition for the
main training and inference path.

## Data layout

The configured data root must contain `train/` and `test/` splits. Each case
contains projection arrays and a `meta_data.json` file:

```text
data/Sparse_challenge/
├── train/
│   └── case_name/
│       ├── projection1.npy
│       ├── projection2.npy
│       └── meta_data.json
└── test/
    └── case_name/
        ├── projection1.npy
        ├── projection2.npy
        └── meta_data.json
```

See `data_construction.txt` for the metadata schema and `data_process/` for the
preprocessing scripts. Datasets, medical images, checkpoints, and generated
outputs are intentionally excluded from Git.

## Training

The default configuration is `configs/default_config.yaml`. Override machine-
specific values on the command line instead of editing source files:

```bash
# Four GPUs by default
./run_train.sh data.data_path=/path/to/dataset

# Select a different GPU count
NUM_GPUS=2 ./run_train.sh \
  data.data_path=/path/to/dataset \
  data.batch_size=2 \
  logging.disable_wandb=true
```

To initialize from an existing checkpoint:

```bash
./run_train.sh \
  data.data_path=/path/to/dataset \
  opt.pretrained_ckpt=/path/to/model_latest.pth
```

Hydra writes each run beneath `experiments_out/`. The batch size in the
configuration is the global batch size and must be divisible by the number of
distributed workers.

## Inference

Reconstruct a volume and optionally render comparison views:

```bash
python inference.py \
  --proj_dir /path/to/case \
  --ckpt_path /path/to/model_latest.pth \
  --config_path configs/default_config.yaml \
  --output_path output/volume_pred.nii.gz
```

Export Gaussian positions and densities:

```bash
python inference_save_pcd.py \
  --proj_dir /path/to/case \
  --ckpt_path /path/to/model_latest.pth \
  --config_path configs/default_config.yaml \
  --output_path output/gaussians.npy
```

Process all immediate case directories under one data directory:

```bash
./run_inference.sh /path/to/cases /path/to/model_latest.pth
```

The inference scripts currently instantiate `scene/gaussian_predictor.py`.
Use checkpoints created for that architecture. The canonical training script
uses `scene/gaussian_predictor_multichannel.py`; architecture unification is
outside this conservative cleanup and should be validated separately.

## Reproducibility notes

- The projection generator depends on TIGRE and a CUDA-capable GPU.
- The R²-Gaussian extensions are compiled locally and are not committed.
- Weights & Biases logging is disabled by default. Set
  `logging.disable_wandb=false` to enable it.
- The legacy experiment and demo scripts may require additional dependencies.

## Acknowledgements and license

This code builds on TIGRE, R²-Gaussian, Splatter Image, and related Gaussian
splatting implementations. See `THIRD_PARTY_NOTICES.md` and the license files
within vendored components for attribution and license terms.

The repository-level license is provided in `LICENSE`.
