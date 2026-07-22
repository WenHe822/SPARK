# Third-party notices

This repository contains or adapts code from third-party research projects.
Their original notices and license terms continue to apply to the corresponding
files.

## TIGRE

- Project: [CERN/TIGRE](https://github.com/CERN/TIGRE)
- Location: `TIGRE-2.3/`
- License: see `TIGRE-2.3/LICENSE.txt` and `TIGRE-2.3/Python/LICENSE`

## R²-Gaussian and CUDA extensions

- Location: `r2_gaussian/`
- X-ray Gaussian rasterization and voxelization license:
  `r2_gaussian/submodules/xray-gaussian-rasterization-voxelization/LICENSE.md`
- Simple KNN license:
  `r2_gaussian/submodules/simple-knn/LICENSE.md`

## Splatter Image

- Project: [Splatter Image](https://github.com/szymanowiczs/splatter-image)
- Adapted concepts and utility/model code are present in `scene/`, `utils/`,
  and `gradio_app.py`.

## Score SDE and EDM

Parts of the U-Net implementation in `scene/gaussian_predictor.py` and
`scene/gaussian_predictor_multichannel.py` reference:

- [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)
- [EDM](https://github.com/NVlabs/edm)

Refer to the copyright headers and upstream repositories for the exact terms
applicable to individual files. This notice is informational and does not
replace any original license text.
