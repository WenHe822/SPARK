#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import sys
import random
import numpy as np
import os.path as osp
import torch

sys.path.append("./")
from r2_gaussian.gaussian import GaussianModel
from r2_gaussian.arguments import ModelParams
from r2_gaussian.dataset.dataset_readers import sceneLoadTypeCallbacks
from r2_gaussian.utils.camera_utils import cameraList_from_camInfos
from r2_gaussian.utils.general_utils import t2a


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        shuffle=True,
    ):
        self.model_path = args.model_path

        self.train_cameras = {}
        self.test_cameras = {}

        # Read scene info
        '''{
  "scene_info": {
    "train_cameras": [
      {
        "uid": "integer",
        "R": "numpy array (3x3) - Rotation matrix",
        "T": "numpy array (3x1) - Translation vector",
        "angle": "float - Projection angle",
        "FovY": "float - Field of View in Y direction",
        "FovX": "float - Field of View in X direction",
        "image": "numpy array (3D) - Projection image data (channels x height x width)",
        "image_path": "string - Path to the projection image file",
        "image_name": "string - Name of the projection image file",
        "width": "integer - Image width",
        "height": "integer - Image height",
        "mode": "integer - Scanner mode ID (e.g., 0 for parallel, 1 for cone)",
        "scanner_cfg": {
          "mode": "string - Scanner mode type ('parallel' or 'cone')",
          "DSD": "float - Distance from Source to Detector",
          "DSO": "float - Distance from Source to Origin",
          "nDetector": "list of integers (2 elements) - Number of detector pixels [vertical, horizontal]",
          "sDetector": "list of floats (2 elements) - Size of detector pixels [vertical, horizontal]",
          "nVoxel": "list of integers (3 elements) - Number of voxels [depth, height, width]",
          "sVoxel": "list of floats (3 elements) - Size of voxels [depth, height, width]",
          "offOrigin": "list of floats (3 elements) - Offset of the volume origin [depth, height, width]",
          "offDetector": "list of floats (2 elements) - Offset of the detector [horizontal, vertical]",
          "accuracy": "float - Accuracy parameter for projection",
          "totalAngle": "float - Total scan angle",
          "startAngle": "float - Starting scan angle",
          "noise": "boolean - Flag indicating if noise is added",
          "filter": "string or null - Filter type used in projection",
          "dVoxel": "list of floats (3 elements) - Derived voxel size",
          "dDetector": "list of floats (2 elements) - Derived detector pixel size"
          // ... potentially other scanner configuration parameters
        }
      },
      // ... more CameraInfo objects for training projections
    ],
    "test_cameras": [
      {
        // ... similar CameraInfo structure as in train_cameras, but for test projections
      },
      // ... more CameraInfo objects for testing projections
    ],
    "vol": "torch.Tensor (or numpy array) (3D) - Ground truth volume data (depth x height x width)",
    "scanner_cfg": {
      "mode": "string - Scanner mode type ('parallel' or 'cone')",
      "DSD": "float - Distance from Source to Detector",
      "DSO": "float - Distance from Source to Origin",
      "nDetector": "list of integers (2 elements) - Number of detector pixels [vertical, horizontal]",
      "sDetector": "list of floats (2 elements) - Size of detector pixels [vertical, horizontal]",
      "nVoxel": "list of integers (3 elements) - Number of voxels [depth, height, width]",
      "sVoxel": "list of floats (3 elements) - Size of voxels [depth, height, width]",
      "offOrigin": "list of floats (3 elements) - Offset of the volume origin [depth, height, width]",
      "offDetector": "list of floats (2 elements) - Offset of the detector [horizontal, vertical]",
      "accuracy": "float - Accuracy parameter for projection",
      "totalAngle": "float - Total scan angle",
      "startAngle": "float - Starting scan angle",
      "noise": "boolean - Flag indicating if noise is added",
      "filter": "string or null - Filter type used in projection",
      "dVoxel": "list of floats (3 elements) - Derived voxel size",
      "dDetector": "list of floats (2 elements) - Derived detector pixel size"
      // ... potentially other scanner configuration parameters, same as in CameraInfo
    },
    "scene_scale": "float - Scaling factor applied to the entire scene"
  }
}'''
        if osp.exists(osp.join(args.source_path, "meta_data.json")):
            # Blender format
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path,
                args.eval,
            )#返回一个SceneInfo对象
        elif args.source_path.split(".")[-1] in ["pickle", "pkl"]:
            # NAF format
            scene_info = sceneLoadTypeCallbacks["NAF"](
                args.source_path,
                args.eval,
            )
        else:
            assert False, f"Could not recognize scene type: {args.source_path}."

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        # Load cameras
        print("Loading Training Cameras")
        self.train_cameras = cameraList_from_camInfos(scene_info.train_cameras, args)
        print("Loading Test Cameras")
        self.test_cameras = cameraList_from_camInfos(scene_info.test_cameras, args)

        # Set up some parameters
        self.vol_gt = scene_info.vol
        self.scanner_cfg = scene_info.scanner_cfg
        self.scene_scale = scene_info.scene_scale
        self.bbox = torch.stack(
            [
                torch.tensor(self.scanner_cfg["offOrigin"])
                - torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
                torch.tensor(self.scanner_cfg["offOrigin"])
                + torch.tensor(self.scanner_cfg["sVoxel"]) / 2,
            ],
            dim=0,
        )

    def save(self, iteration, queryfunc):
        point_cloud_path = osp.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(
            osp.join(point_cloud_path, "point_cloud.pickle")
        )  # Save pickle rather than ply
        if queryfunc is not None:
            vol_pred = queryfunc(self.gaussians)["vol"]
            vol_gt = self.vol_gt
            np.save(osp.join(point_cloud_path, "vol_gt.npy"), t2a(vol_gt))
            np.save(
                osp.join(point_cloud_path, "vol_pred.npy"),
                t2a(vol_pred),
            )

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras
