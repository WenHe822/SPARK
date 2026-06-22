import numpy as np

import sys
import torch


def get_loop_cameras(num_imgs_in_loop, radius=2.0, 
                     max_elevation=np.pi/6, elevation_freq=0.5,
                     azimuth_freq=2.0):

    all_cameras_c2w_cmo = []

    for i in range(num_imgs_in_loop):
        azimuth_angle = np.pi * 2 * azimuth_freq * i / num_imgs_in_loop
        elevation_angle = max_elevation * np.sin(
            np.pi * i * 2 * elevation_freq / num_imgs_in_loop)
        x = np.cos(azimuth_angle) * radius * np.cos(elevation_angle)
        y = np.sin(azimuth_angle) * radius * np.cos(elevation_angle)
        z = np.sin(elevation_angle) * radius

        camera_T_c2w = np.array([x, y, z], dtype=np.float32)

        # in COLMAP / OpenCV convention: z away from camera, y down, x right
        camera_z = - camera_T_c2w / radius
        up = np.array([0, 0, -1], dtype=np.float32)
        camera_x = np.cross(up, camera_z)
        camera_x = camera_x / np.linalg.norm(camera_x)
        camera_y = np.cross(camera_z, camera_x)

        camera_c2w_cmo = np.hstack([camera_x[:, None], 
                                    camera_y[:, None], 
                                    camera_z[:, None], 
                                    camera_T_c2w[:, None]])
        camera_c2w_cmo = np.vstack([camera_c2w_cmo, np.array([0, 0, 0, 1], dtype=np.float32)[None, :]])

        all_cameras_c2w_cmo.append(camera_c2w_cmo)

    return all_cameras_c2w_cmo

sys.path.append("./")
from datasets.cameras import Camera


def loadCam( id, cam_info):
    gt_image = torch.from_numpy(cam_info.image)[None]
    
    # 处理mask投影数据（如果存在）
    mask_image = None
    if hasattr(cam_info, 'mask_image') and cam_info.mask_image is not None:
        mask_image = torch.from_numpy(cam_info.mask_image)[None]

    return Camera(
        colmap_id=cam_info.uid,
        scanner_cfg=cam_info.scanner_cfg,
        R=cam_info.R,
        T=cam_info.T,
        angle=cam_info.angle,
        mode=cam_info.mode,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        image_name=cam_info.image_name,
        uid=id,
        data_device='cuda',
        mask_image=mask_image,  # 添加mask_image参数
    )


def cameraList_from_camInfos(cam_infos):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam( id, c))

    return camera_list



def camera_to_JSON(id, camera: Camera):
    Rt = np.eye(4)
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T

    W2C = Rt
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "mode": camera.mode,
        "position_w2c": pos.tolist(),
        "rotation_w2c": serializable_array_2d,
        "FovY": camera.FovY,
        "FovX": camera.FovX,
    }
    return camera_entry
