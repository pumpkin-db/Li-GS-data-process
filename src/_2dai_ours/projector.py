"""
鱼眼→透视投影

复用 src/ours/ocam_model.py 中的 OCamModel，
将一张鱼眼图像投影为指定 FOV 的透视图像。

约定：
    - 投影方向为相机光轴前方（back face：perspective +Z 对应 fisheye -Z）
    - 输出图像坐标系与标准透视相机相同（X右/Y下/Z前）
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

# 复用父模块的 OCamModel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ours'))
from ocam_model import OCamModel


# 仅翻转 Z 轴：perspective +Z → fisheye -Z（X 保持同向，避免水平镜像）
_BACK_FACE_R = np.array([[1, 0,  0],
                         [0, 1,  0],
                         [0, 0, -1]], dtype=np.float64)


def build_pinhole_K(width: int, height: int,
                    fov_h_deg: float, fov_v_deg: float = None) -> np.ndarray:
    """
    构建透视相机内参矩阵。

    Args:
        width, height: 输出图像尺寸
        fov_h_deg:     水平视场角（度），决定 fx
        fov_v_deg:     垂直视场角（度），决定 fy；None 表示与水平相同
    """
    if fov_v_deg is None:
        fov_v_deg = fov_h_deg
    fx = (width  / 2.0) / np.tan(np.radians(fov_h_deg / 2.0))
    fy = (height / 2.0) / np.tan(np.radians(fov_v_deg / 2.0))
    return np.array([[fx,  0,   width  / 2.0],
                     [0,   fy,  height / 2.0],
                     [0,   0,   1           ]], dtype=np.float64)


def project_fisheye(src_image: np.ndarray,
                    ocam_params: dict,
                    output_width: int,
                    output_height: int,
                    fov_h_deg: float,
                    fov_v_deg: float = None) -> np.ndarray:
    """
    将鱼眼图像投影为透视图像（沿光轴方向，back face）。

    Args:
        src_image:     鱼眼原图（HxWx3 BGR，cv2 读取格式）
        ocam_params:   calibration.load_dual_camera_yaml 返回的单相机参数 dict
        output_width:  输出图像宽度（像素）
        output_height: 输出图像高度（像素）
        fov_h_deg:     水平视场角（度）
        fov_v_deg:     垂直视场角（度）；None 表示与水平相同

    Returns:
        projected: (output_height, output_width, 3) BGR 图像
    """
    ocam = OCamModel(ocam_params)

    K = build_pinhole_K(output_width, output_height, fov_h_deg, fov_v_deg)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # (H*W, 2) 像素坐标
    us = np.arange(output_width,  dtype=np.float64)
    vs = np.arange(output_height, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)  # (H, W)
    # 方向向量（透视相机坐标系，Z=前）
    dirs_persp = np.stack([
        (uu - cx) / fx,
        (vv - cy) / fy,
        np.ones_like(uu),
    ], axis=-1).reshape(-1, 3)  # (H*W, 3)

    # 转换到鱼眼相机坐标系（back face 旋转）
    dirs_fisheye = dirs_persp @ _BACK_FACE_R.T  # (H*W, 3)

    # 用 OCamModel 将方向向量映射到鱼眼图像像素坐标
    uv_fisheye = ocam.world2cam_batch(dirs_fisheye)  # (H*W, 2)

    # 构建 remap 所需的映射图
    map_x = uv_fisheye[:, 0].reshape(output_height, output_width).astype(np.float32)
    map_y = uv_fisheye[:, 1].reshape(output_height, output_width).astype(np.float32)

    # 双线性插值采样
    projected = cv2.remap(src_image, map_x, map_y,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))
    return projected
