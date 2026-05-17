"""
位姿计算模块

将 POS 文件给出的机体（LiDAR）位姿，转换为 COLMAP 所需的相机位姿。

坐标约定：
    POS 文件给出机体位姿：
        P_world = R_bw @ P_body + t_bw   (body→world)
        R_bw = Rz(yaw) @ Ry(pitch) @ Rx(roll)，ZYX 欧拉角
        t_bw = [X, Y, Z]（米）

    YAML 外参定义（camera→body 约定）：
        P_body = extrinsicR @ P_cam + extrinsicT

    推导 world→camera（COLMAP 格式）：
        P_cam = extrinsicR.T @ P_body - extrinsicR.T @ extrinsicT
        R_wc  = extrinsicR.T @ R_bw.T
        t_wc  = extrinsicR.T @ (-R_bw.T @ t_bw - extrinsicT)

    相机中心在世界坐标系中的位置：
        cam_center = t_bw + R_bw @ extrinsicT
"""

import numpy as np
from pathlib import Path
from typing import List, Dict


# ──────────────────────────────────────────────
# POS 文件加载
# ──────────────────────────────────────────────

def load_pos_file(pos_path: str) -> List[Dict]:
    """
    读取 camera_pos.cam 格式的位姿文件。

    文件格式（每行）：
        name  0  0  X  Y  Z  roll  pitch  yaw  timestamp
        （前两列固定为 0，欧拉角单位：弧度，ZYX 顺序）

    Returns:
        list of dict:
            image_name  str
            R_bw        (3,3) body-to-world 旋转矩阵
            t_bw        (3,)  body 在世界坐标系中的位置
            timestamp   float
    """
    poses = []
    with open(pos_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            name  = parts[0]
            x, y, z       = float(parts[3]), float(parts[4]), float(parts[5])
            roll, pitch, yaw = float(parts[6]), float(parts[7]), float(parts[8])
            ts    = float(parts[9])

            poses.append({
                'image_name': name,
                'R_bw': _euler_to_rotation_matrix(roll, pitch, yaw),
                't_bw': np.array([x, y, z], dtype=np.float64),
                'timestamp': ts,
            })
    return poses


# ──────────────────────────────────────────────
# 核心位姿转换
# ──────────────────────────────────────────────

def body_to_camera_pose(R_bw: np.ndarray, t_bw: np.ndarray,
                        ext_R: np.ndarray, ext_T: np.ndarray) -> tuple:
    """
    将机体位姿转换为 COLMAP world-to-camera 位姿。

    Args:
        R_bw:  (3,3) body-to-world 旋转矩阵（来自 POS 文件）
        t_bw:  (3,)  机体在世界坐标系中的位置（来自 POS 文件）
        ext_R: (3,3) extrinsicR，camera→body 旋转（YAML 约定）
        ext_T: (3,)  extrinsicT，相机原点在 body 坐标系下的位置

    Returns:
        R_wc: (3,3) world-to-camera 旋转（COLMAP 格式）
        t_wc: (3,)  COLMAP 平移向量
    """
    R_wc = ext_R.T @ R_bw.T
    t_wc = ext_R.T @ (-R_bw.T @ t_bw - ext_T)
    return R_wc, t_wc


def rotation_to_quaternion(R: np.ndarray) -> tuple:
    """
    旋转矩阵 → 四元数 (qw, qx, qy, qz)，COLMAP 标量优先格式。
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return float(qw), float(qx), float(qy), float(qz)


# ──────────────────────────────────────────────
# 辅助
# ──────────────────────────────────────────────

def _euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX 欧拉角 → body-to-world 旋转矩阵。R = Rz(yaw) @ Ry(pitch) @ Rx(roll)"""
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    Rx = np.array([[1,  0,   0 ], [0,  cr, -sr], [0,  sr,  cr]], dtype=np.float64)
    Ry = np.array([[cp, 0,   sp], [0,   1,   0 ], [-sp, 0,  cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0 ], [sy,  cy,  0 ], [0,   0,   1]], dtype=np.float64)
    return Rz @ Ry @ Rx
