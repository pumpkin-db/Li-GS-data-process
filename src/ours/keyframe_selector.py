"""
关键帧筛选模块
从所有帧中筛选出位姿变化显著的关键帧，控制训练图像数量
"""

import numpy as np
from typing import List, Dict, Optional
from pathlib import Path


class KeyFrameSelector:
    """
    基于位姿变化的关键帧筛选器。

    策略：从第一帧开始，每当与上一个关键帧的累积位移 > translation_threshold
    或旋转角度 > rotation_threshold 时，将当前帧加入关键帧集合。
    如果总数超过 max_frames，改用等间隔采样。
    """

    def __init__(
        self,
        translation_threshold: float = 0.3,   # 位移阈值（米）
        rotation_threshold: float = 5.0,       # 旋转阈值（度）
        max_frames: int = 30,                  # 最大关键帧数
        fixed_interval: Optional[int] = None,  # 强制等间隔（覆盖阈值策略）
    ):
        self.translation_threshold = translation_threshold
        self.rotation_threshold = np.radians(rotation_threshold)
        self.max_frames = max_frames
        self.fixed_interval = fixed_interval

    def select(self, poses: List[Dict], images: List[Path]) -> List[int]:
        """
        从 poses 列表中筛选关键帧索引。

        Args:
            poses : load_poses() 返回的列表，每个元素含 position, rotation
            images: load_image_list() 返回的 Path 列表

        Returns:
            keyframe_indices: 关键帧在原始列表中的整数索引
        """
        n = min(len(poses), len(images))
        if n == 0:
            return []

        if self.fixed_interval is not None:
            indices = list(range(0, n, self.fixed_interval))
        else:
            indices = self._select_by_pose_change(poses, n)

        # 若超过 max_frames，改为等间隔降采样
        if len(indices) > self.max_frames:
            step = len(indices) / self.max_frames
            indices = [indices[int(i * step)] for i in range(self.max_frames)]

        print(f"关键帧筛选: {n} 帧 → {len(indices)} 关键帧")
        for i, idx in enumerate(indices):
            p = poses[idx]['position']
            print(f"  KF {i:3d}: 帧 {idx:4d}  pos=({p[0]:6.2f},{p[1]:6.2f},{p[2]:6.2f})")
        return indices

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _select_by_pose_change(self, poses: List[Dict], n: int) -> List[int]:
        """位姿变化阈值策略：累计位移或旋转超过阈值时加入关键帧。"""
        indices = [0]  # 始终保留第一帧
        last_pos = poses[0]['position'].copy()
        last_R   = poses[0]['rotation'].copy()

        for i in range(1, n):
            cur_pos = poses[i]['position']
            cur_R   = poses[i]['rotation']

            trans = np.linalg.norm(cur_pos - last_pos)
            angle = _rotation_angle_diff(last_R, cur_R)

            if trans >= self.translation_threshold or angle >= self.rotation_threshold:
                indices.append(i)
                last_pos = cur_pos.copy()
                last_R   = cur_R.copy()

        # 始终包含最后一帧（保证覆盖完整路径）
        if indices[-1] != n - 1:
            indices.append(n - 1)

        return indices


def _rotation_angle_diff(R1: np.ndarray, R2: np.ndarray) -> float:
    """计算两个旋转矩阵之间的旋转角度（弧度）。"""
    R_rel = R1.T @ R2
    # trace(R) = 1 + 2*cos(theta)
    cos_theta = (np.trace(R_rel) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)
