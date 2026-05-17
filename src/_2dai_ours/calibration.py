"""
加载双相机 YAML 标定文件（CAMERA_886203051.yaml 格式）
返回 cam1 和 cam2 各自的 OCam 参数 + 外参，可直接传给 OCamModel。
"""

import yaml
import numpy as np
from pathlib import Path


def load_dual_camera_yaml(yaml_path: str) -> tuple:
    """
    从 YAML 文件中提取两个相机的参数。

    Args:
        yaml_path: CALIBRATION_CAMERA/CAMERA_*.yaml 路径

    Returns:
        cam1_params, cam2_params: 两个 dict，格式与 OCamModel 兼容：
            xc, yc          主点（像素）
            pol             camToWorld 多项式系数（np.ndarray）
            width, height   图像尺寸
            rotate          旋转角
            extrinsic_R     (3,3) ndarray，body→camera 旋转
            extrinsic_T     (3,)  ndarray，body→camera 平移
    """
    with open(Path(yaml_path), 'r') as f:
        data = yaml.safe_load(f)

    def _extract(idx: int) -> dict:
        i = str(idx)
        return {
            'xc':          data[f'principal{i}'][0],
            'yc':          data[f'principal{i}'][1],
            'pol':         np.array(data[f'camToWorld{i}'], dtype=np.float64),
            'width':       data['width'],
            'height':      data['height'],
            'rotate':      data.get('rotate', 0),
            'extrinsic_R': np.array(data[f'extrinsicR{i}'], dtype=np.float64).reshape(3, 3),
            'extrinsic_T': np.array(data[f'extrinsicT{i}'], dtype=np.float64),
        }

    return _extract(1), _extract(2)
