"""
OCamCalib 鱼眼相机模型

坐标约定（Scaramuzza 工具箱）：
  - 相机光轴方向为 -Z（即相机正前方是 Zc < 0）
  - 主点 (xc, yc) 是图像中心（像素）
  - pol(r) = a0 + a1*r + a2*r² + ...  r 为像素半径，返回 Zc 分量（负值为前向）
  - world2cam 用 pol 的数值求逆实现，比 invpol 更可靠

cam2world（图像→3D方向向量）：
  1. mx = u - xc,  my = v - yc
  2. r = sqrt(mx² + my²)
  3. Zc = pol(r)
  4. 方向（未归一化）= [mx, my, Zc]

world2cam（3D→图像坐标）：
  1. 目标角度 α = atan2(sqrt(Xc²+Yc²), -Zc)   (-Zc 因为光轴朝 -Z)
  2. 在预计算查找表中插值得到对应像素半径 r
  3. phi = atan2(Yc, Xc)
  4. u = r*cos(phi) + xc,  v = r*sin(phi) + yc
"""

import numpy as np
from typing import List, Dict


class OCamModel:
    """OCamCalib 鱼眼相机正反投影模型。
    
    使用 camToWorld 系数进行正向投影（图像->3D），
    使用查找表进行反向投影（3D->图像）。
    """

    def __init__(self, ocam_params: Dict):
        """
        Args:
            ocam_params: data_loader.load_ocam_intrinsic() 的返回值
        """
        self.xc  = ocam_params['xc']
        self.yc  = ocam_params['yc']
        # 使用 camToWorld 系数（更稳定），而不是 worldToCam
        # 如果提供了 camToWorld，使用它；否则尝试使用 pol（worldToCam）
        if 'camToWorld' in ocam_params:
            self.pol = np.array(ocam_params['camToWorld'], dtype=np.float64)
        else:
            self.pol = np.array(ocam_params['pol'], dtype=np.float64)
        self.width  = ocam_params['width']
        self.height = ocam_params['height']
        self.rotate = ocam_params.get('rotate', 0)

        # 预计算角度→像素半径查找表（用于 world2cam）
        self._lut_alpha, self._lut_r = self._build_lut(n_samples=5000)

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def cam2world_batch(self, uv: np.ndarray) -> np.ndarray:
        """
        批量图像坐标 → 3D 方向向量（已归一化）。
        使用 camToWorld 多项式系数。

        Args:
            uv: (N, 2) 图像坐标 (u, v)

        Returns:
            rays: (N, 3) 单位方向向量
        """
        mx = uv[:, 0] - self.xc
        my = uv[:, 1] - self.yc
        r  = np.sqrt(mx**2 + my**2)
        Zc = self._eval_pol(r)  # 使用 camToWorld 系数

        rays = np.stack([mx, my, Zc], axis=1)
        norms = np.linalg.norm(rays, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return rays / norms

    def world2cam_batch(self, points: np.ndarray) -> np.ndarray:
        """
        批量 3D 方向向量（相机坐标系）→ 图像坐标。
        使用查找表方法（基于 camToWorld 的反函数）。

        Args:
            points: (N, 3) 3D 点或方向向量（不需要归一化）

        Returns:
            uv: (N, 2) 图像坐标 (u, v)，超出视野的点坐标可能超出图像范围
        """
        Xc = points[:, 0]
        Yc = points[:, 1]
        Zc = points[:, 2]

        l = np.sqrt(Xc**2 + Yc**2)
        # 角度 α：从光轴（-Z）到当前方向
        alpha = np.arctan2(l, -Zc)

        # 查找表插值得到像素半径
        # 注意：_lut_alpha 是角度数组，_lut_r 是对应的半径数组
        r = np.interp(alpha, self._lut_alpha, self._lut_r)

        # 防止 l=0（在光轴上）时除零
        safe_l = np.where(l < 1e-10, 1.0, l)
        cos_phi = Xc / safe_l
        sin_phi = Yc / safe_l
        cos_phi = np.where(l < 1e-10, 0.0, cos_phi)
        sin_phi = np.where(l < 1e-10, 0.0, sin_phi)

        u = r * cos_phi + self.xc
        v = r * sin_phi + self.yc

        return np.stack([u, v], axis=1)

    def get_max_fov_deg(self) -> float:
        """估算相机最大视场角（度），基于图像边缘半径。"""
        # 最大有效半径（图像内切圆）
        r_max = min(self.width, self.height) / 2.0
        Zc = self._eval_pol(r_max)
        alpha = np.arctan2(r_max, -Zc)
        return float(np.degrees(alpha))

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _eval_pol(self, r: np.ndarray) -> np.ndarray:
        """计算 pol(r) = a0 + a1*r + a2*r² + ..."""
        result = np.zeros_like(r, dtype=np.float64)
        r_pow  = np.ones_like(r, dtype=np.float64)
        for coeff in self.pol:
            result += coeff * r_pow
            r_pow  *= r
        return result

    def _build_lut(self, n_samples: int = 5000):
        """
        预计算视场角 → 像素半径的查找表（用于 world2cam）。
        
        使用 camToWorld 多项式计算 (r, Zc) 对应关系，
        然后计算 alpha = atan2(r, -Zc)，构建 alpha -> r 的查找表。
        
        返回 (alpha_values, r_values) 两个有序数组，供 np.interp 使用。
        """
        # 最大半径（图像半对角线的 1.1 倍）
        r_max = np.sqrt((self.width / 2)**2 + (self.height / 2)**2) * 1.1
        r_vals = np.linspace(0, r_max, n_samples)
        
        # 使用 camToWorld 系数计算 Zc
        Zc_vals = self._eval_pol(r_vals)
        
        # 计算角度 alpha = atan2(r, -Zc)（与光轴 -Z 的夹角）
        alpha_vals = np.arctan2(r_vals, -Zc_vals)
        
        # 保留单调递增的部分（确保可以反查）
        # 找到 alpha 单调递增的区间
        valid_mask = np.concatenate([[True], np.diff(alpha_vals) > 1e-10])
        
        alpha_lut = alpha_vals[valid_mask]
        r_lut = r_vals[valid_mask]
        
        return alpha_lut, r_lut
