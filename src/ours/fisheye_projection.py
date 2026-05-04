"""
鱼眼图像 5 面投影模块

将鱼眼图像重投影为 5 张标准透视图（前/后/左/右/上），
每张配有独立的相机内外参，供 COLMAP 格式导出和 3DGS 训练。

核心思路：
  - 对每个面定义一个虚拟透视相机（相对鱼眼相机的旋转 R_fc）
  - 对输出图像的每个像素，反向求其对应的鱼眼图像坐标（逆映射）
  - 用 cv2.remap() 高效重采样，与 OpenCV fisheye.remap 原理相同，
    但映射坐标由 OCamCalib 模型计算（不需要 Kannala-Brandt 参数）

面坐标系约定（相机坐标系：X右 Y下 Z前）： #可能有错误，慎重参考！！！！！！！！！
  front : R_fc = I            (朝 +Z)
  back  : R_fc = Ry(180°)    (朝 -Z)
  left  : R_fc = Ry(-90°)    (朝 -X)
  right : R_fc = Ry(+90°)    (朝 +X)
  up    : R_fc = Rx(-90°)    (朝 -Y)

其中 R_fc 将面相机坐标转换到鱼眼相机坐标（行向量为面的轴在鱼眼系中的方向）。
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any

from .ocam_model import OCamModel


# 5 个面相对于鱼眼相机的旋转矩阵（面→鱼眼，行=面的轴在鱼眼坐标中的坐标）
# 相机坐标系: X=右, Y=下, Z=前(光轴 -Z 方向)
# 注意：OCamCalib 光轴是 -Z，所以"前方"在鱼眼坐标中是 [0,0,-1]
# 这里各面的 Z_face（即面的朝向）用 OCamCalib 约定表示
FACE_ROTATIONS = {
    # R_fc[i,:] = 面坐标轴 i 在鱼眼坐标系中的方向
    # 行0=X_face, 行1=Y_face, 行2=Z_face（面的前向 = 投影中心方向）
    'front': np.array([[ 1,  0,  0],   # X_face = 鱼眼 X
                        [ 0,  1,  0],   # Y_face = 鱼眼 Y
                        [ 0,  0,  1]], dtype=np.float64),  # Z_face = 鱼眼 Z (光轴)
    'back':  np.array([[-1,  0,  0],
                        [ 0,  1,  0],
                        [ 0,  0, -1]], dtype=np.float64),
    'left':  np.array([[ 0,  0,  1],
                        [ 0,  1,  0],
                        [-1,  0,  0]], dtype=np.float64),
    'right': np.array([[ 0,  0, -1],
                        [ 0,  1,  0],
                        [ 1,  0,  0]], dtype=np.float64),
    'up':    np.array([[ 1,  0,  0],
                        [ 0,  0,  1],
                        [ 0, -1,  0]], dtype=np.float64),
}

# 只投影 back 面（相机实际朝向前方，但外参矩阵的Z轴朝负方向）
FACE_NAMES = ['back']


class FisheyeProjector:
    """
    鱼眼图像 → 5 面透视图投影器。

    使用 OCamCalib 模型计算逆映射，然后用 cv2.remap 重采样。
    """

    def __init__(
        self,
        ocam: OCamModel,
        output_size: Tuple[int, int] = (1800, 1800),
        face_fov_deg: float = 90.0,
    ):
        """
        Args:
            ocam        : OCamCalib 模型
            output_size : 每个面输出图像的 (width, height)
            face_fov_deg: 每个面的水平/垂直视场角（度），推荐 90-100°
        """
        self.ocam       = ocam
        self.out_w, self.out_h = output_size
        self.fov_deg    = face_fov_deg

        # 透视相机内参（所有面相同）
        self.K_face = _make_perspective_K(self.out_w, self.out_h, face_fov_deg)

        # 预计算所有面的 remap 映射（耗时，只做一次）
        print("预计算 5 面 remap 映射...")
        self._remaps = {
            face: self._compute_remap(FACE_ROTATIONS[face])
            for face in FACE_NAMES
        }
        print("remap 预计算完成。")

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def project_keyframes(
        self,
        poses: List[Dict],
        image_paths: List[Path],
        output_dir: Path,
    ) -> Dict[str, Any]:
        """
        批量处理所有关键帧，生成 5 面透视图并返回相机参数。

        Returns:
            projection_data: {
                'face_images': list of str,          # 输出图像文件名（相对 output_dir/images/）
                'face_extrinsics': list of (4,4),    # 每张图的世界→相机变换矩阵（COLMAP 约定）
                'K_face': (3,3),                     # 所有面共用的内参矩阵
                'out_w': int, 'out_h': int,
            }
        """
        images_dir = output_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)

        face_image_names = []
        face_extrinsics  = []

        total = len(image_paths) * len(FACE_NAMES)
        done  = 0

        for seq_idx, (img_path, pose) in enumerate(zip(image_paths, poses)):
            # seq_idx: 关键帧列表中的索引（0, 1, 2, ...）
            fisheye_img = self._load_fisheye(img_path)

            for face_name in FACE_NAMES:
                # 生成透视图
                out_img = self._project_one_face(fisheye_img, face_name)

                # 保存图像，使用 seq_idx 作为编号，确保顺序一致
                face_offset = FACE_NAMES.index(face_name)
                img_id      = seq_idx * len(FACE_NAMES) + face_offset
                img_filename = f'{img_id:04d}.jpg'
                cv2.imwrite(str(images_dir / img_filename), cv2.flip(out_img, 1))

                # 计算该面在世界坐标下的外参（COLMAP world-to-camera 格式）
                # 注意：pose['rotation'] 是 camera-to-world，需要转置为 world-to-camera
                R_wc, t_wc = _compute_face_extrinsic(
                    pose['rotation'], pose['position'], FACE_ROTATIONS[face_name]
                )
                T_wc = np.eye(4, dtype=np.float64)
                T_wc[:3, :3] = R_wc
                T_wc[:3,  3] = t_wc

                face_image_names.append(img_filename)
                face_extrinsics.append(T_wc)

                done += 1
                if done % 10 == 0 or done == total:
                    print(f"  投影进度: {done}/{total}")

        print(f"投影完成，共 {len(face_image_names)} 张图像。")
        return {
            'face_images':    face_image_names,
            'face_extrinsics': face_extrinsics,
            'K_face':          self.K_face,
            'out_w':           self.out_w,
            'out_h':           self.out_h,
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _load_fisheye(self, path: Path) -> np.ndarray:
        """读取鱼眼图像，按 rotate 参数旋转到标准方向。"""
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {path}")
        rotate = self.ocam.rotate
        if rotate == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif rotate == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def _compute_remap(self, R_fc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算单个面的 cv2.remap 映射图。

        对输出图像每个像素 (u_out, v_out)：
          1. 用面内参反投影到 3D 方向（面相机坐标系）：ray_face = K_inv @ [u,v,1]
          2. 转换到鱼眼坐标系：ray_fisheye = R_fc.T @ ray_face
          3. 用 OCamCalib 模型投影到鱼眼图像坐标：(src_u, src_v)

        Returns:
            map_x, map_y: (H, W) float32 数组，供 cv2.remap 使用
        """
        H, W = self.out_h, self.out_w
        K_inv = np.linalg.inv(self.K_face)

        # 输出图像所有像素的齐次坐标 (3, H*W)
        # 注意：OpenCV图像坐标中，u是列索引(从左到右)，v是行索引(从上到下)
        # 在相机坐标系中：X向右，Y向下，Z向前
        # 所以 u 对应 X，v 对应 Y
        u_grid = np.arange(W, dtype=np.float64)  # 列坐标 (X方向)
        v_grid = np.arange(H, dtype=np.float64)  # 行坐标 (Y方向)
        uu, vv = np.meshgrid(u_grid, v_grid)
        ones   = np.ones_like(uu)
        pixels = np.stack([uu.ravel(), vv.ravel(), ones.ravel()], axis=0)  # (3, H*W)

        # 反投影到面相机坐标系的 3D 方向
        # K_inv @ [u, v, 1] 得到的是相机坐标系下的射线方向
        rays_face = K_inv @ pixels             # (3, H*W)

        # 转换到鱼眼坐标系（R_fc 行向量 = 面轴在鱼眼坐标中的方向）
        # ray_fisheye = R_fc.T @ ray_face
        rays_fisheye = R_fc.T @ rays_face      # (3, H*W)
        rays_fisheye = rays_fisheye.T          # (H*W, 3)

        # OCamCalib 前向投影（3D→鱼眼图像坐标）
        src_uv = self.ocam.world2cam_batch(rays_fisheye)  # (H*W, 2)

        map_x = src_uv[:, 0].reshape(H, W).astype(np.float32)
        map_y = src_uv[:, 1].reshape(H, W).astype(np.float32)
        return map_x, map_y

    def _project_one_face(self, fisheye_img: np.ndarray, face_name: str) -> np.ndarray:
        """用预计算的 remap 图将鱼眼图像重采样为透视图。"""
        map_x, map_y = self._remaps[face_name]
        return cv2.remap(fisheye_img, map_x, map_y,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(0, 0, 0))


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------

def _make_perspective_K(w: int, h: int, fov_deg: float) -> np.ndarray:
    """
    根据输出分辨率和视场角构建透视相机内参矩阵。
    fx = fy = (w/2) / tan(fov/2)
    """
    f  = (w / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    cx = w / 2.0
    cy = h / 2.0
    return np.array([[f,  0, cx],
                     [0,  f, cy],
                     [0,  0,  1]], dtype=np.float64)


def _compute_face_extrinsic(
    R_cam2world: np.ndarray,
    t_world: np.ndarray,
    R_fc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算面相机的 COLMAP world-to-camera 外参 (R_wc, t_wc)。

    R_cam2world : (3,3) 鱼眼相机 camera-to-world 旋转
    t_world     : (3,) 相机在世界坐标中的位置
    R_fc        : (3,3) 面→鱼眼 旋转（FACE_ROTATIONS 中的值）

    COLMAP 约定: X_cam = R_wc @ X_world + t_wc
    
    推导:
    - R_fc 是 面→鱼眼，所以 R_fc.T 是 鱼眼→面
    - R_world2fisheye = R_cam2world.T (world-to-fisheye)
    - R_wc = R_fc.T @ R_world2fisheye (world-to-face)
    
    对应: R_wc = R_fc.T @ R_cam2world.T
          t_wc = -R_wc @ t_world
    """
    R_world2fisheye = R_cam2world.T
    R_wc = R_fc.T @ R_world2fisheye     # (3,3)
    t_wc = -R_wc @ t_world              # (3,)
    
    return R_wc, t_wc
