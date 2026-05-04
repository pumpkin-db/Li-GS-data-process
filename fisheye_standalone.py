"""
鱼眼图像 → back面透视图 独立脚本

只需输入一张鱼眼图片和相机内参文件，即可生成 back 面透视投影图。
输出到 output/fisheye_to_plat/ 目录。

用法：
    1. 修改下方的 IMAGE_PATH 和 INTRINSIC_PATH
    2. 运行: python fisheye_standalone.py

依赖: numpy, opencv-python, pyyaml
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# 直接导入子模块，绕过 ours/__init__.py（避免触发 open3d 等重依赖）
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, 'src', 'ours'))
from ocam_model import OCamModel
from data_loader import DataLoader


# ============================================================
# 面旋转矩阵 & 透视内参（来自 fisheye_projection.py，内联以保持独立）
# ============================================================

# 5 个面相对于鱼眼相机的旋转矩阵（面→鱼眼，行=面的轴在鱼眼坐标中的方向）
# 相机坐标系: X=右, Y=下, Z=前
FACE_ROTATIONS = {
    'front': np.array([[ 1,  0,  0],
                        [ 0,  1,  0],
                        [ 0,  0,  1]], dtype=np.float64),
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


def _make_perspective_K(w: int, h: int, fov_deg: float) -> np.ndarray:
    """构建透视相机内参矩阵。fx = fy = (w/2) / tan(fov/2)"""
    f  = (w / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    cx = w / 2.0
    cy = h / 2.0
    return np.array([[f,  0, cx],
                     [0,  f, cy],
                     [0,  0,  1]], dtype=np.float64)


# ============================================================
# ★ 配置参数 - 在这里修改输入输出
# ============================================================

# 输入：鱼眼图像目录
IMAGE_DIR = r'E:\_cloud\house\camera1'
IMAGE_EXTENSION = '.jpg'         # 图像文件扩展名

# 输入：相机内参文件路径
#   .yaml → OCamCalib YAML 格式（含 camToWorld1, principal1 等）
#   .txt  → OCamCalib calib_results.txt 格式
INTRINSIC_PATH = r'E:\_cloud\house\20260428003947_gs\CALIBRATION_CAMERA\CAMERA_698802819.yaml'

# 输出参数
OUTPUT_SIZE = (3600, 3600)      # 每面输出分辨率 (width, height)
FACE_FOV_DEG = 90.0             # 每面视场角（度）
SKIP_FRAMES = 1                 # 抽帧间隔: 1=每帧, 2=隔一帧, 3=隔两帧...

# ============================================================


def compute_remap(ocam: OCamModel, K_face: np.ndarray,
                  R_fc: np.ndarray, out_w: int, out_h: int):
    """
    计算单个面的 cv2.remap 映射图（与 FisheyeProjector._compute_remap 一致）。

    对输出图像每个像素 (u_out, v_out)：
      1. 用面内参反投影到 3D 方向（面相机坐标系）
      2. 转换到鱼眼坐标系
      3. 用 OCamCalib 模型投影到鱼眼图像坐标

    Returns:
        map_x, map_y: (H, W) float32 数组
    """
    K_inv = np.linalg.inv(K_face)

    u_grid = np.arange(out_w, dtype=np.float64)
    v_grid = np.arange(out_h, dtype=np.float64)
    uu, vv = np.meshgrid(u_grid, v_grid)
    ones = np.ones_like(uu)
    pixels = np.stack([uu.ravel(), vv.ravel(), ones.ravel()], axis=0)  # (3, H*W)

    # 反投影到面相机坐标系
    rays_face = K_inv @ pixels              # (3, H*W)
    # 转换到鱼眼坐标系
    rays_fisheye = R_fc.T @ rays_face       # (3, H*W)
    rays_fisheye = rays_fisheye.T           # (H*W, 3)

    # OCamCalib 投影到鱼眼图像坐标
    src_uv = ocam.world2cam_batch(rays_fisheye)  # (H*W, 2)

    map_x = src_uv[:, 0].reshape(out_h, out_w).astype(np.float32)
    map_y = src_uv[:, 1].reshape(out_h, out_w).astype(np.float32)
    return map_x, map_y


def _parse_index(stem: str) -> int:
    """从文件名提取数字索引（与 DataLoader 一致）"""
    import re
    m = re.search(r'\d+', stem)
    return int(m.group()) if m else 0


def _scan_images(dir_path: str, ext: str) -> list:
    """扫描目录，返回按数字编号排序的图片路径列表"""
    p = Path(dir_path)
    files = sorted(p.glob(f'*{ext}'), key=lambda x: _parse_index(x.stem))
    if not files:
        raise FileNotFoundError(f"目录中没有 {ext} 文件: {dir_path}")
    return files


def main():
    # 处理 Windows → WSL 路径转换
    image_dir = _fix_wsl_path(IMAGE_DIR)
    intrinsic_path = _fix_wsl_path(INTRINSIC_PATH)

    print("=" * 60)
    print("鱼眼图像 → back面透视图（批量）")
    print("=" * 60)
    print(f"输入目录: {image_dir}")
    print(f"内参文件: {intrinsic_path}")
    print(f"输出分辨率: {OUTPUT_SIZE[0]}×{OUTPUT_SIZE[1]}")

    # ---- 1. 加载相机内参 ----
    print("\n[1/4] 加载相机内参...")
    loader = DataLoader({})
    ocam_params = loader.load_ocam_intrinsic(intrinsic_path)
    ocam = OCamModel(ocam_params)
    print(f"  ✓ 图像尺寸: {ocam.width}×{ocam.height}")
    print(f"  ✓ 主点: ({ocam.xc:.1f}, {ocam.yc:.1f})")
    print(f"  ✓ 估计最大视场角: {ocam.get_max_fov_deg():.1f}°")
    if ocam.rotate:
        print(f"  ✓ 旋转补偿: {ocam.rotate}°")

    # ---- 2. 扫描图像列表 ----
    print(f"\n[2/4] 扫描图像...")
    image_paths = _scan_images(image_dir, IMAGE_EXTENSION)
    if SKIP_FRAMES > 1:
        image_paths = image_paths[::SKIP_FRAMES]
        print(f"  ✓ 找到 {len(image_paths) * SKIP_FRAMES} 张，抽帧 1/{SKIP_FRAMES}，实际处理 {len(image_paths)} 张")
    else:
        print(f"  ✓ 找到 {len(image_paths)} 张")

    # ---- 3. 预计算 remap（只算一次，所有图片共用） ----
    print(f"\n[3/4] 预计算 back 面 remap 映射...")
    out_w, out_h = OUTPUT_SIZE
    K_face = _make_perspective_K(out_w, out_h, FACE_FOV_DEG)
    R_fc = FACE_ROTATIONS['back']
    map_x, map_y = compute_remap(ocam, K_face, R_fc, out_w, out_h)
    print(f"  ✓ 完成")

    # ---- 4. 逐张投影 ----
    print(f"\n[4/4] 投影 {len(image_paths)} 张...")
    output_dir = Path(_script_dir) / 'output' / 'fisheye_to_plat'
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(image_paths)
    for idx, img_path in enumerate(image_paths):
        # 加载鱼眼图像
        fisheye_img = cv2.imread(str(img_path))
        if fisheye_img is None:
            print(f"  ⚠ 跳过无法读取: {img_path}")
            continue

        # 旋转到标准方向
        if ocam.rotate == 90:
            fisheye_img = cv2.rotate(fisheye_img, cv2.ROTATE_90_CLOCKWISE)
        elif ocam.rotate == 180:
            fisheye_img = cv2.rotate(fisheye_img, cv2.ROTATE_180)
        elif ocam.rotate == 270:
            fisheye_img = cv2.rotate(fisheye_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 重采样
        out_img = cv2.remap(fisheye_img, map_x, map_y,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0))
        out_img = cv2.flip(out_img, 1)

        # 保存（命名: 0001.jpg, 0002.jpg, ...）
        out_path = output_dir / f'{idx + 1:04d}.jpg'
        cv2.imwrite(str(out_path), out_img)

        if (idx + 1) % 10 == 0 or idx == total - 1:
            print(f"  进度: {idx + 1}/{total}")

    print(f"\n完成！共 {total} 张，输出目录: {output_dir}")


def _fix_wsl_path(path: str) -> str:
    """
    将 Windows 路径转为 WSL 路径（与 DataLoader._fix_windows_path 一致）。
    仅在使用 WSL 时转换：D:/xxx → /mnt/d/xxx
    """
    import platform

    # 检测是否在 WSL 中
    is_wsl = False
    if platform.system() == "Linux":
        try:
            with open('/proc/version', 'r') as f:
                version_info = f.read().lower()
                is_wsl = 'microsoft' in version_info or 'wsl' in version_info
        except Exception:
            pass

    if is_wsl:
        path = path.replace('\\', '/')
        if len(path) >= 2 and path[1] == ':':
            drive = path[0].lower()
            path = f"/mnt/{drive}{path[2:]}"

    return os.path.expanduser(path)


if __name__ == '__main__':
    main()
