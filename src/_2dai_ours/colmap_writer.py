"""
COLMAP 格式文件写入器

输出：
    cameras.txt  — 两个 PINHOLE 相机
    images.txt   — 按 左相机/右相机 交替排列
    points3D.txt — 从手动滤波后的点云生成（空 TRACK）
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional


def write_cameras(cam1_K: np.ndarray, cam2_K: np.ndarray,
                  width: int, height: int,
                  out_dir: str) -> None:
    """
    写入 cameras.txt（两个 PINHOLE 相机）。

    Args:
        cam1_K: (3,3) 相机1 内参矩阵
        cam2_K: (3,3) 相机2 内参矩阵
        width, height: 投影图像尺寸（两个相机相同）
        out_dir: 输出目录
    """
    path = Path(out_dir) / 'cameras.txt'
    with open(path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 2\n")
        for cam_id, K in [(1, cam1_K), (2, cam2_K)]:
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            f.write(f"{cam_id} PINHOLE {width} {height} "
                    f"{fx:.10f} {fy:.10f} {cx:.10f} {cy:.10f}\n")
    print(f"  ✓ cameras.txt ({out_dir})")


def write_images(image_entries: List[Dict], out_dir: str) -> None:
    """
    写入 images.txt。

    Args:
        image_entries: 每个 dict 包含：
            image_id  int
            qw, qx, qy, qz  float  (world-to-camera 四元数)
            tx, ty, tz       float  (world-to-camera 平移)
            camera_id  int   (1 或 2)
            name       str   (图像文件名)
        out_dir: 输出目录
    """
    path = Path(out_dir) / 'images.txt'
    with open(path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_entries)}\n")
        for e in image_entries:
            f.write(f"{e['image_id']} "
                    f"{e['qw']:.10f} {e['qx']:.10f} {e['qy']:.10f} {e['qz']:.10f} "
                    f"{e['tx']:.10f} {e['ty']:.10f} {e['tz']:.10f} "
                    f"{e['camera_id']} {e['name']}\n")
            f.write("\n")  # 空行（无 POINTS2D）
    print(f"  ✓ images.txt ({len(image_entries)} 张)")


def write_points3d(pointcloud: np.ndarray, out_dir: str) -> None:
    """
    写入 points3D.txt（空 TRACK）。

    Args:
        pointcloud: (N, 6) ndarray，列为 X Y Z R G B
                    RGB 范围 0~1 (float) 或 0~255 (uint8/int)，自动判断
        out_dir: 输出目录
    """
    path = Path(out_dir) / 'points3D.txt'
    n = len(pointcloud)

    # 判断 RGB 范围并统一转为 0~255 整数
    rgb = pointcloud[:, 3:6]
    if rgb.max() <= 1.01:
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
    else:
        rgb = rgb.clip(0, 255).astype(np.uint8)

    xyz = pointcloud[:, :3]

    with open(path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
                "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {n}\n")
        for i in range(n):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            f.write(f"{i+1} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.0\n")

    print(f"  ✓ points3D.txt ({n:,} 个点)")
