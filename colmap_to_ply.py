"""
将 COLMAP points3D.txt 转换为 .ply 点云文件

用法：
    python colmap_to_ply.py E:\_cloud\Colmap\sparse\0_20260509_1016\points3D.txt

输出：同目录下的 points3D.ply
"""

import sys
import numpy as np
from pathlib import Path


def load_points3d(path: Path) -> np.ndarray:
    xs, ys, zs, rs, gs, bs = [], [], [], [], [], []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            # POINT3D_ID X Y Z R G B ERROR [TRACK...]
            xs.append(float(parts[1]))
            ys.append(float(parts[2]))
            zs.append(float(parts[3]))
            rs.append(int(parts[4]))
            gs.append(int(parts[5]))
            bs.append(int(parts[6]))
    return np.column_stack([xs, ys, zs, rs, gs, bs])


def save_ply(points: np.ndarray, path: Path):
    n = len(points)
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        xyz = points[:, :3]
        rgb = points[:, 3:6].astype(np.uint8)
        for i in range(n):
            f.write(f"{xyz[i,0]:.6f} {xyz[i,1]:.6f} {xyz[i,2]:.6f} "
                    f"{rgb[i,0]} {rgb[i,1]} {rgb[i,2]}\n")


def main():
    if len(sys.argv) < 2:
        raw = input("请输入 points3D.txt 路径: ")
    else:
        raw = sys.argv[1]

    src = Path(raw.strip().strip('"').strip("'"))
    if not src.exists():
        print(f"[错误] 找不到文件: {src}")
        sys.exit(1)

    print(f"读取 {src} ...")
    points = load_points3d(src)
    print(f"点数：{len(points):,}")

    dst = src.with_suffix('.ply')
    print(f"写入 {dst} ...")
    save_ply(points, dst)
    print("完成！")


if __name__ == '__main__':
    main()
