"""
将 cloud.txt（X Y Z R G B）转换为 COLMAP points3D.txt 格式（空 TRACK）
用于 3DGS LiDAR 点云初始化验证（不含深度监督）

用法：
    python cloud_txt_to_points3d.py E:\_cloud\Colmap\sparse\0\cloud.txt
    python cloud_txt_to_points3d.py E:\_cloud\Colmap\sparse\0\cloud.txt --voxel 0.05
    python cloud_txt_to_points3d.py E:\_cloud\Colmap\sparse\0\cloud.txt --sample 200000
    python cloud_txt_to_points3d.py E:\_cloud\Colmap\sparse\0\cloud.txt --no_filter

输出：cloud.txt 同目录下的 points3D.txt（自动备份原有的为 points3D_sfm_backup.txt）
"""

import argparse
import sys
import numpy as np
from pathlib import Path


def voxel_downsample(points, voxel_size):
    """每个体素只保留距体素中心最近的真实点，颜色不变。"""
    xyz = points[:, :3]
    mins = xyz.min(axis=0)
    voxel_idx = np.floor((xyz - mins) / voxel_size).astype(np.int64)

    voxel_centers = mins + (voxel_idx + 0.5) * voxel_size
    dist_sq = ((xyz - voxel_centers) ** 2).sum(axis=1)

    max_idx = voxel_idx.max(axis=0) + 1
    flat_key = (voxel_idx[:, 0] * max_idx[1] * max_idx[2]
                + voxel_idx[:, 1] * max_idx[2]
                + voxel_idx[:, 2])

    order = np.lexsort((dist_sq, flat_key))
    _, first = np.unique(flat_key[order], return_index=True)
    return points[order[first]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cloud', help='cloud.txt 文件路径')
    parser.add_argument('--voxel', type=float, default=None,
                        help='体素滤波大小（米），推荐 0.05~0.1，不指定则用 --sample')
    parser.add_argument('--sample', type=int, default=300000,
                        help='随机采样点数，默认 300000（--voxel 优先）')
    parser.add_argument('--no_filter', action='store_true',
                        help='跳过所有降采样，直接转换全部点')
    args = parser.parse_args()

    cloud_path = Path(args.cloud.strip().strip('"'))
    if not cloud_path.exists():
        print(f"[错误] 找不到文件: {cloud_path}")
        sys.exit(1)

    out_dir = cloud_path.parent
    backup_path = out_dir / 'points3D_sfm_backup.txt'
    out_path = out_dir / 'points3D.txt'

    # 读取 cloud.txt
    print(f"读取 {cloud_path} ...")
    points = np.loadtxt(cloud_path, dtype=np.float64)
    print(f"原始点数：{len(points):,}")

    # 降采样
    if args.no_filter:
        print("跳过降采样，保留全部点。")
    elif args.voxel:
        print(f"体素滤波（voxel={args.voxel}m）...")
        points = voxel_downsample(points, args.voxel)
        print(f"滤波后点数：{len(points):,}")
    elif len(points) > args.sample:
        print(f"随机采样至 {args.sample:,} 点...")
        idx = np.random.choice(len(points), args.sample, replace=False)
        points = points[idx]
        print(f"采样后点数：{len(points):,}")

    # 备份原有 points3D.txt
    if out_path.exists():
        out_path.rename(backup_path)
        print(f"已备份原 points3D.txt → {backup_path.name}")

    # 写入新的 points3D.txt
    print(f"写入 {out_path} ...")
    n = len(points)
    with open(out_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {n}\n")
        for i, p in enumerate(points, 1):
            x, y, z = p[0], p[1], p[2]
            r, g, b = int(p[3]), int(p[4]), int(p[5])
            f.write(f"{i} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.0\n")

    print(f"完成！共写入 {n:,} 个点。")
    print(f"\n训练命令（去掉 --depth_loss）：")
    print(f"python simple_trainer.py default ^")
    print(f"    --data_dir E:\\_cloud\\Colmap ^")
    print(f"    --colmap_dir sparse/0 ^")
    print(f"    --result_dir E:\\_cloud\\Colmap\\results ^")
    print(f"    --max_steps 30000 ^")
    print(f"    --strategy.refine_stop_iter 25000 ^")
    print(f"    --use_bilateral_grid ^")
    print(f"    --random_bkgd")


if __name__ == '__main__':
    main()
