"""
相机位姿可视化工具

从 COLMAP images.txt 提取相机中心，叠加到点云 PLY 中，输出可在 CloudCompare 打开的 PLY 文件。

相机中心用红色标记，前向箭头（+Z 方向）用黄色标记。

用法：
    python viz_camera_poses.py <images.txt> [pointcloud.ply] [--output out.ply] [--arrow_len 1.0]

示例：
    python viz_camera_poses.py E:\_cloud\20260507041844_raw\colmap_output\sparse\0\images.txt
    python viz_camera_poses.py "E:\_cloud\20260507041844_raw\colmap_output\sparse\0\images.txt" "E:\_cloud\20260507041844_raw\output_voxel\20260507041844_raw_rgb_0_voxel.las" --arrow_len 1.0

"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path


def _fix_path(p: str) -> str:
    import platform
    if platform.system() == 'Linux':
        try:
            with open('/proc/version') as f:
                if 'microsoft' in f.read().lower():
                    p = p.replace('\\', '/')
                    if len(p) >= 2 and p[1] == ':':
                        p = f"/mnt/{p[0].lower()}{p[2:]}"
        except Exception:
            pass
    return p


def quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
    """COLMAP 四元数 (qw, qx, qy, qz) → 3x3 旋转矩阵（world-to-camera）"""
    return np.array([
        [1-2*(qy**2+qz**2),  2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),  1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),  2*(qy*qz+qx*qw),  1-2*(qx**2+qy**2)],
    ], dtype=np.float64)


def read_images_txt(path: str):
    """
    读取 COLMAP images.txt，返回每张图的：
        center    : (3,) 相机中心在世界坐标系中的位置
        forward   : (3,) 相机前向（+Z in camera coords → world）
        camera_id : int  相机 ID（1=cam1，2=cam2）
        name      : str  图像文件名
    """
    entries = []
    with open(path) as f:
        lines = [l.rstrip() for l in f if not l.startswith('#') and l.strip()]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) >= 9:
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz     = float(parts[5]), float(parts[6]), float(parts[7])
            camera_id = int(parts[8]) if len(parts) > 8 else 1
            name      = parts[9]      if len(parts) > 9 else ''
            R = quat_to_rot(qw, qx, qy, qz)
            center  = -R.T @ np.array([tx, ty, tz])
            forward = R.T @ np.array([0.0, 0.0, 1.0])
            entries.append({'center': center, 'forward': forward,
                            'camera_id': camera_id, 'name': name})
        i += 1  # 空白 POINTS2D 行已被过滤，直接逐行读取
    return entries


def load_pointcloud(path: str) -> np.ndarray:
    """读取点云（ASCII PLY 或 LAS），返回 (N, 6) [X Y Z R G B]，RGB 范围 0~1。"""
    suffix = Path(path).suffix.lower()

    if suffix == '.las':
        import laspy
        las = laspy.read(path)
        x = np.array(las.x, dtype=np.float64)
        y = np.array(las.y, dtype=np.float64)
        z = np.array(las.z, dtype=np.float64)
        try:
            r = np.array(las.red,   dtype=np.float32) / 65535.0
            g = np.array(las.green, dtype=np.float32) / 65535.0
            b = np.array(las.blue,  dtype=np.float32) / 65535.0
        except AttributeError:
            r = g = b = np.full(len(x), 0.5, dtype=np.float32)
        return np.column_stack([x, y, z, r, g, b])

    elif suffix == '.ply':
        pts, rgb = [], []
        with open(path) as f:
            in_header = True
            has_color = False
            for line in f:
                line = line.strip()
                if in_header:
                    if 'red' in line or 'green' in line:
                        has_color = True
                    if line == 'end_header':
                        in_header = False
                else:
                    vals = list(map(float, line.split()))
                    pts.append(vals[:3])
                    if has_color and len(vals) >= 6:
                        rgb.append(vals[3:6])
                    else:
                        rgb.append([0.5, 0.5, 0.5])
        pts = np.array(pts, dtype=np.float64)
        rgb = np.array(rgb, dtype=np.float64)
        if rgb.max() > 1.01:
            rgb /= 255.0
        return np.column_stack([pts, rgb])

    else:
        raise ValueError(f"不支持的点云格式: {suffix}，仅支持 .ply 和 .las")


def write_ply(path: str, xyzrgb: np.ndarray) -> None:
    """写入 ASCII PLY，xyzrgb 形状 (N,6)，RGB 范围 0~1。"""
    n = len(xyzrgb)
    rgb = (xyzrgb[:, 3:6].clip(0, 1) * 255).astype(np.uint8)
    xyz = xyzrgb[:, :3]
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def main():
    parser = argparse.ArgumentParser(description='相机位姿可视化')
    parser.add_argument('images_txt',    help='COLMAP images.txt 路径')
    parser.add_argument('pointcloud',    nargs='?', default=None, help='背景点云 PLY（可选）')
    parser.add_argument('--output',      default=None, help='输出 PLY 路径（默认与 images.txt 同目录）')
    parser.add_argument('--arrow_len',   type=float, default=0.5, help='前向箭头长度（米），0=不画箭头')
    parser.add_argument('--pc_sample',   type=int, default=5000000, help='背景点云最大点数（随机降采样）')
    args = parser.parse_args()

    images_txt = _fix_path(args.images_txt)
    out_path   = args.output
    if out_path is None:
        out_path = str(Path(images_txt).parent / 'viz_cameras.ply')
    out_path = _fix_path(out_path)

    # ── 读取 images.txt ────────────────────────────
    print(f"读取 {images_txt} ...")
    entries = read_images_txt(images_txt)
    print(f"  共 {len(entries)} 张图像")
    if not entries:
        print("  [错误] 未读到任何相机位姿，检查文件格式。")
        sys.exit(1)

    centers   = np.array([e['center']    for e in entries])
    forwards  = np.array([e['forward']   for e in entries])
    cam_ids   = np.array([e['camera_id'] for e in entries])

    print(f"  X 范围: [{centers[:,0].min():.2f}, {centers[:,0].max():.2f}]")
    print(f"  Y 范围: [{centers[:,1].min():.2f}, {centers[:,1].max():.2f}]")
    print(f"  Z 范围: [{centers[:,2].min():.2f}, {centers[:,2].max():.2f}]")
    for cid in np.unique(cam_ids):
        mask = cam_ids == cid
        mean_fwd = forwards[mask].mean(axis=0)
        print(f"  camera_id={cid}: {mask.sum()} 张  平均朝向={mean_fwd.round(3)}")

    # 颜色约定：
    #   camera1 中心 = 红(1,0,0)   箭头 = 橙(1,0.5,0)
    #   camera2 中心 = 蓝(0,0,1)   箭头 = 青(0,1,1)
    CENTER_COLORS = {1: [1.0, 0.0, 0.0], 2: [0.0, 0.0, 1.0]}
    ARROW_COLORS  = {1: [1.0, 0.5, 0.0], 2: [0.0, 1.0, 1.0]}

    # ── 构建标记点 ─────────────────────────────────
    all_pts = []

    # 背景点云
    if args.pointcloud:
        pc_path = _fix_path(args.pointcloud)
        print(f"读取点云 {pc_path} ...")
        pc = load_pointcloud(pc_path)
        print(f"  {len(pc):,} 个点")
        if len(pc) > args.pc_sample:
            idx = np.random.choice(len(pc), args.pc_sample, replace=False)
            pc = pc[idx]
            print(f"  随机降采样至 {len(pc):,} 个点")
        all_pts.append(pc)

    # 相机中心（按 camera_id 着色）
    for cid, color in CENTER_COLORS.items():
        mask = cam_ids == cid
        if not mask.any():
            continue
        pts = np.zeros((mask.sum(), 6))
        pts[:, :3] = centers[mask]
        pts[:, 3:6] = color
        all_pts.append(pts)

    # 前向箭头（按 camera_id 着色）
    if args.arrow_len > 0:
        steps = np.linspace(0, args.arrow_len, 5)[1:]
        for cid, color in ARROW_COLORS.items():
            mask = cam_ids == cid
            if not mask.any():
                continue
            for s in steps:
                tip = centers[mask] + forwards[mask] * s
                pts = np.zeros((len(tip), 6))
                pts[:, :3] = tip
                pts[:, 3:6] = color
                all_pts.append(pts)

    combined = np.vstack(all_pts)

    # ── 写出 PLY ──────────────────────────────────
    print(f"\n写出 {out_path} ({len(combined):,} 个点)...")
    write_ply(out_path, combined)
    print("完成！用 CloudCompare 打开即可查看。")
    print()
    print("说明：")
    print("  红点/橙箭头  = camera1（左眼）中心 / 朝向")
    print("  蓝点/青箭头  = camera2（右眼）中心 / 朝向")
    if args.pointcloud:
        print("  其余彩色点  = 背景点云（已降采样）")


if __name__ == '__main__':
    main()
