"""
独立点云格式转换脚本
输入：单个 .las 或 .ply 点云文件
输出：转换后的文件（.las → .ply，.ply → .las），保存在 output_trans 子目录
"""

from pathlib import Path
import numpy as np

# ============================================================
# 参数设置
# ============================================================

INPUT_FILE = r"E:/_cloud/20260421084017_ours-3/LAS_Rgb/20260421084017_ours-3_rgb_0.las"

# ============================================================


def load_las(path: Path) -> np.ndarray:
    import laspy
    las = laspy.read(str(path))
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


def load_ply(path: Path) -> np.ndarray:
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pcd.has_colors():
        cols = np.asarray(pcd.colors, dtype=np.float32)
    else:
        cols = np.full((len(pts), 3), 0.5, dtype=np.float32)
    return np.column_stack([pts, cols])


def save_ply(points: np.ndarray, path: Path):
    n = len(points)
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        colors = (np.clip(points[:, 3:6], 0, 1) * 255).astype(np.uint8)
        for i in range(n):
            x, y, z = points[i, :3]
            r, g, b = colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def save_las(points: np.ndarray, path: Path):
    import laspy
    xyz = points[:, :3]
    scales = np.array([0.001, 0.001, 0.001])
    offsets = xyz.min(axis=0)
    header = laspy.LasHeader(point_format=2, version="1.2")
    header.offsets = offsets
    header.scales = scales
    las_out = laspy.LasData(header=header)
    las_out.x = xyz[:, 0]
    las_out.y = xyz[:, 1]
    las_out.z = xyz[:, 2]
    las_out.red   = (np.clip(points[:, 3], 0, 1) * 65535).astype(np.uint16)
    las_out.green = (np.clip(points[:, 4], 0, 1) * 65535).astype(np.uint16)
    las_out.blue  = (np.clip(points[:, 5], 0, 1) * 65535).astype(np.uint16)
    las_out.write(str(path))


# ============================================================
# 主流程
# ============================================================

input_path = Path(INPUT_FILE)
if not input_path.exists():
    raise FileNotFoundError(f"文件不存在: {input_path}")

suffix = input_path.suffix.lower()
print(f"输入: {input_path.name}")

if suffix == '.las':
    points = load_las(input_path)
    out_suffix = '.ply'
elif suffix == '.ply':
    points = load_ply(input_path)
    out_suffix = '.las'
else:
    raise ValueError(f"不支持的格式: {suffix}，仅支持 .las 和 .ply")

print(f"点数: {len(points):,}")

output_dir = input_path.parent / 'output_trans'
output_dir.mkdir(exist_ok=True)
output_path = output_dir / (input_path.stem + out_suffix)

if out_suffix == '.ply':
    save_ply(points, output_path)
else:
    save_las(points, output_path)

print(f"已保存: {output_path}")
