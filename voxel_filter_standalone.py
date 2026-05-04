"""
独立体素滤波脚本
输入：单个 .las 或 .ply 点云文件
输出：滤波后的同格式文件（保存在 output_voxel 子目录，文件名加 _voxel 后缀）
"""

from pathlib import Path
import numpy as np

# ============================================================
# 参数设置
# ============================================================

INPUT_FILE = r"E:/_cloud/606/LAS_Rgb/20260329081547_rgb_0.las"

VOXEL_SIZE = 0.02       # 体素大小（米），越小保留越多点

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


def voxel_filter(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """均匀体素滤波：每个体素只保留距体素中心最近的真实点，颜色不变。"""
    xyz = points[:, :3]
    mins = xyz.min(axis=0)
    voxel_idx = np.floor((xyz - mins) / voxel_size).astype(np.int64)

    # 计算每个点到其体素中心的距离²
    voxel_centers = mins + (voxel_idx + 0.5) * voxel_size
    dist_sq = ((xyz - voxel_centers) ** 2).sum(axis=1)

    # 线性化体素 key
    max_idx = voxel_idx.max(axis=0) + 1
    flat_key = (voxel_idx[:, 0] * max_idx[1] * max_idx[2]
                + voxel_idx[:, 1] * max_idx[2]
                + voxel_idx[:, 2])

    # 按体素分组，组内取距离最小的点
    order = np.lexsort((dist_sq, flat_key))
    _, first = np.unique(flat_key[order], return_index=True)
    selected = order[first]

    return points[selected]


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


def save_las(points: np.ndarray, src_path: Path, out_path: Path):
    import laspy
    src = laspy.read(str(src_path))
    header = laspy.LasHeader(point_format=src.point_format, version=src.header.version)
    header.offsets = src.header.offsets
    header.scales = src.header.scales
    las_out = laspy.LasData(header=header)
    las_out.x = points[:, 0]
    las_out.y = points[:, 1]
    las_out.z = points[:, 2]
    try:
        las_out.red   = (np.clip(points[:, 3], 0, 1) * 65535).astype(np.uint16)
        las_out.green = (np.clip(points[:, 4], 0, 1) * 65535).astype(np.uint16)
        las_out.blue  = (np.clip(points[:, 5], 0, 1) * 65535).astype(np.uint16)
    except Exception:
        pass
    las_out.write(str(out_path))


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
elif suffix == '.ply':
    points = load_ply(input_path)
else:
    raise ValueError(f"不支持的格式: {suffix}，仅支持 .las 和 .ply")

print(f"原始点数: {len(points):,}")

filtered = voxel_filter(points, VOXEL_SIZE)
print(f"滤波后点数: {len(filtered):,}  (保留率 {len(filtered)/len(points)*100:.1f}%，体素大小 {VOXEL_SIZE}m)")

output_dir = input_path.parent / 'output_voxel'
output_dir.mkdir(exist_ok=True)
output_path = output_dir / (input_path.stem + '_voxel' + input_path.suffix)

if suffix == '.las':
    save_las(filtered, input_path, output_path)
else:
    save_ply(filtered, output_path)
print(f"已保存: {output_path}")
