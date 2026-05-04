"""
外参诊断脚本：将 points3D.ply 中的点投影到图像，验证外参是否正确。
输出: output/<scene>/debug_projection_XXX.jpg
"""
import numpy as np
import cv2
from pathlib import Path

# ============================================================
# 修改这里：指向你的场景输出目录
# ============================================================
OUTPUT_DIR = r"D:\Li-GS_data_process\output\20260421084017_ours-3"
IMAGES_SUBDIR = r"projected_images\images"  # 训练用图像目录
MAX_POINTS = 120000   # 最多投影多少个点（太多会很慢）
TEST_FRAME_INDICES = [0, 25,50]  # 测试第几帧（可改）


def read_cameras_txt(path):
    cameras = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            cam_id = int(parts[0])
            w, h = int(parts[2]), int(parts[3])
            fx, fy, cx, cy = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            cameras[cam_id] = {'w': w, 'h': h, 'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
    return cameras


def read_images_txt(path):
    images = []
    with open(path) as f:
        lines = [l.rstrip('\n') for l in f if not l.startswith('#') and l.strip()]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) >= 10:
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            cam_id = int(parts[8])
            name = parts[9]
            R = _quat_to_rot(qw, qx, qy, qz)
            images.append({'R': R, 't': np.array([tx, ty, tz]), 'cam_id': cam_id, 'name': name})
        i += 2
    return images


def _quat_to_rot(qw, qx, qy, qz):
    return np.array([
        [1-2*(qy*qy+qz*qz),   2*(qx*qy-qw*qz),   2*(qx*qz+qw*qy)],
        [  2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz),   2*(qy*qz-qw*qx)],
        [  2*(qx*qz-qw*qy),   2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)]
    ])


def read_ply_ascii(path, max_pts):
    pts, cols = [], []
    with open(path, 'r') as f:
        in_header = True
        for line in f:
            if in_header:
                if line.strip() == 'end_header':
                    in_header = False
                continue
            if len(pts) >= max_pts:
                break
            v = line.strip().split()
            if len(v) >= 6:
                pts.append([float(v[0]), float(v[1]), float(v[2])])
                cols.append([int(v[3]), int(v[4]), int(v[5])])
    return np.array(pts, dtype=np.float64), np.array(cols, dtype=np.uint8)


def project_and_draw(img, points, colors, R, t, fx, fy, cx, cy, w, h):
    P_cam = (R @ points.T).T + t       # (N, 3) world → camera
    front = P_cam[:, 2] > 0.1          # 只投影相机前方的点
    u = fx * (P_cam[:, 0] / np.where(front, P_cam[:, 2], 1)) + cx
    v = fy * (P_cam[:, 1] / np.where(front, P_cam[:, 2], 1)) + cy

    result = img.copy()
    n_drawn = 0
    for i in np.where(front)[0]:
        ui, vi = int(round(u[i])), int(round(v[i]))
        if 0 <= ui < w and 0 <= vi < h:
            r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])
            cv2.circle(result, (ui, vi), 4, (b, g, r), -1)  # BGR for OpenCV
            n_drawn += 1
    return result, n_drawn, int(front.sum())


# ============================================================
# 主流程
# ============================================================
output_dir = Path(OUTPUT_DIR)
cameras = read_cameras_txt(output_dir / 'cameras.txt')
images  = read_images_txt(output_dir  / 'images.txt')
points, colors = read_ply_ascii(output_dir / 'points3D.ply', MAX_POINTS)

print(f"加载: {len(images)} 帧位姿, {len(points)} 个点")

for idx in TEST_FRAME_INDICES:
    if idx >= len(images):
        continue
    info = images[idx]
    cam  = cameras[info['cam_id']]
    img_path = output_dir / IMAGES_SUBDIR / info['name']

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[!] 无法读取图像: {img_path}")
        continue

    result, n_drawn, n_front = project_and_draw(
        img, points, colors,
        info['R'], info['t'],
        cam['fx'], cam['fy'], cam['cx'], cam['cy'],
        cam['w'], cam['h']
    )

    out_name = f"debug_projection_{idx:04d}.jpg"
    cv2.imwrite(str(output_dir / out_name), result)
    print(f"帧 {idx:4d} ({info['name']:12s}): 前方 {n_front} 点, 投影到图像内 {n_drawn} 点 → {out_name}")

print("\n请打开输出的 debug_projection_*.jpg 检查：")
print("  ✓ 彩色点落在对应物体/地面上 → 外参正确")
print("  ✗ 彩色点和图像内容完全不对应 → 外参有误")
