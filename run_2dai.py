"""
双相机设备（2dai）数据处理入口

处理流程：
  1. 加载双相机 YAML 标定（cam1 + cam2 OCam 参数 + 外参）
  2. 加载 POS 文件（机体位姿）
  3. 对每一帧：
       a. 投影左眼（camera1）鱼眼图 → 透视图，保存为 {2i-1:04d}.jpg
       b. 投影右眼（camera2）鱼眼图 → 透视图，保存为 {2i  :04d}.jpg
       c. 用 extrinsicR1/T1 计算左相机 COLMAP 位姿
       d. 用 extrinsicR2/T2 计算右相机 COLMAP 位姿
  4. 加载手动滤波后的点云（PLY 或 LAS）
  5. 写出 cameras.txt / images.txt / points3D.txt

用法：
    直接修改下方 CONFIG 后运行：
        python run_2dai.py

注意：
    - 点云需先用 voxel_filter_standalone.py 手动滤波，再填写 pointcloud 路径
    - 解算完成后填写 pos_file 路径，解算前保持 None（只做投影预览）
"""

import os
import sys
import re
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from _2dai_ours.calibration    import load_dual_camera_yaml
from _2dai_ours.pose_calculator import load_pos_file, body_to_camera_pose, rotation_to_quaternion
from _2dai_ours.projector       import project_fisheye, build_pinhole_K
from _2dai_ours.colmap_writer   import write_cameras, write_images, write_points3d


# =============================================================================
# ★ 配置参数 - 在这里修改
# =============================================================================

CONFIG = {
    # ── 数据路径 ──────────────────────────────────────────
    'calib_yaml':   r'E:\_cloud\20260507041844_raw\CALIBRATION_CAMERA\CAMERA_886203051.yaml',
    'camera1_dir':  r'E:\_cloud\20260507041844_raw\CAMERA\camera1',
    'camera2_dir':  r'E:\_cloud\20260507041844_raw\CAMERA\camera2',

    # 解算完成后填写 POS 文件路径；None = 仅做投影，不生成 COLMAP 文件
    'pos_file':  r'E:\_cloud\20260507041844_raw\POS\20260509_215512\camera_pos.cam',

    # 手动滤波后的点云路径（.ply 或 .las）；None = 跳过 points3D.txt
    'pointcloud':  r'E:\_cloud\20260507041844_raw\output_voxel\20260507041844_raw_rgb_0_voxel.las',

    # ── 输出目录 ──────────────────────────────────────────
    'output_dir':   r'E:\_cloud\20260507041844_raw\colmap_output',

    # ── 投影参数 ──────────────────────────────────────────
    'fov_h_degrees':      60.0,   # 水平视场角（度），决定 fx
    'fov_v_degrees':      90.0,   # 垂直视场角（度），决定 fy；改为 None 则与水平相同
    'output_width':       2048,   # 输出图像宽度（像素）
    'output_height':      2048,   # 输出图像高度（像素）

    # ── 抽帧 ──────────────────────────────────────────────
    # 1 = 每帧都投影，2 = 每隔一帧，N = 每 N 帧取一帧
    'skip_frames':        1,
}


# =============================================================================
# 路径工具
# =============================================================================

def _fix_path(p: str) -> str:
    """Windows 路径 → 当前系统路径（WSL 兼容）。"""
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


def _sort_key(path: Path) -> int:
    m = re.search(r'\d+', path.stem)
    return int(m.group()) if m else 0


# =============================================================================
# 点云加载（支持 PLY ASCII / LAS）
# =============================================================================

def _load_pointcloud(path: str) -> np.ndarray:
    path = _fix_path(path)
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == '.ply':
        pts, rgb = [], []
        with open(p) as f:
            in_header = True
            n_vertex = 0
            has_color = False
            for line in f:
                line = line.strip()
                if in_header:
                    if line.startswith('element vertex'):
                        n_vertex = int(line.split()[-1])
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
                        rgb.append([128, 128, 128])
        pts = np.array(pts, dtype=np.float64)
        rgb = np.array(rgb, dtype=np.float64)
        if rgb.max() > 1.01:
            rgb = rgb / 255.0
        return np.column_stack([pts, rgb])

    elif suffix == '.las':
        import laspy
        las = laspy.read(str(p))
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

    else:
        raise ValueError(f"不支持的点云格式: {suffix}，仅支持 .ply 和 .las")


# =============================================================================
# 主流程
# =============================================================================

def main():
    cfg = CONFIG

    calib_yaml  = _fix_path(cfg['calib_yaml'])
    cam1_dir    = Path(_fix_path(cfg['camera1_dir']))
    cam2_dir    = Path(_fix_path(cfg['camera2_dir']))
    output_dir  = Path(_fix_path(cfg['output_dir']))
    fov_h       = cfg['fov_h_degrees']
    fov_v       = cfg['fov_v_degrees']
    out_w       = cfg['output_width']
    out_h       = cfg['output_height']
    skip        = cfg['skip_frames']

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / 'images'
    images_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("双相机（2dai）数据处理")
    print("=" * 60)

    # ──────────────────────────────────────
    # 1. 加载标定
    # ──────────────────────────────────────
    print("\n[1] 加载双相机标定...")
    cam1_params, cam2_params = load_dual_camera_yaml(calib_yaml)
    K1 = build_pinhole_K(out_w, out_h, fov_h, fov_v)
    K2 = build_pinhole_K(out_w, out_h, fov_h, fov_v)
    print(f"  cam1 主点: ({cam1_params['xc']:.1f}, {cam1_params['yc']:.1f})")
    print(f"  cam2 主点: ({cam2_params['xc']:.1f}, {cam2_params['yc']:.1f})")
    print(f"  透视内参 fx={K1[0,0]:.1f} fy={K1[1,1]:.1f}  fov_h={fov_h}° fov_v={fov_v}°  {out_w}×{out_h}")

    # ──────────────────────────────────────
    # 2. 加载图像列表
    # ──────────────────────────────────────
    print("\n[2] 扫描图像目录...")
    imgs1 = sorted(cam1_dir.glob('*.jpg'), key=_sort_key)
    imgs2 = sorted(cam2_dir.glob('*.jpg'), key=_sort_key)
    if len(imgs1) != len(imgs2):
        print(f"  [警告] camera1 有 {len(imgs1)} 张，camera2 有 {len(imgs2)} 张，数量不一致！")
    n_total = min(len(imgs1), len(imgs2))
    # 按 skip 抽帧
    frame_indices = list(range(0, n_total, skip))
    print(f"  共 {n_total} 帧，抽帧间隔={skip}，实际处理 {len(frame_indices)} 帧")

    # ──────────────────────────────────────
    # 3. 投影
    # ──────────────────────────────────────
    print(f"\n[3] 鱼眼投影 → {images_dir.name}/")
    projected_names = []  # [(cam1_name, cam2_name), ...]

    for seq, fi in enumerate(frame_indices):
        img1_path = imgs1[fi]
        img2_path = imgs2[fi]

        # 输出文件名：奇数=左相机，偶数=右相机
        name1 = f"{seq * 2 + 1:04d}.jpg"
        name2 = f"{seq * 2 + 2:04d}.jpg"

        # 左相机
        src1 = cv2.imread(str(img1_path))
        if src1 is None:
            print(f"  [警告] 读取失败: {img1_path.name}")
            continue
        proj1 = project_fisheye(src1, cam1_params, out_w, out_h, fov_h, fov_v)
        cv2.imwrite(str(images_dir / name1), proj1)

        # 右相机
        src2 = cv2.imread(str(img2_path))
        if src2 is None:
            print(f"  [警告] 读取失败: {img2_path.name}")
            continue
        proj2 = project_fisheye(src2, cam2_params, out_w, out_h, fov_h, fov_v)
        cv2.imwrite(str(images_dir / name2), proj2)

        projected_names.append((name1, name2))

        if (seq + 1) % 50 == 0 or seq == len(frame_indices) - 1:
            print(f"  {seq + 1}/{len(frame_indices)} 帧完成...")

    print(f"  ✓ 投影完成，共 {len(projected_names) * 2} 张图像")

    # ──────────────────────────────────────
    # 4. 位姿 + COLMAP 文件
    # ──────────────────────────────────────
    if cfg['pos_file'] is None:
        print("\n[4] pos_file 未设置，跳过 COLMAP 文件生成。")
        print("    解算完成后，填写 CONFIG['pos_file'] 再次运行即可。")
    else:
        print(f"\n[4] 加载 POS 文件并计算位姿...")
        pos_path = _fix_path(cfg['pos_file'])
        body_poses = load_pos_file(pos_path)

        # 用图像原始编号（1.jpg, 2.jpg...）匹配位姿
        pose_dict = {p['image_name']: p for p in body_poses}

        image_entries = []
        image_id = 1

        for seq, fi in enumerate(frame_indices):
            raw_name = imgs1[fi].name   # e.g. "123.jpg"（camera1 和 camera2 同名）
            if raw_name not in pose_dict:
                print(f"  [警告] 未找到位姿: {raw_name}，跳过该帧")
                continue

            pose = pose_dict[raw_name]
            R_bw = pose['R_bw']
            t_bw = pose['t_bw']

            name1, name2 = projected_names[seq]

            # 左相机位姿
            R_wc1, t_wc1 = body_to_camera_pose(
                R_bw, t_bw,
                cam1_params['extrinsic_R'],
                cam1_params['extrinsic_T'],
            )
            qw1, qx1, qy1, qz1 = rotation_to_quaternion(R_wc1)
            image_entries.append({
                'image_id': image_id, 'camera_id': 1, 'name': name1,
                'qw': qw1, 'qx': qx1, 'qy': qy1, 'qz': qz1,
                'tx': t_wc1[0], 'ty': t_wc1[1], 'tz': t_wc1[2],
            })
            image_id += 1

            # 右相机位姿
            R_wc2, t_wc2 = body_to_camera_pose(
                R_bw, t_bw,
                cam2_params['extrinsic_R'],
                cam2_params['extrinsic_T'],
            )
            qw2, qx2, qy2, qz2 = rotation_to_quaternion(R_wc2)
            image_entries.append({
                'image_id': image_id, 'camera_id': 2, 'name': name2,
                'qw': qw2, 'qx': qx2, 'qy': qy2, 'qz': qz2,
                'tx': t_wc2[0], 'ty': t_wc2[1], 'tz': t_wc2[2],
            })
            image_id += 1

        print(f"  生成 {len(image_entries)} 条位姿（{len(image_entries)//2} 帧 × 2 相机）")

        colmap_dir = output_dir / 'sparse' / '0'
        colmap_dir.mkdir(parents=True, exist_ok=True)

        write_cameras(K1, K2, out_w, out_h, str(colmap_dir))
        write_images(image_entries, str(colmap_dir))

        # ── 点云 ────────────────────────────────
        if cfg['pointcloud'] is None:
            print("\n[5] pointcloud 未设置，跳过 points3D.txt。")
            print("    用 voxel_filter_standalone.py 滤波后，填写 CONFIG['pointcloud'] 再次运行。")
        else:
            print(f"\n[5] 加载点云并写入 points3D.txt...")
            pc = _load_pointcloud(cfg['pointcloud'])
            print(f"  点云: {len(pc):,} 个点")
            write_points3d(pc, str(colmap_dir))

        print(f"\n  COLMAP 输出目录: {colmap_dir}")

    # ──────────────────────────────────────
    # 完成
    # ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("完成！")
    print(f"  投影图像: {images_dir}")
    if cfg['pos_file']:
        print(f"  COLMAP:   {output_dir / 'sparse' / '0'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
