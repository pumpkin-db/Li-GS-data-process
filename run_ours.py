"""
Li-GS 自采数据集处理入口脚本

处理流程：
1. DataLoader 加载原始数据（点云、位姿、相机内参、图像列表）
2. KeyFrameSelector 筛选关键帧（146张 → ~30张）  #已禁用
3. OCamModel + FisheyeProjector 进行单面投影（30张 → 30张透视图，仅 back 面）
4. DynamicVoxelFilter 对点云进行动态体素滤波
5. COLMAPExporter 导出 COLMAP 格式

使用方法：
    python run_ours.py --scene 20260418035854_ours
    
    或直接修改本文件的 CONFIG 字典后运行：
    python run_ours.py
"""

import argparse
import os
import sys
import yaml
import numpy as np
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ours import (
    DataLoader,
    KeyFrameSelector,
    OCamModel,
    FisheyeProjector,
    DynamicVoxelFilter,
    COLMAPExporter,
    rotation_matrix_to_quaternion
)


# =============================================================================
# 配置参数 - 直接在这里修改参数，无需命令行
# =============================================================================

CONFIG = {
    # 场景设置
    'scene': '606_ours',  # 场景名称
    'config_path': None,              # 配置文件路径（None 表示使用默认）
    'output_dir': None,               # 输出目录（None 表示使用默认 output/{scene}）
                                    #例如：r'D:\Li-GS_data_process\output\我的输出目录'
    
    # 关键帧筛选参数 # 已禁用！！！
    'keyframe': {
        'translation_threshold': 0.05,  # 位移阈值（米）
        'rotation_threshold': 5.0,       # 旋转阈值（度）
        'fixed_interval': None,          # 固定间隔采样（如 5 表示每5帧取1帧，None 表示使用阈值策略）
        'max_frames': 200,               # 最大关键帧数量
    },
    
    # 投影参数（仅 back 面）
    'projection': {
        'resolution': (3600, 3600),      # 投影图像分辨率 (width, height)
        'fov_degrees': 90.0,             # 视场角（度）
    },
    
    # 动态体素滤波参数
    'voxel_filter': {
        'feature_rich_ratio': 0.0005,      # Feature-rich 区域比例（前 5%）
        'voxel_size_rich': 0.02,         # Feature-rich 区域体素大小（米）
        'voxel_size_normal': 0.2,       # 普通区域体素大小（米）
        'kde_bandwidth': 300.0,           # KDE 带宽（像素）
        'kde_resize_factor': 0.125,       # KDE 图像缩放因子（加速计算）
        'sift_nfeatures': 500,             # SIFT 最大特征点数（0 表示无限制）
        'max_workers': 4,                # 并行线程数
        'depth_tolerance': 0.1,         # 深度容差（米），0 表示禁用遮挡过滤
    },
}


# =============================================================================
# 辅助函数
# =============================================================================

def load_config(scene_name: str, config_path: str = None) -> dict:
    """加载场景配置"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'ours_scenes.yaml')
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if scene_name not in config.get('scenes', {}):
        raise ValueError(f"场景 '{scene_name}' 未在配置中找到。可用场景: {list(config.get('scenes', {}).keys())}")
    
    return config['scenes'][scene_name]


# =============================================================================
# 主处理流程
# =============================================================================

def process_scene(config: dict = None):
    """
    处理单个场景
    
    Args:
        config: 配置字典（默认使用 CONFIG）
    """
    if config is None:
        config = CONFIG
    
    # 提取配置
    scene_name = config['scene']
    config_path = config['config_path']
    output_dir = config['output_dir']
    kf_cfg = config['keyframe']
    proj_cfg = config['projection']
    vf_cfg = config['voxel_filter']
    
    # 加载配置
    print(f"=" * 60)
    print(f"Li-GS 自采数据处理")
    print(f"=" * 60)
    print(f"场景: {scene_name}")
    
    scene_config = load_config(scene_name, config_path)
    
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'output', scene_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 获取投影分辨率
    projection_res = proj_cfg['resolution']
    print(f"投影分辨率: {projection_res[0]}x{projection_res[1]}")
    
    # ============================================================
    # 步骤 1: 加载原始数据
    # ============================================================
    print(f"\n[1/5] 加载原始数据...")
    print(f"  - 点云: {scene_config['pointcloud']}")
    print(f"  - 图像: {scene_config['images']}")
    print(f"  - 位姿: {scene_config['poses']}")
    print(f"  - 相机: {scene_config['camera_intrinsic']}")
    
    loader = DataLoader(scene_config)
    data = loader.load_all()
    
    print(f"  ✓ 点云点数: {len(data['pointcloud'])}")
    print(f"  ✓ 图像数量: {len(data['images'])}")
    print(f"  ✓ 位姿数量: {len(data['poses'])}")
    
    # ============================================================
    # 步骤 2: 关键帧筛选（已禁用，使用所有帧）
    # ============================================================
    print(f"\n[2/5] 关键帧筛选（已禁用，使用所有帧）...")
    
    # 保留关键帧筛选代码但不使用，直接使用所有帧
    # selector = KeyFrameSelector(...)
    # keyframe_indices = selector.select(data['poses'], data['images'])
    
    # 使用所有帧
    keyframe_indices = list(range(len(data['poses'])))
    keyframe_paths = [data['images'][i] for i in keyframe_indices]
    keyframe_poses = [data['poses'][i] for i in keyframe_indices]
    
    print(f"  ✓ 使用所有帧: {len(keyframe_indices)} 帧")

    # 应用坐标系修正：经 debug_projection.py 验证的经验值
    # Ry_90 修正相机朝向，Rz_m90 修正 roll 90° 误差
    # extrinsicR1 尝试过（Rz_180 @ extrinsicR1），但引入 ~1.9° 残差，效果更差
    Ry_90  = np.array([[ 0,  0,  1], [ 0,  1,  0], [-1,  0,  0]], dtype=np.float64)
    Rz_m90 = np.array([[ 0,  1,  0], [-1,  0,  0], [ 0,  0,  1]], dtype=np.float64)
    correction = Ry_90 @ Rz_m90

    # 精细旋转补偿参数（可手动调整）
    # 正值 = 点云向右/向下偏，需要往反方向修正；负值反之
    _fine_px_right = 28.0    # 点云水平偏移（像素，正=往左修正，负=往右修正）
    _fine_px_down  = 55.0    # 点云垂直偏移（像素，正=往上修正，负=往下修正）
    _fine_roll_deg = -1.0    # 点云滚转（度，负=逆时针修正，正=顺时针修正）

    _fx = projection_res[0] / (2.0 * np.tan(np.radians(proj_cfg['fov_degrees'] / 2.0)))
    _theta_ry = np.arctan(_fine_px_right / _fx)
    _theta_rx = np.arctan(_fine_px_down  / _fx)
    _theta_rz = np.radians(_fine_roll_deg)
    Ry_fine = np.array([[ np.cos(_theta_ry), 0, np.sin(_theta_ry)],
                         [ 0,                 1, 0                 ],
                         [-np.sin(_theta_ry), 0, np.cos(_theta_ry)]], dtype=np.float64)
    Rx_fine = np.array([[1, 0,                  0                 ],
                         [0, np.cos(_theta_rx), -np.sin(_theta_rx)],
                         [0, np.sin(_theta_rx),  np.cos(_theta_rx)]], dtype=np.float64)
    Rz_fine = np.array([[ np.cos(_theta_rz), -np.sin(_theta_rz), 0],
                         [ np.sin(_theta_rz),  np.cos(_theta_rz), 0],
                         [ 0,                  0,                  1]], dtype=np.float64)
    correction = correction @ Ry_fine @ Rx_fine @ Rz_fine

    # 相机相对 LiDAR 的物理安装偏移（来自 extrinsicT1 标定值）
    # P_cam = extrinsicR1 @ P_lidar + extrinsicT1
    # => 相机原点在 LiDAR 坐标系中：cam_offset = -extrinsicR1.T @ extrinsicT1
    ext_R = data['ocam']['extrinsic_R']
    ext_T = data['ocam']['extrinsic_T']
    if ext_R is not None and ext_T is not None:
        cam_offset_in_lidar = -ext_R.T @ ext_T  # 约 [0, -0.057, -0.066] m
        # Z 分量（深度方向）旋转到世界坐标后会引入水平偏差，只保留 Y 分量
        cam_offset_in_lidar[0] = 0.0
        cam_offset_in_lidar[2] = 0.0
        print(f"  ✓ 相机安装偏移（仅Y分量）: "
              f"Y={cam_offset_in_lidar[1]*100:.1f}cm")
    else:
        cam_offset_in_lidar = None
        print(f"  ⚠ extrinsicT1 不可用，跳过相机位置偏移修正")

    corrected_poses = []
    for pose in keyframe_poses:
        p = dict(pose)
        p['rotation'] = pose['rotation'] @ correction
        if cam_offset_in_lidar is not None:
            # 将 LiDAR 坐标系偏移转到世界坐标系，修正相机位置
            p['position'] = pose['position'] + pose['rotation'] @ cam_offset_in_lidar
        corrected_poses.append(p)
    keyframe_poses = corrected_poses
    print(f"  ✓ 已应用 Ry(+90°) @ Rz(-90°) 修正 + extrinsicT1 位置偏移")

    # ============================================================
    # 步骤 3: 鱼眼 5 面投影
    # ============================================================
    print(f"\n[3/5] 鱼眼投影...")
    print(f"  - 输入关键帧: {len(keyframe_paths)}")
    print(f"  - 输出透视图: {len(keyframe_paths)} × 1 = {len(keyframe_paths)} (仅 back 面)")
    
    # 创建 OCamCalib 模型
    ocam_model = OCamModel(data['ocam'])
    
    # 创建投影器
    projector = FisheyeProjector(
        ocam=ocam_model,
        output_size=projection_res,
        face_fov_deg=proj_cfg['fov_degrees']
    )
    
    # 投影目录
    projection_dir = os.path.join(output_dir, 'projected_images')
    os.makedirs(projection_dir, exist_ok=True)
    
    # 执行投影
    projection_info = projector.project_keyframes(
        poses=keyframe_poses,
        image_paths=keyframe_paths,
        output_dir=Path(projection_dir)
    )
    
    print(f"  ✓ 投影完成，保存至: {projection_dir}")
    
    # ============================================================
    # 步骤 4: 动态体素滤波
    # ============================================================
    print(f"\n[4/5] 动态体素滤波...")
    print(f"  - Feature-rich 比例: {vf_cfg['feature_rich_ratio']}")
    print(f"  - 小体素大小: {vf_cfg['voxel_size_rich']}m")
    print(f"  - 大体素大小: {vf_cfg['voxel_size_normal']}m")
    print(f"  - 并行线程数: {vf_cfg['max_workers']}")
    
    voxel_filter = DynamicVoxelFilter(
        feature_rich_ratio=vf_cfg['feature_rich_ratio'],
        voxel_size_rich=vf_cfg['voxel_size_rich'],
        voxel_size_normal=vf_cfg['voxel_size_normal'],
        kde_bandwidth=vf_cfg['kde_bandwidth'],
        kde_resize_factor=vf_cfg['kde_resize_factor'],
        sift_nfeatures=vf_cfg['sift_nfeatures'],
        max_workers=vf_cfg['max_workers'],
        depth_tolerance=vf_cfg.get('depth_tolerance', 0.02)
    )
    
    # 准备投影图像信息（用于体素滤波）
    projected_images_info = []
    images_dir = Path(projection_dir) / 'images'
    for i, img_name in enumerate(projection_info['face_images']):
        projected_images_info.append({
            'image_path': str(images_dir / img_name),
            'camera_matrix': projection_info['K_face'],
            'extrinsic': projection_info['face_extrinsics'][i],
            'image_shape': (projection_info['out_h'], projection_info['out_w'])
        })
    
    # 点云与相机位姿同属 LiDAR SLAM 世界坐标系，无需额外对齐
    # extrinsicR1/T1 是物理传感器安装偏移（~6cm），不是坐标系差异
    # 传 lidar_to_cam_transform=None，让各帧的 world-to-camera 外参正常参与投影
    filtered_points = voxel_filter.filter(
        pointcloud=data['pointcloud'],
        projected_images=projected_images_info,
        lidar_to_cam_transform=None
    )
    
    print(f"  ✓ 原始点云: {len(data['pointcloud'])} 点")
    print(f"  ✓ 滤波后: {len(filtered_points)} 点")
    print(f"    保留率: {len(filtered_points) / len(data['pointcloud']) * 100:.1f}%")
    
    # ============================================================
    # 步骤 5: 导出 COLMAP 格式
    # ============================================================
    print(f"\n[5/5] 导出 COLMAP 格式...")
    
    exporter = COLMAPExporter(output_dir=Path(output_dir))
    
    # 构建相机列表（COLMAP格式）
    cameras_list = [{
        'camera_id': 1,
        'model': 'PINHOLE',
        'width': projection_info['out_w'],
        'height': projection_info['out_h'],
        'params': [
            projection_info['K_face'][0, 0],  # fx
            projection_info['K_face'][1, 1],  # fy
            projection_info['K_face'][0, 2],  # cx
            projection_info['K_face'][1, 2]   # cy
        ]
    }]
    
    # 构建图像列表（COLMAP格式）
    images_list = []
    for i, (img_name, extrinsic) in enumerate(zip(projection_info['face_images'], projection_info['face_extrinsics'])):
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        # 旋转矩阵转四元数
        qw, qx, qy, qz = rotation_matrix_to_quaternion(R)
        images_list.append({
            'image_id': i + 1,
            'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
            'tx': t[0], 'ty': t[1], 'tz': t[2],
            'camera_id': 1,
            'name': img_name
        })
    
    # 计算每个点最近的相机（供 COLMAP TRACK 可视化）
    print("  计算点云-相机关联（TRACK）...")
    cam_centers = np.array([
        -E[:3, :3].T @ E[:3, 3]
        for E in projection_info['face_extrinsics']
    ])  # (M, 3)
    pts_xyz = filtered_points[:, :3]
    chunk = 100_000
    nearest = np.empty(len(pts_xyz), dtype=np.int32)
    for s in range(0, len(pts_xyz), chunk):
        e = min(s + chunk, len(pts_xyz))
        diff = pts_xyz[s:e, None, :] - cam_centers[None, :, :]  # (chunk, M, 3)
        nearest[s:e] = np.argmin(np.sum(diff ** 2, axis=2), axis=1)
    tracks = [[int(nearest[i]) + 1] for i in range(len(pts_xyz))]  # image_id 1-indexed

    # 导出所有文件
    exporter.export_all(
        cameras=cameras_list,
        images=images_list,
        points=filtered_points[:, :3],  # XYZ
        colors=(filtered_points[:, 3:6] * 255).astype(np.uint8) if filtered_points.shape[1] >= 6 else None,
        tracks=tracks,
    )
    
    print(f"  ✓ cameras.txt")
    print(f"  ✓ images.txt")
    print(f"  ✓ points3D.txt")
    
    # 额外导出 PLY 格式（用于可视化）
    exporter.export_pointcloud_ply(
        points=filtered_points[:, :3],
        colors=(filtered_points[:, 3:6] * 255).astype(np.uint8) if filtered_points.shape[1] >= 6 else None,
        filename="points3D.ply"
    )
    print(f"  ✓ points3D.ply（用于可视化）")
    
    # ============================================================
    # 完成
    # ============================================================
    print(f"\n" + "=" * 60)
    print(f"处理完成!")
    print(f"=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"  - 投影图像: {projection_dir} (仅 back 面)")
    print(f"  - COLMAP 数据: {output_dir}")
    print(f"\n下一步: 使用 gsplat 进行训练")
    print(f"  python -m gsplat.examples.simple_trainer --data_dir {output_dir}")
    
    return output_dir


# =============================================================================
# 入口点
# =============================================================================

if __name__ == '__main__':
    # 支持两种方式运行：
    # 1. 直接运行：使用 CONFIG 字典中的参数
    # 2. 命令行运行：python run_ours.py --scene xxx
    
    parser = argparse.ArgumentParser(description='Li-GS 自采数据集处理')
    parser.add_argument('--scene', type=str, default=None, help='场景名称（覆盖 CONFIG）')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    # 如果提供了命令行参数，覆盖 CONFIG
    if args.scene:
        CONFIG['scene'] = args.scene
    if args.config:
        CONFIG['config_path'] = args.config
    if args.output:
        CONFIG['output_dir'] = args.output
    
    # 运行处理
    process_scene(CONFIG)
