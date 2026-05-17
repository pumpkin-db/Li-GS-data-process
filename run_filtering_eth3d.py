"""
Li-GS ETH3D 动态体素滤波入口脚本

使用方法:
    python run_filtering_eth3d.py --scene courtyard
    python run_filtering_eth3d.py --scene facade
    python run_filtering_eth3d.py --scene terrains
    python run_filtering_eth3d.py --scene courtyard --config config/my_scenes.yaml

配置文件格式 (YAML):
    scenes:
      scene_name:
        scan: "path/to/scan.ply"
        images: "path/to/images_dir"
        cameras: "path/to/cameras.txt"
        images_txt: "path/to/images.txt"
"""

'''
2026.5.12 bug修复：
scan1.ply 在扫描仪自身的坐标系中，而 images.txt 里的相机外参在世界坐标系中——两者根本不对齐，投影完全是错的。

scan_alignment.mlp 的作用：
它是 MeshLab 项目文件（XML），记录每个原始扫描到世界坐标系的 4×4 变换矩阵
对 courtyard 的 scan1.ply，矩阵约为小旋转（~4.8°）+ 平移 (3.82, -7.24, 1.87) 米
不应用这个变换，投影后点云会出现在图像的完全错误位置
'''

import argparse
import os
import sys
import yaml
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from voxel_filtering_eth3d import (
    DynamicVoxelFilter,
    load_colmap_cameras,
    load_colmap_images,
    build_camera_matrix,
    save_colmap_points3d,
    load_scan_alignment,
    apply_scan_alignment,
)
import open3d as o3d
import numpy as np
import shutil


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def fix_images_txt_paths(input_path: str, output_path: str):
    """
    修复 images.txt 中的图像路径，去掉子目录，只保留文件名
    
    ETH3D 的 images.txt 中图像路径可能包含子目录（如 dslr_images_undistorted/DSC_0286.JPG）
    但 Li-GS 输出时只复制图像到 images/ 目录下，不包含子目录
    因此需要修改 images.txt 中的路径，使其与实际路径一致
    
    COLMAP images.txt 格式：
    - 每两行一组：图像信息行 + 点观测行
    - 图像信息行：IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    - 点观测行：POINTS2D[] as (X, Y, POINT3D_ID) 或空行
    
    Args:
        input_path: 输入的 images.txt 路径（ETH3D 原版）
        output_path: 输出的 images.txt 路径（修复后）
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    with open(output_path, 'w') as f:
        i = 0
        while i < len(lines):
            line = lines[i].rstrip('\n')
            
            # 保留注释和空行
            if not line.strip() or line.strip().startswith('#'):
                f.write(lines[i])
                i += 1
                continue
            
            # 处理图像信息行（至少10个字段）
            parts = line.split()
            if len(parts) >= 10:
                # 提取图像名（去掉子目录）
                # 如: dslr_images_undistorted/DSC_0286.JPG -> DSC_0286.JPG
                img_name_with_subdir = parts[9]
                img_name = os.path.basename(img_name_with_subdir)
                
                # 替换路径
                parts[9] = img_name
                f.write(' '.join(parts) + '\n')
                
                # 写入点观测行（下一行）
                i += 1
                if i < len(lines):
                    f.write(lines[i])
            else:
                # 其他格式，直接写入
                f.write(lines[i])
            
            i += 1


def process_scene(
    scene_name: str,
    scene_config: dict,
    output_base_path: str = r'D:\Li-GS_data_process\output',
    filter_params: dict = None,
    max_workers: int = 4
):
    """
    处理单个场景
    
    Args:
        scene_name: 场景名称
        scene_config: 场景配置字典
        output_base_path: 输出根目录
        filter_params: 滤波器参数字典
    """
    print(f"=" * 60)
    print(f"处理场景: {scene_name}")
    print(f"=" * 60)
    
    # 提取路径
    scan_path = scene_config['scan']
    images_dir = scene_config['images']
    cameras_path = scene_config['cameras']
    images_txt_path = scene_config['images_txt']
    
    # 检查路径是否存在
    paths_to_check = [
        ('扫描点云', scan_path),
        ('图像目录', images_dir),
        ('相机参数', cameras_path),
        ('图像位姿', images_txt_path)
    ]
    
    for name, path in paths_to_check:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} 不存在: {path}")
        print(f"  ✓ {name}: {path}")
    
    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    
    # 加载点云
    pcd = o3d.io.read_point_cloud(scan_path)
    print(f"  点云点数: {len(pcd.points)}")
    print(f"  点云有颜色: {pcd.has_colors()}")

    # 应用 scan_alignment 变换（ETH3D 原始扫描在扫描仪坐标系，需变换到世界坐标系）
    # scan_alignment.mlp 与扫描 PLY 文件位于同一目录
    scan_dir = os.path.dirname(os.path.abspath(scan_path))
    mlp_path = os.path.join(scan_dir, 'scan_alignment.mlp')
    if os.path.exists(mlp_path):
        print(f"  检测到 scan_alignment.mlp，应用扫描仪→世界坐标变换...")
        alignment_matrix = load_scan_alignment(mlp_path, os.path.basename(scan_path))
        apply_scan_alignment(pcd, alignment_matrix)
        print(f"  变换矩阵:\n{alignment_matrix}")
    else:
        print(f"  未找到 scan_alignment.mlp（{mlp_path}），跳过坐标对齐")

    # 加载相机参数
    cameras = load_colmap_cameras(cameras_path)
    print(f"  相机数量: {len(cameras)}")
    
    # 加载图像位姿
    images = load_colmap_images(images_txt_path)
    print(f"  图像数量: {len(images)}")
    
    # 2. 准备参数
    print("\n[2/4] 准备参数...")
    
    # 获取第一个相机参数
    camera_id = list(cameras.keys())[0]
    camera_params = cameras[camera_id]
    print(f"  相机模型: {camera_params['model']}")
    print(f"  图像尺寸: {camera_params['width']} x {camera_params['height']}")
    
    # 构建相机内参矩阵
    K = build_camera_matrix(camera_params)
    print(f"  相机内参矩阵:\n{K}")
    
    # 准备图像路径和外参
    image_paths = []
    extrinsics = []
    
    for img_id in sorted(images.keys()):
        img_info = images[img_id]
        # images.txt 中的 name 可能包含子目录（如 dslr_images_undistorted/DSC_0286.JPG）
        # 需要正确处理路径拼接
        img_name = img_info['name']
        
        # 如果 name 包含目录分隔符，直接拼接到 images_dir 的父目录
        if '/' in img_name or '\\' in img_name:
            # 使用 images_dir 的父目录作为基础路径
            base_dir = os.path.dirname(images_dir)  # 去掉 dslr_images_undistorted
            img_path = os.path.join(base_dir, img_name)
        else:
            # 简单的文件名，直接拼接
            img_path = os.path.join(images_dir, img_name)
        
        # 标准化路径分隔符
        img_path = os.path.normpath(img_path)
        
        if os.path.exists(img_path):
            image_paths.append(img_path)
            extrinsics.append(img_info['extrinsic'])
        else:
            print(f"  警告: 图像不存在 {img_path}")
    
    print(f"  有效图像: {len(image_paths)}")
    
    image_shape = (camera_params['height'], camera_params['width'])
    
    # 3. 执行动态体素滤波
    print("\n[3/4] 执行动态体素滤波...")
    
    # 默认参数
    default_params = {
        'feature_rich_ratio': 0.05,    # 前5%
        'voxel_size_rich': 0.02,       # feature-rich: 0.02m
        'voxel_size_normal': 0.05,     # 普通: 0.05m
        'sift_nfeatures': 5000,        # 默认限制5000个特征点
        'kde_bandwidth': 50.0,         # KDE带宽50像素
        'depth_tolerance': 0.02         # 深度容差
    }
    
    # 合并用户参数
    if filter_params:
        default_params.update(filter_params)
    
    filter = DynamicVoxelFilter(**default_params)
    
    print(f"  参数:")
    for key, value in default_params.items():
        print(f"    {key}: {value}")
    print(f"    max_workers: {max_workers}")
    
    filtered_pcd = filter.filter(
        pcd, image_paths, K, extrinsics, image_shape, max_workers
    )
    
    # 4. 保存结果
    print("\n[4/4] 保存结果...")
    
    output_dir = os.path.join(output_base_path, scene_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存滤波后的点云为 PLY 格式（便于可视化）
    output_ply = os.path.join(output_dir, 'points3D.ply')
    o3d.io.write_point_cloud(output_ply, filtered_pcd)
    print(f"  保存点云 (PLY): {output_ply}")
    
    # 保存滤波后的点云为 COLMAP points3D.txt 格式
    output_points3d_txt = os.path.join(output_dir, 'points3D.txt')
    save_colmap_points3d(filtered_pcd, output_points3d_txt)
    print(f"  保存点云 (COLMAP): {output_points3d_txt}")
    
    # 复制图像
    output_images_dir = os.path.join(output_dir, 'images')
    os.makedirs(output_images_dir, exist_ok=True)
    
    for img_path in image_paths:
        dst_path = os.path.join(output_images_dir, os.path.basename(img_path))
        if not os.path.exists(dst_path):
            shutil.copy2(img_path, dst_path)
    print(f"  复制 {len(image_paths)} 张图像到: {output_images_dir}")
    
    # 复制相机参数
    shutil.copy2(cameras_path, os.path.join(output_dir, 'cameras.txt'))
    
    # 复制并修复 images.txt（去掉图像路径中的子目录）
    output_images_txt = os.path.join(output_dir, 'images.txt')
    fix_images_txt_paths(images_txt_path, output_images_txt)
    print(f"  复制并修复 images.txt: {output_images_txt}")
    
    # 保存场景配置（方便追溯）
    config_backup = {
        'scene_name': scene_name,
        'scan': scan_path,
        'images': images_dir,
        'cameras': cameras_path,
        'images_txt': images_txt_path,
        'filter_params': default_params,
        'input_points': len(pcd.points),
        'output_points': len(filtered_pcd.points),
        'compression_ratio': len(filtered_pcd.points) / len(pcd.points) * 100
    }
    
    import json
    with open(os.path.join(output_dir, 'scene_info.json'), 'w') as f:
        json.dump(config_backup, f, indent=2)
    
    print(f"\n处理完成！")
    print(f"  输入: {len(pcd.points)} 点")
    print(f"  输出: {len(filtered_pcd.points)} 点")
    print(f"  压缩比: {len(filtered_pcd.points) / len(pcd.points) * 100:.2f}%")
    print(f"  输出目录: {output_dir}")
    
    return filtered_pcd


def main():
    parser = argparse.ArgumentParser(
        description='Li-GS ETH3D 动态体素滤波',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_filtering_eth3d.py --scene courtyard
  python run_filtering_eth3d.py --scene facade --voxel-rich 0.015
  python run_filtering_eth3d.py --scene terrains --config my_config.yaml
        """
    )
    
    parser.add_argument(
        '--scene', '-s',
        type=str,
        required=True,
        help='场景名称 (如: courtyard, facade, terrains)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/eth3d_scenes.yaml',
        help='配置文件路径 (默认: config/eth3d_scenes.yaml)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='输出目录 (默认: output)'
    )
    
    # 滤波器参数
    parser.add_argument(
        '--feature-ratio',
        type=float,
        default=0.03,
        help='Feature-rich 区域比例 (默认: 0.05 = 5%)'
    )

    parser.add_argument(
        '--depth-tolerance',
        type=float,
        default=0.02,
        help='Depth-tolerance 深度容差 (默认: 0.02m)'
    )

    parser.add_argument(
        '--voxel-rich',
        type=float,
        default=0.02,
        help='Feature-rich 区域体素大小 (默认: 0.02m)'
    )
    
    parser.add_argument(
        '--voxel-normal',
        type=float,
        default=0.05,
        help='普通区域体素大小 (默认: 0.05m)'
    )
    
    parser.add_argument(
        '--kde-bandwidth',
        type=float,
        default=200.0,
        help='KDE 带宽 (默认: 200像素)'
    )
    
    parser.add_argument(
        '--max-features',
        type=int,
        default=1000,
        help='SIFT 最大特征点数量 (默认: 500, 0表示无限制)'
    )
    
    parser.add_argument(
        '--kde-resize',
        type=float,
        default=0.125,
        help='KDE计算时图像缩放因子 (默认: 0.125 = 1/8)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=3,
        help='并行处理图像的线程数 (默认: 4, 设为1禁用并行)'
    )
    
    args = parser.parse_args()
    
    # 检查配置文件
    config_path = args.config
    if not os.path.isabs(config_path):
        # 相对路径，相对于脚本目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
    
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)
    
    print(f"加载配置: {config_path}")
    config = load_config(config_path)
    
    # 检查场景是否存在
    if 'scenes' not in config or args.scene not in config['scenes']:
        available = list(config.get('scenes', {}).keys())
        print(f"错误: 场景 '{args.scene}' 不存在于配置文件中")
        print(f"可用场景: {', '.join(available) if available else '无'}")
        sys.exit(1)
    
    scene_config = config['scenes'][args.scene]
    
    # 准备滤波器参数
    filter_params = {
        'feature_rich_ratio': args.feature_ratio,
        'voxel_size_rich': args.voxel_rich,
        'voxel_size_normal': args.voxel_normal,
        'kde_bandwidth': args.kde_bandwidth,
        'sift_nfeatures': args.max_features,
        'kde_resize_factor': args.kde_resize,
        'depth_tolerance': args.depth_tolerance
    }
    
    # 处理输出路径
    output_path = args.output
    if not os.path.isabs(output_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, output_path)
    
    try:
        process_scene(
            scene_name=args.scene,
            scene_config=scene_config,
            output_base_path=output_path,
            filter_params=filter_params,
            max_workers=args.max_workers
        )
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
