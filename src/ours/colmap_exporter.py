"""
COLMAP 格式导出模块 - 将处理结果导出为 COLMAP 格式

职责：
1. 生成 cameras.txt（相机内参）
2. 生成 images.txt（图像位姿）
3. 生成 points3D.txt（点云）
4. 生成 points3D.ply（点云可视化）

输入：
- 相机参数（来自 fisheye_projection）
- 图像位姿（来自 fisheye_projection）
- 点云（来自 voxel_filter）

输出：
- COLMAP 格式的文本文件
- PLY 格式的点云文件

注意：此模块只负责格式转换，不读取原始数据、不投影、不滤波

================================================================================
输出文件结构说明
================================================================================

最终输出目录结构（由 run_ours.py 统一管理）：

D:\Li-GS_data_process\output\[scene_name]\          # 场景输出根目录
├── images\                                          # 【fisheye_projection 输出】投影后的平面图像
│   ├── 0000_front.jpg                               # 第0张投影图像
│   ├── 0001_back.jpg
│   ├── 0002_left.jpg
│   └── ...                                          # 共：关键帧数 × 5 张图像
├── cameras.txt                                      # 【本模块输出】相机内参
├── images.txt                                       # 【本模块输出】图像位姿
├── points3D.txt                                     # 【本模块输出】点云（COLMAP格式）
└── points3D.ply                                     # 【本模块输出】点云（PLY格式，可视化）

总计 5 个输出项：
1. images/ 文件夹（fisheye_projection 生成，供高斯训练使用）
2. cameras.txt（COLMAP 格式）
3. images.txt（COLMAP 格式）
4. points3D.txt（COLMAP 格式）
5. points3D.ply（PLY 格式，用于可视化）

命名规则：
- 使用 {seq_id:04d}_{face_name}.jpg 格式
- 30张关键帧 × 5面 = 150张图像

所有文件都位于 D:\Li-GS_data_process\output\[scene_name]\ 目录下

================================================================================
COLMAP 格式说明
================================================================================

cameras.txt:
    # Camera list with one line of data per camera:
    #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    1 PINHOLE 1800 1800 900.0 900.0 900.0 900.0
    
images.txt:
    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    1 0.5 0.3 0.1 0.2 1.0 2.0 3.0 1 0000_front.jpg
    
points3D.txt:
    # 3D point list with one line of data per point:
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
    1 0.5 1.2 3.4 255 128 64 0.5

================================================================================
"""

from pathlib import Path
from typing import List, Dict, Optional
import numpy as np


class COLMAPExporter:
    """COLMAP 格式导出器"""
    
    def __init__(self, output_dir: Path):
        """
        初始化导出器
        
        Args:
            output_dir: 场景输出根目录 (output/[scene_name]/)
                         所有文件将输出到此目录下
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_cameras(
        self,
        cameras: List[Dict],
        camera_model: str = "PINHOLE"
    ) -> Path:
        """
        导出 cameras.txt
        
        输出路径: output_dir / "cameras.txt"
        
        Args:
            cameras: 相机列表，每个相机为字典：
                {
                    'camera_id': int,
                    'width': int,
                    'height': int,
                    'params': List[float]  # [fx, fy, cx, cy] for PINHOLE
                }
            camera_model: 相机模型类型（"PINHOLE", "OPENCV", etc.）
            
        Returns:
            输出文件路径
        """
        output_path = self.output_dir / "cameras.txt"
        
        with open(output_path, 'w') as f:
            # 写入文件头
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            
            for cam in cameras:
                camera_id = cam['camera_id']
                width = cam['width']
                height = cam['height']
                params = cam['params']
                
                # 构建参数字符串
                params_str = ' '.join([f"{p:.6f}" for p in params])
                
                # 写入相机行
                f.write(f"{camera_id} {camera_model} {width} {height} {params_str}\n")
        
        print(f"  导出 cameras.txt: {output_path}")
        return output_path
    
    def export_images(
        self,
        images: List[Dict]
    ) -> Path:
        """
        导出 images.txt
        
        输出路径: output_dir / "images.txt"
        
        Args:
            images: 图像列表，每个图像为字典：
                {
                    'image_id': int,
                    'qw', 'qx', 'qy', 'qz': float,  # 四元数
                    'tx', 'ty', 'tz': float,        # 平移
                    'camera_id': int,
                    'name': str  # 图像文件名（如 "0000_front.jpg"）
                }
                
        Returns:
            输出文件路径
        """
        output_path = self.output_dir / "images.txt"
        
        with open(output_path, 'w') as f:
            # 写入文件头
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            
            for img in images:
                image_id = img['image_id']
                qw, qx, qy, qz = img['qw'], img['qx'], img['qy'], img['qz']
                tx, ty, tz = img['tx'], img['ty'], img['tz']
                camera_id = img['camera_id']
                name = img['name']
                
                # 写入图像信息行
                f.write(f"{image_id} {qw:.10f} {qx:.10f} {qy:.10f} {qz:.10f} ")
                f.write(f"{tx:.10f} {ty:.10f} {tz:.10f} {camera_id} {name}\n")
                
                # 写入点观测行（空，因为我们没有2D-3D对应关系）
                f.write("\n")
        
        print(f"  导出 images.txt: {output_path}")
        return output_path
    
    def export_points3D(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        tracks: Optional[List] = None,
    ) -> Path:
        """
        导出 points3D.txt（COLMAP 格式）

        Args:
            points: (N, 3) 点云坐标
            colors: (N, 3) 点云颜色 [0-255]，可选
            tracks: 每个点的 image_id 列表，例如 [[1], [3, 7], ...]
                    提供后 COLMAP 3D viewer 才能同时显示点云和相机位姿
        """
        output_path = self.output_dir / "points3D.txt"
        n_points = len(points)

        if colors is None:
            colors = np.full((n_points, 3), 255, dtype=np.uint8)
        else:
            colors = colors.astype(np.uint8)

        with open(output_path, 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write(f"# Number of points: {n_points}\n")

            for i in range(n_points):
                point_id = i + 1
                x, y, z = points[i]
                r, g, b = colors[i]
                track_str = ''
                if tracks is not None and i < len(tracks) and tracks[i]:
                    track_str = ' ' + ' '.join(f'{img_id} 0' for img_id in tracks[i])
                f.write(f"{point_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.000000{track_str}\n")

        print(f"  导出 points3D.txt: {output_path} ({n_points} 点)")
        return output_path
    
    def export_pointcloud_ply(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        filename: str = "points3D.ply"
    ) -> Path:
        """
        导出点云为 PLY 格式（用于可视化）
        
        这是动态体素滤波后的点云，命名为 points3D.ply
        输出路径: output_dir / filename
        
        Args:
            points: (N, 3) 点云坐标
            colors: (N, 3) 点云颜色 [0-255]，可选
            filename: 输出文件名（默认: points3D.ply）
            
        Returns:
            输出文件路径
        """
        output_path = self.output_dir / filename
        
        n_points = len(points)
        
        # 处理颜色
        if colors is None:
            # 默认白色
            colors = np.full((n_points, 3), 255, dtype=np.uint8)
        else:
            colors = colors.astype(np.uint8)
        
        # 手动写入二进制 PLY 格式（更高效）
        # 或者使用 ASCII 格式（更易读）
        # 这里使用 ASCII 格式以便调试
        
        with open(output_path, 'w') as f:
            # 写入 PLY 头部
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n_points}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            # 写入顶点数据
            for i in range(n_points):
                x, y, z = points[i]
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
        
        print(f"  导出 {filename}: {output_path} ({n_points} 点)")
        return output_path
    
    def export_all(
        self,
        cameras: List[Dict],
        images: List[Dict],
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        tracks: Optional[List] = None,
    ) -> Dict[str, Path]:
        """
        导出所有 COLMAP 文件和 PLY 文件
        
        这是本模块的主入口函数，由 run_ours.py 调用。
        
        输出文件：
        - cameras.txt
        - images.txt
        - points3D.txt
        - points3D.ply
        
        Args:
            cameras: 相机列表
            images: 图像列表
            points: 点云坐标 (N, 3)
            colors: 点云颜色 (N, 3) [0-255]，可选
            
        Returns:
            字典，包含所有输出文件路径
        """
        results = {}
        
        # 导出 cameras.txt
        results['cameras'] = self.export_cameras(cameras)
        
        # 导出 images.txt
        results['images'] = self.export_images(images)
        
        # 导出 points3D.txt
        results['points3D'] = self.export_points3D(points, colors, tracks=tracks)
        
        # 导出 points3D.ply
        results['points3D_ply'] = self.export_pointcloud_ply(points, colors)
        
        return results


def rotation_matrix_to_quaternion(R: np.ndarray) -> tuple:
    """
    将旋转矩阵转换为四元数 (qw, qx, qy, qz)
    
    COLMAP 使用 Hamilton 四元数，qw 为实部
    
    Args:
        R: (3, 3) 旋转矩阵
        
    Returns:
        (qw, qx, qy, qz) 四元数
    """
    # 使用标准算法从旋转矩阵计算四元数
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    return qw, qx, qy, qz
