"""
数据加载模块
负责加载点云、位姿、相机内参和图像列表
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import re


class DataLoader:
    """
    数据加载器
    
    加载 Li-GS 所需的所有数据：
    - 彩色点云 (LAS 格式)
    - 相机位姿 (camera_pos.cam)
    - OCamCalib 内参 (calib_results.txt)
    - 图像列表
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 包含以下键的字典
                - pointcloud: LAS 文件路径
                - poses: camera_pos.cam 文件路径
                - camera_intrinsic: calib_results.txt 文件路径
                - images: 图像目录路径
                - image_extension: 图像扩展名 (默认 .jpg)
                - start_index: 起始图像编号 (默认 1)
                - end_index: 结束图像编号 (默认 None)
        """
        self.config = config

    def load_all(self) -> Dict[str, Any]:
        """加载所有数据，返回字典"""
        pointcloud = self.load_pointcloud(self.config['pointcloud'])
        ocam = self.load_ocam_intrinsic(self.config['camera_intrinsic'])
        
        # 先加载图像列表（按文件名排序）
        images = self.load_image_list(
            self.config['images'],
            self.config.get('image_extension', '.jpg'),
            self.config.get('start_index', 1),
            self.config.get('end_index', None),
        )
        
        # 加载位姿，然后按图像顺序重新排序位姿
        # 确保位姿和图像一一对应
        poses = self.load_poses(self.config['poses'])
        poses = self._match_poses_to_images(poses, images)
        
        return {
            'pointcloud': pointcloud,   # (N, 6): X Y Z R G B
            'poses': poses,             # list of pose dicts (与images顺序一致)
            'ocam': ocam,               # dict with cam model params
            'images': images,           # list of Path
        }
    
    def _match_poses_to_images(self, poses: List[Dict], images: List[Path]) -> List[Dict]:
        """
        将位姿列表按图像文件名重新排序，确保位姿和图像一一对应。
        
        Args:
            poses: 位姿列表（每个包含 'image_name'）
            images: 图像路径列表（已按文件名排序）
            
        Returns:
            重新排序后的位姿列表，与 images 顺序一致
        """
        # 创建位姿字典：image_name -> pose
        pose_dict = {pose['image_name']: pose for pose in poses}
        
        # 按图像顺序提取对应的位姿
        matched_poses = []
        missing_poses = []
        
        for img_path in images:
            img_name = img_path.name
            if img_name in pose_dict:
                matched_poses.append(pose_dict[img_name])
            else:
                missing_poses.append(img_name)
        
        if missing_poses:
            print(f"  警告: {len(missing_poses)} 张图像没有找到对应的位姿")
            for name in missing_poses[:5]:  # 只显示前5个
                print(f"    - {name}")
        
        print(f"  位姿匹配: {len(matched_poses)} / {len(images)} 张图像")
        return matched_poses

    # ------------------------------------------------------------------
    # 各子加载方法
    # ------------------------------------------------------------------

    def load_pointcloud(self, las_path: str) -> np.ndarray:
        """
        读取 LAS 彩色点云，返回 (N, 6) 数组: [X, Y, Z, R, G, B]
        R/G/B 归一化到 [0, 1]。
        
        支持读取单个文件或整个目录（合并多个 .las 文件）。
        """
        import laspy
        las_path = _fix_windows_path(las_path)
        path = Path(las_path)
        
        # 收集所有 .las 文件
        if path.is_dir():
            las_files = sorted(path.glob('*.las'))
            if not las_files:
                raise FileNotFoundError(f"目录中没有 .las 文件: {las_path}")
            print(f"发现 {len(las_files)} 个 .las 文件，将合并加载...")
        else:
            las_files = [path]
        
        # 加载并合并所有点云
        all_points = []
        for las_file in las_files:
            las = laspy.read(str(las_file))
            x = np.array(las.x, dtype=np.float64)
            y = np.array(las.y, dtype=np.float64)
            z = np.array(las.z, dtype=np.float64)

            # LAS 颜色通常是 uint16 (0-65535)，归一化到 [0,1]
            try:
                r = np.array(las.red,   dtype=np.float32) / 65535.0
                g = np.array(las.green, dtype=np.float32) / 65535.0
                b = np.array(las.blue,  dtype=np.float32) / 65535.0
            except AttributeError:
                r = g = b = np.ones(len(x), dtype=np.float32) * 0.5

            points = np.column_stack([x, y, z, r, g, b])
            all_points.append(points)
            print(f"  加载: {las_file.name} -> {len(points):,} 个点")
        
        # 合并所有点云
        merged_points = np.vstack(all_points)
        print(f"合并点云: {len(merged_points):,} 个点 (来自 {len(las_files)} 个文件)")
        return merged_points

    def load_poses(self, cam_path: str) -> List[Dict]:
        """
        解析 camera_pos.cam，返回位姿列表。

        每条记录：
            {
                'image_name': str,       # e.g. '5.jpg'
                'position': np.array,    # [X, Y, Z] 世界坐标（米）
                'euler': np.ndarray,     # [roll, pitch, yaw]（弧度）
                'rotation': np.ndarray,  # (3,3) camera-to-world 旋转矩阵
                'timestamp': float,
            }
        """
        cam_path = _fix_windows_path(cam_path)
        poses = []
        with open(cam_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 10:
                    continue
                name = parts[0]
                # 格式: name 0 0 X Y Z roll pitch yaw timestamp
                X     = float(parts[3])
                Y     = float(parts[4])
                Z     = float(parts[5])
                roll  = float(parts[6])
                pitch = float(parts[7])
                yaw   = float(parts[8])
                ts    = float(parts[9])

                position = np.array([X, Y, Z], dtype=np.float64)
                euler    = np.array([roll, pitch, yaw], dtype=np.float64)
                R        = _euler_to_rotation_matrix(roll, pitch, yaw)

                poses.append({
                    'image_name': name,
                    'position':   position,
                    'euler':      euler,
                    'rotation':   R,
                    'timestamp':  ts,
                })

        print(f"加载位姿: {len(poses)} 条，路径: {Path(cam_path).name}")
        return poses

    def load_ocam_intrinsic(self, txt_path: str) -> Dict:
        """
        解析相机标定文件。
        支持 OCamCalib 格式 (calib_results.txt) 和 YAML 格式。
        返回包含多项式系数、主点、图像尺寸等的字典。
        
        注意: 使用 camToWorld 系数进行计算（更稳定），
        worldToCam 通过查找表实现。
        """
        import yaml
        txt_path = _fix_windows_path(txt_path)
        path = Path(txt_path)
        
        # 根据文件扩展名判断格式
        if path.suffix.lower() in ['.yaml', '.yml']:
            # YAML 格式
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            
            # 提取参数
            xc, yc = data['principal1']
            # 使用 camToWorld 系数（更稳定）
            if 'camToWorld1' in data:
                pol = np.array(data['camToWorld1'], dtype=np.float64)
            else:
                # 如果没有 camToWorld，使用 worldToCam（不推荐）
                pol = np.array(data['worldToCam1'], dtype=np.float64)
            width = data['width']
            height = data['height']
            rotate = data.get('rotate', 0)
            
            # 外参（如果有）
            ext_T = None
            ext_R = None
            if 'extrinsicT1' in data:
                ext_T = np.array(data['extrinsicT1'], dtype=np.float64)
            if 'extrinsicR1' in data:
                ext_R = np.array(data['extrinsicR1'], dtype=np.float64).reshape(3, 3)
            
            print(f"加载相机内参 (YAML): {width}×{height}, 主点=({xc:.1f},{yc:.1f}), rotate={rotate}°")
        else:
            # OCamCalib 格式 (calib_results.txt)
            with open(path, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]

            # 第1行: 多项式阶数
            pol_order = int(lines[0])
            # 第2行: 多项式系数 (从低到高) - 这是 camToWorld 系数
            pol = np.array([float(x) for x in lines[1].split()], dtype=np.float64)
            # 第3行: 主点 (xc, yc)
            xc, yc = [float(x) for x in lines[2].split()]
            # 第4行: 图像尺寸 (height, width)
            height, width = [int(x) for x in lines[3].split()]
            # 第5行: 仿射变换 (c, d, e)
            c, d, e = [float(x) for x in lines[4].split()]
            # 第6行: 旋转角度 (rotate)
            rotate = int(lines[5])

            # 外参 (如果有)
            ext_T = None
            ext_R = None
            if len(lines) >= 8:
                ext_T = np.array([float(x) for x in lines[6].split()], dtype=np.float64)
                ext_R = np.array([float(x) for x in lines[7].split()], dtype=np.float64).reshape(3, 3)

            print(f"加载相机内参 (OCamCalib): {width}×{height}, 主点=({xc:.1f},{yc:.1f}), rotate={rotate}°")
        
        return {
            'xc': xc, 'yc': yc,
            'pol': pol,  # 这是 camToWorld 系数
            'width': width, 'height': height,
            'rotate': rotate,
            'extrinsic_T': ext_T,
            'extrinsic_R': ext_R,
        }

    def load_image_list(self, image_dir: str, ext: str = '.jpg',
                        start: int = 1, end: int = None) -> List[Path]:
        """扫描图像目录，返回按数字编号排序的 Path 列表。"""
        image_dir = _fix_windows_path(image_dir)
        p = Path(image_dir)
        files = sorted(p.glob(f'*{ext}'), key=lambda x: _parse_index(x.stem))
        if start is not None:
            files = [f for f in files if _parse_index(f.stem) >= start]
        if end is not None:
            files = [f for f in files if _parse_index(f.stem) <= end]
        print(f"加载图像列表: {len(files)} 张，路径: {p.name}")
        return files


# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------

def _fix_windows_path(path: str) -> str:
    """
    将 Windows 路径转为当前系统路径。
    
    自动检测运行环境：
    - 如果在 WSL 中运行，将 D:/ 转换为 /mnt/d/
    - 如果在 Windows 原生环境运行，保持原样
    """
    import platform
    import os
    
    # 检测是否在 WSL 中运行
    is_wsl = False
    if platform.system() == "Linux":
        # 检查 /proc/version 是否包含 Microsoft 或 WSL
        try:
            with open('/proc/version', 'r') as f:
                version_info = f.read().lower()
                is_wsl = 'microsoft' in version_info or 'wsl' in version_info
        except:
            pass
    
    # 转换路径
    if is_wsl:
        # WSL 环境：将 D:\\path 或 D:/path 转换为 /mnt/d/path
        path = path.replace('\\\\', '/')
        if len(path) >= 2 and path[1] == ':':
            drive = path[0].lower()
            path = f"/mnt/{drive}{path[2:]}"
    else:
        # Windows 原生环境：确保使用正确的分隔符
        path = path.replace('/', '\\\\')
    
    # 展开用户目录
    path = os.path.expanduser(path)
    
    return path


def _parse_index(stem: str) -> int:
    import re
    m = re.search(r'\d+', stem)
    return int(m.group()) if m else 0


def _euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX 欧拉角 → camera-to-world 旋转矩阵。R = Rz(yaw) @ Ry(pitch) @ Rx(roll)"""
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    Rx = np.array([[1,  0,   0], [0,  cr, -sr], [0,  sr,  cr]], dtype=np.float64)
    Ry = np.array([[ cp, 0, sp], [0,   1,   0], [-sp, 0,  cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy,  cy,  0], [0,   0,   1]], dtype=np.float64)
    return Rz @ Ry @ Rx
