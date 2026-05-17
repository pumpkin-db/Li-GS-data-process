"""
Li-GS 动态体素滤波模块 (ETH3D版本)
基于图像特征密度对点云进行分区降采样

参考论文: Li-GS: a fast 3D Gaussian reconstruction method assisted by LiDAR point clouds
"""

import numpy as np
import cv2
import open3d as o3d
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict
from scipy.stats import gaussian_kde
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class DynamicVoxelFilter:
    """
    动态体素滤波器
    
    根据图像特征密度对点云进行分区降采样：
    - feature-rich 区域：小体素（0.02m）
    - 普通区域：大体素（0.05m）
    """
    
    def __init__(
        self,
        feature_rich_ratio: float = 0.05,  # feature-rich blocks 比例（前5%）
        voxel_size_rich: float = 0.02,      # feature-rich 区域体素大小
        voxel_size_normal: float = 0.05,    # 普通区域体素大小
        sift_nfeatures: int = 0,            # SIFT特征点数量（0表示无限制）
        kde_bandwidth: float = 50.0,        # KDE带宽（像素）
        depth_tolerance: float = 0.1,       # 深度容差（用于判断遮挡）
        kde_resize_factor: float = 0.25     # KDE计算时图像缩放因子（默认1/4）
    ):
        self.feature_rich_ratio = feature_rich_ratio
        self.voxel_size_rich = voxel_size_rich
        self.voxel_size_normal = voxel_size_normal
        self.sift_nfeatures = sift_nfeatures
        self.kde_bandwidth = kde_bandwidth
        self.depth_tolerance = depth_tolerance
        self.kde_resize_factor = kde_resize_factor
        
        # 初始化 SIFT 检测器
        self.sift = cv2.SIFT_create(nfeatures=sift_nfeatures)
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        使用 SIFT 提取图像特征点
        
        Args:
            image_path: 图像路径
            
        Returns:
            特征点坐标数组 (N, 2) - (x, y)
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        keypoints, _ = self.sift.detectAndCompute(img, None)
        
        if len(keypoints) == 0:
            return np.array([])
        
        # 提取特征点坐标
        features = np.array([kp.pt for kp in keypoints])  # (N, 2)
        return features
    
    def compute_feature_density(
        self, 
        features: np.ndarray, 
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        使用核密度估计（KDE）计算特征密度分布
        
        优化：先在缩小后的图像上计算KDE，再插值回原尺寸
        
        公式: p(x) = sum_f K((x - f) / h)
        
        Args:
            features: 特征点坐标 (N, 2)
            image_shape: 图像尺寸 (height, width)
            
        Returns:
            密度图 (height, width)
        """
        if len(features) == 0:
            return np.zeros(image_shape)
        
        original_height, original_width = image_shape
        
        # 缩小图像尺寸以提高KDE计算速度
        scale = self.kde_resize_factor
        small_height = int(original_height * scale)
        small_width = int(original_width * scale)
        
        # 特征点坐标也相应缩放
        scaled_features = features * scale
        
        # 在缩小后的尺寸上创建网格
        x_grid = np.arange(small_width)
        y_grid = np.arange(small_height)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.vstack([xx.ravel(), yy.ravel()])  # (2, small_height*small_width)
        
        # 使用高斯核进行KDE（在缩小后的尺寸上）
        # bw_method 是各维度 std 的乘数因子，需分维度计算 std 再取算术平均，
        # 避免将 x/y 坐标混合展平后求 std 导致带宽在各方向不均匀
        std_x = np.std(scaled_features[:, 0])
        std_y = np.std(scaled_features[:, 1])
        representative_std = (std_x + std_y) / 2.0
        if representative_std < 1e-6:
            representative_std = 1.0  # 防止特征点全部重叠时除零
        bw_factor = (self.kde_bandwidth * scale) / representative_std
        kde = gaussian_kde(scaled_features.T, bw_method=bw_factor)
        
        # 计算密度
        density_small = kde(grid_points)  # (small_height*small_width,)
        density_small = density_small.reshape(small_height, small_width)
        
        # 插值回原尺寸
        density = cv2.resize(density_small.astype(np.float32), (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        
        # 归一化
        if density.max() > 0:
            density = density / density.max()
        
        return density
    
    def get_feature_rich_mask(
        self, 
        density: np.ndarray, 
        ratio: float = None
    ) -> np.ndarray:
        """
        获取 feature-rich 区域的掩码
        
        取密度最高的前 ratio 比例的像素作为 feature-rich blocks
        
        Args:
            density: 密度图
            ratio: feature-rich 比例（默认使用 self.feature_rich_ratio）
            
        Returns:
            二值掩码 (height, width)，True表示feature-rich区域
        """
        if ratio is None:
            ratio = self.feature_rich_ratio
        
        # 计算阈值
        threshold = np.percentile(density, (1 - ratio) * 100)
        
        # 生成掩码
        mask = density >= threshold
        
        return mask
    
    def project_points_to_image(
        self,
        points: np.ndarray,
        camera_matrix: np.ndarray,
        extrinsic: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        将3D点云投影到图像平面（向量化实现）
        
        Args:
            points: 3D点云 (N, 3)
            camera_matrix: 相机内参矩阵 (3, 3)
            extrinsic: 相机外参矩阵 (4, 4) - 世界到相机的变换
            image_shape: 图像尺寸 (height, width)
            
        Returns:
            projected: 投影后的2D坐标 (N, 2) - (u, v)
            depths: 深度值 (N,)
            valid_mask: 有效投影掩码 (N,)
        """
        height, width = image_shape
        
        # 齐次坐标
        points_h = np.hstack([points, np.ones((len(points), 1))])  # (N, 4)
        
        # 世界坐标 -> 相机坐标
        points_cam = (extrinsic @ points_h.T).T  # (N, 4)
        points_cam = points_cam[:, :3]  # (N, 3)
        
        # 深度值（Z坐标）
        depths = points_cam[:, 2]
        
        # 只保留相机前方的点
        valid_depth = depths > 0
        
        # 相机坐标 -> 图像坐标
        points_2d = (camera_matrix @ points_cam.T).T  # (N, 3)
        points_2d = points_2d[:, :2] / points_2d[:, 2:3]  # (N, 2) - 归一化
        
        # 检查是否在图像范围内
        u = points_2d[:, 0]
        v = points_2d[:, 1]
        in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        
        valid_mask = valid_depth & in_image
        
        projected = points_2d
        
        return projected, depths, valid_mask
    
    def frustum_culling(
        self,
        points: np.ndarray,
        extrinsic: np.ndarray,
        fov_x: float = 60.0,
        fov_y: float = 60.0,
        near_plane: float = 0.1,
        far_plane: float = 100.0
    ) -> np.ndarray:
        """
        视锥体剔除 - 快速剔除明显不可见的点
        
        Args:
            points: 3D点云 (N, 3)
            extrinsic: 相机外参矩阵 (4, 4) - 世界到相机的变换
            fov_x: 水平视场角（度）
            fov_y: 垂直视场角（度）
            near_plane: 近裁剪面
            far_plane: 远裁剪面
            
        Returns:
            visible_mask: 可见点掩码 (N,)
        """
        # 齐次坐标
        points_h = np.hstack([points, np.ones((len(points), 1))])  # (N, 4)
        
        # 世界坐标 -> 相机坐标
        points_cam = (extrinsic @ points_h.T).T  # (N, 4)
        points_cam = points_cam[:, :3]  # (N, 3)
        
        # 深度检查
        z = points_cam[:, 2]
        valid_depth = (z > near_plane) & (z < far_plane)
        
        # 视锥体角度检查（简化版）
        # tan(fov/2) = x/z 或 y/z 的阈值
        tan_fov_x = np.tan(np.radians(fov_x / 2))
        tan_fov_y = np.tan(np.radians(fov_y / 2))
        
        x, y = points_cam[:, 0], points_cam[:, 1]
        in_frustum_x = np.abs(x) < z * tan_fov_x * 1.2  # 留一些余量
        in_frustum_y = np.abs(y) < z * tan_fov_y * 1.2
        
        return valid_depth & in_frustum_x & in_frustum_y
    
    def _process_single_image(
        self,
        img_path: str,
        extrinsic: np.ndarray,
        points: np.ndarray,
        camera_matrix: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, int, int, int]:
        """
        处理单张图像，返回可见点的索引数组及统计信息
        
        Args:
            img_path: 图像路径
            extrinsic: 相机外参矩阵 (4, 4)
            points: 3D点云 (N, 3)
            camera_matrix: 相机内参矩阵 (3, 3)
            image_shape: 图像尺寸 (height, width)
            
        Returns:
            visible_feature_rich_indices: 落在feature-rich区域的点的索引数组
            n_features: 提取的SIFT特征点数量
            n_valid_projected: 有效投影的点数量
            n_feature_rich_pixels: feature-rich像素数量
        """
        # 1. 提取特征
        features = self.extract_features(img_path)
        n_features = len(features)
        
        if n_features == 0:
            return np.array([], dtype=np.int64), 0, 0, 0
        
        # 2. 计算密度
        density = self.compute_feature_density(features, image_shape)
        
        # 3. 获取 feature-rich 掩码
        feature_rich_mask = self.get_feature_rich_mask(density)
        n_feature_rich_pixels = np.sum(feature_rich_mask)
        
        # 4. 投影点云（向量化）
        projected, depths, valid_mask = self.project_points_to_image(
            points, camera_matrix, extrinsic, image_shape
        )
        
        # 5. 向量化处理所有有效投影点
        valid_indices = np.where(valid_mask)[0]
        n_valid_projected = len(valid_indices)
        
        if n_valid_projected == 0:
            return np.array([], dtype=np.int64), n_features, 0, n_feature_rich_pixels
        
        # 获取有效点的投影坐标
        valid_projected = projected[valid_indices]
        valid_depths = depths[valid_indices]
        
        # 转换为整数坐标并裁剪到图像范围内
        u_int = np.clip(valid_projected[:, 0].astype(int), 0, image_shape[1] - 1)
        v_int = np.clip(valid_projected[:, 1].astype(int), 0, image_shape[0] - 1)
        
        # 深度容差判断：对于每个像素，只保留最前面的点（考虑depth_tolerance）
        # 论文算法：对于投影到同一像素的所有点，记录最小深度d
        # 只有满足 |Z_i - d| < depth_tolerance 的点才被认为是可见的
        if self.depth_tolerance > 0:
            # 使用字典按像素分组，找出每个像素的最小深度
            pixel_to_min_depth = {}
            for i in range(len(u_int)):
                pixel_key = (v_int[i], u_int[i])  # (row, col)
                depth = valid_depths[i]
                if pixel_key not in pixel_to_min_depth or depth < pixel_to_min_depth[pixel_key]:
                    pixel_to_min_depth[pixel_key] = depth
            
            # 只保留深度接近最小深度的点（在depth_tolerance范围内）
            visible_mask = np.zeros(len(u_int), dtype=bool)
            for i in range(len(u_int)):
                pixel_key = (v_int[i], u_int[i])
                min_depth = pixel_to_min_depth[pixel_key]
                if np.abs(valid_depths[i] - min_depth) < self.depth_tolerance:
                    visible_mask[i] = True
            
            # 更新有效索引和坐标
            visible_indices_in_valid = np.where(visible_mask)[0]
            if len(visible_indices_in_valid) == 0:
                return np.array([], dtype=np.int64), n_features, 0, n_feature_rich_pixels
            
            # 更新用于后续处理的数组
            valid_indices = valid_indices[visible_indices_in_valid]
            u_int = u_int[visible_indices_in_valid]
            v_int = v_int[visible_indices_in_valid]
        
        # 检查哪些点落在 feature-rich 区域
        in_feature_rich = feature_rich_mask[v_int, u_int]
        
        # 返回落在 feature-rich 区域的点的原始索引
        return valid_indices[in_feature_rich], n_features, n_valid_projected, n_feature_rich_pixels
    
    def label_points_by_feature_density(
        self,
        points: np.ndarray,
        image_paths: List[str],
        camera_matrix: np.ndarray,
        extrinsics: List[np.ndarray],
        image_shape: Tuple[int, int],
        max_workers: int = 4
    ) -> np.ndarray:
        """
        根据图像特征密度标记点云（优化版：多线程并行处理图像）
        
        对每个图像：
        1. 提取SIFT特征
        2. 计算KDE密度
        3. 确定feature-rich区域
        4. 投影点云并标记落在feature-rich区域的点
        
        Args:
            points: 3D点云 (N, 3)
            image_paths: 图像路径列表
            camera_matrix: 相机内参矩阵 (3, 3)
            extrinsics: 相机外参矩阵列表 [(4, 4), ...]
            image_shape: 图像尺寸 (height, width)
            max_workers: 最大并行线程数
            
        Returns:
            labels: 点云标签 (N,)，True表示feature-rich
        """
        n_points = len(points)
        labels = np.zeros(n_points, dtype=bool)
        
        print(f"处理 {len(image_paths)} 张图像...")
        print(f"并行线程数: {max_workers}")
        
        # 使用线程池并行处理图像
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(
                    self._process_single_image,
                    img_path, extrinsic, points, camera_matrix, image_shape
                ): i 
                for i, (img_path, extrinsic) in enumerate(zip(image_paths, extrinsics))
            }
            
            # 收集结果
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                img_path = image_paths[i]
                
                try:
                    result = future.result()
                    feature_rich_indices = result[0]
                    n_features = result[1]
                    n_valid_projected = result[2]
                    n_feature_rich_pixels = result[3]
                    
                    # 标记这些点为 feature-rich
                    if len(feature_rich_indices) > 0:
                        labels[feature_rich_indices] = True
                    
                    # 计算 feature-rich 像素比例
                    total_pixels = image_shape[0] * image_shape[1]
                    feature_rich_ratio = n_feature_rich_pixels / total_pixels * 100
                    
                    print(f"  图像 {i+1}/{len(image_paths)}: {os.path.basename(img_path)} - "
                          f"SIFT: {n_features:5d} | "
                          f"投影: {n_valid_projected:6d} | "
                          f"Feature-rich像素: {n_feature_rich_pixels:6d} ({feature_rich_ratio:4.1f}%) | "
                          f"标记: {len(feature_rich_indices):6d} 点")
                    
                except Exception as e:
                    print(f"  图像 {i+1}/{len(image_paths)}: {os.path.basename(img_path)} - 错误: {e}")
        
        n_feature_rich_points = np.sum(labels)
        print(f"\nFeature-rich 点云数: {n_feature_rich_points} / {n_points} ({n_feature_rich_points/n_points*100:.2f}%)")
        
        return labels
    
    @staticmethod
    def _nearest_to_center_downsample(
        points: np.ndarray,
        colors: np.ndarray,
        voxel_size: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """每个体素只保留距体素中心最近的真实点，颜色不变。"""
        if len(points) == 0:
            return points, colors

        mins = points.min(axis=0)
        voxel_idx = np.floor((points - mins) / voxel_size).astype(np.int64)
        voxel_centers = mins + (voxel_idx + 0.5) * voxel_size
        dist_sq = ((points - voxel_centers) ** 2).sum(axis=1)

        max_idx = voxel_idx.max(axis=0) + 1
        flat_key = (voxel_idx[:, 0] * max_idx[1] * max_idx[2]
                    + voxel_idx[:, 1] * max_idx[2]
                    + voxel_idx[:, 2])

        order = np.lexsort((dist_sq, flat_key))
        _, first = np.unique(flat_key[order], return_index=True)
        selected = order[first]

        out_colors = colors[selected] if colors is not None else None
        return points[selected], out_colors

    def voxel_downsample_with_labels(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        colors: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """根据标签对点云进行分区体素降采样（保留最近真实点，颜色不变）"""
        rich_colors  = colors[labels]  if colors is not None else None
        normal_colors = colors[~labels] if colors is not None else None

        print(f"\n降采样:")
        print(f"  Feature-rich 点: {labels.sum()} -> 体素大小 {self.voxel_size_rich}m")
        print(f"  普通点: {(~labels).sum()} -> 体素大小 {self.voxel_size_normal}m")

        rp, rc = self._nearest_to_center_downsample(
            points[labels], rich_colors, self.voxel_size_rich)
        np_, nc = self._nearest_to_center_downsample(
            points[~labels], normal_colors, self.voxel_size_normal)

        parts_p = [p for p in [rp, np_] if len(p) > 0]
        result_points = np.vstack(parts_p) if parts_p else np.empty((0, 3))

        result_colors = None
        if colors is not None:
            parts_c = [c for c in [rc, nc] if c is not None and len(c) > 0]
            result_colors = np.vstack(parts_c) if parts_c else None

        print(f"  降采样后: {len(result_points)} 点")
        return result_points, result_colors
    
    def filter(
        self,
        point_cloud: o3d.geometry.PointCloud,
        image_paths: List[str],
        camera_matrix: np.ndarray,
        extrinsics: List[np.ndarray],
        image_shape: Tuple[int, int],
        max_workers: int = 4
    ) -> o3d.geometry.PointCloud:
        """
        执行完整的动态体素滤波流程
        
        Args:
            point_cloud: 输入点云
            image_paths: 图像路径列表
            camera_matrix: 相机内参矩阵 (3, 3)
            extrinsics: 相机外参矩阵列表
            image_shape: 图像尺寸 (height, width)
            max_workers: 最大并行线程数
            
        Returns:
            滤波后的点云
        """
        points = np.asarray(point_cloud.points)
        colors = np.asarray(point_cloud.colors) if point_cloud.has_colors() else None
        
        print(f"输入点云: {len(points)} 点")
        
        # 1. 标记点云（多线程并行）
        labels = self.label_points_by_feature_density(
            points, image_paths, camera_matrix, extrinsics, image_shape, max_workers
        )
        
        # 2. 分区降采样
        filtered_points, filtered_colors = self.voxel_downsample_with_labels(
            points, labels, colors
        )
        
        # 3. 创建输出点云
        result = o3d.geometry.PointCloud()
        result.points = o3d.utility.Vector3dVector(filtered_points)
        if filtered_colors is not None:
            result.colors = o3d.utility.Vector3dVector(filtered_colors)
        
        return result


def save_colmap_points3d(point_cloud: o3d.geometry.PointCloud, output_path: str):
    """
    将点云保存为 COLMAP points3D.txt 格式
    
    COLMAP points3D.txt 格式：
    # 3D point list with one line of data per point:
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    
    Args:
        point_cloud: Open3D 点云对象
        output_path: 输出文件路径
    """
    points = np.asarray(point_cloud.points)
    has_colors = point_cloud.has_colors()
    
    if has_colors:
        colors = np.asarray(point_cloud.colors)
        # Open3D 颜色是 0-1 浮点数，转换为 0-255 整数
        colors = (colors * 255).astype(np.uint8)
    else:
        # 默认白色
        colors = np.full((len(points), 3), 255, dtype=np.uint8)
    
    with open(output_path, 'w') as f:
        # 写入文件头
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}\n")
        
        # 写入每个点
        for i in range(len(points)):
            point_id = i + 1  # COLMAP 使用 1-based ID
            x, y, z = points[i]
            r, g, b = colors[i]
            error = 0.0  # 重投影误差，这里设为 0（未知）
            
            # 格式: POINT3D_ID, X, Y, Z, R, G, B, ERROR
            # TRACK 信息为空（因为我们是 LiDAR 点云，没有 SfM 的 track 信息）
            f.write(f"{point_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {error:.6f}\n")
    
    print(f"    写入 {len(points)} 个点到 {output_path}")


def load_colmap_cameras(cameras_path: str) -> Dict[int, Dict]:
    """
    加载 COLMAP cameras.txt 文件
    
    Returns:
        相机参数字典 {camera_id: {model, width, height, params}}
    """
    cameras = {}
    
    with open(cameras_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            
            cameras[camera_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    
    return cameras


def load_colmap_images(images_path: str) -> Dict[int, Dict]:
    """
    加载 COLMAP images.txt 文件
    
    Returns:
        图像位姿字典 {image_id: {qw, qx, qy, qz, tx, ty, tz, camera_id, name, extrinsic}}
    """
    images = {}
    
    with open(images_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line or line.startswith('#'):
            i += 1
            continue
        
        # 读取图像信息行
        parts = line.split()
        image_id = int(parts[0])
        qw, qx, qy, qz = [float(x) for x in parts[1:5]]
        tx, ty, tz = [float(x) for x in parts[5:8]]
        camera_id = int(parts[8])
        name = parts[9]
        
        # 四元数转旋转矩阵
        # COLMAP格式: qw, qx, qy, qz
        R = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        
        t = np.array([tx, ty, tz])
        
        # 构建外参矩阵 (世界到相机)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t
        
        images[image_id] = {
            'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
            'tx': tx, 'ty': ty, 'tz': tz,
            'camera_id': camera_id,
            'name': name,
            'extrinsic': extrinsic
        }
        
        # 跳过点观测行
        i += 2
    
    return images


def build_camera_matrix(camera_params: Dict) -> np.ndarray:
    """
    从 COLMAP 相机参数构建相机内参矩阵
    
    Args:
        camera_params: 包含 model, width, height, params 的字典
        
    Returns:
        相机内参矩阵 (3, 3)
    """
    model = camera_params['model']
    params = camera_params['params']
    
    if model == 'PINHOLE':
        # PINHOLE: fx, fy, cx, cy
        fx, fy, cx, cy = params
    elif model == 'SIMPLE_PINHOLE':
        # SIMPLE_PINHOLE: f, cx, cy
        f, cx, cy = params
        fx = fy = f
    else:
        raise NotImplementedError(f"不支持的相机模型: {model}")
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    return K


def load_scan_alignment(mlp_path: str, scan_filename: str) -> np.ndarray:
    """
    从 ETH3D scan_alignment.mlp 文件中提取指定扫描文件的对齐变换矩阵。

    scan_alignment.mlp 是 MeshLab 项目文件（XML格式），包含将每个原始扫描
    从扫描仪坐标系变换到世界/COLMAP坐标系的 4x4 矩阵。

    scan_clean.ply 不存在时，需直接使用 scan1.ply/scan2.ply，并手动应用此矩阵。

    Args:
        mlp_path:      scan_alignment.mlp 文件的完整路径
        scan_filename: 目标扫描文件名，如 'scan1.ply'（仅文件名，不含路径）

    Returns:
        4x4 numpy 矩阵，将点从扫描仪坐标系变换到世界坐标系
    """
    tree = ET.parse(mlp_path)
    root = tree.getroot()

    target_name = os.path.basename(scan_filename)
    for mesh in root.iter('MLMesh'):
        label = mesh.get('label', '') or mesh.get('filename', '')
        if os.path.basename(label) == target_name:
            matrix_elem = mesh.find('MLMatrix44')
            if matrix_elem is not None:
                values = [float(x) for x in matrix_elem.text.strip().split()]
                return np.array(values, dtype=np.float64).reshape(4, 4)

    raise ValueError(f"在 {mlp_path} 中未找到 '{target_name}' 的变换矩阵")


def apply_scan_alignment(pcd: o3d.geometry.PointCloud, alignment_matrix: np.ndarray) -> o3d.geometry.PointCloud:
    """
    将 4x4 对齐矩阵应用到点云（in-place），使点云从扫描仪坐标系进入世界坐标系。

    Args:
        pcd:              Open3D 点云对象
        alignment_matrix: 4x4 变换矩阵（来自 load_scan_alignment）

    Returns:
        变换后的同一点云对象
    """
    points = np.asarray(pcd.points)
    ones = np.ones((len(points), 1), dtype=np.float64)
    points_h = np.hstack([points, ones])                  # (N, 4)
    points_world = (alignment_matrix @ points_h.T).T[:, :3]  # (N, 3)
    pcd.points = o3d.utility.Vector3dVector(points_world)
    return pcd


def process_eth3d_scene(
    scene_name: str,
    eth3d_base_path: str = r'D:\ETH3D',
    output_base_path: str = r'D:\Li-GS_data_process\output'
):
    """
    处理 ETH3D 场景
    
    Args:
        scene_name: 场景名称（如 'courtyard'）
        eth3d_base_path: ETH3D 数据集根目录
        output_base_path: 输出根目录
    """
    print(f"=" * 60)
    print(f"处理场景: {scene_name}")
    print(f"=" * 60)
    
    # 路径设置
    scene_path = os.path.join(eth3d_base_path, f'{scene_name}_dslr_undistorted', scene_name)
    scan_path = os.path.join(eth3d_base_path, f'{scene_name}_scan_clean', scene_name, 'scan_clean')
    
    images_dir = os.path.join(scene_path, 'images', 'dslr_images_undistorted')
    calibration_dir = os.path.join(scene_path, 'dslr_calibration_undistorted')
    
    cameras_path = os.path.join(calibration_dir, 'cameras.txt')
    images_txt_path = os.path.join(calibration_dir, 'images.txt')
    
    # 检查路径
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"图像目录不存在: {images_dir}")
    if not os.path.exists(cameras_path):
        raise FileNotFoundError(f"相机参数文件不存在: {cameras_path}")
    if not os.path.exists(images_txt_path):
        raise FileNotFoundError(f"图像位姿文件不存在: {images_txt_path}")
    
    # 查找点云文件
    scan_files = [f for f in os.listdir(scan_path) if f.endswith('.ply')]
    if not scan_files:
        raise FileNotFoundError(f"扫描点云文件不存在: {scan_path}")
    
    scan_file = os.path.join(scan_path, scan_files[0])
    print(f"使用点云文件: {scan_files[0]}")
    
    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    
    # 加载点云
    pcd = o3d.io.read_point_cloud(scan_file)
    print(f"  点云点数: {len(pcd.points)}")
    print(f"  点云有颜色: {pcd.has_colors()}")
    
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
        img_path = os.path.join(images_dir, img_info['name'])
        
        if os.path.exists(img_path):
            image_paths.append(img_path)
            extrinsics.append(img_info['extrinsic'])
        else:
            print(f"  警告: 图像不存在 {img_path}")
    
    print(f"  有效图像: {len(image_paths)}")
    
    image_shape = (camera_params['height'], camera_params['width'])
    
    # 3. 执行动态体素滤波
    print("\n[3/4] 执行动态体素滤波...")
    
    filter = DynamicVoxelFilter(
        feature_rich_ratio=0.05,    # 前5%
        voxel_size_rich=0.02,       # feature-rich: 0.02m
        voxel_size_normal=0.05,     # 普通: 0.05m
        sift_nfeatures=3000,        # 限制3000个特征点
        kde_bandwidth=80.0          # KDE带宽80像素
    )
    
    filtered_pcd = filter.filter(
        pcd, image_paths, K, extrinsics, image_shape
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
    import shutil
    output_images_dir = os.path.join(output_dir, 'images')
    os.makedirs(output_images_dir, exist_ok=True)
    
    for img_path in image_paths:
        dst_path = os.path.join(output_images_dir, os.path.basename(img_path))
        shutil.copy2(img_path, dst_path)
    print(f"  复制 {len(image_paths)} 张图像到: {output_images_dir}")
    
    # 复制相机参数
    import shutil
    shutil.copy2(cameras_path, os.path.join(output_dir, 'cameras.txt'))
    shutil.copy2(images_txt_path, os.path.join(output_dir, 'images.txt'))
    print(f"  复制相机参数")
    
    print(f"\n处理完成！")
    print(f"  输入: {len(pcd.points)} 点")
    print(f"  输出: {len(filtered_pcd.points)} 点")
    print(f"  压缩比: {len(filtered_pcd.points) / len(pcd.points) * 100:.2f}%")
    
    return filtered_pcd


if __name__ == '__main__':
    # 处理 courtyard 场景
    try:
        process_eth3d_scene('courtyard')
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
