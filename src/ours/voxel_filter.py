"""
Li-GS 动态体素滤波模块 (自采数据版本)
基于图像特征密度对点云进行分区降采样

参考: voxel_filtering_eth3d.py
"""

import numpy as np
import cv2
import open3d as o3d
import os
from typing import List, Tuple, Dict, Any
from scipy.stats import gaussian_kde
from concurrent.futures import ThreadPoolExecutor, as_completed


class DynamicVoxelFilter:
    """
    动态体素滤波器
    
    根据图像特征密度对点云进行分区降采样：
    - feature-rich 区域：小体素（0.02m）
    - 普通区域：大体素（0.05m）
    """
    
    def __init__(
        self,
        feature_rich_ratio: float = 0.05,   # feature-rich blocks 比例
        voxel_size_rich: float = 0.02,      # feature-rich 区域体素大小
        voxel_size_normal: float = 0.05,    # 普通区域体素大小
        sift_nfeatures: int = 0,            # SIFT特征点数量（0表示无限制）
        kde_bandwidth: float = 50.0,        # KDE带宽（像素）
        kde_resize_factor: float = 0.25,    # KDE计算时图像缩放因子
        max_workers: int = 4,               # 最大并行线程数
        depth_tolerance: float = 0.02       # 深度容差（用于判断遮挡，单位：米）
    ):
        self.feature_rich_ratio = feature_rich_ratio
        self.voxel_size_rich = voxel_size_rich
        self.voxel_size_normal = voxel_size_normal
        self.sift_nfeatures = sift_nfeatures
        self.kde_bandwidth = kde_bandwidth
        self.kde_resize_factor = kde_resize_factor
        self.max_workers = max_workers
        self.depth_tolerance = depth_tolerance
        
        # 初始化 SIFT 检测器
        self.sift = cv2.SIFT_create(nfeatures=sift_nfeatures)
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """使用 SIFT 提取图像特征点"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        keypoints, _ = self.sift.detectAndCompute(img, None)
        
        if len(keypoints) == 0:
            return np.array([])
        
        features = np.array([kp.pt for kp in keypoints])  # (N, 2)
        return features
    
    def compute_feature_density(
        self, 
        features: np.ndarray, 
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """使用核密度估计（KDE）计算特征密度分布"""
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
        grid_points = np.vstack([xx.ravel(), yy.ravel()])
        
        # 使用高斯核进行KDE
        std_x = np.std(scaled_features[:, 0])
        std_y = np.std(scaled_features[:, 1])
        representative_std = (std_x + std_y) / 2.0
        if representative_std < 1e-6:
            representative_std = 1.0
        bw_factor = (self.kde_bandwidth * scale) / representative_std
        kde = gaussian_kde(scaled_features.T, bw_method=bw_factor)
        
        # 计算密度
        density_small = kde(grid_points)
        density_small = density_small.reshape(small_height, small_width)
        
        # 插值回原尺寸
        density = cv2.resize(
            density_small.astype(np.float32), 
            (original_width, original_height), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # 归一化
        if density.max() > 0:
            density = density / density.max()
        
        return density
    
    def get_feature_rich_mask(
        self, 
        density: np.ndarray
    ) -> np.ndarray:
        """获取 feature-rich 区域的掩码"""
        # 计算阈值
        threshold = np.percentile(density, (1 - self.feature_rich_ratio) * 100)
        
        # 生成掩码
        mask = density >= threshold
        
        return mask
    
    def project_points_to_image(
        self,
        points: np.ndarray,
        camera_matrix: np.ndarray,
        extrinsic: np.ndarray,
        image_shape: Tuple[int, int],
        points_in_cam_coords: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        将3D点云投影到图像平面
        
        Args:
            points: 3D点云 (N, 3)
            camera_matrix: 相机内参矩阵 (3, 3)
            extrinsic: 世界坐标系到相机坐标系的变换矩阵 (4, 4)
            image_shape: 图像尺寸 (height, width)
            points_in_cam_coords: 如果为 True，则点云已经在相机坐标系中，跳过 extrinsic 变换
        
        Returns:
            projected: 投影后的2D坐标 (N, 2) - (u, v)
            depths: 深度值 (N,)
            valid_mask: 有效投影掩码 (N,)
        """
        height, width = image_shape
        
        if points_in_cam_coords:
            # 点云已经在相机坐标系中，直接使用
            points_cam = points
        else:
            # 齐次坐标
            points_h = np.hstack([points, np.ones((len(points), 1))])  # (N, 4)
            
            # 世界坐标 -> 相机坐标
            points_cam = (extrinsic @ points_h.T).T  # (N, 4)
            points_cam = points_cam[:, :3]  # (N, 3)
        
        # 深度值（Z坐标）
        depths = points_cam[:, 2]
        
        # 只保留相机前方的点 (Z > 0)
        valid_depth = depths > 0
        
        # 相机坐标 -> 图像坐标
        points_2d = (camera_matrix @ points_cam.T).T  # (N, 3)
        points_2d = points_2d[:, :2] / points_2d[:, 2:3]  # (N, 2)
        
        # 检查是否在图像范围内
        u = points_2d[:, 0]
        v = points_2d[:, 1]
        in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        
        valid_mask = valid_depth & in_image
        
        return points_2d, depths, valid_mask
    
    def _process_single_image(
        self,
        img_path: str,
        extrinsic: np.ndarray,
        points: np.ndarray,
        camera_matrix: np.ndarray,
        image_shape: Tuple[int, int],
        points_in_cam_coords: bool = False
    ) -> Tuple[np.ndarray, int, int, int]:
        """处理单张图像，返回可见点的索引数组及统计信息
        
        注意：当 points_in_cam_coords=True 时，点云已在参考相机坐标系中，
        此时投影统计数字（深度>0、在图像范围内、有效投影）反映的是参考相机视角，
        而非当前处理图像的相机视角。这是设计上的权衡，不影响 feature-rich 标记的正确性。
        """
        # 1. 提取特征
        features = self.extract_features(img_path)
        n_features = len(features)
        
        if n_features == 0:
            return np.array([], dtype=np.int64), 0, 0, 0, 0, 0
        
        # 2. 计算密度
        density = self.compute_feature_density(features, image_shape)
        
        # 3. 获取 feature-rich 掩码
        feature_rich_mask = self.get_feature_rich_mask(density)
        n_feature_rich_pixels = np.sum(feature_rich_mask)
        
        # 4. 投影点云
        projected, depths, valid_mask = self.project_points_to_image(
            points, camera_matrix, extrinsic, image_shape, points_in_cam_coords
        )
        
        # 5. 处理有效投影点（深度>0 且在图像内）
        valid_indices = np.where(valid_mask)[0]
        n_valid_projected = len(valid_indices)
        
        if n_valid_projected == 0:
            return np.array([], dtype=np.int64), n_features, 0, n_feature_rich_pixels
        
        # 获取有效点的投影坐标
        valid_projected = projected[valid_indices]
        
        # 转换为整数坐标并裁剪到图像范围内
        valid_depths = depths[valid_indices]
        u_int = np.clip(valid_projected[:, 0].astype(int), 0, image_shape[1] - 1)
        v_int = np.clip(valid_projected[:, 1].astype(int), 0, image_shape[0] - 1)

        # 深度容差过滤：每个像素只保留最近表面附近的点（遮挡处理）
        if self.depth_tolerance > 0:
            pixel_to_min_depth = {}
            for i in range(len(u_int)):
                pixel_key = (v_int[i], u_int[i])
                depth = valid_depths[i]
                if pixel_key not in pixel_to_min_depth or depth < pixel_to_min_depth[pixel_key]:
                    pixel_to_min_depth[pixel_key] = depth

            visible_mask = np.zeros(len(u_int), dtype=bool)
            for i in range(len(u_int)):
                pixel_key = (v_int[i], u_int[i])
                if np.abs(valid_depths[i] - pixel_to_min_depth[pixel_key]) < self.depth_tolerance:
                    visible_mask[i] = True

            visible_indices_in_valid = np.where(visible_mask)[0]
            if len(visible_indices_in_valid) == 0:
                return np.array([], dtype=np.int64), n_features, 0, n_feature_rich_pixels

            valid_indices = valid_indices[visible_indices_in_valid]
            u_int = u_int[visible_indices_in_valid]
            v_int = v_int[visible_indices_in_valid]

        # 检查哪些点落在 feature-rich 区域
        in_feature_rich = feature_rich_mask[v_int, u_int]
        
        # 返回落在 feature-rich 区域的点的原始索引
        return valid_indices[in_feature_rich], n_features, n_valid_projected, n_feature_rich_pixels
    
    def filter(
        self,
        pointcloud: np.ndarray,  # (N, 6) - XYZRGB
        projected_images: List[Dict[str, Any]],  # 包含 image_path, camera_matrix, extrinsic, image_shape
        lidar_to_cam_transform: np.ndarray = None  # (4, 4) LiDAR 到相机的变换矩阵
    ) -> np.ndarray:
        """
        执行动态体素滤波
        
        Args:
            pointcloud: 点云数组 (N, 6) - XYZRGB
            projected_images: 投影图像信息列表
            lidar_to_cam_transform: LiDAR 到相机的变换矩阵 (4x4)，如果点云在 LiDAR 坐标系中需要提供
            
        Returns:
            滤波后的点云 (M, 6)
        """
        points = pointcloud[:, :3]
        colors = pointcloud[:, 3:6] if pointcloud.shape[1] >= 6 else None
        n_points = len(points)
        
        print(f"输入点云: {n_points} 点")
        print(f"处理 {len(projected_images)} 张图像...")
        print(f"并行线程数: {self.max_workers}")
        
        # 如果提供了 LiDAR 到相机的变换，先将点云转换到相机坐标系
        if lidar_to_cam_transform is not None:
            print("应用 LiDAR 到相机的坐标变换...")
            points_h = np.hstack([points, np.ones((n_points, 1))])  # (N, 4)
            points_cam = (lidar_to_cam_transform @ points_h.T).T  # (N, 4)
            points = points_cam[:, :3]  # (N, 3)
            print(f"  点云已转换到相机坐标系")
        
        # 标记点云
        labels = np.zeros(n_points, dtype=bool)
        
        # 使用线程池并行处理图像
        # 如果使用了 LiDAR 到相机的变换，点云已经在相机坐标系中
        points_in_cam_coords = (lidar_to_cam_transform is not None)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {}
            for i, img_info in enumerate(projected_images):
                future = executor.submit(
                    self._process_single_image,
                    img_info['image_path'],
                    img_info['extrinsic'],
                    points,
                    img_info['camera_matrix'],
                    img_info['image_shape'],
                    points_in_cam_coords
                )
                future_to_idx[future] = i
            
            # 收集结果
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                img_info = projected_images[i]
                img_path = img_info['image_path']
                
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
                    image_shape = img_info['image_shape']
                    total_pixels = image_shape[0] * image_shape[1]
                    feature_rich_ratio = n_feature_rich_pixels / total_pixels * 100
                    
                    print(f"  图像 {i+1}/{len(projected_images)}: {os.path.basename(img_path)} - "
                          f"SIFT: {n_features:5d} | "
                          f"投影: {n_valid_projected:6d} | "
                          f"Feature-rich像素: {n_feature_rich_pixels:6d} ({feature_rich_ratio:4.1f}%) | "
                          f"标记: {len(feature_rich_indices):6d} 点")
                    
                except Exception as e:
                    print(f"  图像 {i+1}/{len(projected_images)}: {os.path.basename(img_path)} - 错误: {e}")
        
        n_feature_rich_points = np.sum(labels)
        print(f"\nFeature-rich 点云数: {n_feature_rich_points} / {n_points} ({n_feature_rich_points/n_points*100:.2f}%)")
        
        # 分区降采样
        return self._voxel_downsample(points, colors, labels)
    
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

        # lexsort: 先按体素 key 分组，组内按距离升序，取第一个
        order = np.lexsort((dist_sq, flat_key))
        _, first = np.unique(flat_key[order], return_index=True)
        selected = order[first]

        out_colors = colors[selected] if colors is not None else None
        return points[selected], out_colors

    def _voxel_downsample(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """根据标签对点云进行分区体素降采样（保留最近真实点，颜色不变）"""
        rich_mask = labels
        normal_mask = ~labels
        rich_colors  = colors[rich_mask]  if colors is not None else None
        normal_colors = colors[normal_mask] if colors is not None else None

        print(f"\n降采样:")
        print(f"  Feature-rich 点: {rich_mask.sum()} -> 体素大小 {self.voxel_size_rich}m")
        print(f"  普通点: {normal_mask.sum()} -> 体素大小 {self.voxel_size_normal}m")

        rp, rc = self._nearest_to_center_downsample(
            points[rich_mask], rich_colors, self.voxel_size_rich)
        np_, nc = self._nearest_to_center_downsample(
            points[normal_mask], normal_colors, self.voxel_size_normal)

        all_points = [p for p in [rp, np_] if len(p) > 0]
        if not all_points:
            return np.array([])

        filtered_points = np.vstack(all_points)

        if colors is not None:
            all_colors = [c for c in [rc, nc] if c is not None and len(c) > 0]
            filtered_colors = np.vstack(all_colors)
            return np.hstack([filtered_points, filtered_colors])

        return filtered_points
