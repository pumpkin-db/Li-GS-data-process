"""
Li-GS 自采数据处理模块

模块结构：
- data_loader: 统一数据加载
- keyframe_selector: 关键帧筛选
- ocam_model: OCamCalib 鱼眼模型
- fisheye_projection: 5面投影
- voxel_filter: 动态体素滤波
- colmap_exporter: COLMAP 格式导出

使用流程：
1. DataLoader 加载原始数据
2. KeyFrameSelector 筛选关键帧
3. OCamModel + FisheyeProjector 进行5面投影
4. DynamicVoxelFilter 对点云滤波
5. COLMAPExporter 导出结果
"""

from .data_loader import DataLoader
from .keyframe_selector import KeyFrameSelector
from .ocam_model import OCamModel
from .fisheye_projection import FisheyeProjector
from .voxel_filter import DynamicVoxelFilter
from .colmap_exporter import COLMAPExporter, rotation_matrix_to_quaternion

__all__ = [
    'DataLoader',
    'KeyFrameSelector',
    'OCamModel',
    'FisheyeProjector',
    'DynamicVoxelFilter',
    'COLMAPExporter',
    'rotation_matrix_to_quaternion'
]
