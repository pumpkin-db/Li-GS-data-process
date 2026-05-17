# Li-GS 数据处理

MID360 LiDAR + 鱼眼相机自采数据的预处理 pipeline，将原始点云和相机位姿转换为 COLMAP 格式，供 3DGS 训练使用。同时支持 ETH3D 数据集和双相机（2dai）设备。

---

## 一、环境配置

### 前置要求

- [Anaconda](https://www.anaconda.com/download) 或 Miniconda
- Python 3.10

### 创建环境

将 `<安装路径>` 替换为你希望存放环境的目录（例如 `D:\Li-GS_data_process\envs\ligs`）：

```bat
conda create --prefix <安装路径> python=3.10 -y
conda activate <安装路径>
```

### 克隆项目

```bat
git clone https://github.com/pumpkin-db/Li-GS-data-process.git
cd Li-GS_data_process
```

### 安装依赖

```bat
pip install -r requirements.txt
```

### 验证

```bat
python test_env.py
```

---

## 二、代码文件说明

### 入口脚本

| 文件 | 说明 |
|---|---|
| `run_ours.py` | 自采数据主流程：加载 LAS 点云 + 位姿 + 鱼眼内参，投影为透视图，体素滤波，导出 COLMAP 格式 |
| `run_2dai.py` | 双相机（2dai）设备数据处理：加载双目 OCam 标定 + POS 位姿，分别投影左右鱼眼图，写出 COLMAP 格式 |
| `run_filtering_eth3d.py` | ETH3D 数据集流程：读取扫描点云 + `scan_alignment.mlp` 对齐变换 + COLMAP 位姿，做动态体素滤波后重新导出 |

### 工具脚本

| 文件 | 说明 |
|---|---|
| `cloud_txt_to_points3d.py` | `cloud.txt`（X Y Z R G B）→ COLMAP `points3D.txt`，支持体素滤波或随机采样降点 |
| `colmap_to_ply.py` | COLMAP `points3D.txt` → `.ply` 点云，方便在 CloudCompare 中查看 |
| `resize_images.py` | 批量缩放图片，支持 `--scale`（整倍缩小）、`--size`（指定宽高）、`--width`（等比缩放），输出到子目录不覆盖原文件 |
| `viz_camera_poses.py` | 从 COLMAP `images.txt` 提取相机中心和朝向，叠加到点云中输出 `.ply`，在 CloudCompare 中可视化相机轨迹 |
| `debug_projection.py` | 将 `points3D.ply` 投影到图像，验证外参是否正确，输出带点的图片 |
| `visualize_sift.py` | 可视化指定图像的 SIFT 特征点分布，输出加 `_sift` 后缀的图片 |
| `fisheye_standalone.py` | 独立鱼眼投影：单张图 + 内参文件 → back 面透视图，不依赖完整 pipeline |
| `pointcloud_convert_standalone.py` | 独立点云格式转换：`.las` ↔ `.ply` |
| `voxel_filter_standalone.py` | 独立体素滤波：对单个点云文件降采样 |
| `thin_images.py` | 删除文件夹中第 2、4、6... 张图片（隔一张删一张，用于减少冗余帧） |
| `rename_picture.py` | 将文件夹内图片按原始顺序重命名为 `0001`、`0002`... |

### 源码模块

| 路径 | 说明 |
|---|---|
| `src/voxel_filtering_eth3d.py` | ETH3D 动态体素滤波核心，含 `load_scan_alignment` / `apply_scan_alignment` |
| `src/_2dai_ours/calibration.py` | 解析双相机 OCam YAML 标定文件 |
| `src/_2dai_ours/pose_calculator.py` | POS 文件解析，机体位姿 → 相机位姿变换 |
| `src/_2dai_ours/projector.py` | 鱼眼图像投影（OCam 模型）→ 透视图 |
| `src/_2dai_ours/colmap_writer.py` | 写出 `cameras.txt` / `images.txt` / `points3D.txt` |

### 配置文件

| 文件 | 说明 |
|---|---|
| `config/ours_scenes.yaml` | 自采数据场景路径配置 |
| `config/eth3d_scenes.yaml` | ETH3D 场景路径配置 |

---

## 三、使用说明

### 自采数据（run_ours.py）

**第一步：** 在 `config/ours_scenes.yaml` 添加场景：

```yaml
scenes:
  my_scene:
    pointcloud: "E:/_cloud/my_scene/LAS_rgb"
    images:     "E:/_cloud/my_scene/CAMERA/camera1"
    poses:      "E:/_cloud/my_scene/POS/camera_pos.cam"
    camera_intrinsic: "E:/_cloud/my_scene/CALIBRATION_CAMERA/CAMERA_xxx.yaml"
    image_extension: ".jpg"
    start_index: 1
    end_index: 146
```

**第二步：** 修改 `run_ours.py` 顶部的 `CONFIG`，设置场景名和滤波参数，然后运行：

```bat
python run_ours.py --scene my_scene
```

**输出目录结构：**

```
output\my_scene\
├── projected_images\images\   # 投影透视图（供训练使用）
├── cameras.txt
├── images.txt
├── points3D.txt
└── points3D.ply
```

**第三步：** 3DGS 训练：

```bat
cd D:\3d_gsplat_02\gsplat\examples
python simple_trainer.py default ^
    --data_dir D:\Li-GS_data_process\output\my_scene ^
    --result_dir D:\Li-GS_data_process\output\my_scene\results ^
    --max_steps 30000
```

---

### 双相机数据（run_2dai.py）

修改 `run_2dai.py` 顶部的 `CONFIG`，填写标定文件、双目图像目录、POS 文件、点云路径和输出目录，然后运行：

```bat
python run_2dai.py
```

> 说明：`pos_file` 设为 `None` 时仅做投影预览（不生成 COLMAP 文件），解算完成后填写路径再次运行即可。点云建议先用 `voxel_filter_standalone.py` 手动降采样后再填写。

**输出目录结构：**

```
colmap_output\
├── images\          # 左右眼交替投影图（0001.jpg=左眼, 0002.jpg=右眼, ...）
└── sparse\0\
    ├── cameras.txt  # 双相机内参（camera_id=1/2）
    ├── images.txt
    └── points3D.txt
```

---

### ETH3D 数据集（run_filtering_eth3d.py）

> **注意（Bug 修复）：** ETH3D 的扫描点云（`scan1.ply`）在扫描仪自身坐标系中，而 `images.txt` 的相机外参在世界坐标系中，两者不对齐。需要在 `config/eth3d_scenes.yaml` 中提供 `scan_alignment` 字段，指向 MeshLab 项目文件（`scan_alignment.mlp`），该文件记录了扫描坐标系到世界坐标系的 4×4 变换矩阵。不应用此变换，投影结果完全错误。

在 `config/eth3d_scenes.yaml` 配置好路径后运行：

```bat
python run_filtering_eth3d.py --scene courtyard
```

---

### 工具脚本

```bat
# cloud.txt → COLMAP points3D.txt（体素滤波，推荐 --voxel 0.05~0.1）
python cloud_txt_to_points3d.py E:\_cloud\Colmap\sparse\0\cloud.txt --voxel 0.05

# COLMAP points3D.txt → .ply（CloudCompare 查看）
python colmap_to_ply.py E:\_cloud\Colmap\sparse\0\points3D.txt

# 批量缩放图片（分辨率减半，输出到 images_x2 子目录）
python resize_images.py D:\my_scene\images --scale 2

# 可视化相机轨迹（叠加到点云，用 CloudCompare 打开输出的 viz_cameras.ply）
python viz_camera_poses.py E:\_cloud\colmap_output\sparse\0\images.txt
python viz_camera_poses.py E:\_cloud\colmap_output\sparse\0\images.txt E:\_cloud\points.ply --arrow_len 1.0

# 隔一张删一张（先用 --dry-run 预览）
python thin_images.py D:\my_scene\images --dry-run
python thin_images.py D:\my_scene\images

# 按顺序重命名为 0001, 0002...
python rename_picture.py D:\my_scene\images

# 验证外参（修改脚本内 OUTPUT_DIR 后运行）
python debug_projection.py

# 单文件体素滤波（修改脚本内 INPUT_FILE 后运行）
python voxel_filter_standalone.py
```
