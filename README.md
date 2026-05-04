# Li-GS 数据处理

MID360 LiDAR + 鱼眼相机自采数据的预处理 pipeline，将原始点云和相机位姿转换为 COLMAP 格式，供 3DGS 训练使用。同时支持 ETH3D 数据集。

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
| `run_filtering_eth3d.py` | ETH3D 数据集流程：读取扫描点云 + COLMAP 位姿，做动态体素滤波后重新导出 |

### 工具脚本

| 文件 | 说明 |
|---|---|
| `debug_projection.py` | 将 `points3D.ply` 投影到图像，验证外参是否正确，输出带点的图片 |
| `visualize_sift.py` | 可视化指定图像的 SIFT 特征点分布，输出加 `_sift` 后缀的图片 |
| `fisheye_standalone.py` | 独立鱼眼投影：单张图 + 内参文件 → back 面透视图，不依赖完整 pipeline |
| `pointcloud_convert_standalone.py` | 独立点云格式转换：`.las` ↔ `.ply` |
| `voxel_filter_standalone.py` | 独立体素滤波：对单个点云文件降采样 |
| `thin_images.py` | 删除文件夹中第 2、4、6... 张图片（隔一张删一张，用于减少冗余帧） |
| `rename_picture.py` | 将文件夹内图片按原始顺序重命名为 `0001`、`0002`... |

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

### ETH3D 数据集（run_filtering_eth3d.py）

在 `config/eth3d_scenes.yaml` 配置好路径后运行：

```bat
python run_filtering_eth3d.py --scene courtyard
```

---

### 工具脚本

```bat
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
