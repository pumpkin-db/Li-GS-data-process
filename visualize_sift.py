"""
SIFT 特征点可视化脚本
输入：3 张图像路径
输出：3 张带特征点的图像（保存到同目录，文件名加 _sift 后缀）
"""
import cv2
import sys
from pathlib import Path

# ============================================================
# 修改这里：填入 3 张图像路径
# ============================================================
IMAGE_PATHS = [
    r"D:\Li-GS_data_process\output\20260421084017_ours-3\projected_images\images\0000.jpg",
    r"D:\Li-GS_data_process\output\20260421084017_ours-3\projected_images\images\0057.jpg",
    r"D:\Li-GS_data_process\output\20260421084017_ours-3\projected_images\images\0169.jpg",
]

# SIFT 参数（与 voxel_filter.py 保持一致）
SIFT_NFEATURES = 3000   # 0 = 不限制

# ============================================================

def extract_and_draw(image_path: str, nfeatures: int) -> str:
    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] 无法读取: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    keypoints = sift.detect(gray, None)

    out = cv2.drawKeypoints(
        img, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    p = Path(image_path)
    out_path = str(p.parent / (p.stem + '_sift' + p.suffix))
    cv2.imwrite(out_path, out)
    print(f"  {p.name}: {len(keypoints)} 个特征点 → {Path(out_path).name}")
    return out_path


print(f"SIFT nfeatures={SIFT_NFEATURES}")
for path in IMAGE_PATHS:
    extract_and_draw(path, SIFT_NFEATURES)
print("完成。")
