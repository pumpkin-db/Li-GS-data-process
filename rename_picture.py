"""
按原始顺序批量重命名图片为 0001, 0002, 0003...

用法：
    python rename_picture.py D:\Li-GS_data_process\output\house\整合图片
    python rename_picture.py   （不带参数时会提示输入路径）
"""

import os
import sys
import re
from pathlib import Path

IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
    '.webp', '.ico', '.heic', '.heif', '.avif', '.jfif',
    '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP', '.TIFF', '.TIF',
    '.WEBP', '.ICO', '.HEIC', '.HEIF', '.AVIF', '.JFIF',
}


def natural_sort_key(name: str) -> list:
    """自然排序: 如 '2.jpg' 排在 '10.jpg' 前面"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', name)]


def main():
    # 获取文件夹路径（支持 Windows 路径，如 D:\my_scene\images）
    if len(sys.argv) > 1:
        raw = " ".join(sys.argv[1:])  # 支持路径中含空格时未加引号的情况
    else:
        raw = input("请输入图片文件夹路径: ")

    folder = Path(raw.strip().strip('"').strip("'"))
    if not folder.is_dir():
        print(f"[错误] 文件夹不存在: {folder}")
        sys.exit(1)

    # 收集图片文件
    images = sorted(
        [f for f in folder.iterdir() if f.is_file() and f.suffix in IMAGE_EXTENSIONS],
        key=lambda f: natural_sort_key(f.name),
    )

    if not images:
        print(f"[提示] 未找到图片文件（支持的格式: {', '.join(set(ext.lower() for ext in IMAGE_EXTENSIONS))}）")
        return

    # 计算需要的位数（至少4位）
    total = len(images)
    digits = max(4, len(str(total)))

    # 预览
    print(f"\n找到 {total} 张图片，将按当前顺序重命名为 0001~{str(total).zfill(digits)}\n")
    print("前5条预览:")
    print(f"{'原文件名':<40} → {'新文件名'}")
    print("-" * 60)
    for i, img in enumerate(images[:5], 1):
        new_name = f"{str(i).zfill(digits)}{img.suffix}"
        print(f"{img.name:<40} → {new_name}")
    if total > 5:
        print(f"... 以及其余 {total - 5} 张图片")

    # 确认
    print()
    confirm = input("确认执行重命名? (y/N): ").strip().lower()
    if confirm != 'y':
        print("已取消。")
        return

    # 第一步: 先重命名为临时文件名，避免命名冲突
    tmp_names = []
    for i, img in enumerate(images, 1):
        tmp_name = folder / f"__tmp_rename_{i:04d}{img.suffix}"
        img.rename(tmp_name)
        tmp_names.append((tmp_name, f"{str(i).zfill(digits)}{img.suffix}"))

    # 第二步: 从临时文件名改为最终文件名
    renamed = 0
    for tmp_path, final_name in tmp_names:
        final_path = folder / final_name
        tmp_path.rename(final_path)
        renamed += 1

    print(f"\n完成! 已重命名 {renamed} 张图片。")


if __name__ == '__main__':
    main()
