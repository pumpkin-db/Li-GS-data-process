"""
删除文件夹中第 2、4、6... 张图片（按文件名排序后每隔一张删一张）
用法：
    python thin_images.py D:\Li-GS_data_process\output\house\筛选图片4
    python thin_images.py D:\Li-GS_data_process\output\house\筛选图片4 --dry-run   # 只预览，不实际删除
"""

import argparse
import os
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def thin_images(folder: Path, dry_run: bool):
    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if not images:
        print("未找到图片文件。")
        return

    to_delete = images[1::2]  # 索引 1, 3, 5 ... 即第 2, 4, 6 张

    print(f"共 {len(images)} 张图片，将删除 {len(to_delete)} 张，保留 {len(images) - len(to_delete)} 张")
    if dry_run:
        print("（dry-run 模式，以下为将被删除的文件）")

    for p in to_delete:
        if dry_run:
            print(f"  [预览] {p.name}")
        else:
            p.unlink()
            print(f"  已删除 {p.name}")

    if not dry_run:
        print("完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="图片所在文件夹路径")
    parser.add_argument("--dry-run", action="store_true", help="只预览，不实际删除")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"错误：{folder} 不是有效目录")
        exit(1)

    thin_images(folder, args.dry_run)
