"""
按原始顺序批量重命名图片为 0001, 0002, 0003...
可选：转换图片格式（支持 HEIC → jpg/png 等）

用法：
    python rename_picture.py D:\my_scene\images
    python rename_picture.py D:\my_scene\images --convert jpg   # 转换为 jpg
    python rename_picture.py D:\my_scene\images --convert png   # 转换为 png
    python rename_picture.py   （不带参数时会提示输入路径）

依赖：
    pip install pillow pillow-heif
"""

import os
import sys
import re
import argparse
from pathlib import Path

IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
    '.webp', '.ico', '.heic', '.heif', '.avif', '.jfif',
    '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP', '.TIFF', '.TIF',
    '.WEBP', '.ICO', '.HEIC', '.HEIF', '.AVIF', '.JFIF',
}


def natural_sort_key(name: str) -> list:
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', name)]


def convert_image(src: Path, dst: Path):
    from PIL import Image
    img = Image.open(src)
    if img.mode in ('RGBA', 'P') and dst.suffix.lower() in ('.jpg', '.jpeg'):
        img = img.convert('RGB')
    img.save(dst)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('folder', nargs='?', default=None)
    parser.add_argument('--convert', choices=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'],
                        default=None, metavar='FORMAT',
                        help='转换输出格式，如 jpg 或 png')
    args, _ = parser.parse_known_args()

    # 获取文件夹路径
    if args.folder:
        raw = args.folder
    else:
        raw = input("请输入图片文件夹路径: ")
    folder = Path(raw.strip().strip('"').strip("'"))

    if not folder.is_dir():
        print(f"[错误] 文件夹不存在: {folder}")
        sys.exit(1)

    # 如果要转换格式，提前注册 HEIC 支持
    if args.convert:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            pass  # 没有 HEIC 文件时不需要

    # 收集图片文件
    images = sorted(
        [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in
         {e.lower() for e in IMAGE_EXTENSIONS}],
        key=lambda f: natural_sort_key(f.name),
    )

    if not images:
        print(f"[提示] 未找到图片文件")
        return

    total = len(images)
    digits = max(4, len(str(total)))
    out_ext = f".{args.convert}" if args.convert else None

    # 预览
    print(f"\n找到 {total} 张图片，将按当前顺序重命名为 0001~{str(total).zfill(digits)}")
    if out_ext:
        print(f"输出格式：{out_ext}")
    print("\n前5条预览:")
    print(f"{'原文件名':<40} → {'新文件名'}")
    print("-" * 60)
    for i, img in enumerate(images[:5], 1):
        new_ext = out_ext if out_ext else img.suffix
        print(f"{img.name:<40} → {str(i).zfill(digits)}{new_ext}")
    if total > 5:
        print(f"... 以及其余 {total - 5} 张图片")

    print()
    confirm = input("确认执行? (y/N): ").strip().lower()
    if confirm != 'y':
        print("已取消。")
        return

    success = 0
    for i, img in enumerate(images, 1):
        new_ext = out_ext if out_ext else img.suffix
        final_name = folder / f"{str(i).zfill(digits)}{new_ext}"
        tmp_name = folder / f"__tmp_rename_{i:04d}{img.suffix}"

        if out_ext and img.suffix.lower() != out_ext:
            # 需要格式转换：直接转换到最终文件名，再删原文件
            try:
                convert_image(img, final_name)
                img.unlink()
                success += 1
            except Exception as e:
                print(f"[警告] 转换失败 {img.name}: {e}")
        else:
            # 不需要转换：先改临时名再改最终名（避免冲突）
            img.rename(tmp_name)
            tmp_name.rename(final_name)
            success += 1

    print(f"\n完成！已处理 {success} / {total} 张图片。")


if __name__ == '__main__':
    main()
