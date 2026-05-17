"""
批量降低图片分辨率

用法：
    python resize_images.py D:\my_scene\images --scale 2      # 分辨率 /2
    python resize_images.py D:\my_scene\images --scale 4      # 分辨率 /4
    python resize_images.py D:\my_scene\images --size 1920 1080  # 指定宽高
    python resize_images.py D:\my_scene\images --width 1920    # 只指定宽，高度等比缩放
    输出到原文件夹下的 resized_x2 / resized_x4 / resized 子目录，不覆盖原文件

依赖：pip install pillow
"""

import argparse
import sys
from pathlib import Path
from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp',
              '.JPG', '.JPEG', '.PNG', '.BMP', '.TIF', '.TIFF', '.WEBP'}


def resize_image(src: Path, dst: Path, target_size: tuple):
    img = Image.open(src)
    img = img.resize(target_size, Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, quality=95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', nargs='?', default=None, help='图片文件夹路径')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--scale', type=int, choices=[2, 4, 8],
                       help='缩放倍数，如 2 表示分辨率除以 2')
    group.add_argument('--size', type=int, nargs=2, metavar=('W', 'H'),
                       help='指定输出宽高，如 --size 1920 1080')
    group.add_argument('--width', type=int,
                       help='指定输出宽度，高度等比缩放')
    args = parser.parse_args()

    if args.folder:
        raw = args.folder
    else:
        raw = input("请输入图片文件夹路径: ")
    folder = Path(raw.strip().strip('"').strip("'"))

    if not folder.is_dir():
        print(f"[错误] 文件夹不存在: {folder}")
        sys.exit(1)

    images = sorted(
        [f for f in folder.iterdir() if f.is_file() and f.suffix in IMAGE_EXTS]
    )

    if not images:
        print("[提示] 未找到图片文件")
        return

    # 确定输出目录名
    if args.scale:
        out_dir = folder.parent / f"{folder.name}_x{args.scale}"
    elif args.size:
        out_dir = folder.parent / f"{folder.name}_{args.size[0]}x{args.size[1]}"
    else:
        out_dir = folder.parent / f"{folder.name}_w{args.width}"

    # 预览
    sample = Image.open(images[0])
    w, h = sample.size
    sample.close()

    if args.scale:
        tw, th = w // args.scale, h // args.scale
    elif args.size:
        tw, th = args.size
    else:
        tw = args.width
        th = int(h * tw / w)

    print(f"\n找到 {len(images)} 张图片")
    print(f"原始分辨率（首张）：{w} × {h}")
    print(f"输出分辨率：{tw} × {th}")
    print(f"输出目录：{out_dir}")

    confirm = input("\n确认执行? (y/N): ").strip().lower()
    if confirm != 'y':
        print("已取消。")
        return

    for i, img_path in enumerate(images, 1):
        dst = out_dir / img_path.name
        try:
            if args.scale:
                img = Image.open(img_path)
                size = (img.width // args.scale, img.height // args.scale)
                img.close()
                resize_image(img_path, dst, size)
            elif args.size:
                resize_image(img_path, dst, (tw, th))
            else:
                img = Image.open(img_path)
                ratio = args.width / img.width
                size = (args.width, int(img.height * ratio))
                img.close()
                resize_image(img_path, dst, size)
            print(f"\r处理中 {i}/{len(images)}...", end='', flush=True)
        except Exception as e:
            print(f"\n[警告] 处理失败 {img_path.name}: {e}")

    print(f"\n完成！{len(images)} 张图片已保存到 {out_dir}")


if __name__ == '__main__':
    main()
