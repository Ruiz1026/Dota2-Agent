#!/usr/bin/env python3
"""
地图图片 Resize 脚本

将 maps/ 目录中的所有图片 resize 为 900x900 像素

支持的格式：JPEG, JPG, PNG
"""

import os
from PIL import Image
from typing import List, Tuple

# 配置
MAPS_DIR = "maps"
TARGET_SIZE = (900, 900)  # 目标尺寸
SUPPORTED_EXTENSIONS = ('.jpeg', '.jpg', '.png', '.bmp', '.tiff')

def get_image_files(directory: str) -> List[str]:
    """获取目录中的所有图片文件"""
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return []

    image_files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(SUPPORTED_EXTENSIONS):
            image_files.append(os.path.join(directory, filename))

    return sorted(image_files)

def resize_image(input_path: str, output_path: str, target_size: Tuple[int, int]) -> bool:
    """
    将图片 resize 到指定尺寸

    Args:
        input_path: 输入图片路径
        output_path: 输出图片路径
        target_size: 目标尺寸 (width, height)

    Returns:
        bool: 是否成功
    """
    try:
        # 打开图片
        with Image.open(input_path) as img:
            # 获取原始尺寸
            original_size = img.size
            print(f"   原始尺寸: {original_size[0]}x{original_size[1]}")

            # 检查是否需要 resize
            if original_size == target_size:
                print(f"   已为目标尺寸，跳过: {os.path.basename(input_path)}")
                return True

            # 使用 LANCZOS 重采样算法进行高质量 resize
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)

            # 保存图片，保持原格式和质量
            if input_path.lower().endswith('.jpg') or input_path.lower().endswith('.jpeg'):
                # JPEG 格式，设置质量为 95
                resized_img.save(output_path, quality=95, optimize=True)
            else:
                # 其他格式
                resized_img.save(output_path)

            print(f"   已 resize: {target_size[0]}x{target_size[1]}")
            return True

    except Exception as e:
        print(f"   处理失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("  Dota 2 地图图片 Resize 工具")
    print("=" * 60)
    print(f"目标尺寸: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} 像素")
    print(f"处理目录: {MAPS_DIR}/")
    print()

    # 获取图片文件
    image_files = get_image_files(MAPS_DIR)

    if not image_files:
        print(f"在 {MAPS_DIR}/ 目录中未找到支持的图片文件")
        print(f"支持的格式: {', '.join(SUPPORTED_EXTENSIONS)}")
        return

    print(f"找到 {len(image_files)} 个图片文件:")
    for img_file in image_files:
        print(f"   - {os.path.basename(img_file)}")
    print()

    # 处理每个图片
    processed = 0
    successful = 0

    for img_file in image_files:
        filename = os.path.basename(img_file)
        print(f"处理: {filename}")

        # resize 并覆盖原文件
        if resize_image(img_file, img_file, TARGET_SIZE):
            successful += 1
        processed += 1
        print()

    # 统计结果
    print("=" * 60)
    print("处理结果:")
    print(f"   总文件数: {len(image_files)}")
    print(f"   成功处理: {successful}")
    print(f"   处理失败: {processed - successful}")

    if successful == len(image_files):
        print("所有图片已成功 resize!")
    else:
        print("部分图片处理失败，请检查上述错误信息")

if __name__ == "__main__":
    main()