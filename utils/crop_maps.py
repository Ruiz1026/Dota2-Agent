#!/usr/bin/env python3
"""
地图图片裁剪脚本

将 maps/ 目录中的所有图片进行裁剪：
- 可配置截去上方、下方、左侧、右侧的像素数

裁剪后直接覆盖原文件
"""

import os
from PIL import Image
from typing import List

# 配置
MAPS_DIR = "maps"
TARGET_FILES = ["740.jpeg"]  # 指定要处理的文件，为空列表则处理所有文件
TOP_CROP = 5  # 截去上方的像素数
BOTTOM_CROP = 0   # 截去下方的像素数
LEFT_CROP = 0  # 截去左侧的像素数
RIGHT_CROP = 0  # 截去右侧的像素数
SUPPORTED_EXTENSIONS = ('.jpeg', '.jpg', '.png', '.bmp', '.tiff')

def get_image_files(directory: str, target_files: List[str] = None) -> List[str]:
    """获取目录中的所有图片文件"""
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return []

    image_files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(SUPPORTED_EXTENSIONS):
            # 如果指定了目标文件，则只处理目标文件
            if target_files and filename not in target_files:
                continue
            image_files.append(os.path.join(directory, filename))

    return sorted(image_files)

def crop_image(input_path: str, top_crop: int = 50, bottom_crop: int = 0, left_crop: int = 0, right_crop: int = 30) -> bool:
    """
    对图片进行裁剪，直接覆盖原文件

    Args:
        input_path: 图片路径
        top_crop: 截去上方的像素数
        bottom_crop: 截去下方的像素数
        left_crop: 截去左侧的像素数
        right_crop: 截去右侧的像素数

    Returns:
        bool: 是否成功
    """
    try:
        # 打开图片
        img = Image.open(input_path)
        
        # 获取原始尺寸
        original_size = img.size  # (width, height)
        print(f"   原始尺寸: {original_size[0]}x{original_size[1]}")

        # 计算裁剪后的尺寸
        # 裁剪框：(left, upper, right, lower)
        left = left_crop
        upper = top_crop
        right = original_size[0] - right_crop
        lower = original_size[1] - bottom_crop

        # 检查裁剪参数是否合理
        if right <= left or lower <= upper:
            print(f"   ❌ 裁剪参数不合理: 裁剪后尺寸为 {right - left}x{lower - upper}")
            img.close()
            return False

        # 进行裁剪
        cropped_img = img.crop((left, upper, right, lower))
        
        # 关闭原图，释放文件句柄
        img.close()

        # 获取裁剪后的尺寸
        cropped_size = cropped_img.size
        print(f"   ✅ 裁剪后尺寸: {cropped_size[0]}x{cropped_size[1]}")

        # 直接覆盖原文件
        if input_path.lower().endswith('.jpg') or input_path.lower().endswith('.jpeg'):
            # JPEG 格式，设置质量为 95
            cropped_img.save(input_path, quality=95, optimize=True)
        else:
            # 其他格式
            cropped_img.save(input_path)

        return True

    except Exception as e:
        print(f"   ❌ 处理失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("  Dota 2 地图图片裁剪工具")
    print("=" * 60)
    print(f"裁剪参数: 上 {TOP_CROP}, 下 {BOTTOM_CROP}, 左 {LEFT_CROP}, 右 {RIGHT_CROP} 像素")
    print(f"处理目录: {MAPS_DIR}/")
    if TARGET_FILES:
        print(f"目标文件: {', '.join(TARGET_FILES)}")
    print()

    # 获取图片文件
    image_files = get_image_files(MAPS_DIR, TARGET_FILES)

    if not image_files:
        print(f"❌ 在 {MAPS_DIR}/ 目录中未找到支持的图片文件")
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

        # 裁剪图片（直接覆盖原文件）
        if crop_image(img_file, TOP_CROP, BOTTOM_CROP, LEFT_CROP, RIGHT_CROP):
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
        print("所有图片已成功裁剪并覆盖原文件!")
    else:
        print("部分图片处理失败，请检查上述错误信息")

if __name__ == "__main__":
    main()