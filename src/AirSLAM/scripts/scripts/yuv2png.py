import argparse
import os

import cv2
import numpy as np

OUTPUT_WIDTH = 768
OUTPUT_HEIGHT = 432


def convert_nv12_to_bgr(yuv_file, width, height):
    """
    将单个 NV12 格式的 YUV 文件转换为 BGR 图像。
    """
    frame_size = width * height * 3 // 2

    with open(yuv_file, "rb") as file:
        yuv_data = np.frombuffer(file.read(frame_size), dtype=np.uint8)

    if yuv_data.size < frame_size:
        raise ValueError(
            f"文件大小不足，无法读取完整 NV12 图像: {yuv_file}，"
            f"期望 {frame_size} 字节，实际 {yuv_data.size} 字节"
        )

    y_size = width * height
    y_plane = yuv_data[:y_size].reshape((height, width))
    uv_plane = yuv_data[y_size : y_size + y_size // 2].reshape((height // 2, width))

    nv12_image = np.vstack([y_plane, uv_plane])
    return cv2.cvtColor(nv12_image, cv2.COLOR_YUV2BGR_NV12)


def convert_file(yuv_file, width, height, output_png):
    """
    转换单个 YUV 文件，缩放到固定分辨率后保存为 PNG。
    """
    bgr_image = convert_nv12_to_bgr(yuv_file, width, height)
    resized_image = cv2.resize(
        bgr_image,
        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    if not cv2.imwrite(output_png, resized_image):
        raise IOError(f"PNG 保存失败: {output_png}")


def batch_convert_yuv_to_png(input_dir, width, height, output_dir):
    """
    批量转换输入目录下的 YUV 文件，输出到指定目录。
    转换后文件名保持一致，仅后缀改为 .png。
    """
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"输入目录不存在: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    yuv_files = sorted(
        file_name
        for file_name in os.listdir(input_dir)
        if file_name.lower().endswith(".yuv")
        and os.path.isfile(os.path.join(input_dir, file_name))
    )

    if not yuv_files:
        raise FileNotFoundError(f"输入目录中未找到 .yuv 文件: {input_dir}")

    success_count = 0
    failed_files = []

    for file_name in yuv_files:
        input_path = os.path.join(input_dir, file_name)
        output_name = os.path.splitext(file_name)[0] + ".png"
        output_path = os.path.join(output_dir, output_name)

        try:
            convert_file(input_path, width, height, output_path)
            print(f"转换成功: {input_path} -> {output_path}")
            success_count += 1
        except Exception as error:
            failed_files.append((input_path, str(error)))
            print(f"转换失败: {input_path}，原因: {error}")

    print(f"批量转换完成，共 {len(yuv_files)} 个文件，成功 {success_count} 个，失败 {len(failed_files)} 个")

    if failed_files:
        raise RuntimeError("存在文件转换失败，请查看上方错误信息")


def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="批量将 NV12 YUV 图像转换为 PNG 图像")
    parser.add_argument("input_dir", help="输入 YUV 文件夹路径")
    parser.add_argument("width", type=int, help="图像宽度")
    parser.add_argument("height", type=int, help="图像高度")
    parser.add_argument("output_dir", help="输出 PNG 文件夹路径")
    return parser.parse_args()


def main():
    """
    主函数：处理命令行参数并执行批量转换。
    """
    # args = parse_args()
    # batch_convert_yuv_to_png(args.input_dir, args.width, args.height, args.output_dir)

    input_dir = '/workspace/workspace/data/rgb_slam/260309_shihu_dingweijingdu_jiantu/left_org_rgb'
    width = 1280
    height = 1088
    output_dir = '/workspace/workspace/AirSLAM/src/datasets/shuanmu_rgb/cam0/data'

    batch_convert_yuv_to_png(input_dir, width, height, output_dir)




if __name__ == "__main__":
    main()
