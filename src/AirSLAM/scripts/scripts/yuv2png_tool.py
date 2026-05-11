import numpy as np
import cv2
import sys
import os

def convert_nv12_to_png(yuv_file, width, height, output_png=None):
    """
    将 NV12 格式的 YUV 文件转换为 PNG 彩色图像
    """
    frame_size = width * height * 3 // 2

    with open(yuv_file, 'rb') as f:
        yuv_data = np.frombuffer(f.read(frame_size), dtype=np.uint8)

    y_size = width * height
    y = yuv_data[:y_size].reshape((height, width))

    # NV12 格式：Y 平面 + 交错 UV 平面（半高，宽度不变）
    uv_data = yuv_data[y_size:y_size + y_size//2].reshape((height//2, width))

    # 直接使用 OpenCV 的 NV12 转 BGR
    bgr = cv2.cvtColor(np.vstack([y.reshape((height, width)), uv_data]), cv2.COLOR_YUV2BGR_NV12)

    if output_png:
        cv2.imwrite(output_png, bgr)
        print(f"NV12 转换完成，保存到: {output_png}")

    return bgr

def main():
    """
    主函数：处理命令行参数
    用法: python script.py <yuv_file> <width> <height> [output_png]
    """
    if len(sys.argv) < 4:
        print("用法: python yuv_to_png.py <yuv文件路径> <宽度> <高度> [输出png路径]")
        print("示例: python yuv_to_png.py input.yuv 1280 1088 output.png")
        print("示例: python yuv_to_png.py input.yuv 1280 1088 (将自动生成输出文件名)")
        sys.exit(1)

    # 解析参数
    yuv_file = sys.argv[1]
    width = int(sys.argv[2])
    height = int(sys.argv[3])

    # 检查 YUV 文件是否存在
    if not os.path.exists(yuv_file):
        print(f"错误: YUV文件 '{yuv_file}' 不存在")
        sys.exit(1)

    # 计算期望的文件大小
    expected_size = width * height * 3 // 2
    actual_size = os.path.getsize(yuv_file)

    print(f"YUV文件: {yuv_file}")
    print(f"图像尺寸: {width}x{height}")
    print(f"期望的一帧大小: {expected_size} 字节")
    print(f"实际文件大小: {actual_size} 字节")

    if actual_size < expected_size:
        print(f"警告: 文件大小不足，可能无法读取完整帧")

    # 确定输出文件路径
    if len(sys.argv) >= 5:
        output_png = sys.argv[4]
    else:
        base_name = os.path.splitext(yuv_file)[0]
        output_png = f"{base_name}.png"
        print(f"未指定输出路径，将保存到: {output_png}")

    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(output_png)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 执行转换
    try:
        result = convert_nv12_to_png(yuv_file, width, height, output_png)
        if result is not None:
            print(f"转换成功！输出图像尺寸: {result.shape}")
            print(f"颜色统计:")
            print(f"  B通道: 均值={result[:,:,0].mean():.1f}, 范围=[{result[:,:,0].min()}, {result[:,:,0].max()}]")
            print(f"  G通道: 均值={result[:,:,1].mean():.1f}, 范围=[{result[:,:,1].min()}, {result[:,:,1].max()}]")
            print(f"  R通道: 均值={result[:,:,2].mean():.1f}, 范围=[{result[:,:,2].min()}, {result[:,:,2].max()}]")

            # 检查图像是否为灰度
            channel_diff = np.abs(result[:,:,0] - result[:,:,1]) + np.abs(result[:,:,1] - result[:,:,2]) + np.abs(result[:,:,0] - result[:,:,2])
            avg_diff = channel_diff.mean() / 3
            print(f"通道差异均值: {avg_diff:.1f} (值越小越可能是灰度图)")
        else:
            print("转换失败")
    except Exception as e:
        print(f"转换过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
