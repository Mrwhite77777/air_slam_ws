import argparse
import os


PROJECT_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT_PATH = os.path.join(PROJECT_SRC_DIR, "datasets", "shuanmu_rgb", "gnss_gt.csv")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_SRC_DIR, "datasets", "results", "gnss_gt_tum.txt")

TIME_OFFSET_SUB = 92179.97
TIME_OFFSET_ADD = 0.008253


def convert_timestamp(timestamp_value):
    """
    按要求修正时间戳。
    """
    return timestamp_value - TIME_OFFSET_SUB + TIME_OFFSET_ADD


def convert_gnss_to_tum(input_path, output_path):
    """
    将 GNSS 轨迹转换为 TUM 格式。
    TUM 格式: timestamp tx ty tz qx qy qz qw
    角度统一按 0 处理，因此四元数固定为 0 0 0 1。
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    converted_lines = []

    with open(input_path, "r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.replace(",", " ").split()
            if len(parts) < 4:
                raise ValueError(f"第 {line_number} 行列数不足，无法提取时间和位置: {line}")

            try:
                timestamp = float(parts[0])
                tx = float(parts[-3])
                ty = float(parts[-2])
                tz = float(parts[-1])
            except ValueError as error:
                raise ValueError(f"第 {line_number} 行存在非数值内容: {line}") from error

            tum_timestamp = convert_timestamp(timestamp)
            converted_lines.append(
                f"{tum_timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} 0.000000 0.000000 0.000000 1.000000"
            )

    if not converted_lines:
        raise ValueError(f"输入文件中没有可转换的数据: {input_path}")

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(converted_lines) + "\n")


def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="将 gnss_gt.csv 转换为 TUM 格式")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help=f"输入文件路径，默认: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"输出文件路径，默认: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


def main():
    """
    主函数。
    """
    args = parse_args()
    convert_gnss_to_tum(args.input, args.output)
    print(f"TUM 结果已保存: {args.output}")


if __name__ == "__main__":
    main()
