import argparse
import csv
import os


DEFAULT_INPUT_CSV = "/home/bhp/workspace/AirSLAM/src/datasets/shuanmu_rgb/imu_data.csv"
DEFAULT_OUTPUT_CSV = "/home/bhp/workspace/AirSLAM/src/datasets/shuanmu_rgb/imu0/data.csv"

OUTPUT_HEADER = [
    "#timestamp [ns]",
    "w_RS_S_x [rad s^-1]",
    "w_RS_S_y [rad s^-1]",
    "w_RS_S_z [rad s^-1]",
    "a_RS_S_x [m s^-2]",
    "a_RS_S_y [m s^-2]",
    "a_RS_S_z [m s^-2]",
]


def format_timestamp(timestamp_value):
    """
    将时间戳格式化为示例中的 6 位小数形式。
    """
    return f"{float(timestamp_value):.6f}"


def convert_imu_csv(input_csv, output_csv):
    """
    将原始 IMU CSV 转换为目标格式并写入指定文件。
    """
    if not os.path.isfile(input_csv):
        raise FileNotFoundError(f"输入文件不存在: {input_csv}")

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(input_csv, "r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)

        required_columns = [
            "Timestamp(ns)",
            "GyroX(rad/s)",
            "GyroY(rad/s)",
            "GyroZ(rad/s)",
            "AccelX(m/s²)",
            "AccelY(m/s²)",
            "AccelZ(m/s²)",
        ]

        missing_columns = [column for column in required_columns if column not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"输入 CSV 缺少必要列: {', '.join(missing_columns)}")

        with open(output_csv, "w", encoding="utf-8", newline="") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(OUTPUT_HEADER)

            for row in reader:
                writer.writerow(
                    [
                        format_timestamp(row["Timestamp(ns)"]),
                        row["GyroX(rad/s)"],
                        row["GyroY(rad/s)"],
                        row["GyroZ(rad/s)"],
                        row["AccelX(m/s²)"],
                        row["AccelY(m/s²)"],
                        row["AccelZ(m/s²)"],
                    ]
                )


def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="将 shuanmu_rgb 的 IMU 数据转换为 imu0/data.csv 格式")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_CSV,
        help=f"输入 CSV 路径，默认: {DEFAULT_INPUT_CSV}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_CSV,
        help=f"输出 CSV 路径，默认: {DEFAULT_OUTPUT_CSV}",
    )
    return parser.parse_args()


def main():
    """
    主函数。
    """
    args = parse_args()
    convert_imu_csv(args.input, args.output)
    print(f"转换完成: {args.output}")


if __name__ == "__main__":
    main()
