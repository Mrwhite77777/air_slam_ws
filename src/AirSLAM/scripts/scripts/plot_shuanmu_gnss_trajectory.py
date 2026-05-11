import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


PROJECT_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT_PATH = os.path.join(PROJECT_SRC_DIR, "datasets", "shuanmu_rgb", "gnss_gt.csv")


def load_positions(input_path):
    """
    读取轨迹文件，并提取后三列位置数据。
    支持空白分隔或逗号分隔。
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    positions = []

    with open(input_path, "r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            normalized_line = line.replace(",", " ")
            parts = normalized_line.split()

            if len(parts) < 3:
                raise ValueError(f"第 {line_number} 行列数不足，无法提取后三列: {line}")

            try:
                x, y, z = map(float, parts[-3:])
            except ValueError as error:
                raise ValueError(f"第 {line_number} 行存在非数值内容: {line}") from error

            positions.append([x, y, z])

    if not positions:
        raise ValueError(f"输入文件中没有可用的位置数据: {input_path}")

    return np.array(positions, dtype=np.float64)


def plot_trajectory(positions, output_path=None):
    """
    绘制 2D XY 轨迹图和 3D 轨迹图。
    默认直接显示；如果提供输出路径则额外保存图片。
    """
    x_values = positions[:, 0]
    y_values = positions[:, 1]
    z_values = positions[:, 2]

    figure = plt.figure(figsize=(14, 6))

    axis_2d = figure.add_subplot(1, 2, 1)
    axis_2d.plot(x_values, y_values, color="tab:blue", linewidth=1.5)
    axis_2d.scatter(x_values[0], y_values[0], color="green", s=40, label="起点")
    axis_2d.scatter(x_values[-1], y_values[-1], color="red", s=40, label="终点")
    axis_2d.set_title("GNSS 2D 轨迹图")
    axis_2d.set_xlabel("X")
    axis_2d.set_ylabel("Y")
    axis_2d.axis("equal")
    axis_2d.grid(True, linestyle="--", alpha=0.4)
    axis_2d.legend()

    axis_3d = figure.add_subplot(1, 2, 2, projection="3d")
    axis_3d.plot(x_values, y_values, z_values, color="tab:orange", linewidth=1.2)
    axis_3d.scatter(x_values[0], y_values[0], z_values[0], color="green", s=30)
    axis_3d.scatter(x_values[-1], y_values[-1], z_values[-1], color="red", s=30)
    axis_3d.set_title("GNSS 3D 轨迹图")
    axis_3d.set_xlabel("X")
    axis_3d.set_ylabel("Y")
    axis_3d.set_zlabel("Z")

    figure.tight_layout()

    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        figure.savefig(output_path, dpi=200)

    plt.show()
    plt.close(figure)


def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="根据 gnss_gt.csv 后三列位置绘制轨迹图")
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help=f"输入轨迹文件路径，默认: {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="可选：输出图片路径；不传则只显示不保存",
    )
    return parser.parse_args()


def main():
    """
    主函数。
    """
    # args = parse_args()
    # positions = load_positions(args.input)
    # plot_trajectory(positions, args.output)
    # if args.output:
    #     print(f"轨迹图已保存: {args.output}")
    # else:
    #     print("轨迹图已显示。")

    positions = load_positions(DEFAULT_INPUT_PATH)
    plot_trajectory(positions)


if __name__ == "__main__":
    main()
