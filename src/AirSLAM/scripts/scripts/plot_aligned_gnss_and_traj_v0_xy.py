import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


PROJECT_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_GNSS_PATH = os.path.join(PROJECT_SRC_DIR, "datasets", "gnss_gt_tum.txt")
DEFAULT_TRAJ_PATH = os.path.join(PROJECT_SRC_DIR, "datasets", "trajectory_v0.txt")


def load_tum_xy(file_path):
    """
    读取 TUM 格式轨迹，提取时间戳与 XY 位置。
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    timestamps = []
    positions = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                raise ValueError(f"第 {line_number} 行列数不足: {line}")

            try:
                timestamps.append(float(parts[0]))
                positions.append([float(parts[1]), float(parts[2])])
            except ValueError as error:
                raise ValueError(f"第 {line_number} 行存在非数值内容: {line}") from error

    return np.array(timestamps, dtype=np.float64), np.array(positions, dtype=np.float64)


def load_traj_v0_xy(file_path):
    """
    读取 trajectory_v0.txt，提取时间戳与第 2 到第 3 列的 XY 位置。
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    timestamps = []
    positions = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.replace(",", " ").split()
            if len(parts) < 4:
                raise ValueError(f"第 {line_number} 行列数不足: {line}")

            try:
                timestamps.append(float(parts[0]))
                positions.append([float(parts[1]), float(parts[2])])
            except ValueError as error:
                raise ValueError(f"第 {line_number} 行存在非数值内容: {line}") from error

    return np.array(timestamps, dtype=np.float64), np.array(positions, dtype=np.float64)


def interpolate_xy(source_times, source_xy, target_times):
    """
    将 XY 轨迹插值到目标时间轴。
    """
    x_interp = np.interp(target_times, source_times, source_xy[:, 0])
    y_interp = np.interp(target_times, source_times, source_xy[:, 1])
    return np.column_stack([x_interp, y_interp])


def align_by_absolute_timestamps(gnss_times, gnss_xy, traj_times, traj_xy, max_timestamp=None):
    """
    使用绝对时间戳进行对齐，并插值到公共时间范围。
    """
    overlap_start = max(gnss_times[0], traj_times[0])
    overlap_end = min(gnss_times[-1], traj_times[-1])

    if max_timestamp is not None:
        overlap_end = min(overlap_end, max_timestamp)

    if overlap_end <= overlap_start:
        raise ValueError("两条轨迹没有有效的绝对时间重叠区间")

    common_times = traj_times[(traj_times >= overlap_start) & (traj_times <= overlap_end)]
    if common_times.size == 0:
        raise ValueError("重叠时间范围内没有可用采样点")

    aligned_gnss_xy = interpolate_xy(gnss_times, gnss_xy, common_times)
    aligned_traj_xy = interpolate_xy(traj_times, traj_xy, common_times)
    return common_times, aligned_gnss_xy, aligned_traj_xy


def align_estimate_by_first_frame(reference_xy, estimate_xy):
    """
    用首帧平移将估计轨迹对齐到参考轨迹。
    """
    translation = reference_xy[0] - estimate_xy[0]
    return estimate_xy + translation


def align_estimate_se2(reference_xy, estimate_xy):
    """
    使用二维刚体变换（旋转 + 平移）将估计轨迹最优对齐到参考轨迹。
    行为类似 evo 的 align，但这里限定在 XY 平面且不估计尺度。
    """
    reference_center = np.mean(reference_xy, axis=0)
    estimate_center = np.mean(estimate_xy, axis=0)

    reference_zero_mean = reference_xy - reference_center
    estimate_zero_mean = estimate_xy - estimate_center

    covariance = estimate_zero_mean.T @ reference_zero_mean
    u_matrix, _, vh_matrix = np.linalg.svd(covariance)
    rotation = vh_matrix.T @ u_matrix.T

    if np.linalg.det(rotation) < 0:
        vh_matrix[-1, :] *= -1
        rotation = vh_matrix.T @ u_matrix.T

    translation = reference_center - estimate_center @ rotation.T
    aligned_xy = estimate_xy @ rotation.T + translation
    return aligned_xy, rotation, translation


def compute_ape_errors(reference_xy, estimate_xy):
    """
    计算逐帧绝对位置误差（XY 平面欧式距离）。
    """
    deltas = estimate_xy - reference_xy
    return np.linalg.norm(deltas, axis=1)


def compute_rpe_errors(reference_xy, estimate_xy, delta=1):
    """
    计算相对位置误差（XY 平面欧式距离）。
    delta 表示相对位移使用的帧间隔，类似 evo 中的 delta。
    """
    if delta < 1:
        raise ValueError("RPE 的 delta 必须大于等于 1")
    if reference_xy.shape[0] <= delta or estimate_xy.shape[0] <= delta:
        raise ValueError("轨迹长度不足，无法计算当前 delta 下的 RPE")

    reference_relative = reference_xy[delta:] - reference_xy[:-delta]
    estimate_relative = estimate_xy[delta:] - estimate_xy[:-delta]
    relative_deltas = estimate_relative - reference_relative
    return np.linalg.norm(relative_deltas, axis=1)


def summarize_errors(errors):
    """
    汇总误差统计，形式参考 evo 常见输出。
    """
    return {
        "count": int(errors.size),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
        "sse": float(np.sum(errors ** 2)),
    }


def print_error_summary(title, summary):
    """
    打印误差统计信息。
    """
    print(title)
    print(f"  count:  {summary['count']}")
    print(f"  rmse:   {summary['rmse']:.6f}")
    print(f"  mean:   {summary['mean']:.6f}")
    print(f"  median: {summary['median']:.6f}")
    print(f"  std:    {summary['std']:.6f}")
    print(f"  min:    {summary['min']:.6f}")
    print(f"  max:    {summary['max']:.6f}")
    print(f"  sse:    {summary['sse']:.6f}")


def plot_xy_trajectories(gnss_xy, traj_xy, output_path=None):
    """
    绘制时间对齐后的 XY 轨迹对比图。
    """
    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(1, 1, 1)

    axis.plot(gnss_xy[:, 0], gnss_xy[:, 1], color="tab:blue", linewidth=1.3, label="GNSS")
    axis.plot(traj_xy[:, 0], traj_xy[:, 1], color="tab:orange", linewidth=1.3, label="Trajectory v0")

    axis.scatter(gnss_xy[0, 0], gnss_xy[0, 1], color="green", s=35, label="Start")
    axis.scatter(gnss_xy[-1, 0], gnss_xy[-1, 1], color="red", s=35, label="End")

    axis.set_title("Aligned XY Trajectories")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.axis("equal")
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend()

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
    parser = argparse.ArgumentParser(description="根据时间戳对齐 GNSS 与 trajectory_v0，并绘制 XY 轨迹图")
    parser.add_argument(
        "--gnss",
        default=DEFAULT_GNSS_PATH,
        help=f"GNSS TUM 轨迹文件路径，默认: {DEFAULT_GNSS_PATH}",
    )
    parser.add_argument(
        "--traj",
        default=DEFAULT_TRAJ_PATH,
        help=f"trajectory_v0 文件路径，默认: {DEFAULT_TRAJ_PATH}",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="可选：输出图片路径；不传则只显示不保存",
    )
    parser.add_argument(
        "--max-timestamp",
        type=float,
        default=827,
        help="可选：仅使用不晚于该绝对时间戳的轨迹进行绘图和误差计算",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        default=False,
        help="启用类似 evo 的二维刚体对齐（SE2）后再绘图和计算误差",
    )
    parser.add_argument(
        "--rpe-delta",
        type=int,
        default=50,
        help="RPE 计算所使用的帧间隔，默认: 1",
    )
    return parser.parse_args()


def main():
    """
    主函数。
    """
    args = parse_args()
    gnss_times, gnss_xy = load_tum_xy(args.gnss)
    traj_times, traj_xy = load_traj_v0_xy(args.traj)
    _, aligned_gnss_xy, aligned_traj_xy = align_by_absolute_timestamps(
        gnss_times,
        gnss_xy,
        traj_times,
        traj_xy,
        args.max_timestamp,
    )
    raw_errors = compute_ape_errors(aligned_gnss_xy, aligned_traj_xy)
    raw_summary = summarize_errors(raw_errors)
    print_error_summary("APE XY（直接对齐时间戳）", raw_summary)
    raw_rpe_errors = compute_rpe_errors(aligned_gnss_xy, aligned_traj_xy, args.rpe_delta)
    raw_rpe_summary = summarize_errors(raw_rpe_errors)
    print_error_summary(f"RPE XY（直接对齐时间戳，delta={args.rpe_delta}）", raw_rpe_summary)

    shifted_traj_xy = align_estimate_by_first_frame(aligned_gnss_xy, aligned_traj_xy)
    shifted_errors = compute_ape_errors(aligned_gnss_xy, shifted_traj_xy)
    shifted_summary = summarize_errors(shifted_errors)
    print_error_summary("APE XY（首帧平移对齐后）", shifted_summary)
    shifted_rpe_errors = compute_rpe_errors(aligned_gnss_xy, shifted_traj_xy, args.rpe_delta)
    shifted_rpe_summary = summarize_errors(shifted_rpe_errors)
    print_error_summary(f"RPE XY（首帧平移对齐后，delta={args.rpe_delta}）", shifted_rpe_summary)

    plot_traj_xy = aligned_traj_xy
    if args.align:
        se2_traj_xy, rotation, translation = align_estimate_se2(aligned_gnss_xy, aligned_traj_xy)
        se2_errors = compute_ape_errors(aligned_gnss_xy, se2_traj_xy)
        se2_summary = summarize_errors(se2_errors)
        print_error_summary("APE XY（SE2 align 后）", se2_summary)
        se2_rpe_errors = compute_rpe_errors(aligned_gnss_xy, se2_traj_xy, args.rpe_delta)
        se2_rpe_summary = summarize_errors(se2_rpe_errors)
        print_error_summary(f"RPE XY（SE2 align 后，delta={args.rpe_delta}）", se2_rpe_summary)
        print("SE2 align 变换参数")
        print(f"  rotation: [[{rotation[0,0]:.6f}, {rotation[0,1]:.6f}], [{rotation[1,0]:.6f}, {rotation[1,1]:.6f}]]")
        print(f"  translation: [{translation[0]:.6f}, {translation[1]:.6f}]")
        plot_traj_xy = se2_traj_xy

    plot_xy_trajectories(aligned_gnss_xy, plot_traj_xy, args.output)

    if args.output:
        print(f"对齐后的 XY 轨迹图已保存: {args.output}")
    else:
        print("对齐后的 XY 轨迹图已显示。")


if __name__ == "__main__":
    main()
