import argparse
import os

import cv2
import numpy as np
import yaml


DEFAULT_CONFIG_PATH = "/workspace/workspace/AirSLAM/src/configs/camera/shuangmu.yaml"


def load_camera_config(config_path, camera_name):
    """
    读取相机配置文件中的指定相机参数。
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        content = file.read()

    if content.startswith("%YAML:1.0"):
        content = content.split("\n", 1)[1]

    config = yaml.safe_load(content)

    if camera_name not in config:
        raise KeyError(f"配置中不存在相机节点: {camera_name}")

    camera_config = config[camera_name]
    intrinsics = camera_config["intrinsics"]
    distortion_coeffs = camera_config["distortion_coeffs"]
    distortion_type = int(config.get("distortion_type", 1))

    camera_matrix = np.array(
        [
            [intrinsics[0], 0.0, intrinsics[2]],
            [0.0, intrinsics[1], intrinsics[3]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    distortion = np.array(distortion_coeffs, dtype=np.float64).reshape(-1, 1)
    return camera_matrix, distortion, distortion_type


def undistort_image(image, camera_matrix, distortion, distortion_type):
    """
    根据畸变模型去除图像畸变。
    """
    height, width = image.shape[:2]

    if distortion_type == 0:
        return image.copy()

    if distortion_type == 1:
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            distortion,
            (width, height),
            0,
            (width, height),
        )
        return cv2.undistort(image, camera_matrix, distortion, None, new_camera_matrix)
        # return cv2.undistort(image, camera_matrix, distortion)
        # return cv2.undistortPoints(image, camera_matrix, distortion)

    if distortion_type == 2:
        distortion = distortion[:4]
        rectify_map_1, rectify_map_2 = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix,
            distortion,
            np.eye(3),
            camera_matrix,
            (width, height),
            cv2.CV_16SC2,
        )
        return cv2.remap(image, rectify_map_1, rectify_map_2, interpolation=cv2.INTER_LINEAR)

    raise ValueError(f"不支持的 distortion_type: {distortion_type}")


def add_title(image, title):
    """
    在图像顶部添加标题，便于对比观察。
    """
    canvas = cv2.copyMakeBorder(
        image,
        50,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    cv2.putText(
        canvas,
        title,
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas


def build_comparison_image(original, undistorted):
    """
    生成左右拼接的对比图。
    """
    original_labeled = add_title(original, "Before Undistort")
    undistorted_labeled = add_title(undistorted, "After Undistort")
    return np.hstack([original_labeled, undistorted_labeled])


def compare_undistort_effect(config_path, camera_name, image_path, output_path=None, show=False):
    """
    读取图像并生成去畸变前后的对比图。
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"图片不存在: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    camera_matrix, distortion, distortion_type = load_camera_config(config_path, camera_name)
    undistorted = undistort_image(image, camera_matrix, distortion, distortion_type)
    comparison = build_comparison_image(image, undistorted)

    if output_path is None:
        image_dir = os.path.dirname(image_path)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(image_dir, f"{image_name}_{camera_name}_undistort_compare.png")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not cv2.imwrite(output_path, comparison):
        raise IOError(f"对比图保存失败: {output_path}")

    if show:
        cv2.imshow("undistort comparison", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output_path


def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="读取相机参数并对比图像去畸变前后效果")
    parser.add_argument("image_path", help="输入图片路径")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"相机配置文件路径，默认: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--camera",
        default="cam0",
        choices=["cam0", "cam1"],
        help="使用哪个相机参数，默认: cam0",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出对比图路径，默认与输入图片同目录自动生成",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="是否弹窗显示对比结果",
    )
    return parser.parse_args()


def main():
    """
    主函数。
    """
    # args = parse_args()
    # output_path = compare_undistort_effect(
    #     config_path=args.config,
    #     camera_name=args.camera,
    #     image_path=args.image_path,
    #     output_path=args.output,
    #     show=args.show,
    # )
    # print(f"对比图已保存: {output_path}")

    config_path = DEFAULT_CONFIG_PATH
    camera_name = "cam1"
    image_path = "/workspace/workspace/AirSLAM/src/datasets/shuanmu_rgb/cam1/data/152791674447_9.png"
    output_path = "/workspace/workspace/AirSLAM/src/datasets/results/test.png"
    show = True

    output_path = compare_undistort_effect(
        config_path,
        camera_name,
        image_path,
        output_path,
        show,
    )
    print(f"对比图已保存: {output_path}")


if __name__ == "__main__":
    main()
