import argparse
import os


DEFAULT_IMAGE_DIR = "/home/bhp/workspace/AirSLAM/src/datasets/shuanmu_rgb/cam0/data"
TIMESTAMP_OFFSET = 8253000


def extract_timestamp(file_name):
    """
    提取文件名中下划线前的原始时间戳数值。
    """
    stem, _ = os.path.splitext(file_name)
    if "_" not in stem:
        raise ValueError(f"文件名中不包含下划线，无法解析时间戳: {file_name}")

    timestamp_text = stem.split("_", 1)[0]
    if not timestamp_text.isdigit():
        raise ValueError(f"下划线前不是纯数字时间戳: {file_name}")

    return int(timestamp_text)


def build_new_filename(file_name):
    """
    根据原文件名生成新的文件名。
    规则：取下划线前的数字，加上固定偏移量后作为新文件名。
    """
    stem, extension = os.path.splitext(file_name)
    new_timestamp = extract_timestamp(file_name) + TIMESTAMP_OFFSET
    return f"{new_timestamp}{extension}"


def collect_rename_pairs(image_dir):
    """
    收集目录下所有文件的重命名映射关系。
    """
    if not os.path.isdir(image_dir):
        raise NotADirectoryError(f"图片目录不存在: {image_dir}")

    rename_pairs = []
    existing_names = set()

    file_names = [
        file_name
        for file_name in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, file_name))
    ]

    for file_name in sorted(file_names, key=extract_timestamp):
        old_path = os.path.join(image_dir, file_name)

        new_name = build_new_filename(file_name)
        new_path = os.path.join(image_dir, new_name)

        if new_name in existing_names:
            raise ValueError(f"生成了重复的新文件名: {new_name}")
        existing_names.add(new_name)

        rename_pairs.append((old_path, new_path))

    return rename_pairs


def rename_images(image_dir, apply_changes=False):
    """
    预览或执行图片重命名。
    """
    rename_pairs = collect_rename_pairs(image_dir)

    for old_path, new_path in rename_pairs:
        print(f"{os.path.basename(old_path)} -> {os.path.basename(new_path)}")

    if not apply_changes:
        print("当前为预览模式，未实际重命名。添加 --apply 后执行。")
        return

    for old_path, new_path in rename_pairs:
        if os.path.exists(new_path) and old_path != new_path:
            raise FileExistsError(f"目标文件已存在，停止重命名: {new_path}")

    for old_path, new_path in rename_pairs:
        if old_path != new_path:
            os.rename(old_path, new_path)

    print(f"重命名完成，共处理 {len(rename_pairs)} 个文件。")


def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="批量重命名 shuanmu_rgb/cam0/data 中的图片时间戳")
    parser.add_argument(
        "--image-dir",
        default=DEFAULT_IMAGE_DIR,
        help=f"图片目录路径，默认: {DEFAULT_IMAGE_DIR}",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="实际执行重命名；默认仅预览",
    )
    return parser.parse_args()


def main():
    """
    主函数。
    """
    # args = parse_args()
    # rename_images(args.image_dir, apply_changes=args.apply)

    image_dir = "/workspace/workspace/AirSLAM/src/datasets/shuanmu_rgb/cam1/data"
    apply = True

    rename_images(image_dir, apply)


if __name__ == "__main__":
    main()
