import os
import cv2
import argparse
import gc
import numpy as np
from tqdm import tqdm

if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool

from imagecorruptions import corrupt

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
SUPPORTED_CORRUPTIONS = ("gaussian_noise", "snow", "fog")


def iter_image_files(input_dir, skip_dir=None):
    skip_dir = os.path.abspath(skip_dir) if skip_dir else None
    for root, _, files in os.walk(input_dir):
        root_abs = os.path.abspath(root)
        if skip_dir and os.path.commonpath([root_abs, skip_dir]) == skip_dir:
            continue

        for f in files:
            if f.lower().endswith(IMAGE_EXTENSIONS):
                yield os.path.join(root, f)


def count_images(input_dir, skip_dir):
    return sum(1 for _ in iter_image_files(input_dir, skip_dir))


def resize_by_max_side(img, max_side):
    if max_side <= 0:
        return img

    height, width = img.shape[:2]
    current_max_side = max(height, width)
    if current_max_side <= max_side:
        return img

    scale = max_side / current_max_side
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def generate_corrupted_dataset(
    input_dir,
    output_dir,
    degradation,
    severity,
    max_side,
    overwrite,
):
    if degradation not in SUPPORTED_CORRUPTIONS:
        raise ValueError(
            f"Unsupported degradation '{degradation}'. "
            f"Choose from: {', '.join(SUPPORTED_CORRUPTIONS)}."
        )

    severity = int(severity)
    if severity < 1 or severity > 5:
        raise ValueError("Severity must be between 1 and 5.")
    severities = [severity]

    input_abs = os.path.abspath(input_dir)
    output_abs = os.path.abspath(output_dir)
    try:
        skip_dir = (
            output_abs
            if os.path.commonpath([input_abs, output_abs]) == input_abs
            else None
        )
    except ValueError:
        skip_dir = None

    total_images = count_images(input_dir, skip_dir)

    print(f"Found {total_images} images.")
    print(f"Degradation: {degradation}")
    print(f"Severity levels: {severities}")
    if skip_dir:
        print(f"Skipping output directory inside source: {skip_dir}")
    if max_side > 0:
        print(f"Resize enabled: longest side <= {max_side}")

    for img_path in tqdm(iter_image_files(input_dir, skip_dir), total=total_images):
        rel_path = os.path.relpath(img_path, input_dir)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read: {img_path}")
            continue

        img = resize_by_max_side(img, max_side)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for sev in severities:
            save_path = os.path.join(
                output_dir,
                f"{degradation}_{sev}",
                rel_path
            )

            if os.path.exists(save_path) and not overwrite:
                continue

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            corrupted = corrupt(
                img,
                corruption_name=degradation,
                severity=sev
            )
            corrupted_bgr = cv2.cvtColor(corrupted, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, corrupted_bgr)
            del corrupted_bgr, corrupted
            gc.collect()

        del img
        gc.collect()

    print("Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--degradation",
        type=str,
        required=True,
        choices=SUPPORTED_CORRUPTIONS,
    )

    parser.add_argument("--severity", type=int, required=True)
    parser.add_argument(
        "--max_side",
        type=int,
        default=0,
        help="Resize images so the longest side is at most this value. 0 keeps original size.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )

    args = parser.parse_args()

    generate_corrupted_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        degradation=args.degradation,
        severity=args.severity,
        max_side=args.max_side,
        overwrite=args.overwrite,
    )
