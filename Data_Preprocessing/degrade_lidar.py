from __future__ import annotations

import argparse
import os
from typing import Callable

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


SUPPORTED_DEGRADATIONS = (
    "random_dropout",
    "coordinate_jitter",
)


def _fallback_tqdm(iterable, **_kwargs):
    return iterable


def _progress_write(progress_bar, message: str) -> None:
    if progress_bar is not None and hasattr(progress_bar, "write"):
        progress_bar.write(message)
    else:
        print(message)


def read_ply_xyz_intensity(ply_path: str) -> np.ndarray:
    """Read an ASCII PLY file and return x, y, z, intensity as [N, 4]."""
    with open(ply_path, "r", encoding="utf-8", errors="ignore") as f:
        header = []
        for line in f:
            stripped = line.strip().lstrip("\ufeff")
            header.append(stripped)
            if stripped == "end_header":
                break
        else:
            raise ValueError("PLY header is missing end_header.")

        if not header or header[0] != "ply":
            raise ValueError("Input file is not a valid PLY file.")

        fmt_lines = [line for line in header if line.startswith("format ")]
        if not fmt_lines or "ascii" not in fmt_lines[0]:
            raise ValueError("Only ASCII PLY files are supported.")

        vertex_count = None
        vertex_props = []
        in_vertex_element = False

        for line in header:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "element":
                in_vertex_element = parts[1] == "vertex"
                if in_vertex_element:
                    vertex_count = int(parts[2])
                continue

            if in_vertex_element and len(parts) >= 3 and parts[0] == "property":
                vertex_props.append(parts[-1])

        if vertex_count is None:
            raise ValueError("PLY header is missing element vertex.")

        required = ["x", "y", "z", "intensity"]
        missing = [name for name in required if name not in vertex_props]
        if missing:
            raise ValueError(f"PLY vertex properties are missing: {missing}")

        prop_indices = [vertex_props.index(name) for name in required]
        points = np.zeros((vertex_count, 4), dtype=np.float64)

        for idx in range(vertex_count):
            line = f.readline()
            if not line:
                raise ValueError(
                    f"Expected {vertex_count} vertex rows, got {idx} rows."
                )

            values = line.split()
            if len(values) < len(vertex_props):
                raise ValueError(f"Malformed vertex row at index {idx}.")

            points[idx] = [float(values[prop_idx]) for prop_idx in prop_indices]

    return points


def write_ply_xyz_intensity(ply_path: str, points: np.ndarray) -> None:
    """Write x, y, z, intensity as an ASCII PLY file."""
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)

    output = np.asarray(points, dtype=np.float64)
    if output.ndim != 2 or output.shape[1] != 4:
        raise ValueError("points must have shape [N, 4].")

    intensities = np.rint(output[:, 3]).astype(np.int64)
    intensities = np.clip(intensities, 0, 65535)

    with open(ply_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {output.shape[0]}\n")
        f.write("property double x\n")
        f.write("property double y\n")
        f.write("property double z\n")
        f.write("property ushort intensity\n")
        f.write("end_header\n")

        for point, intensity in zip(output, intensities):
            f.write(
                f"{point[0]:.10f} {point[1]:.10f} {point[2]:.10f} {int(intensity)}\n"
            )


def collect_ply_files(input_dir: str, skip_dir: str | None = None) -> list[str]:
    ply_files = []
    skip_abs = os.path.abspath(skip_dir) if skip_dir else None

    for root, dirs, files in os.walk(input_dir):
        root_abs = os.path.abspath(root)
        if skip_abs:
            dirs[:] = [
                d
                for d in dirs
                if os.path.commonpath([os.path.abspath(os.path.join(root_abs, d)), skip_abs])
                != skip_abs
            ]

        for filename in files:
            if filename.lower().endswith(".ply"):
                ply_files.append(os.path.join(root, filename))

    return sorted(ply_files)


def _ensure_nonempty_dropout(
    points: np.ndarray, keep_mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    if points.shape[0] == 0:
        return points.copy()

    if np.any(keep_mask):
        return points[keep_mask]

    keep_idx = int(rng.integers(0, points.shape[0]))
    return points[keep_idx : keep_idx + 1]


def random_dropout(
    points: np.ndarray, intensity: float, rng: np.random.Generator
) -> np.ndarray:
    ratio = intensity
    keep_mask = rng.random(points.shape[0]) > ratio
    return _ensure_nonempty_dropout(points, keep_mask, rng)


def coordinate_jitter(
    points: np.ndarray, intensity: float, rng: np.random.Generator
) -> np.ndarray:
    sigma = intensity
    degraded_points = points.copy()
    noise = rng.normal(loc=0.0, scale=sigma, size=degraded_points[:, :3].shape)
    degraded_points[:, :3] += noise
    return degraded_points


def apply_degradation(
    points: np.ndarray,
    degradation: str,
    intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    degrade_fn: dict[str, Callable[[np.ndarray, float, np.random.Generator], np.ndarray]] = {
        "random_dropout": random_dropout,
        "coordinate_jitter": coordinate_jitter,
    }

    if degradation not in degrade_fn:
        raise ValueError(f"Unsupported degradation: {degradation}")

    return degrade_fn[degradation](points, intensity, rng)


def _get_skip_dir(input_dir: str, output_dir: str) -> str | None:
    input_abs = os.path.abspath(input_dir)
    output_abs = os.path.abspath(output_dir)
    try:
        if os.path.commonpath([input_abs, output_abs]) == input_abs:
            return output_abs
    except ValueError:
        return None
    return None


def process_dataset(
    input_dir: str,
    output_dir: str,
    degradation: str,
    intensity: float,
    seed: int,
    overwrite: bool,
) -> None:
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if degradation == "random_dropout" and not 0.0 <= intensity <= 1.0:
        raise ValueError("Dropout intensity must be between zero and one.")
    if degradation == "coordinate_jitter" and intensity < 0.0:
        raise ValueError("Jitter intensity must be non-negative.")
    intensity_tag = f"intensity_{intensity}"

    if degradation in SUPPORTED_DEGRADATIONS:
        degradation_names = [degradation]
    else:
        raise ValueError(
            "degradation must be one of: "
            + ", ".join(SUPPORTED_DEGRADATIONS)
        )

    skip_dir = _get_skip_dir(input_dir, output_dir)
    ply_files = collect_ply_files(input_dir, skip_dir=skip_dir)
    if not ply_files:
        raise FileNotFoundError(f"No .ply files found in input directory: {input_dir}")

    total_outputs = len(ply_files) * len(degradation_names)
    written_count = 0
    skipped_count = 0
    failed_count = 0

    print(f"Found {len(ply_files)} PLY files.")
    print(f"Degradation: {degradation}")
    print(f"Intensity: {intensity}")
    if skip_dir:
        print(f"Skipping output directory inside input: {skip_dir}")

    progress_bar = None
    if tqdm is not None:
        progress_bar = tqdm(
            total=total_outputs,
            desc=f"{degradation}/{intensity_tag}",
            unit="file",
            dynamic_ncols=True,
            leave=True,
        )

    try:
        for file_idx, ply_path in enumerate(ply_files):
            rel_path = os.path.relpath(ply_path, input_dir)
            basename = os.path.basename(ply_path)

            try:
                points = read_ply_xyz_intensity(ply_path)
            except Exception as exc:
                failed_count += len(degradation_names)
                _progress_write(progress_bar, f"[WARNING] Failed to read {ply_path}: {exc}")
                if progress_bar is not None:
                    progress_bar.update(len(degradation_names))
                    progress_bar.set_postfix_str(f"{basename} | failed to read")
                continue

            before_count = int(points.shape[0])

            for degradation_idx, degradation_name in enumerate(degradation_names):
                output_path = os.path.join(
                    output_dir,
                    degradation_name,
                    intensity_tag,
                    rel_path,
                )

                if os.path.exists(output_path) and not overwrite:
                    skipped_count += 1
                    if progress_bar is not None:
                        progress_bar.set_postfix_str(
                            f"{basename} | {degradation_name} | skipped"
                        )
                        progress_bar.update(1)
                    else:
                        print(f"[INFO] Skip existing output: {output_path}")
                    continue

                file_seed = seed + file_idx * len(degradation_names) + degradation_idx
                file_rng = np.random.default_rng(file_seed)

                try:
                    degraded_points = apply_degradation(
                        points=points,
                        degradation=degradation_name,
                        intensity=intensity,
                        rng=file_rng,
                    )
                    write_ply_xyz_intensity(output_path, degraded_points)
                except Exception as exc:
                    failed_count += 1
                    _progress_write(
                        progress_bar,
                        f"[WARNING] Failed to process {ply_path}: {exc}",
                    )
                    if progress_bar is not None:
                        progress_bar.set_postfix_str(
                            f"{basename} | {degradation_name} | failed"
                        )
                        progress_bar.update(1)
                    continue

                after_count = int(degraded_points.shape[0])
                keep_ratio = after_count / before_count if before_count > 0 else 0.0
                written_count += 1

                status = (
                    f"{basename} | {degradation_name} | "
                    f"{before_count}->{after_count} | keep {keep_ratio:.2f}"
                )
                if progress_bar is not None:
                    progress_bar.set_postfix_str(status)
                    progress_bar.update(1)
                else:
                    print(f"[INFO] {status} | output: {output_path}")
    finally:
        if progress_bar is not None:
            progress_bar.close()

    print("Done.")
    print(f"Input files: {len(ply_files)}")
    print(f"Expected output files: {total_outputs}")
    print(f"Output files: {written_count}")
    print(f"Skipped files: {skipped_count}")
    print(f"Failed files: {failed_count}")
    print(f"Degradation: {degradation}")
    print(f"Intensity: {intensity}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply degradation augmentation to ASCII PLY LiDAR point clouds."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input dataset directory. All .ply files are searched recursively.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output dataset directory. Relative input structure is preserved.",
    )
    parser.add_argument(
        "--degradation",
        type=str,
        required=True,
        choices=SUPPORTED_DEGRADATIONS,
        help="Degradation type to apply.",
    )
    parser.add_argument(
        "--intensity",
        type=float,
        required=True,
        help="Caller-provided degradation intensity; no experiment value is preset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Caller-provided random seed; no experiment value is preset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        degradation=args.degradation,
        intensity=args.intensity,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
