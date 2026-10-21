from __future__ import annotations

import argparse
import os

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def random_feature_dropout(
    radar_map: np.ndarray,
    intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    rho = intensity
    mask = rng.random(radar_map.shape) > rho
    return radar_map * mask.astype(radar_map.dtype, copy=False)


def gaussian_measurement_noise(
    radar_map: np.ndarray,
    intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    sigma = intensity
    noise = rng.normal(loc=0.0, scale=sigma, size=radar_map.shape)
    return radar_map + noise.astype(radar_map.dtype, copy=False)


def apply_degradation(
    radar_map: np.ndarray,
    degradation: str,
    intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if degradation == "dropout":
        return random_feature_dropout(radar_map, intensity, rng)
    if degradation == "noise":
        return gaussian_measurement_noise(radar_map, intensity, rng)
    raise ValueError(f"Unsupported degradation: {degradation}")


def collect_radar_pairs(input_dir: str) -> list[tuple[str, str, str]]:
    direct_ang = os.path.join(input_dir, "radar_data_ang.npy")
    direct_vel = os.path.join(input_dir, "radar_data_vel.npy")
    if os.path.isfile(direct_ang) and os.path.isfile(direct_vel):
        return [("radar_data_ang.npy", direct_ang, direct_vel)]

    ang_dir = os.path.join(input_dir, "radar_data_ang")
    vel_dir = os.path.join(input_dir, "radar_data_vel")
    if not os.path.isdir(ang_dir) or not os.path.isdir(vel_dir):
        raise FileNotFoundError(
            "input_dir must contain either radar_data_ang.npy/radar_data_vel.npy "
            "or radar_data_ang/ and radar_data_vel/ subdirectories."
        )

    pairs = []
    for root, _, files in os.walk(ang_dir):
        for filename in files:
            if not filename.lower().endswith(".npy"):
                continue
            ang_path = os.path.join(root, filename)
            rel_path = os.path.relpath(ang_path, ang_dir)
            vel_path = os.path.join(vel_dir, rel_path)
            if os.path.isfile(vel_path):
                pairs.append((rel_path, ang_path, vel_path))
            else:
                print(f"[WARNING] Missing velocity file for angle map: {ang_path}")

    return sorted(pairs, key=lambda item: item[0])


def output_paths(output_dir: str, rel_path: str, single_pair: bool) -> tuple[str, str]:
    if single_pair:
        return (
            os.path.join(output_dir, "radar_data_ang.npy"),
            os.path.join(output_dir, "radar_data_vel.npy"),
        )

    return (
        os.path.join(output_dir, "radar_data_ang", rel_path),
        os.path.join(output_dir, "radar_data_vel", rel_path),
    )


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
    if degradation == "dropout" and not 0.0 <= intensity <= 1.0:
        raise ValueError("Dropout intensity must be between zero and one.")
    if degradation == "noise" and intensity < 0.0:
        raise ValueError("Noise intensity must be non-negative.")

    pairs = collect_radar_pairs(input_dir)
    if not pairs:
        raise FileNotFoundError(f"No paired radar .npy files found in: {input_dir}")

    single_pair = len(pairs) == 1 and pairs[0][0] == "radar_data_ang.npy"
    progress_iter = tqdm(
        pairs,
        desc=f"{degradation}/intensity_{intensity}",
        unit="pair",
        dynamic_ncols=True,
    ) if tqdm else pairs

    written_count = 0
    skipped_count = 0
    failed_count = 0

    print(f"[INFO] Radar pairs: {len(pairs)}")
    print(f"[INFO] Degradation: {degradation}")
    print(f"[INFO] Intensity: {intensity}")

    for pair_idx, (rel_path, ang_path, vel_path) in enumerate(progress_iter):
        out_ang_path, out_vel_path = output_paths(output_dir, rel_path, single_pair)

        if (
            os.path.exists(out_ang_path)
            and os.path.exists(out_vel_path)
            and not overwrite
        ):
            skipped_count += 1
            if tqdm:
                progress_iter.set_postfix_str(f"{rel_path} | skipped")
            else:
                print(f"[INFO] Skip existing pair: {rel_path}")
            continue

        try:
            radar_ang = np.load(ang_path)
            radar_vel = np.load(vel_path)

            rng_ang = np.random.default_rng(seed + pair_idx * 2)
            rng_vel = np.random.default_rng(seed + pair_idx * 2 + 1)
            degraded_ang = apply_degradation(
                radar_ang,
                degradation,
                intensity,
                rng_ang,
            )
            degraded_vel = apply_degradation(
                radar_vel,
                degradation,
                intensity,
                rng_vel,
            )

            if degraded_ang.shape != radar_ang.shape:
                raise RuntimeError("Angle radar shape changed during degradation.")
            if degraded_vel.shape != radar_vel.shape:
                raise RuntimeError("Velocity radar shape changed during degradation.")

            os.makedirs(os.path.dirname(out_ang_path), exist_ok=True)
            os.makedirs(os.path.dirname(out_vel_path), exist_ok=True)
            np.save(out_ang_path, degraded_ang)
            np.save(out_vel_path, degraded_vel)
            written_count += 1

            if tqdm:
                progress_iter.set_postfix_str(
                    f"{rel_path} | ang {radar_ang.shape} | vel {radar_vel.shape}"
                )
            else:
                print(
                    f"[INFO] {rel_path} | "
                    f"ang shape: {radar_ang.shape} | vel shape: {radar_vel.shape}"
                )
        except Exception as exc:
            failed_count += 1
            if tqdm:
                progress_iter.write(f"[WARNING] Failed to process {rel_path}: {exc}")
            else:
                print(f"[WARNING] Failed to process {rel_path}: {exc}")

    print("Done.")
    print(f"[INFO] Written pairs: {written_count}")
    print(f"[INFO] Skipped pairs: {skipped_count}")
    print(f"[INFO] Failed pairs: {failed_count}")
    print(f"[INFO] Output directory: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply degradation to radar angle and velocity numpy feature maps."
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--degradation",
        type=str,
        required=True,
        choices=["dropout", "noise"],
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
