from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np


DEFAULT_HOME_POSITION = np.array([3.07692308, -33.14285714,  41.18681319,  61.8021978,  -89.62637363, 0.0])
DEFAULT_ROBOT_PORT = "/dev/ttyACM0"
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
DEFAULT_FALLBACK_MODELS = "gemini-2.5-flash-lite"

JSON_PATH = "camera_calib/data/calib_poses_data/2026-04-30_14-41-28/samples.json"
ROBOT_PORT = "/dev/ttyACM0"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a planar pixel<->world homography from point correspondences."
    )
    parser.add_argument(
        "--points",
        type=Path,
        default=JSON_PATH,
        help=(
            "JSON/CSV with pixel/world correspondences, or samples.json with "
            "`key` and `gripper_pose.position_m` for Gemini-based collection."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("camera_calib/calibrations"),
        help="Where to save homography files. Default: camera_calib/calibrations",
    )
    parser.add_argument(
        "--ransac-threshold-px",
        type=float,
        default=5.0,
        help="RANSAC reprojection threshold in image pixels. Default: 5.0",
    )
    parser.add_argument("--letters", help="Optional comma-separated subset of letters from --points.")
    parser.add_argument("--image", type=Path, help="Optional image path. If omitted, capture from --camera.")
    parser.add_argument("--camera", type=int, default=5, help="OpenCV camera index for Gemini capture.")
    parser.add_argument(
        "--backend",
        choices=["auto", "dshow", "msmf", "any"],
        default="auto",
        help="OpenCV camera backend. Default: auto.",
    )
    parser.add_argument("--project", default=os.getenv("GOOGLE_CLOUD_PROJECT"), help="Google Cloud project ID.")
    parser.add_argument("--location", default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL, help="Gemini model for letter detection.")
    parser.add_argument("--fallback-models", default=DEFAULT_FALLBACK_MODELS)
    parser.add_argument("--api-max-dim", type=int, default=1280)
    parser.add_argument("--api-jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--correspondences-output",
        type=Path,
        default=Path("camera_calib/letter_points_pixel_world.json"),
        help="Where to save Gemini pixel/world letter correspondences.",
    )
    parser.add_argument(
        "--annotated-output",
        type=Path,
        default=Path("camera_calib/letter_homography_points.jpg"),
        help="Where to save an annotated Gemini detection image.",
    )
    parser.add_argument(
        "--robot-port",
        default=ROBOT_PORT,
        help="Robot serial port used to move to home before Gemini capture.",
    )
    parser.add_argument(
        "--home-deg",
        default="0,-30,30,60,-90,0",
        help="Comma-separated home joint position in degrees. Default: 0,-30,30,60,-90,0",
    )
    parser.add_argument("--skip-home", action="store_true", help="Do not move the robot to home first.")
    return parser.parse_args()


def load_correspondences(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Correspondence file not found: {path}")

    if path.suffix.lower() == ".csv":
        rows = load_csv_rows(path)
    else:
        rows = load_json_rows(path)

    pixels: list[list[float]] = []
    world_xy: list[list[float]] = []
    for index, row in enumerate(rows):
        try:
            pixel = extract_pixel(row)
            world = extract_world(row)
        except (KeyError, TypeError, ValueError) as exc:
            print(f"Warning: skipping row {index}: {exc}")
            continue
        pixels.append([float(pixel[0]), float(pixel[1])])
        world_xy.append([float(world[0]), float(world[1])])

    if len(pixels) < 4:
        raise ValueError(f"Need at least 4 valid correspondences, got {len(pixels)}.")

    return np.asarray(pixels, dtype=np.float64), np.asarray(world_xy, dtype=np.float64)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json_rows(path: Path) -> list[Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("points", "correspondences", "samples"):
            if isinstance(data.get(key), list):
                return data[key]

        pixel_points = data.get("pixels") or data.get("image_points") or data.get("pixel_points")
        world_points = data.get("world") or data.get("world_points") or data.get("world_xy")
        if isinstance(pixel_points, list) and isinstance(world_points, list):
            if len(pixel_points) != len(world_points):
                raise ValueError("pixel and world point arrays must have the same length.")
            return [{"pixel": pixel, "world": world} for pixel, world in zip(pixel_points, world_points)]

    raise ValueError(
        "Unsupported JSON format. Use a list of points, or an object with points/correspondences, "
        "or image_points + world_points arrays."
    )


def looks_like_key_samples(path: Path) -> bool:
    if path.suffix.lower() == ".csv":
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    rows = data.get("samples") if isinstance(data, dict) else data
    return (
        isinstance(rows, list)
        and bool(rows)
        and isinstance(rows[0], dict)
        and "key" in rows[0]
        and "gripper_pose" in rows[0]
    )


def extract_pixel(row: Any) -> list[float]:
    if not isinstance(row, dict):
        raise TypeError("row must be an object/dict")

    for key in ("pixel", "uv", "image", "image_point", "pixel_point"):
        value = row.get(key)
        if is_point_like(value, min_len=2):
            return [float(value[0]), float(value[1])]

    u = first_existing(row, ("u", "px", "pixel_x", "image_x", "col"))
    v = first_existing(row, ("v", "py", "pixel_y", "image_y", "row"))
    return [float(u), float(v)]


def extract_world(row: Any) -> list[float]:
    if not isinstance(row, dict):
        raise TypeError("row must be an object/dict")

    for key in ("world", "world_xy", "world_point", "xy", "xyz"):
        value = row.get(key)
        if is_point_like(value, min_len=2):
            return [float(value[0]), float(value[1])]

    x = first_existing(row, ("x", "world_x", "X"))
    y = first_existing(row, ("y", "world_y", "Y"))
    return [float(x), float(y)]


def first_existing(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    raise KeyError(f"missing one of: {', '.join(keys)}")


def is_point_like(value: Any, *, min_len: int) -> bool:
    return isinstance(value, (list, tuple)) and len(value) >= min_len


def apply_homography(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points_h = np.column_stack([points_xy, np.ones(len(points_xy))])
    mapped_h = points_h @ H.T
    return mapped_h[:, :2] / mapped_h[:, 2:3]


def print_error_stats(label: str, errors: np.ndarray, unit: str) -> None:
    if len(errors) == 0:
        print(f"{label}: no inliers")
        return
    rms = float(np.sqrt(np.mean(errors**2)))
    mean = float(np.mean(errors))
    max_error = float(np.max(errors))
    print(f"{label} mean: {mean:.6f} {unit}")
    print(f"{label} RMS:  {rms:.6f} {unit}")
    print(f"{label} max:  {max_error:.6f} {unit}")


def print_projection_tests(
    labels: list[str] | None,
    pixels: np.ndarray,
    projected_world: np.ndarray,
    expected_world: np.ndarray,
    errors: np.ndarray,
    inliers: np.ndarray,
) -> None:
    print("Pixel -> world sanity checks:")
    for index, (pixel, predicted, expected, error, is_inlier) in enumerate(
        zip(pixels, projected_world, expected_world, errors, inliers)
    ):
        label = labels[index] if labels is not None and index < len(labels) else f"pt{index}"
        status = "inlier" if is_inlier else "outlier"
        print(
            f"  {label}: pixel=({pixel[0]:.1f}, {pixel[1]:.1f}) -> "
            f"world=({predicted[0]:.5f}, {predicted[1]:.5f}), "
            f"expected=({expected[0]:.5f}, {expected[1]:.5f}), "
            f"err={error:.5f} m [{status}]"
        )


def normalize_letter(value: Any) -> str:
    letter = str(value).strip().upper()
    if len(letter) != 1 or not letter.isalpha():
        raise ValueError(f"Expected a single alphabetic key letter, got {value!r}")
    return letter


def load_letter_world_points(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("samples"), list):
        rows = data["samples"]
    elif isinstance(data, list):
        rows = data
    elif isinstance(data, dict) and isinstance(data.get("letters"), dict):
        return {
            normalize_letter(letter): normalize_world_point(world)
            for letter, world in data["letters"].items()
        }
    elif isinstance(data, dict):
        return {
            normalize_letter(letter): normalize_world_point(world)
            for letter, world in data.items()
        }
    else:
        raise ValueError("Unsupported samples/world-point JSON format.")

    world_points: dict[str, list[float]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            print(f"Warning: skipping sample {index}: not an object")
            continue
        try:
            letter = normalize_letter(row["key"])
            world = sample_world_position(row)
        except (KeyError, TypeError, ValueError) as exc:
            print(f"Warning: skipping sample {index}: {exc}")
            continue
        if letter in world_points:
            print(f"Warning: duplicate key {letter}; keeping the first world point.")
            continue
        world_points[letter] = world

    if len(world_points) < 4:
        raise ValueError(f"Need at least 4 key/world samples, got {len(world_points)}.")
    return dict(sorted(world_points.items()))


def normalize_world_point(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"World point must be [x, y] or [x, y, z], got {value!r}")
    return [float(value[0]), float(value[1]), float(value[2]) if len(value) >= 3 else 0.0]


def sample_world_position(sample: dict[str, Any]) -> list[float]:
    pose = sample.get("gripper_pose")
    if not isinstance(pose, dict):
        raise ValueError("missing gripper_pose")
    if "position_m" in pose:
        return normalize_world_point(pose["position_m"])
    if "transform_matrix" in pose:
        transform = np.asarray(pose["transform_matrix"], dtype=float)
        if transform.shape != (4, 4):
            raise ValueError(f"transform_matrix must be 4x4, got {transform.shape}")
        return [float(v) for v in transform[:3, 3]]
    raise ValueError("missing gripper_pose.position_m")


def parse_letters_arg(letters_arg: str | None, world_points: dict[str, list[float]]) -> list[str]:
    if letters_arg is None:
        return list(world_points.keys())
    letters = [normalize_letter(part) for part in letters_arg.split(",") if part.strip()]
    missing = [letter for letter in letters if letter not in world_points]
    if missing:
        raise ValueError(f"These letters are not present in --points: {missing}")
    return list(dict.fromkeys(letters))


def parse_home_degrees(home_degrees: str) -> np.ndarray:
    values = [float(part.strip()) for part in home_degrees.split(",") if part.strip()]
    if len(values) != 6:
        raise ValueError(f"--home-deg must contain 6 comma-separated values, got {len(values)}.")
    return np.deg2rad(values)


def move_robot_home(args: argparse.Namespace) -> Any | None:
    if args.skip_home:
        print("Skipping robot home move because --skip-home was passed.")
        return None

    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from controller import SO101Interface

    #home_position = parse_home_degrees(args.home_deg)
    home_position = np.array(DEFAULT_HOME_POSITION) #degrees
    robot_interface = SO101Interface(port=args.robot_port)
    print(f"Moving robot to home position (deg): {home_position}")
    robot_interface.write_joints(np.deg2rad(home_position))
    time.sleep(2.0)
    return robot_interface


def collect_letter_correspondences(args: argparse.Namespace) -> list[dict[str, Any]]:
    # Lazy import keeps the old pixel/world-only mode independent of Gemini SDK.
    from collect_keyboard_homography_points import (
        call_gemini_for_letters,
        draw_letters,
        load_or_capture_image,
        save_correspondences,
    )

    world_points = load_letter_world_points(args.points)
    letters = parse_letters_arg(args.letters, world_points)
    robot_interface = move_robot_home(args)
    try:
        image, image_source = load_or_capture_image(args)
        fallback_models = [model.strip() for model in args.fallback_models.split(",") if model.strip()]
        pixel_points = call_gemini_for_letters(
            image,
            letters=letters,
            project=args.project,
            location=args.location,
            model=args.model,
            fallback_models=fallback_models,
            api_max_dim=args.api_max_dim,
            api_jpeg_quality=args.api_jpeg_quality,
        )
        rows = save_correspondences(
            args.correspondences_output,
            letters=letters,
            world_points=world_points,
            pixel_points=pixel_points,
            image_source=image_source,
        )
        draw_letters(image, rows, args.annotated_output)
        print(f"Saved Gemini correspondences: {args.correspondences_output}")
        print(f"Saved annotated detections: {args.annotated_output}")
        return rows
    finally:
        if robot_interface is not None:
            robot_interface.close()


def correspondences_to_arrays(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray([extract_pixel(row) for row in rows], dtype=np.float64)
    world_xy = np.asarray([extract_world(row)[:2] for row in rows], dtype=np.float64)
    if len(pixels) < 4:
        raise ValueError(f"Need at least 4 valid correspondences, got {len(pixels)}.")
    return pixels, world_xy


def main() -> None:
    args = parse_args()
    labels: list[str] | None = None
    if looks_like_key_samples(args.points):
        print("Detected samples.json with key + gripper_pose; using Gemini to collect letter pixels.")
        rows = collect_letter_correspondences(args)
        pixel_uv, world_xy = correspondences_to_arrays(rows)
        labels = [str(row.get("letter", row.get("name", f"pt{index}"))) for index, row in enumerate(rows)]
    else:
        try:
            pixel_uv, world_xy = load_correspondences(args.points)
        except ValueError:
            print("No pixel/world correspondences found; treating --points as letter world points.")
            rows = collect_letter_correspondences(args)
            pixel_uv, world_xy = correspondences_to_arrays(rows)
            labels = [str(row.get("letter", row.get("name", f"pt{index}"))) for index, row in enumerate(rows)]

    # RANSAC threshold is easiest to reason about in pixels, so estimate
    # world->pixel first, then invert it to obtain pixel->world.
    H_world_to_pixel, mask = cv.findHomography(
        world_xy,
        pixel_uv,
        method=cv.RANSAC,
        ransacReprojThreshold=args.ransac_threshold_px,
    )
    if H_world_to_pixel is None or mask is None:
        raise RuntimeError("cv.findHomography failed. Check that the points are not degenerate.")

    H_pixel_to_world = np.linalg.inv(H_world_to_pixel)
    inliers = mask.ravel().astype(bool)

    pred_pixels = apply_homography(H_world_to_pixel, world_xy)
    pixel_errors = np.linalg.norm(pred_pixels - pixel_uv, axis=1)

    pred_world = apply_homography(H_pixel_to_world, pixel_uv)
    world_errors = np.linalg.norm(pred_world - world_xy, axis=1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    px_to_world_npy = args.output_dir / "homography_pixel_to_world.npy"
    px_to_world_txt = args.output_dir / "homography_pixel_to_world.txt"
    world_to_px_npy = args.output_dir / "homography_world_to_pixel.npy"
    world_to_px_txt = args.output_dir / "homography_world_to_pixel.txt"

    np.save(px_to_world_npy, H_pixel_to_world)
    np.savetxt(px_to_world_txt, H_pixel_to_world, fmt="%.12g")
    np.save(world_to_px_npy, H_world_to_pixel)
    np.savetxt(world_to_px_txt, H_world_to_pixel, fmt="%.12g")

    print(f"Loaded correspondences: {len(pixel_uv)}")
    print(f"RANSAC inliers: {int(inliers.sum())}/{len(inliers)}")
    print("H_pixel_to_world:")
    print(H_pixel_to_world)
    print("H_world_to_pixel:")
    print(H_world_to_pixel)
    print_error_stats("Pixel reprojection error (inliers)", pixel_errors[inliers], "px")
    print_error_stats("World back-projection error (inliers)", world_errors[inliers], "m")
    print_projection_tests(labels, pixel_uv, pred_world, world_xy, world_errors, inliers)
    print(f"Saved: {px_to_world_npy}")
    print(f"Saved: {px_to_world_txt}")
    print(f"Saved: {world_to_px_npy}")
    print(f"Saved: {world_to_px_txt}")


if __name__ == "__main__":
    main()
