from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_IMAGE_FOLDER = Path("camera_calib/data/raw_calib_data/2026-04-26_12-01-09/images")
DEFAULT_OUTPUT_PREFIX = Path("camera_calib/calibrations/camera_calibration")
IMAGE_GLOB_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


def collect_image_paths(folder: Path, patterns: Iterable[str]) -> list[Path]:
    image_paths: list[Path] = []
    for pattern in patterns:
        image_paths.extend(folder.glob(pattern))
    return sorted(set(path.resolve() for path in image_paths))


def build_checkerboard_object_points(
    rows: int,
    cols: int,
    square_size_m: float,
) -> np.ndarray:
    object_points = np.zeros((rows * cols, 3), dtype=np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    object_points[:, :2] = grid * square_size_m
    return object_points


def find_checkerboard_corners(
    gray: np.ndarray,
    pattern_size: tuple[int, int],
) -> tuple[bool, np.ndarray | None]:
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=(
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK
        ),
    )
    if not found:
        return False, None

    termination = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1e-3,
    )
    refined_corners = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=termination,
    )
    return True, refined_corners


def compute_reprojection_errors(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: tuple[np.ndarray, ...],
    tvecs: tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[float]:
    errors = []
    for objp, imgp, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected_points, _ = cv2.projectPoints(
            objp,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        projected_points = projected_points.reshape(-1, 2)
        observed_points = imgp.reshape(-1, 2)
        point_errors = np.linalg.norm(projected_points - observed_points, axis=1)
        errors.append(float(np.mean(point_errors)))
    return errors


def save_calibration(
    output_prefix: Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    rms_error: float,
    mean_reprojection_error_px: float,
    per_image_errors_px: list[float],
    used_images: list[Path],
    rows: int,
    cols: int,
    square_size_m: float,
) -> None:
    npz_path = output_prefix.with_suffix(".npz")
    json_path = output_prefix.with_suffix(".json")

    np.savez(
        npz_path,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size, dtype=np.int32),
        rms_error=np.array(rms_error, dtype=np.float64),
        mean_reprojection_error_px=np.array(
            mean_reprojection_error_px,
            dtype=np.float64,
        ),
    )

    payload = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
        "image_size": {
            "width": image_size[0],
            "height": image_size[1],
        },
        "rms_error": rms_error,
        "mean_reprojection_error_px": mean_reprojection_error_px,
        "checkerboard": {
            "inner_rows": rows,
            "inner_cols": cols,
            "square_size_m": square_size_m,
        },
        "used_images": [
            {
                "path": str(path),
                "mean_reprojection_error_px": error,
            }
            for path, error in zip(used_images, per_image_errors_px)
        ],
    }
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print()
    print(f"Saved NumPy calibration: {npz_path.resolve()}")
    print(f"Saved JSON calibration:  {json_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate camera intrinsics and distortion from checkerboard images."
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_IMAGE_FOLDER,
        help=f"Folder containing checkerboard images. Default: {DEFAULT_IMAGE_FOLDER}",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=6,
        help="Number of checkerboard inner-corner rows. Default: 6",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=8,
        help="Number of checkerboard inner-corner columns. Default: 8",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=0.014,
        help="Checkerboard square size in meters. Default: 0.014",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT_PREFIX,
        help=(
            "Output path without extension. The script writes .npz and .json. "
            f"Default: {DEFAULT_OUTPUT_PREFIX}"
        ),
    )
    parser.add_argument(
        "--fix-k3",
        action="store_true",
        help="Fix the k3 radial distortion coefficient during calibration.",
    )
    parser.add_argument(
        "--zero-tangent-dist",
        action="store_true",
        help="Force tangential distortion p1,p2 to zero.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_folder = args.images
    image_paths = collect_image_paths(image_folder, IMAGE_GLOB_PATTERNS)

    print("Camera calibration started")
    print(f"Image folder: {image_folder.resolve()}")
    print(
        f"Checkerboard inner corners: rows={args.rows}, cols={args.cols}, "
        f"square_size={args.square_size} m"
    )
    print(f"Found {len(image_paths)} image(s)")

    if not image_paths:
        raise ValueError(f"No calibration images found in {image_folder.resolve()}")

    pattern_size = (args.cols, args.rows)
    template_object_points = build_checkerboard_object_points(
        rows=args.rows,
        cols=args.cols,
        square_size_m=args.square_size,
    )

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    used_images: list[Path] = []
    image_size: tuple[int, int] | None = None

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[{index}/{len(image_paths)}] {image_path.name}: skipped, unreadable")
            continue

        height, width = image.shape[:2]
        if image_size is None:
            image_size = (width, height)
        elif image_size != (width, height):
            print(
                f"[{index}/{len(image_paths)}] {image_path.name}: skipped, "
                f"image size {(width, height)} differs from {image_size}"
            )
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = find_checkerboard_corners(gray, pattern_size)
        if not found or corners is None:
            print(
                f"[{index}/{len(image_paths)}] {image_path.name}: skipped, "
                "checkerboard not found"
            )
            continue

        object_points.append(template_object_points.copy())
        image_points.append(corners)
        used_images.append(image_path)
        print(f"[{index}/{len(image_paths)}] {image_path.name}: checkerboard detected")

    if image_size is None:
        raise RuntimeError("No readable calibration images found.")

    if len(object_points) < 5:
        raise RuntimeError(
            "Need at least 5 valid checkerboard detections for camera calibration. "
            "Use more images with the board at different positions, tilts, and depths."
        )

    flags = 0
    if args.fix_k3:
        flags |= cv2.CALIB_FIX_K3
    if args.zero_tangent_dist:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST

    rms_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=flags,
    )

    per_image_errors_px = compute_reprojection_errors(
        object_points=object_points,
        image_points=image_points,
        rvecs=rvecs,
        tvecs=tvecs,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )
    mean_reprojection_error_px = float(np.mean(per_image_errors_px))

    print()
    print("=== Camera Calibration Result ===")
    print(f"Valid checkerboard detections: {len(used_images)}/{len(image_paths)}")
    print(f"Image size: {image_size[0]} x {image_size[1]} px")
    print(f"OpenCV RMS reprojection error: {rms_error:.6f} px")
    print(f"Mean per-image reprojection error: {mean_reprojection_error_px:.6f} px")
    print()
    print("Camera matrix K:")
    print(camera_matrix)
    print()
    print("Distortion coefficients [k1, k2, p1, p2, k3, ...]:")
    print(dist_coeffs.reshape(-1))
    print()
    print("Worst per-image reprojection errors:")
    worst_indices = np.argsort(per_image_errors_px)[::-1][: min(5, len(used_images))]
    for worst_index in worst_indices:
        print(
            f"  {used_images[int(worst_index)].name}: "
            f"{per_image_errors_px[int(worst_index)]:.6f} px"
        )

    save_calibration(
        output_prefix=args.output_prefix,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=image_size,
        rms_error=float(rms_error),
        mean_reprojection_error_px=mean_reprojection_error_px,
        per_image_errors_px=per_image_errors_px,
        used_images=used_images,
        rows=args.rows,
        cols=args.cols,
        square_size_m=args.square_size,
    )

    print()
    print("Use these in hand_eye_calibration.py:")
    print("K = np.array(" + repr(camera_matrix.tolist()) + ", dtype=np.float64)")
    print("dist = np.array(" + repr(dist_coeffs.reshape(-1).tolist()) + ", dtype=np.float64)")


if __name__ == "__main__":
    main()
