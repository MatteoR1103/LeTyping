from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.optimize import least_squares


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLES_JSON_PATH = PROJECT_ROOT / "camera_calib/data/calib_poses_data/2026-04-29_11-52-32/samples.json"
LANDMARKS_JSON_PATH = PROJECT_ROOT / "camera_calib/data/calib_poses_data/gripper_poses_plane/samples.json"

CHECKERBOARD_ROWS = 9
CHECKERBOARD_COLS = 13
CHECKERBOARD_CORNERS = CHECKERBOARD_ROWS * CHECKERBOARD_COLS

CAMERA_CALIB_PATH = PROJECT_ROOT / "camera_calib/calibrations/camera_calibration.npz"
INITIAL_TGC_PATH = PROJECT_ROOT / "camera_calib/calibrations/rigid_transform_handeye.npy"
OUTPUT_TGC_PATH = PROJECT_ROOT / "camera_calib/calibrations/rigid_transform_nonlinear.npy"
REPORT_PATH = PROJECT_ROOT / "camera_calib/stats/nonlinear_handeye_report.txt"


camera_intrinsics = np.load(CAMERA_CALIB_PATH)
K = camera_intrinsics["camera_matrix"]
dist = camera_intrinsics["dist_coeffs"]


@dataclass(frozen=True)
class DetectedSample:
    label: str
    T_bg: np.ndarray
    rays_c: np.ndarray


def load_samples_json(samples_json_path: Path) -> list[dict]:
    with samples_json_path.open("r", encoding="utf-8") as fh:
        samples = json.load(fh)

    if not isinstance(samples, list):
        raise ValueError(f"Expected {samples_json_path} to contain a JSON list.")

    return samples


def build_homogeneous_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def pose_dict_to_homogeneous_transform(pose_dict: dict) -> np.ndarray:
    if "transform_matrix" in pose_dict:
        transform = np.asarray(pose_dict["transform_matrix"], dtype=np.float64)
        if transform.shape == (4, 4):
            return transform

    if "transform" in pose_dict:
        transform = np.asarray(pose_dict["transform"], dtype=np.float64)
        if transform.shape == (4, 4):
            return transform

    if "matrix" in pose_dict:
        matrix = np.asarray(pose_dict["matrix"], dtype=np.float64)
        if matrix.shape == (4, 4):
            return matrix

    if "rotation" in pose_dict and "translation" in pose_dict:
        R = np.asarray(pose_dict["rotation"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(pose_dict["translation"], dtype=np.float64).reshape(3, 1)
        return build_homogeneous_transform(R, t)

    if "rotation_matrix" in pose_dict and "position_m" in pose_dict:
        R = np.asarray(pose_dict["rotation_matrix"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(pose_dict["position_m"], dtype=np.float64).reshape(3, 1)
        return build_homogeneous_transform(R, t)

    raise ValueError("Pose dictionary does not contain a supported pose format.")


def sample_to_gripper_pose(sample: dict) -> np.ndarray:
    for key in ("gripper_pose", "robot_pose", "pose", "T_base_gripper", "base_T_gripper"):
        value = sample.get(key)
        if isinstance(value, dict):
            return pose_dict_to_homogeneous_transform(value)
        if value is not None:
            matrix = np.asarray(value, dtype=np.float64)
            if matrix.shape == (4, 4):
                return matrix

    raise ValueError(f"Could not find a gripper pose in sample {sample.get('sample_idx')}.")


def sample_to_position(sample: dict) -> np.ndarray:
    if all(key in sample for key in ("ee.x", "ee.y", "ee.z")):
        return np.array([sample["ee.x"], sample["ee.y"], sample["ee.z"]], dtype=np.float64)

    pose = sample.get("gripper_pose")
    if isinstance(pose, dict) and "position_m" in pose:
        return np.asarray(pose["position_m"], dtype=np.float64).reshape(3)

    return sample_to_gripper_pose(sample)[:3, 3].copy()


def sample_to_checkerboard_index(sample: dict, sample_number: int, path: Path) -> tuple[int, int]:
    if "row" not in sample or "col" not in sample:
        raise ValueError(
            f"Landmark sample #{sample_number} in {path.resolve()} is missing row/col."
        )

    row = int(sample["row"])
    col = int(sample["col"])
    if not (0 <= row < CHECKERBOARD_ROWS and 0 <= col < CHECKERBOARD_COLS):
        raise ValueError(
            f"Landmark sample #{sample_number} has row={row}, col={col}; "
            f"valid ranges are row=[0,{CHECKERBOARD_ROWS - 1}], "
            f"col=[0,{CHECKERBOARD_COLS - 1}]."
        )

    return row, col


def load_sparse_landmarks(path: Path) -> tuple[np.ndarray, np.ndarray]:
    samples = load_samples_json(path)
    landmarks_b = np.full((CHECKERBOARD_CORNERS, 3), np.nan, dtype=np.float64)
    seen: set[tuple[int, int]] = set()

    for sample_number, sample in enumerate(samples, start=1):
        row, col = sample_to_checkerboard_index(sample, sample_number, path)
        if (row, col) in seen:
            raise ValueError(f"Duplicate landmark row={row}, col={col} in {path.resolve()}.")
        seen.add((row, col))
        landmarks_b[row * CHECKERBOARD_COLS + col] = sample_to_position(sample)

    valid_indices = np.flatnonzero(~np.isnan(landmarks_b).any(axis=1))
    if len(valid_indices) < 4:
        raise ValueError("Need at least 4 sparse robot-frame landmarks.")

    return landmarks_b, valid_indices


def detect_checkerboard_samples(samples: list[dict], samples_json_path: Path) -> list[DetectedSample]:
    pattern_size = (CHECKERBOARD_COLS, CHECKERBOARD_ROWS)
    termination = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    detected_samples: list[DetectedSample] = []

    for index, sample in enumerate(samples, start=1):
        image_path_value = sample.get("image_path")
        if not image_path_value:
            print(f"[{index}/{len(samples)}] Skipping sample without image_path")
            continue

        image_path = Path(image_path_value)
        if not image_path.is_absolute():
            image_path = (samples_json_path.resolve().parent / image_path).resolve()

        print(f"[{index}/{len(samples)}] Detecting corners in {image_path.name}")
        image = cv.imread(str(image_path), cv.IMREAD_COLOR)
        if image is None:
            print("  Skipping: image could not be read")
            continue

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            print("  Skipping: checkerboard detection failed")
            continue

        refined_corners = cv.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=termination,
        )
        corner_pixels = refined_corners.reshape(-1, 2)
        undistorted = cv.undistortPoints(
            corner_pixels.reshape(-1, 1, 2).astype(np.float64),
            K,
            dist,
        ).reshape(-1, 2)
        rays_c = np.column_stack((undistorted, np.ones(len(undistorted))))
        rays_c /= np.linalg.norm(rays_c, axis=1, keepdims=True)

        try:
            T_bg = sample_to_gripper_pose(sample)
        except ValueError as exc:
            print(f"  Skipping: {exc}")
            continue

        detected_samples.append(
            DetectedSample(
                label=image_path.name,
                T_bg=np.asarray(T_bg, dtype=np.float64).reshape(4, 4),
                rays_c=rays_c,
            )
        )

    return detected_samples


def transform_to_params(T_gc: np.ndarray) -> np.ndarray:
    rvec, _ = cv.Rodrigues(T_gc[:3, :3])
    return np.concatenate([rvec.reshape(3), T_gc[:3, 3]])


def params_to_transform(params: np.ndarray) -> np.ndarray:
    R_gc, _ = cv.Rodrigues(np.asarray(params[:3], dtype=np.float64).reshape(3, 1))
    return build_homogeneous_transform(R_gc, params[3:6])


def rays_in_base(
    detected_samples: list[DetectedSample],
    T_gc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    origins = np.empty((len(detected_samples), CHECKERBOARD_CORNERS, 3), dtype=np.float64)
    directions = np.empty_like(origins)

    for sample_index, sample in enumerate(detected_samples):
        T_bc = sample.T_bg @ T_gc
        origins[sample_index, :, :] = T_bc[:3, 3]
        directions[sample_index] = (T_bc[:3, :3] @ sample.rays_c.T).T
        directions[sample_index] /= np.linalg.norm(
            directions[sample_index],
            axis=1,
            keepdims=True,
        )

    return origins, directions


def closest_point_to_rays(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    I = np.eye(3, dtype=np.float64)

    for origin, direction in zip(origins, directions):
        projector = I - np.outer(direction, direction)
        A += projector
        b += projector @ origin

    return np.linalg.lstsq(A, b, rcond=None)[0]


def point_to_ray_residuals(
    point: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
) -> np.ndarray:
    deltas = point.reshape(1, 3) - origins
    along_ray = np.sum(deltas * directions, axis=1, keepdims=True) * directions
    return deltas - along_ray


def optimization_residuals(
    params: np.ndarray,
    detected_samples: list[DetectedSample],
    landmarks_b: np.ndarray,
    repeatability_weight: float,
    landmark_weight: float,
) -> np.ndarray:
    T_gc = params_to_transform(params)
    origins, directions = rays_in_base(detected_samples, T_gc)
    residual_blocks = []

    repeatability_scale = np.sqrt(repeatability_weight)
    landmark_scale = np.sqrt(landmark_weight)

    for corner_index in range(CHECKERBOARD_CORNERS):
        corner_origins = origins[:, corner_index, :]
        corner_directions = directions[:, corner_index, :]
        fitted_point = closest_point_to_rays(corner_origins, corner_directions)
        residual_blocks.append(
            repeatability_scale
            * point_to_ray_residuals(fitted_point, corner_origins, corner_directions).reshape(-1)
        )

        landmark = landmarks_b[corner_index]
        if not np.isnan(landmark).any() and landmark_weight > 0:
            residual_blocks.append(
                landmark_scale
                * point_to_ray_residuals(landmark, corner_origins, corner_directions).reshape(-1)
            )

    return np.concatenate(residual_blocks)


def calculate_metrics(
    T_gc: np.ndarray,
    detected_samples: list[DetectedSample],
    landmarks_b: np.ndarray,
) -> dict[str, float]:
    origins, directions = rays_in_base(detected_samples, T_gc)
    repeatability_distances = []
    landmark_distances = []

    for corner_index in range(CHECKERBOARD_CORNERS):
        corner_origins = origins[:, corner_index, :]
        corner_directions = directions[:, corner_index, :]
        fitted_point = closest_point_to_rays(corner_origins, corner_directions)
        repeatability = point_to_ray_residuals(
            fitted_point,
            corner_origins,
            corner_directions,
        )
        repeatability_distances.extend(np.linalg.norm(repeatability, axis=1))

        landmark = landmarks_b[corner_index]
        if not np.isnan(landmark).any():
            landmark_residuals = point_to_ray_residuals(
                landmark,
                corner_origins,
                corner_directions,
            )
            landmark_distances.extend(np.linalg.norm(landmark_residuals, axis=1))

    repeatability_distances = np.asarray(repeatability_distances)
    landmark_distances = np.asarray(landmark_distances)

    return {
        "repeatability_mean_m": float(np.mean(repeatability_distances)),
        "repeatability_median_m": float(np.median(repeatability_distances)),
        "repeatability_rms_m": float(np.sqrt(np.mean(repeatability_distances**2))),
        "repeatability_max_m": float(np.max(repeatability_distances)),
        "landmark_mean_m": float(np.mean(landmark_distances)),
        "landmark_median_m": float(np.median(landmark_distances)),
        "landmark_rms_m": float(np.sqrt(np.mean(landmark_distances**2))),
        "landmark_max_m": float(np.max(landmark_distances)),
    }


def rotation_delta_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_delta = R_a.T @ R_b
    cos_angle = (np.trace(R_delta) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def format_metrics(metrics: dict[str, float], prefix: str) -> list[str]:
    return [
        f"{prefix} repeatability mean:   {metrics['repeatability_mean_m'] * 1000:.3f} mm",
        f"{prefix} repeatability median: {metrics['repeatability_median_m'] * 1000:.3f} mm",
        f"{prefix} repeatability RMS:    {metrics['repeatability_rms_m'] * 1000:.3f} mm",
        f"{prefix} repeatability max:    {metrics['repeatability_max_m'] * 1000:.3f} mm",
        f"{prefix} landmark mean:        {metrics['landmark_mean_m'] * 1000:.3f} mm",
        f"{prefix} landmark median:      {metrics['landmark_median_m'] * 1000:.3f} mm",
        f"{prefix} landmark RMS:         {metrics['landmark_rms_m'] * 1000:.3f} mm",
        f"{prefix} landmark max:         {metrics['landmark_max_m'] * 1000:.3f} mm",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize ^gT_c by making checkerboard corner rays consistent across "
            "images and tying sparse row/col landmarks to measured robot-frame points."
        )
    )
    parser.add_argument("--samples-json", type=Path, default=SAMPLES_JSON_PATH)
    parser.add_argument("--landmarks-json", type=Path, default=LANDMARKS_JSON_PATH)
    parser.add_argument("--initial-transform", type=Path, default=INITIAL_TGC_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_TGC_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--repeatability-weight", type=float, default=1.0)
    parser.add_argument(
        "--landmark-weight",
        type=float,
        default=0.2,
        help="Lower this if FK/contact landmarks are noisy. Default: 0.2",
    )
    parser.add_argument(
        "--loss",
        choices=("linear", "soft_l1", "huber", "cauchy", "arctan"),
        default="soft_l1",
    )
    parser.add_argument(
        "--f-scale-m",
        type=float,
        default=0.005,
        help="Robust loss transition in meters. Default: 0.005",
    )
    parser.add_argument("--max-nfev", type=int, default=200)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    samples = load_samples_json(args.samples_json)
    landmarks_b, landmark_indices = load_sparse_landmarks(args.landmarks_json)
    T_initial = np.load(args.initial_transform)

    print("Nonlinear hand-eye optimization started")
    print(f"Samples JSON: {args.samples_json.resolve()}")
    print(f"Landmarks JSON: {args.landmarks_json.resolve()}")
    print(f"Initial transform: {args.initial_transform.resolve()}")
    print(f"Sparse robot-frame landmarks: {len(landmark_indices)}/{CHECKERBOARD_CORNERS}")
    print(
        f"Weights: repeatability={args.repeatability_weight}, "
        f"landmark={args.landmark_weight}"
    )

    detected_samples = detect_checkerboard_samples(samples, args.samples_json)
    if len(detected_samples) < 2:
        raise RuntimeError("Need at least two detected checkerboards for ray consistency.")

    print(f"Detected checkerboard samples: {len(detected_samples)}/{len(samples)}")

    initial_metrics = calculate_metrics(T_initial, detected_samples, landmarks_b)
    x0 = transform_to_params(T_initial)
    initial_residuals = optimization_residuals(
        x0,
        detected_samples,
        landmarks_b,
        args.repeatability_weight,
        args.landmark_weight,
    )
    initial_linear_cost = 0.5 * float(initial_residuals @ initial_residuals)
    result = least_squares(
        optimization_residuals,
        x0,
        args=(detected_samples, landmarks_b, args.repeatability_weight, args.landmark_weight),
        loss=args.loss,
        f_scale=args.f_scale_m,
        max_nfev=args.max_nfev,
        verbose=1,
    )
    T_optimized = params_to_transform(result.x)
    final_residuals = optimization_residuals(
        result.x,
        detected_samples,
        landmarks_b,
        args.repeatability_weight,
        args.landmark_weight,
    )
    final_linear_cost = 0.5 * float(final_residuals @ final_residuals)
    optimized_metrics = calculate_metrics(T_optimized, detected_samples, landmarks_b)

    delta_t = T_optimized[:3, 3] - T_initial[:3, 3]
    delta_r_deg = rotation_delta_deg(T_initial[:3, :3], T_optimized[:3, :3])

    report_lines = [
        "=== Nonlinear Hand-Eye Optimization Result ===",
        f"success: {result.success}",
        f"message: {result.message}",
        f"function evaluations: {result.nfev}",
        f"initial linear cost: {initial_linear_cost:.10f}",
        f"final linear cost: {final_linear_cost:.10f}",
        f"final robust cost: {result.cost:.10f}",
        f"detected samples: {len(detected_samples)}/{len(samples)}",
        f"sparse landmarks: {len(landmark_indices)}/{CHECKERBOARD_CORNERS}",
        "",
        *format_metrics(initial_metrics, "initial"),
        "",
        *format_metrics(optimized_metrics, "optimized"),
        "",
        f"delta translation [m]: {delta_t}",
        f"delta rotation [deg]: {delta_r_deg:.4f}",
        "",
        "Initial T_gc:",
        str(T_initial),
        "",
        "Optimized T_gc:",
        str(T_optimized),
    ]

    print()
    print("\n".join(report_lines))

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Saved report to {args.report.resolve()}")

    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, T_optimized)
        print(f"Saved optimized transform to {args.output.resolve()}")


if __name__ == "__main__":
    main()
