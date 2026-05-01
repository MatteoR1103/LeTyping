from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares


# Camera intrinsics. These defaults mirror camera_calib/calibrations/camera_calibration.npz
# at the time this script was written. Pass --camera-calib to load another calibration file.
K = np.array(
    [
        [339.231277, 0.0, 315.729369],
        [0.0, 338.306822, 240.968736],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
dist = np.array([0.076909, -0.113363, -0.000896, -0.001614, 0.033428], dtype=np.float64)


DEFAULT_HOME_POSITION_DEG = np.array(
    [3.07692308, -33.14285714, 41.18681319, 61.8021978, -89.62637363, 0.0],
    dtype=np.float64,
)
DEFAULT_URDF_PATH = Path("cfg/arm_model/so101_new_calib.urdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refine T_gripper_camera with one annotated keyboard image and known "
            "base-frame keyboard key positions."
        )
    )
    parser.add_argument("--key-world-positions", type=Path, default=Path("key_world_positions.json"))
    parser.add_argument("--key-pixel-annotations", type=Path, default=Path("key_pixel_annotations.json"))
    parser.add_argument("--handeye-initial", type=Path, default=Path("handeye_initial.json"))
    parser.add_argument(
        "--image-gripper-pose",
        type=Path,
        default=Path("image_gripper_pose.json"),
        help=(
            "JSON containing T_base_gripper for the image. If absent, the script "
            "computes it from DEFAULT_HOME_POSITION_DEG with FK."
        ),
    )
    parser.add_argument("--image", type=Path, default=Path("keyboard_image.png"))
    parser.add_argument("--camera-calib", type=Path, default=None)
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--output", type=Path, default=Path("handeye_refined.json"))
    parser.add_argument("--debug-image", type=Path, default=Path("handeye_refinement_debug.png"))
    parser.add_argument("--lambda-rot", type=float, default=1.0)
    parser.add_argument("--lambda-trans", type=float, default=500.0)
    parser.add_argument("--max-rot-deg", type=float, default=15.0)
    parser.add_argument("--max-trans-m", type=float, default=0.05)
    parser.add_argument("--loss", choices=["linear", "soft_l1", "huber", "cauchy", "arctan"], default="huber")
    parser.add_argument("--f-scale", type=float, default=0.005)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_homogeneous_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def load_4x4_transform_from_json(path: Path, preferred_key: str | None = None) -> np.ndarray:
    data = load_json(path)
    if preferred_key is not None and isinstance(data, dict) and preferred_key in data:
        matrix = np.asarray(data[preferred_key], dtype=np.float64)
    elif isinstance(data, dict):
        matrix = None
        for key in (
            "T_gripper_camera",
            "T_base_gripper",
            "T",
            "transform_matrix",
            "transform",
            "matrix",
        ):
            if key in data:
                matrix = np.asarray(data[key], dtype=np.float64)
                break
        if matrix is None and "rotation_matrix" in data and "position_m" in data:
            return build_homogeneous_transform(data["rotation_matrix"], data["position_m"])
        if matrix is None:
            raise KeyError(f"No 4x4 transform key found in {path}")
    else:
        matrix = np.asarray(data, dtype=np.float64)

    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform in {path}, got shape {matrix.shape}")
    return matrix


def load_transform(path: Path, preferred_key: str | None = None) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        matrix = np.load(path)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 transform in {path}, got shape {matrix.shape}")
        return np.asarray(matrix, dtype=np.float64)
    return load_4x4_transform_from_json(path, preferred_key)


def correction_vector_to_se3(x: np.ndarray) -> np.ndarray:
    """Convert [rx, ry, rz, tx, ty, tz] into a homogeneous SE(3) correction."""
    x = np.asarray(x, dtype=np.float64).reshape(6)
    R_delta, _ = cv2.Rodrigues(x[:3].reshape(3, 1))
    return build_homogeneous_transform(R_delta, x[3:])


def estimate_plane_from_points(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit n^T x + d = 0 from base-frame key positions using PCA/SVD."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) < 3:
        raise ValueError("Need at least 3 world key positions to estimate a plane.")
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    if normal[2] < 0.0:
        normal = -normal
    d = -float(normal @ centroid)
    return normal, d, centroid


def undistort_pixels(pixels_uv: np.ndarray, K_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels_uv, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(pixels, K_matrix, dist_coeffs).reshape(-1, 2)


def back_project_pixel_to_camera_ray(pixel_uv: np.ndarray, K_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    xy = undistort_pixels(np.asarray(pixel_uv, dtype=np.float64).reshape(1, 2), K_matrix, dist_coeffs)[0]
    ray_c = np.array([xy[0], xy[1], 1.0], dtype=np.float64)
    return ray_c / np.linalg.norm(ray_c)


def transform_ray(
    origin: np.ndarray,
    direction: np.ndarray,
    T_to_from: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a ray by T_to_from, e.g. camera ray -> base ray with T_base_camera."""
    origin_to = T_to_from[:3, :3] @ np.asarray(origin, dtype=np.float64).reshape(3) + T_to_from[:3, 3]
    direction_to = T_to_from[:3, :3] @ np.asarray(direction, dtype=np.float64).reshape(3)
    direction_to /= np.linalg.norm(direction_to)
    return origin_to, direction_to


def ray_plane_intersection(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    eps: float = 1e-9,
) -> np.ndarray | None:
    denom = float(plane_normal @ ray_direction)
    if abs(denom) < eps:
        return None
    t = -float(plane_normal @ ray_origin + plane_d) / denom
    if t < 0.0:
        return None
    return ray_origin + t * ray_direction


def normalize_key(label: Any) -> str:
    return str(label).strip().upper()


def parse_point3(value: Any) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64).reshape(-1)
    if point.shape[0] < 3:
        raise ValueError(f"Expected a 3D point, got {value!r}")
    return point[:3]


def load_key_world_positions(path: Path) -> dict[str, np.ndarray]:
    """Load key -> base-frame XYZ in metres.

    Supports the requested mapping format plus common repo formats such as
    lists containing {letter/key/name, world} or {key, gripper_pose.position_m}.
    """
    data = load_json(path)
    positions: dict[str, np.ndarray] = {}
    if isinstance(data, dict) and "samples" in data:
        rows = data["samples"]
    elif isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = None
    else:
        raise ValueError(f"Unsupported world-position JSON format: {path}")

    if rows is None:
        for key, point in data.items():
            if isinstance(point, dict):
                if "world" in point:
                    point = point["world"]
                elif "position_m" in point:
                    point = point["position_m"]
                elif "gripper_pose" in point:
                    point = point["gripper_pose"]["position_m"]
            positions[normalize_key(key)] = parse_point3(point)
        return positions

    for row in rows:
        if not isinstance(row, dict):
            continue
        label = row.get("key", row.get("letter", row.get("name")))
        if label is None:
            continue
        if "world" in row:
            point = row["world"]
        elif "world_point" in row:
            point = row["world_point"]
        elif "position_m" in row:
            point = row["position_m"]
        elif "gripper_pose" in row and isinstance(row["gripper_pose"], dict):
            point = row["gripper_pose"]["position_m"]
        else:
            continue
        positions[normalize_key(label)] = parse_point3(point)
    return positions


def parse_point2(value: Any) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64).reshape(-1)
    if point.shape[0] < 2:
        raise ValueError(f"Expected a pixel coordinate, got {value!r}")
    return point[:2]


def load_key_pixel_annotations(path: Path) -> dict[str, np.ndarray]:
    """Load key -> annotated pixel [u, v] in the original image coordinate system."""
    data = load_json(path)
    pixels: dict[str, np.ndarray] = {}
    if isinstance(data, dict) and "annotations" in data:
        rows = data["annotations"]
    elif isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = None
    else:
        raise ValueError(f"Unsupported pixel-annotation JSON format: {path}")

    if rows is None:
        for key, pixel in data.items():
            if isinstance(pixel, dict):
                pixel = pixel.get("pixel", pixel.get("uv", pixel.get("center")))
            pixels[normalize_key(key)] = parse_point2(pixel)
        return pixels

    for row in rows:
        if not isinstance(row, dict):
            continue
        label = row.get("key", row.get("letter", row.get("name")))
        pixel = row.get("pixel", row.get("uv", row.get("center")))
        if label is None or pixel is None:
            continue
        pixels[normalize_key(label)] = parse_point2(pixel)
    return pixels


def load_camera_intrinsics(camera_calib_path: Path | None) -> tuple[np.ndarray, np.ndarray]:
    if camera_calib_path is None:
        return K.copy(), dist.copy()
    data = np.load(camera_calib_path)
    return np.asarray(data["camera_matrix"], dtype=np.float64), np.asarray(data["dist_coeffs"], dtype=np.float64)


def compute_home_T_base_gripper(urdf_path: Path) -> np.ndarray:
    """Compute T_base_gripper from FK at DEFAULT_HOME_POSITION_DEG.

    The default home joint vector mirrors DEFAULT_HOME_POSITION in main_pipeline.py
    and hover_control_pipeline.py. RobotKinematics.forward_kinematics expects
    joint angles in degrees.
    """
    repo_src = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(repo_src))
    from traj_generation import RobotKinematics  # pylint: disable=import-outside-toplevel

    kinematics = RobotKinematics(urdf_path=urdf_path)
    return np.asarray(kinematics.forward_kinematics(DEFAULT_HOME_POSITION_DEG), dtype=np.float64)


def load_or_compute_T_base_gripper(image_gripper_pose_path: Path, urdf_path: Path) -> np.ndarray:
    if image_gripper_pose_path.exists():
        return load_transform(image_gripper_pose_path, preferred_key="T_base_gripper")
    print(
        f"{image_gripper_pose_path} not found; computing T_base_gripper from "
        "DEFAULT_HOME_POSITION_DEG using FK."
    )
    return compute_home_T_base_gripper(urdf_path)


def build_refined_T_gripper_camera(T_gripper_camera_initial: np.ndarray, correction: np.ndarray) -> np.ndarray:
    # Frame convention:
    # T_gripper_camera maps a point expressed in camera frame into gripper frame.
    # The small correction Delta_T is right-multiplied:
    # T_gripper_camera_refined = T_gripper_camera_initial @ Delta_T.
    return T_gripper_camera_initial @ correction_vector_to_se3(correction)


def predict_key_points(
    correction: np.ndarray,
    key_labels: list[str],
    pixels: dict[str, np.ndarray],
    T_base_gripper: np.ndarray,
    T_gripper_camera_initial: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    K_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> dict[str, np.ndarray | None]:
    T_gripper_camera_refined = build_refined_T_gripper_camera(T_gripper_camera_initial, correction)
    # Frame convention:
    # T_base_gripper maps gripper-frame coordinates into robot base/world frame.
    # T_gripper_camera maps camera-frame coordinates into gripper frame.
    # T_base_camera maps camera-frame coordinates directly into robot base/world frame.
    T_base_camera = T_base_gripper @ T_gripper_camera_refined

    predictions: dict[str, np.ndarray | None] = {}
    for label in key_labels:
        ray_c = back_project_pixel_to_camera_ray(pixels[label], K_matrix, dist_coeffs)
        origin_b, direction_b = transform_ray(np.zeros(3), ray_c, T_base_camera)
        predictions[label] = ray_plane_intersection(origin_b, direction_b, plane_normal, plane_d)
    return predictions


def compute_position_errors(
    correction: np.ndarray,
    key_labels: list[str],
    world_positions: dict[str, np.ndarray],
    pixels: dict[str, np.ndarray],
    T_base_gripper: np.ndarray,
    T_gripper_camera_initial: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    K_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    predictions = predict_key_points(
        correction,
        key_labels,
        pixels,
        T_base_gripper,
        T_gripper_camera_initial,
        plane_normal,
        plane_d,
        K_matrix,
        dist_coeffs,
    )
    errors = []
    for label in key_labels:
        prediction = predictions[label]
        if prediction is None:
            errors.append(np.array([1.0, 1.0, 1.0], dtype=np.float64))
        else:
            errors.append(np.asarray(prediction - world_positions[label], dtype=np.float64))
    return np.vstack(errors)


def residuals(
    correction: np.ndarray,
    key_labels: list[str],
    world_positions: dict[str, np.ndarray],
    pixels: dict[str, np.ndarray],
    T_base_gripper: np.ndarray,
    T_gripper_camera_initial: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    K_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    lambda_rot: float,
    lambda_trans: float,
) -> np.ndarray:
    key_residuals = compute_position_errors(
        correction,
        key_labels,
        world_positions,
        pixels,
        T_base_gripper,
        T_gripper_camera_initial,
        plane_normal,
        plane_d,
        K_matrix,
        dist_coeffs,
    ).reshape(-1)
    regularization = np.concatenate(
        [
            np.sqrt(lambda_rot) * correction[:3],
            np.sqrt(lambda_trans) * correction[3:],
        ]
    )
    return np.concatenate([key_residuals, regularization])


def summarize_errors(name: str, errors_xyz: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(errors_xyz, axis=1)
    summary = {
        "mean_m": float(np.mean(norms)),
        "median_m": float(np.median(norms)),
        "max_m": float(np.max(norms)),
    }
    print(
        f"{name}: mean={summary['mean_m'] * 1000.0:.2f} mm, "
        f"median={summary['median_m'] * 1000.0:.2f} mm, "
        f"max={summary['max_m'] * 1000.0:.2f} mm"
    )
    return summary


def save_refined_transform(
    path: Path,
    T_gripper_camera_initial: np.ndarray,
    T_gripper_camera_refined: np.ndarray,
    correction: np.ndarray,
    matching_keys: list[str],
    plane_normal: np.ndarray,
    plane_d: float,
    before_summary: dict[str, float],
    after_summary: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True) if path.parent != Path(".") else None
    output = {
        "frame_convention": {
            "T_gripper_camera": "maps camera-frame coordinates into gripper-frame coordinates",
            "T_base_gripper": "maps gripper-frame coordinates into robot base/world coordinates",
            "T_base_camera": "T_base_gripper @ T_gripper_camera",
        },
        "T_gripper_camera_initial": T_gripper_camera_initial.tolist(),
        "T_gripper_camera": T_gripper_camera_refined.tolist(),
        "correction_right_multiply": {
            "rotation_vector_rad": correction[:3].tolist(),
            "translation_m": correction[3:].tolist(),
        },
        "matching_keys": matching_keys,
        "keyboard_plane": {
            "normal_base": plane_normal.tolist(),
            "d": float(plane_d),
            "equation": "normal_base dot x_base + d = 0",
        },
        "errors_before": before_summary,
        "errors_after": after_summary,
        "note": (
            "This is a task-specific refinement from one fixed keyboard image. "
            "It can overfit the keyboard plane, annotations, and camera pose used here; "
            "do not treat it as a general hand-eye recalibration without validation."
        ),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def load_debug_image(image_path: Path, pixels: dict[str, np.ndarray]) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is not None:
        return image
    max_uv = np.max(np.vstack(list(pixels.values())), axis=0)
    width = int(max(640, max_uv[0] + 80))
    height = int(max(480, max_uv[1] + 80))
    print(f"{image_path} could not be read; creating a blank debug canvas {width}x{height}.")
    return np.full((height, width, 3), 255, dtype=np.uint8)


def save_debug_visualization(
    image_path: Path,
    output_path: Path,
    key_labels: list[str],
    pixels: dict[str, np.ndarray],
    errors_after_xyz: np.ndarray,
) -> None:
    image = load_debug_image(image_path, pixels)
    error_mm = np.linalg.norm(errors_after_xyz, axis=1) * 1000.0
    for label, err_mm in zip(key_labels, error_mm):
        uv = np.round(pixels[label]).astype(int)
        cv2.circle(image, tuple(uv), 5, (0, 0, 255), -1)
        cv2.putText(
            image,
            f"{label} {err_mm:.1f}mm",
            tuple(uv + np.array([8, -8])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True) if output_path.parent != Path(".") else None
    cv2.imwrite(str(output_path), image)


def main() -> None:
    args = parse_args()
    K_matrix, dist_coeffs = load_camera_intrinsics(args.camera_calib)

    world_positions = load_key_world_positions(args.key_world_positions)
    pixel_annotations = load_key_pixel_annotations(args.key_pixel_annotations)
    matching_keys = sorted(set(world_positions).intersection(pixel_annotations))
    if len(matching_keys) < 6:
        warnings.warn(
            f"Only {len(matching_keys)} matching keys found; optimization may be underconstrained.",
            RuntimeWarning,
            stacklevel=2,
        )
    if len(matching_keys) < 3:
        raise ValueError("Need at least 3 matching keys for plane estimation and refinement.")

    plane_points = np.vstack([world_positions[key] for key in world_positions])
    plane_normal, plane_d, plane_centroid = estimate_plane_from_points(plane_points)
    print(f"Matching keys ({len(matching_keys)}): {', '.join(matching_keys)}")
    print(f"Keyboard plane normal in base frame: {plane_normal}")
    print(f"Keyboard plane d: {plane_d:.6f}, centroid: {plane_centroid}")

    T_gripper_camera_initial = load_transform(args.handeye_initial, preferred_key="T_gripper_camera")
    T_base_gripper = load_or_compute_T_base_gripper(args.image_gripper_pose, args.urdf_path)

    zero_correction = np.zeros(6, dtype=np.float64)
    before_errors = compute_position_errors(
        zero_correction,
        matching_keys,
        world_positions,
        pixel_annotations,
        T_base_gripper,
        T_gripper_camera_initial,
        plane_normal,
        plane_d,
        K_matrix,
        dist_coeffs,
    )
    before_summary = summarize_errors("Before refinement", before_errors)

    max_rot = np.deg2rad(args.max_rot_deg)
    lower_bounds = np.array([-max_rot, -max_rot, -max_rot, -args.max_trans_m, -args.max_trans_m, -args.max_trans_m])
    upper_bounds = np.array([max_rot, max_rot, max_rot, args.max_trans_m, args.max_trans_m, args.max_trans_m])
    result = least_squares(
        residuals,
        zero_correction,
        bounds=(lower_bounds, upper_bounds),
        args=(
            matching_keys,
            world_positions,
            pixel_annotations,
            T_base_gripper,
            T_gripper_camera_initial,
            plane_normal,
            plane_d,
            K_matrix,
            dist_coeffs,
            args.lambda_rot,
            args.lambda_trans,
        ),
        loss=args.loss,
        f_scale=args.f_scale,
        x_scale="jac",
        max_nfev=300,
        verbose=1,
    )

    correction = result.x
    after_errors = compute_position_errors(
        correction,
        matching_keys,
        world_positions,
        pixel_annotations,
        T_base_gripper,
        T_gripper_camera_initial,
        plane_normal,
        plane_d,
        K_matrix,
        dist_coeffs,
    )
    after_summary = summarize_errors("After refinement", after_errors)

    rotation_correction_deg = np.degrees(correction[:3])
    translation_correction_mm = correction[3:] * 1000.0
    print("Final correction:")
    print(f"  rotation vector [deg]: {rotation_correction_deg}")
    print(f"  rotation magnitude [deg]: {np.linalg.norm(rotation_correction_deg):.4f}")
    print(f"  translation [mm]: {translation_correction_mm}")
    print(f"  translation magnitude [mm]: {np.linalg.norm(translation_correction_mm):.4f}")

    T_gripper_camera_refined = build_refined_T_gripper_camera(T_gripper_camera_initial, correction)
    save_refined_transform(
        args.output,
        T_gripper_camera_initial,
        T_gripper_camera_refined,
        correction,
        matching_keys,
        plane_normal,
        plane_d,
        before_summary,
        after_summary,
    )
    save_debug_visualization(args.image, args.debug_image, matching_keys, pixel_annotations, after_errors)
    print(f"Saved refined transform: {args.output}")
    print(f"Saved debug image: {args.debug_image}")


if __name__ == "__main__":
    main()
