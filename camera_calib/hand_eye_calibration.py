from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


CALIBRATION_DIR = Path(__file__).resolve().parent

IMAGE_FOLDER = CALIBRATION_DIR / "handeye_samples_poses/images"
IMAGE_GLOB_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
SAMPLES_JSON_PATH = CALIBRATION_DIR / "handeye_samples_poses_2404/samples.json"
IMAGE_SUFFIXES = tuple(pattern.replace("*", "") for pattern in IMAGE_GLOB_PATTERNS)

# Checkerboard configuration.
# These are the number of INNER corners, not the number of squares.
CHECKERBOARD_ROWS = 6
CHECKERBOARD_COLS = 8
SQUARE_SIZE_METERS = 0.014
CAMERA_CALIB_FILE = CALIBRATION_DIR / "camera_calibration.npz"

# Camera intrinsics (replace with your real calibration).

camera_intrinsics = np.load(CAMERA_CALIB_FILE)
K = camera_intrinsics["camera_matrix"]
dist = camera_intrinsics["dist_coeffs"]

# Hand-eye method. OpenCV returns ^gT_c, the transform from camera frame to gripper frame.
HAND_EYE_METHOD = cv2.CALIB_HAND_EYE_TSAI

# Optional PnP-quality gate. Set to None to disable this pre-filter.
MAX_PNP_REPROJECTION_ERROR_PX: float | None = 3.0

# OpenCV's calibrateHandEye does not provide a RANSAC/robust flag, so this script
# wraps it in an iterative consistency filter before the final solve.
ENABLE_HAND_EYE_OUTLIER_REJECTION = True
HAND_EYE_OUTLIER_MAX_ITERATIONS = 5
HAND_EYE_OUTLIER_SIGMA_THRESHOLD = 3.5
HAND_EYE_OUTLIER_MIN_SAMPLES = 3
HAND_EYE_OUTLIER_TRANSLATION_FLOOR_M = 0.02
HAND_EYE_OUTLIER_ROTATION_FLOOR_DEG = 5.0
HAND_EYE_OUTLIER_TRANSLATION_CEILING_M: float = 0.02
HAND_EYE_OUTLIER_ROTATION_CEILING_DEG: float = 4

def build_homogeneous_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def rotation_distance_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_delta = np.asarray(R_a, dtype=np.float64).reshape(3, 3).T @ np.asarray(
        R_b, dtype=np.float64
    ).reshape(3, 3)
    cos_angle = (np.trace(R_delta) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def robust_upper_threshold(
    values: np.ndarray,
    sigma_threshold: float,
    floor: float,
) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    return max(floor, median + sigma_threshold * robust_sigma)


def rotation_medoid(rotations: list[np.ndarray]) -> np.ndarray:
    if len(rotations) == 1:
        return rotations[0]

    distance_sums = []
    for candidate in rotations:
        distance_sums.append(
            sum(rotation_distance_deg(candidate, other) for other in rotations)
        )
    return rotations[int(np.argmin(distance_sums))]


def rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    return R


def build_checkerboard_object_points(
    rows: int,
    cols: int,
    square_size_m: float,
) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * square_size_m
    return objp


def collect_image_paths(folder: Path, patterns: Iterable[str]) -> list[Path]:
    image_paths: list[Path] = []
    for pattern in patterns:
        image_paths.extend(folder.glob(pattern))
    return sorted(set(path.resolve() for path in image_paths))


def load_samples_json(samples_json_path: Path) -> list[dict]:
    with samples_json_path.open("r", encoding="utf-8") as fh:
        samples = json.load(fh)

    if not isinstance(samples, list):
        raise ValueError(f"Expected {samples_json_path} to contain a JSON list of samples.")

    return samples


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

    raise ValueError("Pose dictionary does not contain a supported 4x4 or (rotation, translation) pose.")


def sample_to_gripper_pose(sample: dict) -> np.ndarray:
    for key in ("gripper_pose", "robot_pose", "pose", "T_base_gripper", "base_T_gripper"):
        value = sample.get(key)
        if isinstance(value, dict):
            return pose_dict_to_homogeneous_transform(value)
        if value is not None:
            matrix = np.asarray(value, dtype=np.float64)
            if matrix.shape == (4, 4):
                return matrix

    joint_state = sample.get("joint_state") or sample.get("observation")
    if isinstance(joint_state, dict):
        joint_names = ", ".join(sorted(joint_state.keys()))
        raise ValueError(
            "This samples.json provides joint angles, not gripper poses. "
            "To run hand-eye calibration you still need the gripper pose ^bT_g for each sample, "
            "either stored directly in the JSON or computed from these joints with your robot's forward kinematics. "
            f"Joint fields found: {joint_names}"
        )

    raise ValueError("Could not find a gripper pose in the sample.")


def parse_robot_pose(pose: np.ndarray | tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(pose, tuple) and len(pose) == 2:
        R, t = pose
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        t = np.asarray(t, dtype=np.float64).reshape(3, 1)
        return R, t

    T = np.asarray(pose, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(
            "Each robot pose must be either a 4x4 homogeneous matrix or a tuple (R, t)."
        )

    R = T[:3, :3]
    t = T[:3, 3].reshape(3, 1)
    return R, t


def compute_mean_reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    projected_points = projected_points.reshape(-1, 2)
    image_points = image_points.reshape(-1, 2)
    errors = np.linalg.norm(projected_points - image_points, axis=1)
    return float(np.mean(errors))


def run_hand_eye_calibration(
    R_gripper2base: list[np.ndarray],
    t_gripper2base: list[np.ndarray],
    R_target2cam: list[np.ndarray],
    t_target2cam: list[np.ndarray],
    indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    selected_R_gripper2base = [R_gripper2base[i] for i in indices]
    selected_t_gripper2base = [t_gripper2base[i] for i in indices]
    selected_R_target2cam = [R_target2cam[i] for i in indices]
    selected_t_target2cam = [t_target2cam[i] for i in indices]

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base=selected_R_gripper2base,
        t_gripper2base=selected_t_gripper2base,
        R_target2cam=selected_R_target2cam,
        t_target2cam=selected_t_target2cam,
        method=HAND_EYE_METHOD,
    )
    return R_cam2gripper, np.asarray(t_cam2gripper, dtype=np.float64).reshape(3, 1)


def compute_base_target_residuals(
    R_gripper2base: list[np.ndarray],
    t_gripper2base: list[np.ndarray],
    R_target2cam: list[np.ndarray],
    t_target2cam: list[np.ndarray],
    R_cam2gripper: np.ndarray,
    t_cam2gripper: np.ndarray,
    indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    T_cam2gripper = build_homogeneous_transform(R_cam2gripper, t_cam2gripper)
    T_base_target_poses = []

    for index in indices:
        T_gripper2base = build_homogeneous_transform(
            R_gripper2base[index],
            t_gripper2base[index],
        )
        T_target2cam = build_homogeneous_transform(
            R_target2cam[index],
            t_target2cam[index],
        )
        T_base_target_poses.append(T_gripper2base @ T_cam2gripper @ T_target2cam)

    reference_translation = np.median(
        np.array([T[:3, 3] for T in T_base_target_poses]),
        axis=0,
    )
    reference_rotation = rotation_medoid([T[:3, :3] for T in T_base_target_poses])

    translation_errors = []
    rotation_errors = []
    for T_base_target in T_base_target_poses:
        translation_errors.append(
            np.linalg.norm(T_base_target[:3, 3] - reference_translation)
        )
        rotation_errors.append(
            rotation_distance_deg(reference_rotation, T_base_target[:3, :3])
        )

    return np.asarray(translation_errors), np.asarray(rotation_errors)


def calibrate_hand_eye_with_outlier_rejection(
    R_gripper2base: list[np.ndarray],
    t_gripper2base: list[np.ndarray],
    R_target2cam: list[np.ndarray],
    t_target2cam: list[np.ndarray],
    sample_labels: list[str],
) -> tuple[np.ndarray, np.ndarray, list[int], list[dict]]:
    kept_indices = list(range(len(R_gripper2base)))
    rejected_samples: list[dict] = []

    if not ENABLE_HAND_EYE_OUTLIER_REJECTION:
        R_cam2gripper, t_cam2gripper = run_hand_eye_calibration(
            R_gripper2base,
            t_gripper2base,
            R_target2cam,
            t_target2cam,
            kept_indices,
        )
        return R_cam2gripper, t_cam2gripper, kept_indices, rejected_samples

    print()
    print("Hand-eye outlier rejection enabled")

    for iteration in range(1, HAND_EYE_OUTLIER_MAX_ITERATIONS + 1):
        if len(kept_indices) <= HAND_EYE_OUTLIER_MIN_SAMPLES:
            print(
                "  Stopping rejection: already at the minimum number of samples "
                f"({HAND_EYE_OUTLIER_MIN_SAMPLES})"
            )
            break

        R_cam2gripper, t_cam2gripper = run_hand_eye_calibration(
            R_gripper2base,
            t_gripper2base,
            R_target2cam,
            t_target2cam,
            kept_indices,
        )
        translation_errors, rotation_errors = compute_base_target_residuals(
            R_gripper2base,
            t_gripper2base,
            R_target2cam,
            t_target2cam,
            R_cam2gripper,
            t_cam2gripper,
            kept_indices,
        )

        translation_threshold = robust_upper_threshold(
            translation_errors,
            HAND_EYE_OUTLIER_SIGMA_THRESHOLD,
            HAND_EYE_OUTLIER_TRANSLATION_FLOOR_M,
        )
        rotation_threshold = robust_upper_threshold(
            rotation_errors,
            HAND_EYE_OUTLIER_SIGMA_THRESHOLD,
            HAND_EYE_OUTLIER_ROTATION_FLOOR_DEG,
        )
        if HAND_EYE_OUTLIER_TRANSLATION_CEILING_M is not None:
            translation_threshold = min(
                translation_threshold,
                HAND_EYE_OUTLIER_TRANSLATION_CEILING_M,
            )
        if HAND_EYE_OUTLIER_ROTATION_CEILING_DEG is not None:
            rotation_threshold = min(
                rotation_threshold,
                HAND_EYE_OUTLIER_ROTATION_CEILING_DEG,
            )

        outlier_positions = np.flatnonzero(
            (translation_errors > translation_threshold)
            | (rotation_errors > rotation_threshold)
        )

        print(
            f"  Iteration {iteration}: median residual "
            f"{np.median(translation_errors):.4f} m, "
            f"{np.median(rotation_errors):.2f} deg; max residual "
            f"{np.max(translation_errors):.4f} m, "
            f"{np.max(rotation_errors):.2f} deg; thresholds "
            f"{translation_threshold:.4f} m, {rotation_threshold:.2f} deg"
        )

        if len(outlier_positions) == 0:
            print("  No more hand-eye outliers found")
            break

        max_rejectable = len(kept_indices) - HAND_EYE_OUTLIER_MIN_SAMPLES
        if max_rejectable <= 0:
            break

        outlier_scores = np.maximum(
            translation_errors[outlier_positions] / translation_threshold,
            rotation_errors[outlier_positions] / rotation_threshold,
        )
        order = np.argsort(outlier_scores)[::-1]
        outlier_positions = outlier_positions[order[:max_rejectable]]
        positions_to_reject = set(int(position) for position in outlier_positions)

        next_kept_indices = []
        for position, original_index in enumerate(kept_indices):
            if position in positions_to_reject:
                rejected_samples.append(
                    {
                        "index": original_index,
                        "label": sample_labels[original_index],
                        "translation_error_m": float(translation_errors[position]),
                        "rotation_error_deg": float(rotation_errors[position]),
                    }
                )
            else:
                next_kept_indices.append(original_index)

        print(f"  Rejected {len(positions_to_reject)} sample(s)")
        kept_indices = next_kept_indices

    R_cam2gripper, t_cam2gripper = run_hand_eye_calibration(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        kept_indices,
    )
    return R_cam2gripper, t_cam2gripper, kept_indices, rejected_samples


def main() -> None:
    print("Hand-eye calibration started")
    print(f"Samples JSON: {SAMPLES_JSON_PATH.resolve()}")
    print(
        f"Checkerboard inner corners: rows={CHECKERBOARD_ROWS}, cols={CHECKERBOARD_COLS}, "
        f"square_size={SQUARE_SIZE_METERS} m"
    )
    samples = load_samples_json(SAMPLES_JSON_PATH)
    if not samples:
        raise ValueError(f"No samples found in {SAMPLES_JSON_PATH.resolve()}")

    print(f"Found {len(samples)} sample(s) in JSON")

    pattern_size = (CHECKERBOARD_COLS, CHECKERBOARD_ROWS)
    object_points = build_checkerboard_object_points(
        rows=CHECKERBOARD_ROWS,
        cols=CHECKERBOARD_COLS,
        square_size_m=SQUARE_SIZE_METERS,
    )

    termination = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1e-3,
    )

    R_gripper2base: list[np.ndarray] = []
    t_gripper2base: list[np.ndarray] = []
    R_target2cam: list[np.ndarray] = []
    t_target2cam: list[np.ndarray] = []
    sample_labels: list[str] = []

    for index, sample in enumerate(samples, start=1):
        image_path_value = sample.get("image_path")
        if not image_path_value:
            print()
            print(f"[{index}/{len(samples)}] Skipping sample without image_path")
            continue

        image_path = Path(image_path_value)
        if not image_path.is_absolute():
            image_path = (SAMPLES_JSON_PATH.resolve().parent / image_path).resolve()

        print()
        print(f"[{index}/{len(samples)}] Processing {image_path.name}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print("  Skipping: image could not be read")
            continue

        try:
            robot_pose = sample_to_gripper_pose(sample)
        except ValueError as exc:
            print(f"  Skipping: {exc}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if not found:
            print("  Skipping: checkerboard detection failed")
            continue

        refined_corners = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=termination,
        )

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            refined_corners,
            K,
            dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            print("  Skipping: solvePnP failed")
            continue

        # solvePnP returns the pose of the target in the camera frame:
        # X_cam = R_target2cam * X_target + t_target2cam
        R_tc = rodrigues_to_matrix(rvec)
        t_tc = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

        print("translation")
        print(t_tc)
        # For eye-in-hand calibration, OpenCV calibrateHandEye expects:
        # - R_gripper2base, t_gripper2base: pose ^bT_g
        # - R_target2cam, t_target2cam: pose ^cT_t
        R_bg, t_bg = parse_robot_pose(robot_pose)

        reproj_error_px = compute_mean_reprojection_error(
            object_points=object_points,
            image_points=refined_corners,
            rvec=rvec,
            tvec=tvec,
            K=K,
            dist=dist,
        )

        if (
            MAX_PNP_REPROJECTION_ERROR_PX is not None
            and reproj_error_px > MAX_PNP_REPROJECTION_ERROR_PX
        ):
            print(
                "  Skipping: mean reprojection error "
                f"{reproj_error_px:.4f} px exceeds "
                f"{MAX_PNP_REPROJECTION_ERROR_PX:.4f} px"
            )
            continue
        
        R_gripper2base.append(R_bg)
        t_gripper2base.append(t_bg)
        R_target2cam.append(R_tc)
        t_target2cam.append(t_tc)
        sample_labels.append(image_path.name)

        print("  Checkerboard detected")
        print(f"  Mean reprojection error: {reproj_error_px:.4f} px")
        print(f"  Stored robot pose ^bT_g and target pose ^cT_t")

    print()
    print(f"Valid checkerboard detections: {len(R_target2cam)}")
    print(f"Valid robot pose pairs used:   {len(R_gripper2base)}")

    if len(R_target2cam) != len(R_gripper2base):
        raise RuntimeError(
            "Mismatch between valid checkerboard detections and valid robot poses."
        )

    if len(R_target2cam) < 3:
        raise RuntimeError(
            "Need at least 3 valid pose pairs for hand-eye calibration. "
            "In practice, use many more with diverse wrist motions."
        )

    R_cam2gripper, t_cam2gripper, kept_indices, rejected_samples = (
        calibrate_hand_eye_with_outlier_rejection(
            R_gripper2base=R_gripper2base,
            t_gripper2base=t_gripper2base,
            R_target2cam=R_target2cam,
            t_target2cam=t_target2cam,
            sample_labels=sample_labels,
        )
    )

    T_cam2gripper = build_homogeneous_transform(R_cam2gripper, t_cam2gripper)

    np.save("calib/rigid_transform", T_cam2gripper)

    print()
    print("=== Hand-Eye Calibration Result ===")
    print(f"Samples used in final solve: {len(kept_indices)}/{len(R_gripper2base)}")
    if rejected_samples:
        print("Rejected hand-eye outliers:")
        for rejected_sample in rejected_samples:
            print(
                "  "
                f"{rejected_sample['label']}: "
                f"{rejected_sample['translation_error_m']:.4f} m, "
                f"{rejected_sample['rotation_error_deg']:.2f} deg"
            )
    print()
    print("OpenCV returned ^gT_c, the transform from camera frame to gripper frame.")
    print()
    print("Rotation matrix R_cam2gripper:")
    print(R_cam2gripper)
    print()
    print("Translation vector t_cam2gripper [m]:")
    print(t_cam2gripper)
    print()
    print("Homogeneous transform T_cam2gripper (^gT_c):")
    print(T_cam2gripper)
    print()
    print(
        "Usage: if p_c is a point in homogeneous camera coordinates [x, y, z, 1]^T, "
        "then p_g = T_cam2gripper @ p_c gives the same point expressed in the gripper frame."
    )


if __name__ == "__main__":
    main()
