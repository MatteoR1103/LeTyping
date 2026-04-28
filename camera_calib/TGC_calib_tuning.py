import numpy as np
import cv2 as cv
from pathlib import Path
import sys
import argparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera_calib.hand_eye_calibration import (
    build_checkerboard_object_points,
    load_samples_json,
    parse_robot_pose,
    sample_to_gripper_pose,
)

try:
    from src.track_to_wld import convert_to_ray, find_intersection
except ImportError:
    from track_to_wld import convert_to_ray, find_intersection

# IMAGE FOLDER PATH
IMAGE_GLOB_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
SAMPLES_JSON_PATH = PROJECT_ROOT / "camera_calib/data/calib_poses_data/handeye_samples_poses_2604_2/samples.json"
ACTUAL_POSES_JSON_PATH = PROJECT_ROOT / "camera_calib/data/calib_poses_data/handeye_samples_poses_2704_z/samples.json"
IMAGE_SUFFIXES = tuple(pattern.replace("*", "") for pattern in IMAGE_GLOB_PATTERNS)

# Checkerboard configuration.
# These are the number of INNER corners, not the number of squares.
CHECKERBOARD_ROWS = 6
CHECKERBOARD_COLS = 8
SQUARE_SIZE_METERS = 0.014

#CALIBRATION PATHS 
RIGID_T_PATH = PROJECT_ROOT / "camera_calib/calibrations/rigid_transform.npy"
CAMERA_CALIB_PATH = PROJECT_ROOT / "camera_calib/calibrations/camera_calibration.npz"

#SAVING PATHS
STATS_SAVE_PATH = PROJECT_ROOT / "camera_calib/stats/pixel_to_ray_stats.txt"
ROW_COL_STATS_SAVE_PATH = PROJECT_ROOT / "camera_calib/stats/pixel_to_ray_row_col_stats.txt"



#LOAD INTRINSICS AND GRIPPER-TO-CAM INTRINSICS
camera_intrinsics = np.load(CAMERA_CALIB_PATH)
K = camera_intrinsics["camera_matrix"]
K_INV = np.linalg.inv(K)
dist = camera_intrinsics["dist_coeffs"]

tilting_angle = 40
tilting_angle = np.deg2rad(tilting_angle)

c_theta = np.cos(tilting_angle)
s_theta = np.sin(tilting_angle)

R_GC = np.array([[-1.0 , 0,       0],
                 [0, -c_theta, -s_theta],
                 [0, -s_theta, c_theta]] ,
                dtype=np.float64)

t_GC = np.array([-0.007, 0.052, -0.04])

VALIDATION_GRID_STEPS = 17
VALIDATION_PASSES = 3
SPACING_ERROR_WEIGHT = 1.0

#LOAD HEURISTIC PLANE INFO 
PLANE_N = np.array([0,0,1.0])
PLANE_P0 = np.array([0.0, 0.0, -0.033459])
WINDOW_NAME = "Ray intersection"


def format_point(point: np.ndarray | None, unit: str = "m") -> str:
    if point is None:
        return "None"
    return f"[{point[0]: .4f}, {point[1]: .4f}, {point[2]: .4f}] {unit}"


def draw_text_lines(
    image: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (12, 24),
) -> None:
    x, y = origin
    for line in lines:
        cv.putText(
            image,
            line,
            (x, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            3,
            cv.LINE_AA,
        )
        cv.putText(
            image,
            line,
            (x, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv.LINE_AA,
        )
        y += 20


def calculate_corner_world_positions(
    corners: np.ndarray,
    T_WC: np.ndarray,
    plane_n: np.ndarray,
    plane_p0: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    corner_pixels = corners.reshape(-1, 2)
    corner_world_positions = np.full((len(corner_pixels), 3), np.nan, dtype=float)
    intersection_statuses = ["hit"] * len(corner_pixels)

    undistorted_pixels = cv.undistortPoints(
        corner_pixels.reshape(-1, 1, 2).astype(np.float64),
        K,
        dist,
    ).reshape(-1, 2)
    ray_c = np.column_stack(
        (undistorted_pixels, np.ones(len(undistorted_pixels), dtype=float))
    )
    ray_o = T_WC[:3, 3]
    ray_w = (T_WC[:3, :3] @ ray_c.T).T
    ray_w /= np.linalg.norm(ray_w, axis=1, keepdims=True)

    denom = ray_w @ plane_n
    num = np.dot(plane_n, plane_p0 - ray_o)

    for corner_index, ray_denom in enumerate(denom):
        if abs(ray_denom) < 1e-5:
            if abs(num) < 1e-5:
                intersection_statuses[corner_index] = "ray lies in plane"
            else:
                intersection_statuses[corner_index] = "parallel, no intersection"
            continue

        t = num / ray_denom
        if t < 0:
            intersection_statuses[corner_index] = "intersection behind ray origin"
            continue

        corner_world_positions[corner_index] = ray_o + t * ray_w[corner_index]

    return corner_world_positions, intersection_statuses


def make_gripper_to_camera_transform(translation: np.ndarray) -> np.ndarray:
    T_GC = np.eye(4)
    T_GC[:3, :3] = R_GC
    T_GC[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return T_GC


def collect_detected_samples(
    samples: list[dict],
    pattern_size: tuple[int, int],
    termination: tuple[int, int, float],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    detected_samples = []

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

        robot_pose = sample_to_gripper_pose(sample)
        R_WG, t_BG = parse_robot_pose(robot_pose)
        T_WG = np.eye(4)
        T_WG[:3, :3] = R_WG
        T_WG[:3, 3] = t_BG.reshape(3)

        detected_samples.append((image_path.name, refined_corners, T_WG))

    return detected_samples


def corner_samples_for_translation(
    detected_samples: list[tuple[str, np.ndarray, np.ndarray]],
    translation: np.ndarray,
    plane_n: np.ndarray,
    plane_p0: np.ndarray,
) -> list[np.ndarray]:
    T_GC_candidate = make_gripper_to_camera_transform(translation)
    corner_position_samples = []

    for _, refined_corners, T_WG in detected_samples:
        T_WC = T_WG @ T_GC_candidate
        corner_world_positions, _ = calculate_corner_world_positions(
            corners=refined_corners,
            T_WC=T_WC,
            plane_n=plane_n,
            plane_p0=plane_p0,
        )
        corner_position_samples.append(corner_world_positions)

    return corner_position_samples


def calculate_repeatability_error(corner_position_samples: list[np.ndarray]) -> float:
    if len(corner_position_samples) < 2:
        return np.inf

    samples = np.stack(corner_position_samples, axis=0)
    combined_stds = []

    for corner_index in range(samples.shape[1]):
        corner_samples = samples[:, corner_index, :]
        valid_samples = corner_samples[~np.isnan(corner_samples).any(axis=1)]
        if len(valid_samples) < 2:
            continue

        std_xy = np.std(valid_samples[:, :2], axis=0)
        combined_stds.append(np.linalg.norm(std_xy))

    if not combined_stds:
        return np.inf

    return float(np.mean(combined_stds))


def calculate_spacing_error(corner_position_samples: list[np.ndarray]) -> float:
    if not corner_position_samples:
        return np.inf

    samples = np.stack(corner_position_samples, axis=0)
    means = np.nanmean(samples, axis=0)
    if np.isnan(means).any():
        return np.inf

    grid_means = means.reshape(CHECKERBOARD_ROWS, CHECKERBOARD_COLS, 3)
    spacing_errors = []

    horizontal_diffs = np.diff(grid_means[:, :, :2], axis=1)
    vertical_diffs = np.diff(grid_means[:, :, :2], axis=0)

    horizontal_distances = np.linalg.norm(horizontal_diffs, axis=2)
    vertical_distances = np.linalg.norm(vertical_diffs, axis=2)

    spacing_errors.extend(np.abs(horizontal_distances.reshape(-1) - SQUARE_SIZE_METERS))
    spacing_errors.extend(np.abs(vertical_distances.reshape(-1) - SQUARE_SIZE_METERS))

    return float(np.mean(spacing_errors))


def evaluate_translation_candidate(
    detected_samples: list[tuple[str, np.ndarray, np.ndarray]],
    translation: np.ndarray,
    plane_n: np.ndarray,
    plane_p0: np.ndarray,
    actual_positions: np.ndarray | None = None,
    actual_error_weight: float = 1.0,
) -> tuple[float, float, float, float]:
    corner_position_samples = corner_samples_for_translation(
        detected_samples=detected_samples,
        translation=translation,
        plane_n=plane_n,
        plane_p0=plane_p0,
    )
    repeatability_error = calculate_repeatability_error(corner_position_samples)
    spacing_error = calculate_spacing_error(corner_position_samples)
    actual_error = (
        calculate_actual_pose_error(corner_position_samples, actual_positions)
        if actual_positions is not None
        else 0.0
    )
    score = (
        repeatability_error
        + SPACING_ERROR_WEIGHT * spacing_error
        + actual_error_weight * actual_error
    )
    return score, repeatability_error, spacing_error, actual_error


def search_camera_translation(
    detected_samples: list[tuple[str, np.ndarray, np.ndarray]],
    initial_translation: np.ndarray,
    search_radius: np.ndarray,
    plane_n: np.ndarray,
    plane_p0: np.ndarray,
    actual_positions: np.ndarray | None = None,
    actual_error_weight: float = 1.0,
) -> tuple[np.ndarray, float, float, float, float]:
    best_translation = initial_translation.copy()
    best_score = np.inf
    best_repeatability = np.inf
    best_spacing = np.inf
    best_actual_error = np.inf

    for pass_index in range(VALIDATION_PASSES):
        grid_x = np.linspace(
            best_translation[0] - search_radius[0],
            best_translation[0] + search_radius[0],
            VALIDATION_GRID_STEPS,
        )
        grid_y = np.linspace(
            best_translation[1] - search_radius[1],
            best_translation[1] + search_radius[1],
            VALIDATION_GRID_STEPS,
        )
        grid_z = np.linspace(
            best_translation[2] - search_radius[2],
            best_translation[2] + search_radius[2],
            VALIDATION_GRID_STEPS,
        )

        pass_best_translation = best_translation.copy()
        pass_best_score = np.inf
        pass_best_repeatability = np.inf
        pass_best_spacing = np.inf
        pass_best_actual_error = np.inf

        for x in grid_x:
            for y in grid_y:
                for z in grid_z:
                    translation = np.array([x, y, z], dtype=float)
                    score, repeatability, spacing, actual_error = evaluate_translation_candidate(
                        detected_samples=detected_samples,
                        translation=translation,
                        plane_n=plane_n,
                        plane_p0=plane_p0,
                        actual_positions=actual_positions,
                        actual_error_weight=actual_error_weight,
                    )
                    if score < pass_best_score:
                        pass_best_score = score
                        pass_best_repeatability = repeatability
                        pass_best_spacing = spacing
                        pass_best_actual_error = actual_error
                        pass_best_translation = translation

        best_translation = pass_best_translation
        best_score = pass_best_score
        best_repeatability = pass_best_repeatability
        best_spacing = pass_best_spacing
        best_actual_error = pass_best_actual_error
        print(
            f"validation pass {pass_index + 1}/{VALIDATION_PASSES}: "
            f"t_GC={best_translation}, "
            f"score={best_score * 1000:.3f} mm, "
            f"repeatability={best_repeatability * 1000:.3f} mm, "
            f"spacing={best_spacing * 1000:.3f} mm, "
            f"actual_error={best_actual_error * 1000:.3f} mm"
        )
        search_radius = search_radius / 3.0

    return best_translation, best_score, best_repeatability, best_spacing, best_actual_error


def print_corner_position_statistics(corner_position_samples: list[np.ndarray]) -> float:
    if not corner_position_samples:
        print()
        print("No corner world positions collected, so no statistics were calculated.")
        return np.inf

    samples = np.stack(corner_position_samples, axis=0)

    print()
    print("Corner world position statistics")
    print(f"Samples used: {samples.shape[0]}")
    
    means = []
    stds = []
    variances = []
    
    for corner_index in range(samples.shape[1]):
        corner_samples = samples[:, corner_index, :]
        valid_samples = corner_samples[~np.isnan(corner_samples).any(axis=1)]

        if len(valid_samples) == 0:
            print(f"corner {corner_index + 1:02d}: no valid world positions")
            continue

        mean = np.mean(valid_samples, axis=0)
        std = np.std(valid_samples, axis=0)
        variance = np.var(valid_samples, axis=0)

        means.append(mean)
        stds.append(std)
        variances.append(variance)

    if not means:
        print("No valid corner world positions were found.")
        return np.inf
    
    means = np.stack(means, axis=0)
    stds = np.stack(stds, axis=0)
    variances = np.stack(variances, axis=0)

    std_combined_xy = float(np.mean(np.linalg.norm(stds[:, :2], axis=1)))

    stats = np.concatenate((means, stds, variances),axis=1)
    fmt = ["%.10f"] * (stats.shape[1])
    np.savetxt(STATS_SAVE_PATH, stats, fmt)
    print_row_column_position_statistics(samples)
    return std_combined_xy


def calculate_mean_corner_world_positions(corner_position_samples: list[np.ndarray]) -> np.ndarray:
    if not corner_position_samples:
        return np.empty((0, 3), dtype=float)

    samples = np.stack(corner_position_samples, axis=0)
    means = np.full((samples.shape[1], 3), np.nan, dtype=float)

    for corner_index in range(samples.shape[1]):
        corner_samples = samples[:, corner_index, :]
        valid_samples = corner_samples[~np.isnan(corner_samples).any(axis=1)]
        if len(valid_samples) > 0:
            means[corner_index] = np.mean(valid_samples, axis=0)

    return means


def extract_actual_position(sample: dict) -> np.ndarray:
    if all(key in sample for key in ("ee.x", "ee.y", "ee.z")):
        return np.array([sample["ee.x"], sample["ee.y"], sample["ee.z"]], dtype=float)

    gripper_pose = sample.get("gripper_pose")
    if isinstance(gripper_pose, dict) and "position_m" in gripper_pose:
        return np.asarray(gripper_pose["position_m"], dtype=float).reshape(3)

    raise ValueError(f"Could not find an actual gripper position in sample {sample.get('sample_idx')}")


def load_actual_world_positions(actual_poses_json_path: Path) -> np.ndarray:
    actual_pose_samples = load_samples_json(actual_poses_json_path)
    expected_count = CHECKERBOARD_ROWS * CHECKERBOARD_COLS
    if len(actual_pose_samples) != expected_count:
        raise ValueError(
            f"Expected {expected_count} actual poses in {actual_poses_json_path.resolve()}, "
            f"but found {len(actual_pose_samples)}."
        )

    return np.stack([extract_actual_position(sample) for sample in actual_pose_samples], axis=0)


def calculate_actual_pose_error(
    corner_position_samples: list[np.ndarray],
    actual_positions: np.ndarray,
) -> float:
    predicted_positions = calculate_mean_corner_world_positions(corner_position_samples)

    if predicted_positions.shape != actual_positions.shape:
        return np.inf

    valid_mask = ~np.isnan(predicted_positions).any(axis=1)
    if not np.any(valid_mask):
        return np.inf

    position_errors = np.linalg.norm(
        predicted_positions[valid_mask] - actual_positions[valid_mask],
        axis=1,
    )
    return float(np.mean(position_errors))


def print_actual_pose_error(
    corner_position_samples: list[np.ndarray],
    actual_poses_json_path: Path,
) -> float:
    actual_positions = load_actual_world_positions(actual_poses_json_path)
    predicted_positions = calculate_mean_corner_world_positions(corner_position_samples)

    if predicted_positions.shape != actual_positions.shape:
        raise ValueError(
            f"Predicted positions shape {predicted_positions.shape} does not match "
            f"actual positions shape {actual_positions.shape}."
        )

    valid_mask = ~np.isnan(predicted_positions).any(axis=1)
    if not np.any(valid_mask):
        print()
        print("No valid predicted positions were available for actual-pose error calculation.")
        return np.inf

    average_error = calculate_actual_pose_error(corner_position_samples, actual_positions)

    print()
    print("Actual pose prediction error")
    print(f"Actual poses: {actual_poses_json_path.resolve()}")
    print(f"Valid points compared: {int(np.sum(valid_mask))}/{len(actual_positions)}")
    print(f"Average world-coordinate error: {average_error:.6f} m ({average_error * 1000:.3f} mm)")
    return average_error


def valid_flattened_values(values: np.ndarray) -> np.ndarray:
    return values.reshape(-1)[~np.isnan(values.reshape(-1))]


def print_row_column_position_statistics(samples: np.ndarray) -> None:
    
    grid_samples = samples.reshape(
        samples.shape[0],
        CHECKERBOARD_ROWS,
        CHECKERBOARD_COLS,
        3,
    )
    output_lines = [
        "# Row stats use world X because X should stay almost constant within one row.",
        "# Format: row_index mean_x_m std_x_m var_x_m n_values",
    ]

    print()
    print("Row consistency statistics")
    print("Rows: world X should stay almost constant within each row.")
    for row_index in range(CHECKERBOARD_ROWS):
        row_x_values = valid_flattened_values(grid_samples[:, row_index, :, 0])
        if len(row_x_values) == 0:
            print(f"row {row_index + 1:02d}: no valid X values")
            continue

        mean = np.mean(row_x_values)
        std = np.std(row_x_values)
        variance = np.var(row_x_values)
        output_lines.append(
            f"row {row_index + 1:02d} {mean:.10f} {std:.10f} {variance:.10f} {len(row_x_values)}"
        )

    output_lines.extend(
        [
            "",
            "# Column stats use world Y because Y should stay almost constant within one column.",
            "# Format: col_index mean_y_m std_y_m var_y_m n_values",
        ]
    )

    print()
    print("Column consistency statistics")
    print("Columns: world Y should stay almost constant within each column.")
    for col_index in range(CHECKERBOARD_COLS):
        col_y_values = valid_flattened_values(grid_samples[:, :, col_index, 1])
        if len(col_y_values) == 0:
            print(f"col {col_index + 1:02d}: no valid Y values")
            continue

        mean = np.mean(col_y_values)
        std = np.std(col_y_values)
        variance = np.var(col_y_values)
        output_lines.append(
            f"col {col_index + 1:02d} {mean:.10f} {std:.10f} {variance:.10f} {len(col_y_values)}"
        )

    ROW_COL_STATS_SAVE_PATH.write_text("\n".join(output_lines) + "\n")
    print(f"Saved row/column statistics to {ROW_COL_STATS_SAVE_PATH.resolve()}")

def show_corner_intersections(
    image: np.ndarray,
    corners: np.ndarray,
    corner_world_positions: np.ndarray,
    intersection_statuses: list[str],
    sample_label: str,
) -> bool:
    corner_pixels = corners.reshape(-1, 2)

    for corner_index, pixel in enumerate(corner_pixels, start=1):
        x_threed = corner_world_positions[corner_index - 1]
        if np.isnan(x_threed).any():
            x_threed = None
        status = intersection_statuses[corner_index - 1]

        display = image.copy()
        for other_pixel in corner_pixels:
            cv.circle(display, tuple(other_pixel.astype(int)), 2, (120, 120, 120), -1)
        cv.circle(display, tuple(pixel.astype(int)), 6, (0, 0, 255), -1)
        cv.circle(display, tuple(pixel.astype(int)), 9, (255, 255, 255), 2)

        lines = [
            f"{sample_label}  corner {corner_index}/{len(corner_pixels)}",
            f"pixel: [{pixel[0]:.2f}, {pixel[1]:.2f}]",
            f"world: {format_point(x_threed)}",
            f"status: {status}",
            "any key: next corner   q/esc: quit",
        ]
        draw_text_lines(display, lines)

        print(
            f"{sample_label} corner {corner_index:02d}: "
            f"pixel=[{pixel[0]:.2f}, {pixel[1]:.2f}], "
            f"world={format_point(x_threed)}, status={status}"
        )

        cv.imshow(WINDOW_NAME, display)
        key = cv.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            return False

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project detected checkerboard corners to world coordinates."
    )
    parser.add_argument(
        "--validate-translation",
        "--tune-translation",
        action="store_true",
        help="Run the grid-search validation loop to tune t_GC before reporting results.",
    )
    parser.add_argument(
        "--t-gc",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Override the configured gripper-to-camera translation in meters.",
    )
    parser.add_argument(
        "--actual-poses-json",
        type=Path,
        default=ACTUAL_POSES_JSON_PATH,
        help=f"JSON file with 48 actual gripper poses in row-major checkerboard order. Default: {ACTUAL_POSES_JSON_PATH}",
    )
    parser.add_argument(
        "--actual-error-weight",
        type=float,
        default=1.0,
        help="Weight for the actual-pose error term in the validation score. Default: 1.0",
    )
    return parser.parse_args()


def main()->None: 
    args = parse_args()
    selected_t_GC = np.array(args.t_gc, dtype=float) if args.t_gc is not None else t_GC

    #FIND CHESS CORNERS IN A LOADED IMAGE
    print("CORNER LOCALIZATION SCRIPT STARTED")
    print(f"IMAGE PATH: {SAMPLES_JSON_PATH.resolve()}")
    print(f"Using configured t_GC: {selected_t_GC}")
    print(
        f"Checkerboard inner corners: rows={CHECKERBOARD_ROWS}, cols={CHECKERBOARD_COLS}, "
        f"square_size={SQUARE_SIZE_METERS} m"
    )
    samples = load_samples_json(SAMPLES_JSON_PATH)
    if not samples:
        raise ValueError(f"No samples found in {SAMPLES_JSON_PATH.resolve()}")

    print(f"Found {len(samples)} sample(s) in JSON")

    pattern_size = (CHECKERBOARD_COLS, CHECKERBOARD_ROWS)

    termination = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
        30,
        1e-3,
    )
    
    plane_n = PLANE_N
    plane_p0 = PLANE_P0
    
    #DO A VALIDATION PASS TO FIND THE EXTRINSICS OF THE CAMERA 
    detected_samples = collect_detected_samples(
        samples=samples,
        pattern_size=pattern_size,
        termination=termination,
    )
    if not detected_samples:
        raise ValueError("No valid checkerboard detections were found.")

    if args.validate_translation:
        if len(detected_samples) < 2:
            raise ValueError("Need at least two valid checkerboard detections for validation.")

        actual_positions = load_actual_world_positions(args.actual_poses_json)
        search_radius = np.array([0.01, 0.02, 0.02])
        selected_t_GC, best_score, best_repeatability, best_spacing, best_actual_error = search_camera_translation(
            detected_samples=detected_samples,
            initial_translation=selected_t_GC,
            search_radius=search_radius,
            plane_n=plane_n,
            plane_p0=plane_p0,
            actual_positions=actual_positions,
            actual_error_weight=args.actual_error_weight,
        )

        print()
        print("Best validation result")
        print(f"t_GC: {selected_t_GC}")
        print(f"score: {best_score * 1000:.3f} mm")
        print(f"repeatability error: {best_repeatability * 1000:.3f} mm")
        print(f"checkerboard spacing error: {best_spacing * 1000:.3f} mm")
        print(f"actual pose error: {best_actual_error * 1000:.3f} mm")
        print(f"actual pose error weight: {args.actual_error_weight}")
        T_GC = np.eye(4)
        T_GC[:3,:3] = R_GC
        T_GC[:3,3] = selected_t_GC
        print(f"Final transform : {T_GC}")
        np.save(RIGID_T_PATH, T_GC)
        print(f"Saved hand-tuned calibration to {RIGID_T_PATH}")
    else:
        print()
        print("Skipping translation validation search.")
        print("Use --validate-translation to run the grid-search tuning loop.")
        if len(detected_samples) >= 2:
            actual_positions = load_actual_world_positions(args.actual_poses_json)
            score, repeatability, spacing, actual_error = evaluate_translation_candidate(
                detected_samples=detected_samples,
                translation=selected_t_GC,
                plane_n=plane_n,
                plane_p0=plane_p0,
                actual_positions=actual_positions,
                actual_error_weight=args.actual_error_weight,
            )
            print(f"current t_GC score: {score * 1000:.3f} mm")
            print(f"current repeatability error: {repeatability * 1000:.3f} mm")
            print(f"current checkerboard spacing error: {spacing * 1000:.3f} mm")
            print(f"current actual pose error: {actual_error * 1000:.3f} mm")

    best_corner_position_samples = corner_samples_for_translation(
        detected_samples=detected_samples,
        translation=selected_t_GC,
        plane_n=plane_n,
        plane_p0=plane_p0,
    )
    print_corner_position_statistics(best_corner_position_samples)
    print_actual_pose_error(best_corner_position_samples, args.actual_poses_json)
    cv.destroyAllWindows()
        

if __name__ == "__main__":
  main()
