import numpy as np
import cv2 as cv
from pathlib import Path
import sys


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
    from .track_to_wld import convert_to_ray, find_intersection
except ImportError:
    from track_to_wld import convert_to_ray, find_intersection

# IMAGE FOLDER PATH
IMAGE_GLOB_PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
SAMPLES_JSON_PATH = PROJECT_ROOT / "camera_calib/handeye_samples_poses/samples.json"
IMAGE_SUFFIXES = tuple(pattern.replace("*", "") for pattern in IMAGE_GLOB_PATTERNS)

# Checkerboard configuration.
# These are the number of INNER corners, not the number of squares.
CHECKERBOARD_ROWS = 6
CHECKERBOARD_COLS = 8
SQUARE_SIZE_METERS = 0.014

#CALIBRATION PATHS 
RIGID_T_PATH = PROJECT_ROOT / "camera_calib/rigid_transform.npy"
CAMERA_CALIB_PATH = PROJECT_ROOT / "camera_calib/camera_calibration.npz"

#CAMERA PARAMS
CAMERA_NO = 4
KLT_PARAMS = dict(winSize  = (21, 21),
                  maxLevel = 2, 
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.001)
                  )

#LOAD INTRINSICS AND GRIPPER-TO-CAM INTRINSICS
camera_intrinsics = np.load(CAMERA_CALIB_PATH)
K = camera_intrinsics["camera_matrix"]
dist = camera_intrinsics["dist_coeffs"]

T_GC = np.load(RIGID_T_PATH)

#LOAD HEURISTIC PLANE INFO 
PLANE_N = np.array([0,0,1.0])
PLANE_P0 = np.array([0.0, 0.0, -0.033])
WINDOW_NAME = "Ray intersection"


def format_point(point: np.ndarray | None) -> str:
    if point is None:
        return "None"
    return f"[{point[0]: .4f}, {point[1]: .4f}, {point[2]: .4f}] m"


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


def show_corner_intersections(
    image: np.ndarray,
    corners: np.ndarray,
    T_WC: np.ndarray,
    plane_n: np.ndarray,
    plane_p0: np.ndarray,
    sample_label: str,
) -> bool:
    corner_pixels = corners.reshape(-1, 2)

    for corner_index, pixel in enumerate(corner_pixels, start=1):
        ray_o, ray_d = convert_to_ray(pixel, T_WC=T_WC)
        x_threed, _, status = find_intersection(
            plane_n=plane_n,
            plane_p0=plane_p0,
            ray_o=ray_o,
            ray_d=ray_d,
        )

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


def main()->None: 
    #FIND CHESS CORNERS IN A LOADED IMAGE
    print("CORNER LOCALIZATION SCRIPT STARTED")
    print(f"IMAGE PATH: {SAMPLES_JSON_PATH.resolve()}")
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
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
        30,
        1e-3,
    )
    
    plane_n = PLANE_N
    plane_p0 = PLANE_P0
    
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

        image = cv.imread(str(image_path), cv.IMREAD_COLOR)
        if image is None:
            print("  Skipping: image could not be read")
            continue
        
        #GET THE ROBOT POSE CORRESPONDING TO THE IMAGE
        robot_pose = sample_to_gripper_pose(sample)

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        found, corners = cv.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE,
        )

        if not found:
            print("Checkerboard detection failed")
            break

        refined_corners = cv.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=termination,
        )
        print(f"Ref corners shape: {refined_corners.shape}")

        R_WG, t_BG = parse_robot_pose(robot_pose)
        T_WG = np.eye(4)
        T_WG[:3,:3] = R_WG
        T_WG[:3, 3] = t_BG.reshape(3)

        # T_GC is the calibrated camera pose in the gripper frame, so this gives
        # the camera pose in the world/base frame for this sample.
        T_WC = T_WG @ T_GC
        keep_going = show_corner_intersections(
            image=image,
            corners=refined_corners,
            T_WC=T_WC,
            plane_n=plane_n,
            plane_p0=plane_p0,
            sample_label=image_path.name,
        )
        if not keep_going:
            break
    
    cv.destroyAllWindows()
        

if __name__ == "__main__":
  main()
