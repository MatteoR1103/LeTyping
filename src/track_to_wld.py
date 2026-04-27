from __future__ import annotations

import argparse
import os

import cv2 as cv
import numpy as np
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower

try:
    from .gemini_keyboard_localizer import (
        GeminiLocalizationResult,
        call_gemini,
        classical_validation,
        parse_fallback_models,
        parse_gemini_response,
        parse_target_letters,
    )
except ImportError:
    from gemini_keyboard_localizer import (
        GeminiLocalizationResult,
        call_gemini,
        classical_validation,
        parse_fallback_models,
        parse_gemini_response,
        parse_target_letters,
    )


RIGID_T_PATH = "camera_calib/calibrations/rigid_transform.npy"
CAMERA_CALIB_PATH = "camera_calib/calibrations/camera_calibration.npz"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
URDF_PATH = "cfg/arm_model/so101_new_calib.urdf"
GRIPPER_LINK = "gripper_frame_link"

CAMERA_NO = 1
WINDOW_NAME = "track to world"
DEFAULT_LIVE_MODEL = "gemini-3-flash-preview"
RAY_BUFFER_SIZE = 50
KLT_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.001),
)

camera_intrinsics = np.load(CAMERA_CALIB_PATH)
K = camera_intrinsics["camera_matrix"]
dist = camera_intrinsics["dist_coeffs"]
T_GC = np.load(RIGID_T_PATH)

PLANE_N = np.array([0.0, 0.0, 1.0])
PLANE_P0 = np.array([0.0, 0.0, -0.033459])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use Gemini once to initialize the tracked keyboard pixel, then track "
            "it with KLT and estimate its world position."
        )
    )
    parser.add_argument(
        "--letter",
        required=True,
        help="Single target keyboard letter to localize and track, for example X.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_NO,
        help=f"OpenCV camera index. Default: {CAMERA_NO}.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LIVE_MODEL,
        help=f"Gemini model used for the initial localization. Default: {DEFAULT_LIVE_MODEL}",
    )
    parser.add_argument(
        "--fallback-models",
        default="",
        help="Optional comma-separated fallback Gemini models. Default: none.",
    )
    parser.add_argument(
        "--project",
        default=os.getenv("GOOGLE_CLOUD_PROJECT"),
        help="Google Cloud project for Vertex AI. Defaults to GOOGLE_CLOUD_PROJECT.",
    )
    parser.add_argument(
        "--location",
        default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        help="Google Cloud location for Vertex AI. Defaults to GOOGLE_CLOUD_LOCATION or global.",
    )
    parser.add_argument(
        "--urdf-path",
        default=URDF_PATH,
        help=(
            "Path to the SO101 URDF file. Default: SO101/so101_new_calib.urdf "
            "(or SO101_URDF_PATH if set)."
        ),
    )
    parser.add_argument(
        "--robot-port",
        default=os.getenv("ROBOT_PORT"),
        help="Serial port for the SO follower arm, for example /dev/ttyACM0. Defaults to ROBOT_PORT.",
    )
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="Test Gemini + KLT without connecting to the real robot. Uses a fixed T_WG = I pose.",
    )
    return parser.parse_args()


def resolve_urdf_path(path: str) -> str:
    candidates = [path]
    if not os.path.isabs(path):
        candidates.append(os.path.join(REPO_ROOT, path))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError(
        "SO101 URDF not found. Copy `so101_new_calib.urdf` from the SO-ARM100 repo into "
        f"`{os.path.join(REPO_ROOT, 'SO101')}` or pass `--urdf-path /absolute/path/to/so101_new_calib.urdf`."
    )


def convert_to_ray(pixel: np.ndarray, T_WC: np.ndarray, K: np.ndarray = K) -> tuple[np.ndarray, np.ndarray]:
    pixel_h = np.array([pixel[0], pixel[1], 1.0])
    K_inv = np.linalg.inv(K)
    R_WC = T_WC[:3, :3]
    t_WC = T_WC[:3, 3]

    ray_c = K_inv @ pixel_h
    ray_w = R_WC @ ray_c
    ray_w /= np.linalg.norm(ray_w)
    return t_WC, ray_w


def find_intersection(
    plane_n: np.ndarray,
    plane_p0: np.ndarray,
    ray_o: np.ndarray,
    ray_d: np.ndarray,
    eps: float = 1e-5,
) -> tuple[np.ndarray | None, float | None, str]:
    denom = np.dot(plane_n, ray_d)
    num = np.dot(plane_n, plane_p0 - ray_o)

    if abs(denom) < eps:
        if abs(num) < eps:
            return None, None, "ray lies in plane"
        return None, None, "parallel, no intersection"

    t = num / denom
    if t < 0:
        return None, t, "intersection behind ray origin"

    x = ray_o + t * ray_d
    return x, t, "hit"


def trackForward(pixel_coord: np.ndarray, prevImg: np.ndarray, nextImg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    klt_pixel_coord = pixel_coord[None, :].astype(np.float32)
    next_pt, status, _ = cv.calcOpticalFlowPyrLK(
        prevImg=prevImg,
        nextImg=nextImg,
        prevPts=klt_pixel_coord,
        nextPts=None,
        **KLT_PARAMS,
    )
    return next_pt, status


def forward_kinematics(kinematics: RobotKinematics, current_joints: np.ndarray) -> np.ndarray:
    return kinematics.forward_kinematics(current_joints)


def read_joints(robot: SOFollower) -> np.ndarray:
    obs = robot.get_observation()
    return np.array(
        [float(value) for key, value in obs.items() if key.endswith(".pos")],
        dtype=float,
    )


def update(origins: list[np.ndarray], directions: list[np.ndarray], height: float) -> np.ndarray:
    A = np.zeros((3, 3))
    b = np.zeros(3)
    I = np.eye(3)
    
    for o, d in zip(origins, directions):
        d = d.reshape(3, 1) 
        
        I_min_ddT = I - (d @ d.T)
        A += I_min_ddT
        b += I_min_ddT @ o
        
    A_2x2 = A[:2, :2]
    
    b_2x1 = b[:2] - (A[:2, 2] * height)
    
    xy, _, _, _ = np.linalg.lstsq(A_2x2, b_2x1, rcond=None)
    x_threed = np.array([xy[0], xy[1], height])
    
    return x_threed


def read_frame(cap: cv.VideoCapture, *, error_message: str) -> np.ndarray:
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(error_message)
    return frame


def put_status_lines(
    frame: np.ndarray,
    lines: list[str],
    *,
    color: tuple[int, int, int] = (0, 220, 255),
) -> None:
    y = 30
    for line in lines:
        cv.putText(
            frame,
            line,
            (12, y),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv.LINE_AA,
        )
        y += 30


def capture_initial_frame_with_preview(cap: cv.VideoCapture, letter: str) -> np.ndarray | None:
    print("Camera preview is open. Press SPACE to run Gemini on the current frame, or q to quit.")

    while True:
        frame = read_frame(cap, error_message="Camera stream ended during preview.")
        preview = frame.copy()
        put_status_lines(
            preview,
            [
                "Live camera preview",
                f"Target letter: {letter}",
                "SPACE: localize with Gemini",
                "q: quit",
            ],
        )
        cv.imshow(WINDOW_NAME, preview)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            return None
        if key in (ord(" "), 13):
            return frame


def show_gemini_busy_frame(frame: np.ndarray, letter: str) -> None:
    busy_frame = frame.copy()
    put_status_lines(
        busy_frame,
        [
            f"Calling Gemini for letter {letter}...",
            "Please wait. Tracking will start after initialization.",
        ],
    )
    cv.imshow(WINDOW_NAME, busy_frame)
    cv.waitKey(1)


def parse_single_letter(letter_arg: str) -> str:
    target_letters = parse_target_letters(letter_arg)
    if len(target_letters) != 1:
        raise ValueError("track_to_wld expects exactly one target letter, for example --letter X.")
    return target_letters[0]


def localize_with_gemini(
    frame: np.ndarray,
    *,
    letter: str,
    model: str,
    fallback_models: list[str],
    project: str | None,
    location: str,
) -> GeminiLocalizationResult:
    image_height, image_width = frame.shape[:2]
    gemini_call = call_gemini(
        image=frame,
        image_width=image_width,
        image_height=image_height,
        target_letters=[letter],
        model=model,
        fallback_models=fallback_models,
        project=project,
        location=location,
    )
    result = parse_gemini_response(
        gemini_call.response_text,
        image_width=image_width,
        image_height=image_height,
        expected_letters=[letter],
    )[0]

    if not result.found or result.center is None:
        raise RuntimeError(f"Gemini did not find the target letter `{letter}`.")

    validation = classical_validation(frame, result)
    print(
        f"Initial localization: center=({result.center['x']}, {result.center['y']}), "
        f"cv_check={'PASS' if validation.passed else 'FAIL'}"
    )
    return result


def point_from_result(result: GeminiLocalizationResult) -> np.ndarray:
    if result.center is None:
        raise ValueError("Cannot initialize tracking without a Gemini center point.")
    return np.array([result.center["x"], result.center["y"]], dtype=np.float32)


def main() -> None:
    cap: cv.VideoCapture | None = None
    robot: SOFollower | None = None

    try:
        args = parse_args()
        letter = parse_single_letter(args.letter)
        fallback_models = parse_fallback_models(args.fallback_models)
        if not args.no_robot and not args.robot_port:
            raise ValueError(
                "Missing robot port. Pass `--robot-port /dev/ttyACM0` or set ROBOT_PORT in the environment."
            )
        urdf_path = None if args.no_robot else resolve_urdf_path(args.urdf_path)

        plane_n = PLANE_N
        plane_p0 = PLANE_P0

        kinematics = None
        if not args.no_robot:
            kinematics = RobotKinematics(
                urdf_path=urdf_path,
                target_frame_name=GRIPPER_LINK,
            )

            config = SOFollowerRobotConfig(port=args.robot_port, id = "zi_padrone")
            robot = SOFollower(config)
            robot.connect()
        else:
            print("Running in no-robot mode: using a fixed T_WG = I pose for testing.")

        cap = cv.VideoCapture(args.camera)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {args.camera}")

        initial_frame = capture_initial_frame_with_preview(cap, letter)
        if initial_frame is None:
            return

        show_gemini_busy_frame(initial_frame, letter)
        initial_result = localize_with_gemini(
            initial_frame,
            letter=letter,
            model=args.model,
            fallback_models=fallback_models,
            project=args.project,
            location=args.location,
        )
        current_pixel = point_from_result(initial_result)
        last_frame = cv.cvtColor(initial_frame, cv.COLOR_BGR2GRAY)
        origins_buffer: list[np.ndarray] = []
        directions_buffer: list[np.ndarray] = []
        x_threed_fixed: np.ndarray | None = None
        print(
            f"Tracking initialized at pixel ({current_pixel[0]:.1f}, {current_pixel[1]:.1f}). "
            "Press q to quit."
        )

        while True:
            frame = read_frame(cap, error_message="Camera stream ended or returned no frame.")
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            new_pixel, status = trackForward(
                pixel_coord=current_pixel,
                prevImg=last_frame,
                nextImg=gray_frame,
            )
            if status[0, 0] == 0:
                print("KLT not able to track through")
                break

            new_pixel = new_pixel[0]

            if robot is not None:
                joints = read_joints(robot)
                T_WG = forward_kinematics(kinematics=kinematics, current_joints=joints)
            else:
                T_WG = np.eye(4)

            T_WC = T_WG @ T_GC
            ray_o, ray_d = convert_to_ray(new_pixel, T_WC=T_WC)
            origins_buffer.append(ray_o)
            directions_buffer.append(ray_d)
            if len(origins_buffer) > RAY_BUFFER_SIZE:
                origins_buffer.pop(0)
                directions_buffer.pop(0)

            if len(origins_buffer) == RAY_BUFFER_SIZE:
                x_threed = update(origins_buffer, directions_buffer,0.02)
                estimator_status = f"least-squares ({RAY_BUFFER_SIZE})"
            else:
                if x_threed_fixed is None:
                    x_threed_fixed, _, estimator_status = find_intersection(
                        plane_n=plane_n,
                        plane_p0=plane_p0,
                        ray_o=ray_o,
                        ray_d=ray_d,
                    )
                else:
                    estimator_status = f"bootstrap ({len(origins_buffer)}/{RAY_BUFFER_SIZE})"
                x_threed = x_threed_fixed

            current_pixel = new_pixel
            cv.circle(frame, tuple(np.round(new_pixel).astype(int)), 2, (0, 0, 255), -1)
            if x_threed is None:
                world_text = "World: unavailable"
            else:
                world_text = f"World: ({x_threed[0]:.3f}, {x_threed[1]:.3f}, {x_threed[2]:.3f})"
            put_status_lines(
                frame,
                [
                    f"Letter: {letter}",
                    f"Pixel: ({new_pixel[0]:.1f}, {new_pixel[1]:.1f})",
                    world_text,
                    f"Estimator: {estimator_status}",
                    "q: quit",
                ],
                color=(0, 220, 0),
            )
            cv.imshow(WINDOW_NAME, frame)

            last_frame = gray_frame
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if cap is not None:
            cap.release()
        cv.destroyAllWindows()
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    main()
