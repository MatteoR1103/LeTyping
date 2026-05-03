from __future__ import annotations

import argparse
import os

import cv2 as cv
import numpy as np

try:
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower
    _LEROBOT_AVAILABLE = True
except ImportError:
    RobotKinematics = None  # type: ignore[assignment,misc]
    SOFollowerRobotConfig = None  # type: ignore[assignment,misc]
    SOFollower = None  # type: ignore[assignment,misc]
    _LEROBOT_AVAILABLE = False

try:
    from .gemini_keyboard_localizer import (
        localize_with_gemini,
        parse_fallback_models,
        parse_single_letter,
        point_from_result,
    )
except ImportError:
    from gemini_keyboard_localizer import (
        localize_with_gemini,
        parse_fallback_models,
        parse_single_letter,
        point_from_result,
    )

try:
    from .utils.general_utils import (
        capture_initial_frame_with_preview,
        put_status_lines,
        read_frame,
        resolve_capture_backend,
        resolve_urdf_path,
        show_gemini_busy_frame,
    )
except ImportError:
    from utils.general_utils import (
        capture_initial_frame_with_preview,
        put_status_lines,
        read_frame,
        resolve_capture_backend,
        resolve_urdf_path,
        show_gemini_busy_frame,
    )

try:
    from .utils.tracking_utils import (
        convert_to_ray, 
        find_intersection, 
        trackForward, 
        update_LS, 
    )
except ImportError:
    from utils.tracking_utils import (
        convert_to_ray, 
        find_intersection, 
        trackForward, 
        update_LS, 
    )

RIGID_T_PATH = "camera_calib/calibrations/rigid_transform.npy"
CAMERA_CALIB_PATH = "camera_calib/calibrations/camera_calibration.npz"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
URDF_PATH = "cfg/arm_model/so101_new_calib.urdf"
GRIPPER_LINK = "gripper_frame_link"

ROBOT_PORT = "/dev/ttyACM0"

CAMERA_NO = 1
WINDOW_NAME = "track to world"
DEFAULT_LIVE_MODEL = "gemini-3-flash-preview"
RAY_BUFFER_SIZE = 20

T_GC = np.load(RIGID_T_PATH)

PLANE_N = np.array([0.0, 0.0, 1.0])
PLANE_P0 = np.array([0.0, 0.0, -0.033459])

KEYBOARD_HEIGHT = 0.02

def parse_args() -> argparse.Namespace:
    """
    Parser to use the tracking module as a standalone script, not tied to robot operation. 
    Can be used for testing of the vision module 
    """
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
        "--backend",
        choices=["auto", "dshow", "msmf", "any"],
        default="auto",
        help="OpenCV camera backend. Default: auto.",
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
        default=ROBOT_PORT,
        help="Serial port for the SO follower arm, for example /dev/ttyACM0. Defaults to ROBOT_PORT.",
    )
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="Test Gemini + KLT without connecting to the real robot. Uses a fixed T_WG = I pose.",
    )
    return parser.parse_args()

def read_joints(robot: SOFollower) -> np.ndarray:
    """
    Reads joints locations using SO API function get_observation
    """
    obs = robot.get_observation()
    return np.array(
        [float(value) for key, value in obs.items() if key.endswith(".pos")],
        dtype=float,
    )

def main() -> None:
    """
    Loop function to run track_to_wld as a standalone script, without robot control and trajectory generation
    to inspect valid working conditions of the vision component.
    Estimates one keyboard key position in world coordinates.
    """

    try:
        args = parse_args()
        fallback_models = parse_fallback_models(args.fallback_models)
        letter=args.letter
        camera=args.camera
        model=args.model
        fallback_models=fallback_models
        project=args.project
        location=args.location
        urdf_path=args.urdf_path
        robot_port=args.robot_port
        no_robot=args.no_robot
        keyboard_height=KEYBOARD_HEIGHT
        backend=args.backend
        return_debug=True
        return_on_estimate=False
        ray_buffer_size = RAY_BUFFER_SIZE
        frame_width = 640
        frame_height = 480

        cap: cv.VideoCapture | None = None
        robot: SOFollower | None = None
        last_estimate: np.ndarray | None = None
        last_debug: dict[str, object] | None = None
        x_threed_fixed = None

        letter = parse_single_letter(letter)
        fallback_models = fallback_models or []
        if ray_buffer_size < 1:
            raise ValueError("ray_buffer_size must be at least 1.")
        plane_n = PLANE_N
        plane_p0 = PLANE_P0
        
        keyboard_p0 = plane_p0.copy()
        keyboard_p0[2] += keyboard_height 

        kinematics = None
        if not no_robot:
            if not _LEROBOT_AVAILABLE:
                raise RuntimeError("lerobot is required when --no-robot is not set.")
            if not robot_port:
                raise ValueError(
                    "Missing robot port. Pass `--robot-port /dev/ttyACM0` or set ROBOT_PORT."
                )
            kinematics = RobotKinematics(
                urdf_path=resolve_urdf_path(urdf_path),
                target_frame_name=GRIPPER_LINK,
            )
            config = SOFollowerRobotConfig(port=robot_port, id="zi_padrone")
            robot = SOFollower(config)
            robot.connect()
            #disable torque to move the robot freely 
            robot.bus.disable_torque()
            
        else:
            print("Running in no-robot mode: using a fixed T_WG = I pose for testing.")

        cap = cv.VideoCapture(camera, resolve_capture_backend(backend))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera} with backend `{backend}`.")

        initial_frame = capture_initial_frame_with_preview(cap, letter)
        if initial_frame is None:
            raise RuntimeError("Key world estimation cancelled before Gemini localization.")

        show_gemini_busy_frame(initial_frame, letter)
        initial_result = localize_with_gemini(
            initial_frame,
            letter=letter,
            model=model,
            fallback_models=fallback_models,
            project=project,
            location=location,
        )
        current_pixel = point_from_result(initial_result)
        print(f"Localized pixel: ({current_pixel[0]:.1f}, {current_pixel[1]:.1f})")

        origins_buffer: list[np.ndarray] = []
        directions_buffer: list[np.ndarray] = []

        if robot is not None:
            # JOINTS ARE IN DEGREES
            joints = read_joints(robot)
            print("Initial joints")
            print(joints)
            T_WG =  kinematics.forward_kinematics(joints)
            print("Initial transform")
            print(T_WG)
        else:
            T_WG = np.eye(4)

        T_WC = T_WG @ T_GC
        ray_o, ray_d = convert_to_ray(current_pixel, T_WC=T_WC)
        origins_buffer.append(ray_o)
        directions_buffer.append(ray_d)
        x_threed, _, estimator_status = find_intersection(
            plane_n=plane_n,
            plane_p0=keyboard_p0,
            ray_o=ray_o,
            ray_d=ray_d,
        )
        if x_threed is None:
            raise RuntimeError(f"Initial Gemini ray-plane estimate failed: {estimator_status}.")

        last_estimate = np.asarray(x_threed, dtype=float).reshape(3)
        last_debug = {
            "pixel": current_pixel.copy(),
            "initial_pixel": current_pixel.copy(),
            "bbox": initial_result.bounding_box,
            "estimator_status": "gemini-plane-bootstrap",
        }
        print(
            "Initial Gemini world estimate: "
            f"({last_estimate[0]:.4f}, {last_estimate[1]:.4f}, {last_estimate[2]:.4f})"
        )

        last_frame = cv.cvtColor(initial_frame, cv.COLOR_BGR2GRAY)
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
                if last_estimate is None:
                    raise RuntimeError("KLT lost the key before a world estimate was available.")
                print("KLT not able to track through")
                break

            new_pixel = new_pixel[0]

            if robot is not None:
                joints = read_joints(robot)
                T_WG = kinematics.forward_kinematics(joints)
            else:
                T_WG = np.eye(4)

            T_WC = T_WG @ T_GC
            ray_o, ray_d = convert_to_ray(new_pixel, T_WC=T_WC)
            origins_buffer.append(ray_o)
            directions_buffer.append(ray_d)
            
            if len(origins_buffer) > ray_buffer_size:
                origins_buffer.pop(0)
                directions_buffer.pop(0)

            if len(origins_buffer) == RAY_BUFFER_SIZE:
                x_threed = update_LS(origins=origins_buffer,
                                  directions=directions_buffer,
                                  height=keyboard_p0[2])
                
                estimator_status = f"least-squares ({RAY_BUFFER_SIZE})"
            else:
                if x_threed_fixed is None:
                    x_threed_fixed, _, estimator_status = find_intersection(
                        plane_n=plane_n,
                        plane_p0=keyboard_p0,
                        ray_o=ray_o,
                        ray_d=ray_d,
                    )
                else:
                    estimator_status = f"bootstrap ({len(origins_buffer)}/{RAY_BUFFER_SIZE})"
                x_threed = x_threed_fixed

            current_pixel = new_pixel
            if x_threed is not None:
                last_estimate = np.asarray(x_threed, dtype=float).reshape(3)
                last_debug = {
                    "pixel": np.asarray(new_pixel, dtype=float).reshape(2),
                    "initial_pixel": point_from_result(initial_result),
                    "bbox": initial_result.bounding_box,
                    "estimator_status": estimator_status,
                }

            cv.circle(frame, tuple(np.round(new_pixel).astype(int)), 2, (0, 0, 255), -1)
            if last_estimate is None:
                world_text = "World: unavailable"
            else:
                world_text = (
                    f"World: ({last_estimate[0]:.3f}, "
                    f"{last_estimate[1]:.3f}, {last_estimate[2]:.3f})"
                )
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

            if (
                return_on_estimate
                and last_estimate is not None
                and len(origins_buffer) == ray_buffer_size
            ):
                print(
                    "Updated key_pos world: "
                    f"({last_estimate[0]:.4f}, {last_estimate[1]:.4f}, {last_estimate[2]:.4f})"
                )
                if return_debug:
                    return last_estimate, last_debug or {}
                return last_estimate

            last_frame = gray_frame
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

        if last_estimate is None:
            raise RuntimeError("No key world estimate was produced.")

    
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc
    
    finally:
        if cap is not None:
            cap.release()
        cv.destroyAllWindows()
        if robot is not None:
            robot.disconnect()

if __name__ == "__main__":
    main()
