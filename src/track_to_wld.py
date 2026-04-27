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
try: 
    from .controller import SO101Interface
except ImportError: 
    from controller import SO101Interface

RIGID_T_PATH = "camera_calib/calibrations/rigid_transform.npy"
CAMERA_CALIB_PATH = "camera_calib/calibrations/camera_calibration.npz"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
URDF_PATH = "cfg/arm_model/so101_new_calib.urdf"
GRIPPER_LINK = "gripper_frame_link"

ROBOT_PORT = "/dev/ttyACM0"

CAMERA_NO = 5
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

KEYBOARD_HEIGHT = 0.02


class KeyWorldTracker:
    """Keep Gemini-initialized KLT tracking alive while another loop moves the robot."""

    def __init__(
        self,
        *,
        letter: str,
        camera: int = CAMERA_NO,
        model: str = DEFAULT_LIVE_MODEL,
        fallback_models: list[str] | None = None,
        project: str | None = None,
        location: str = "global",
        keyboard_height: float = KEYBOARD_HEIGHT,
        backend: str = "auto",
        frame_width: int = 640,
        frame_height: int = 480,
        ray_buffer_size: int = RAY_BUFFER_SIZE,
    ) -> None:
        if ray_buffer_size < 1:
            raise ValueError("ray_buffer_size must be at least 1.")
        self.letter = parse_single_letter(letter)
        self.camera = camera
        self.model = model
        self.fallback_models = fallback_models or []
        self.project = project
        self.location = location
        self.keyboard_height = keyboard_height
        self.backend = backend
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.ray_buffer_size = ray_buffer_size
        self.plane_n = PLANE_N
        self.plane_p0 = PLANE_P0
        
        #ADJUSTED KEYBOARD HEIGHT FOR PLANE INTERSECTION
        self.keyboard_p0 = self.plane_p0.copy()
        self.keyboard_p0[2] += self.keyboard_height 
        
        self.cap: cv.VideoCapture | None = None
        self.current_pixel: np.ndarray | None = None
        self.last_frame: np.ndarray | None = None
        self.last_estimate: np.ndarray | None = None
        self.last_debug: dict[str, object] = {}
        self.origins_buffer: list[np.ndarray] = []
        self.directions_buffer: list[np.ndarray] = []

    def start(self, robot_interface: SO101Interface, kinematics: RobotKinematics) -> tuple[np.ndarray, np.ndarray]:
        """
        Bootstraps the tracking and world estimation module of the pipeline.
        -Starts the video capture with cv2 
        -Opens preview and prompts gemini-flash-3 for the key location 
        -Finds the first world coordinate by intersecting the ray with the plane
        """

        #CAPTURING FRAME FROM CV2
        self.cap = cv.VideoCapture(self.camera, resolve_capture_backend(self.backend))
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera} with backend `{self.backend}`.")

        initial_frame = capture_initial_frame_with_preview(self.cap, self.letter)
        if initial_frame is None:
            raise RuntimeError("Key world tracking cancelled before Gemini localization.")

        show_gemini_busy_frame(initial_frame, self.letter)
        
        #LOCALIZATION WITH GEMINI
        initial_result = localize_with_gemini(
            initial_frame,
            letter=self.letter,
            model=self.model,
            fallback_models=self.fallback_models,
            project=self.project,
            location=self.location,
        )
        self.current_pixel = point_from_result(initial_result)
        print(f"Localized pixel: ({self.current_pixel[0]:.1f}, {self.current_pixel[1]:.1f})")


        self.last_frame = cv.cvtColor(initial_frame, cv.COLOR_BGR2GRAY)

        if robot_interface.robot is not None:
            joints = read_joints(robot_interface.robot) #degrees 
            print(f"Initial joints: {joints}")
            T_WG = kinematics.forward_kinematics(joints) #expects degrees
            print("Initial transform")
            print(T_WG)
        else:
            T_WG = np.eye(4)
        
        T_WC = T_WG @ T_GC
        
        # RAY 
        ray_o, ray_d = convert_to_ray(self.current_pixel, T_WC=T_WC)
        # FILLING BUFFER
        self.origins_buffer.append(ray_o)
        self.directions_buffer.append(ray_d)
        
        # 3D ESTIMATE 
        x_threed, _, estimator_status = find_intersection(
            plane_n=self.plane_n,
            plane_p0=self.keyboard_p0,
            ray_o=ray_o,
            ray_d=ray_d,
        )
        if x_threed is None:
            raise RuntimeError(f"Initial Gemini ray-plane estimate failed: {estimator_status}.")

        self.last_estimate = np.asarray(x_threed, dtype=float).reshape(3)
        self.last_debug = {
            "pixel": self.current_pixel.copy(),
            "initial_pixel": self.current_pixel.copy(),
            "bbox": initial_result.bounding_box,
            "estimator_status": "gemini-plane-bootstrap",
        }
        print(
            "Initial Gemini world estimate: "
            f"({self.last_estimate[0]:.4f}, {self.last_estimate[1]:.4f}, {self.last_estimate[2]:.4f})"
        )
        return self.last_estimate, joints
    #
    def update(self, robot_interface: SO101Interface, kinematics: RobotKinematics) -> np.ndarray:
        """
        """   
        frame = read_frame(self.cap, error_message="Camera stream ended or returned no frame.")
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        new_pixel, status = trackForward(
            pixel_coord=self.current_pixel,
            prevImg=self.last_frame,
            nextImg=gray_frame,
        )
        if status[0, 0] == 0:
            print("KLT not able to track through")
            return self.last_estimate

        new_pixel = new_pixel[0]
        if robot_interface.robot is not None:
            joints = read_joints(robot_interface.robot)
            print(f"Initial joints: {joints}")
            T_WG = kinematics.forward_kinematics(joints)
            print("Initial transform")
            print(T_WG)
        else:
            T_WG = np.eye(4)

        T_WC = T_WG @ T_GC
        
        ray_o, ray_d = convert_to_ray(new_pixel, T_WC=T_WC)
        self.origins_buffer.append(ray_o)
        self.directions_buffer.append(ray_d)
        if len(self.origins_buffer) > self.ray_buffer_size:
            self.origins_buffer.pop(0)
            self.directions_buffer.pop(0)

        if len(self.origins_buffer) == self.ray_buffer_size:
            x_threed = update(origins=self.origins_buffer, 
                              directions=self.directions_buffer, 
                              height=self.keyboard_p0[2],
                              )
            
            estimator_status = f"least-squares ({self.ray_buffer_size})"
        else:
            x_threed, _, intersection_status = find_intersection(
                plane_n=self.plane_n,
                plane_p0=self.keyboard_p0,
                ray_o=ray_o,
                ray_d=ray_d,
            )
            estimator_status = f"plane-bootstrap ({len(self.origins_buffer)}/{self.ray_buffer_size})"
            if x_threed is None:
                estimator_status = intersection_status

        self.current_pixel = new_pixel
        self.last_frame = gray_frame
        if x_threed is not None:
            self.last_estimate = np.asarray(x_threed, dtype=float).reshape(3)
            self.last_debug = {
                "pixel": np.asarray(new_pixel, dtype=float).reshape(2),
                "initial_pixel": self.last_debug.get("initial_pixel"),
                "bbox": self.last_debug.get("bbox"),
                "estimator_status": estimator_status,
            }

        cv.circle(frame, tuple(np.round(new_pixel).astype(int)), 2, (0, 0, 255), -1)
        if self.last_estimate is not None:
            put_status_lines(
                frame,
                [
                    f"Letter: {self.letter}",
                    f"Pixel: ({new_pixel[0]:.1f}, {new_pixel[1]:.1f})",
                    (
                        "World: "
                        f"({self.last_estimate[0]:.3f}, {self.last_estimate[1]:.3f}, "
                        f"{self.last_estimate[2]:.3f})"
                    ),
                    f"Estimator: {estimator_status}",
                ],
                color=(0, 220, 0),
            )
        cv.imshow(WINDOW_NAME, frame)
        cv.waitKey(1)
        return self.last_estimate

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv.destroyAllWindows()

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


def resolve_capture_backend(backend_name: str) -> int:
    normalized = backend_name.strip().lower()
    if normalized in {"auto", "any"}:
        return cv.CAP_ANY
    if normalized == "dshow":
        return cv.CAP_DSHOW
    if normalized == "msmf":
        return cv.CAP_MSMF
    raise ValueError(f"Unsupported camera backend: {backend_name}")



def convert_to_ray(
    pixel: np.ndarray,
    T_WC: np.ndarray,
    K: np.ndarray = K,
    dist: np.ndarray = dist,
) -> tuple[np.ndarray, np.ndarray]:
    R_WC = T_WC[:3, :3]
    t_WC = T_WC[:3, 3]

    pixel_for_cv = np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2)
    undistorted = cv.undistortPoints(pixel_for_cv, K, dist).reshape(2)
    ray_c = np.array([undistorted[0], undistorted[1], 1.0], dtype=np.float64)
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

def estimate_key_world_position(
    *,
    letter: str,
    camera: int = CAMERA_NO,
    model: str = DEFAULT_LIVE_MODEL,
    fallback_models: list[str] | None = None,
    project: str | None = None,
    location: str = "global",
    urdf_path: str = URDF_PATH,
    robot_port: str | None = None,
    no_robot: bool = False,
    keyboard_height: float = KEYBOARD_HEIGHT,
    backend: str = "auto",
    frame_width: int = 640,
    frame_height: int = 480,
    ray_buffer_size: int = RAY_BUFFER_SIZE,
    return_debug: bool = False,
    return_on_estimate: bool = True,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    
    """
    Loop function to run track_to_wld as a standalone script, without robot control and trajectory generation
    to inspect valid working conditions of the vision component.
    Estimates one keyboard key position in world coordinates.
    """
    
    cap: cv.VideoCapture | None = None
    robot: SOFollower | None = None
    last_estimate: np.ndarray | None = None
    last_debug: dict[str, object] | None = None
    x_threed_fixed = None
    try:
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
                x_threed = update(origins=origins_buffer,
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
        if return_debug:
            return last_estimate, last_debug or {}
        return last_estimate

    finally:
        if cap is not None:
            cap.release()
        cv.destroyAllWindows()
        if robot is not None:
            robot.disconnect()


def main() -> None:
    try:
        args = parse_args()
        fallback_models = parse_fallback_models(args.fallback_models)
        estimate_key_world_position(
            letter=args.letter,
            camera=args.camera,
            model=args.model,
            fallback_models=fallback_models,
            project=args.project,
            location=args.location,
            urdf_path=args.urdf_path,
            robot_port=args.robot_port,
            no_robot=args.no_robot,
            keyboard_height=KEYBOARD_HEIGHT,
            backend=args.backend,
            return_debug=True,
            return_on_estimate=False,
        )
    except RuntimeError as exc:
        if "cancelled" in str(exc):
            return
        raise SystemExit(f"Error: {exc}") from exc
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
