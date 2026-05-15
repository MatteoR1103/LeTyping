from __future__ import annotations

import cv2 as cv
import numpy as np
from collections import deque
from pathlib import Path
import time

try:
    from .traj_generation import RobotKinematics

except ImportError:
    from traj_generation import RobotKinematics

try:
    from .utils.general_utils import (
        resolve_capture_backend,
        capture_initial_frame_with_preview,
        show_gemini_busy_frame,
        read_frame, 
    )
except ImportError:
    from utils.general_utils import (
        resolve_capture_backend,
        capture_initial_frame_with_preview,
        show_gemini_busy_frame,
        read_frame, 
    )

try:
    from .gemini_keyboard_localizer import (
        call_gemini,
        classical_validation,
        localize_with_gemini,
        parse_gemini_response,
        point_from_result,
    )
except ImportError:
    from gemini_keyboard_localizer import (
        call_gemini,
        classical_validation,
        localize_with_gemini,
        parse_gemini_response,
        point_from_result,
    )

try:
    from .utils.tracking_utils import (
        convert_to_ray, 
        find_intersection, 
        trackForward, 
        update_LS, 
        show_initial_localizations,
        show_tracking_view,
        template_match
    )
except ImportError:
    from utils.tracking_utils import (
        convert_to_ray, 
        find_intersection, 
        trackForward, 
        update_LS, 
        show_initial_localizations,
        show_tracking_view,
        template_match,
    )

try: 
    from .controller import SO101Interface
except ImportError: 
    from controller import SO101Interface


def read_joints(robot: SO101Interface) -> np.ndarray:
    obs = robot.get_observation()
    return np.array(
        [float(value) for key, value in obs.items() if key.endswith(".pos")],
        dtype=float,
    )

RIGID_T_PATH = "camera_calib/calibrations/rigid_nonlinear_refined.npy"
CAMERA_NO = 5
WINDOW_NAME = "track to world"
DEFAULT_LIVE_MODEL = "gemini-3.1-pro-preview"
RAY_BUFFER_SIZE = 25

DEBUG_VIZ = True

T_GC = np.load(RIGID_T_PATH)

#PLANE INFO
PLANE_N = np.array([0.0, 0.0, 1.0])
PLANE_P0 = np.array([0.0, 0.0, -0.032459])


print(f"Plane height being used: {PLANE_P0[2]}")
KEYBOARD_HEIGHT = 0.02

class KeyWorldTracker:
    """
    Class that is responsible of bootstrapping and maintaining a pixel and world estimate of the keyboard
    -start method used for bootstrapping
    -update method used as a callback from the controller at each step to update the world position of the key
    using LS 
    """

    def __init__(
        self,
        *,
        letter: str,
        camera: int = CAMERA_NO,
        model: str = DEFAULT_LIVE_MODEL,
        fallback_models: list[str] | None = None,
        gemini_backend: str = "standard",
        project: str | None = None,
        location: str = "global",
        keyboard_height: float = KEYBOARD_HEIGHT,
        backend: str = "auto",
        frame_width: int = 640,
        frame_height: int = 480,
        ray_buffer_size: int = RAY_BUFFER_SIZE,
        matching_roi: int = 200,
    ) -> None:
        if ray_buffer_size < 1:
            raise ValueError("ray_buffer_size must be at least 1.")
        requested_letters = [letter.strip().upper() for letter in letter.split(",") if letter.strip()]
        if not requested_letters or any(
            not ((len(letter) == 1 and letter.isalpha()) or letter in {"SPACE", "ENTER"})
            for letter in requested_letters
        ):
            raise ValueError("Expected one or more letters, SPACE, or ENTER, for example A,SPACE,ENTER,R,L.")
        self.letters = []
        seen_letters: set[str] = set()
        for current_letter in requested_letters:
            if current_letter in seen_letters:
                continue
            seen_letters.add(current_letter)
            self.letters.append(current_letter)
        self.letter = ",".join(self.letters)
        self.camera = camera
        self.model = model
        self.fallback_models = fallback_models or []
        self.gemini_backend = gemini_backend
        self.project = project
        self.location = location
        self.keyboard_height = keyboard_height
        self.backend = backend
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.ray_buffer_size = ray_buffer_size
        self.plane_n = PLANE_N

        #ADJUSTED KEYBOARD HEIGHT FOR PLANE INTERSECTION
        self.keyboard_p0 = PLANE_P0.copy()
        self.keyboard_p0[2] += self.keyboard_height 
        
        self.cap: cv.VideoCapture | None = None
        self.current_pixel: np.ndarray | None = None
        self.last_frame: np.ndarray | None = None
        self.initial_frame_gray: np.ndarray | None = None
        self.last_estimate: np.ndarray | None = None
        self.targets = []
        self.targets_by_letter: dict[str, dict] = {}
        self.templates = {}
        self.matching_roi = matching_roi
        self.origins_buffer = deque(maxlen=self.ray_buffer_size)
        self.directions_buffer = deque(maxlen=self.ray_buffer_size)
        self._update_count = 0
        
        self.set_homing = False

        print("Localization mode: ray")
        print(f"Handeye transformation being used: {T_GC}")
        
        print(f"Table plane height being used: {PLANE_P0[2]}")
    
    def start(self, robot_interface: SO101Interface, kinematics: RobotKinematics) -> tuple[np.ndarray, np.ndarray]:
        """
        Bootstraps the tracking and world estimation module of the pipeline.
        -Starts the video capture with cv2 
        -Opens preview and prompts gemini-flash-3 for the key location 
        -Finds the first world coordinates of all the keys by intersecting each ray with the keyboard plane

        args: 
        -robot_interface (SO101Interface): the custom robot Interface for the SO101 robot that serves as the 
        wrapper to read and send actions to the joints
        -kinematics (RobotKinematics): the kinematics used for FK and IK 

        returns: 
        -last_estimate : 3D world estimate of the key
        -joints: joint positions observation at the time of the estimate
        """

        # CAPTURING FRAME FROM CV2
        self.cap = cv.VideoCapture(self.camera, resolve_capture_backend(self.backend))
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera} with backend `{self.backend}`.")
        
        if self.set_homing: 
            while True: 
                frame = read_frame(self.cap, error_message="Camera stream ended during preview.")
                cv.imshow("set homing", frame)
                cv.waitKey(1)
                print(f"Current joints: {np.rad2deg(robot_interface.read_joints()[0])}")

        initial_frame = capture_initial_frame_with_preview(self.cap, self.letter)
        if initial_frame is None:
            raise RuntimeError("Key world tracking cancelled before Gemini localization.")
        
        
        # Capture the camera pose associated with the frame sent to Gemini.
        if robot_interface.robot is not None:
            joints_at_gemini = read_joints(robot_interface.robot) #degrees
            print()
            print(f"Initial joints: {joints_at_gemini}")
            T_WG_at_gemini = kinematics.forward_kinematics(joints_at_gemini) #expects degrees
            print()
            print(f"Initial transform:{T_WG_at_gemini}")
        else:
            joints_at_gemini = np.array([])
            T_WG_at_gemini = np.eye(4)

        T_WC_at_gemini = T_WG_at_gemini @ T_GC
        
        show_gemini_busy_frame(initial_frame, self.letter)
        
        #LOCALIZATION WITH GEMINI
        if len(self.letters) == 1:
            initial_results = [
                localize_with_gemini(
                    initial_frame,
                    letter=self.letters[0],
                    model=self.model,
                    fallback_models=self.fallback_models,
                    project=self.project,
                    location=self.location,
                    gemini_backend=self.gemini_backend,
                )
            ]
        else:
            image_height, image_width = initial_frame.shape[:2]
            gemini_call = call_gemini(
                image=initial_frame,
                image_width=image_width,
                image_height=image_height,
                target_letters=self.letters,
                model=self.model,
                fallback_models=self.fallback_models,
                project=self.project,
                location=self.location,
                gemini_backend=self.gemini_backend,
            )
            initial_results = parse_gemini_response(
                gemini_call.response_text,
                image_width=image_width,
                image_height=image_height,
                expected_letters=self.letters,
            )
            for result in initial_results:
                if not result.found or result.center is None:
                    raise RuntimeError(f"Gemini did not find the target letter `{result.target_letter}`.")
                validation = classical_validation(initial_frame, result)
                print(
                    f"Initial localization ({result.target_letter}): "
                    f"center=({result.center['x']}, {result.center['y']}), "
                    f"cv_check={'PASS' if validation.passed else 'FAIL'}"
                )

        #ALL PIXEL LOCATIONS
        current_pixels = [point_from_result(result) for result in initial_results]
        self.current_pixel = current_pixels[0]
        
        for result, current_pixel in zip(initial_results, current_pixels):
            print(f"Localized pixel ({result.target_letter}): ({current_pixel[0]:.1f}, {current_pixel[1]:.1f})")
        annotated_initial = initial_frame.copy()
        for result, current_pixel in zip(initial_results, current_pixels):
            center = tuple(np.round(current_pixel).astype(int))
            cv.circle(annotated_initial, center, 5, (0, 0, 255), -1)
            cv.putText(
                annotated_initial,
                str(result.target_letter),
                (center[0] + 7, center[1] - 7),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
        output_dir = Path("camera")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"initial_gemini_pixels_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv.imwrite(str(output_path), annotated_initial)
        print(f"Saved initial Gemini pixels image: {output_path}")
        show_initial_localizations(
            initial_frame,
            initial_results,
            current_pixels,
            window_name=WINDOW_NAME,
        )
        
        self.initial_frame_gray = cv.cvtColor(initial_frame, cv.COLOR_BGR2GRAY)
        
        # Initialize templates and preserve the same pixel anchor returned by point_from_result.
        image_h, image_w = self.initial_frame_gray.shape[:2]
        for result in initial_results:
            xmin, ymin, xmax, ymax = result.bounding_box
            #Add space for context
            xmin -=10
            ymin -=10
            xmax +=10
            ymax +=10
            xmin = max(0, min(image_w - 1, xmin))
            xmax = max(0, min(image_w - 1, xmax))
            ymin = max(0, min(image_h - 1, ymin))
            ymax = max(0, min(image_h - 1, ymax))
            if result.target_letter == "SPACE":
                centerx = (xmin+xmax)//2
                centery = (ymin+ymax)//2
                len_x = (xmax-xmin)//2
                len_y = (ymax-ymin)//2
                template = self.initial_frame_gray[centery-len_x:centery+len_x+1, centerx-len_y:centerx+len_y]
            else: 
                template = self.initial_frame_gray[ymin:ymax + 1, xmin:xmax + 1]
            anchor_offset = point_from_result(result) - np.array([xmin, ymin], dtype=np.float32)
            self.templates[result.target_letter] = {
                "template": template,
                "anchor_offset": anchor_offset.astype(np.float32),
            }

        self.last_frame = self.initial_frame_gray.copy()

        joints = joints_at_gemini
        T_WC = T_WC_at_gemini
        key_positions = []
        for index, (result, current_pixel) in enumerate(zip(initial_results, current_pixels)):
            # RAY COMPUTATION
            ray_o, ray_d = convert_to_ray(current_pixel, T_WC=T_WC)

            # FILLING BUFFER FOR THE FIRST KEY THAT MAY BE TRACKED LATER
            if index == 0:
                self.origins_buffer.append(ray_o)
                self.directions_buffer.append(ray_d)

            x_threed, _, estimator_status = find_intersection(
                plane_n=self.plane_n,
                plane_p0=self.keyboard_p0,
                ray_o=ray_o,
                ray_d=ray_d,
            )

            if x_threed is None:
                raise RuntimeError(f"Initial Gemini ray-plane estimate failed: {estimator_status}.")

            key_positions.append(np.asarray(x_threed, dtype=float).reshape(3))
            print(
                f"Initial Gemini world estimate ({result.target_letter}): "
                f"({key_positions[-1][0]:.4f}, {key_positions[-1][1]:.4f}, {key_positions[-1][2]:.4f})"
            )

        self.targets = [
            {
                "index": index,
                "letter": result.target_letter,
                "pixel": current_pixel.copy(),
                "world": key_position.copy(),
            }
            for index, (result, current_pixel, key_position) in enumerate(zip(initial_results, current_pixels, key_positions))
        ]
        self.targets_by_letter = {
            target["letter"]: target
            for target in self.targets
        }

        #LOG THE LAST LOCATION FOUND BY THE LOCALIZATION MODULE
        self.last_estimate = key_positions[0].copy()
        if len(key_positions) == 1:
            return self.last_estimate, joints
        return np.asarray(key_positions, dtype=float), joints

    def _track_pixel_on_frame(
        self,
        *,
        pixel: np.ndarray,
        letter: str,
        prev_gray: np.ndarray,
        current_gray: np.ndarray,
    ) -> np.ndarray:
        tracked_pixel = np.asarray(pixel, dtype=np.float32).reshape(2)
        new_pixel, status = trackForward(
            pixel_coord=tracked_pixel,
            prevImg=prev_gray,
            nextImg=current_gray,
        )
        if status is not None and status[0, 0] != 0 and new_pixel is not None:
            tracked_pixel = new_pixel[0]

        template_info = self.templates.get(letter)
        if template_info is not None:
            tracked_pixel = template_match(
                template_info=template_info,
                current_gray=current_gray,
                current_pixel=tracked_pixel,
                matching_roi=self.matching_roi,
            )
        return np.asarray(tracked_pixel, dtype=np.float32).reshape(2)

    def _estimate_world_from_pixel(
        self,
        *,
        pixel: np.ndarray,
        T_WC: np.ndarray,
        origins_buffer: deque | None = None,
        directions_buffer: deque | None = None,
    ) -> np.ndarray:
        ray_o, ray_d = convert_to_ray(pixel, T_WC=T_WC)
        if origins_buffer is not None and directions_buffer is not None:
            origins_buffer.append(ray_o)
            directions_buffer.append(ray_d)

        if origins_buffer is not None and directions_buffer is not None and len(origins_buffer) == self.ray_buffer_size:
            x_threed = update_LS(
                origins=list(origins_buffer),
                directions=list(directions_buffer),
                height=self.keyboard_p0[2],
            )
        else:
            x_threed, _, estimator_status = find_intersection(
                plane_n=self.plane_n,
                plane_p0=self.keyboard_p0,
                ray_o=ray_o,
                ray_d=ray_d,
            )
            if x_threed is None:
                raise RuntimeError(f"Ray-plane estimate failed: {estimator_status}.")

        return np.asarray(x_threed, dtype=float).reshape(3)

    def set_target(
        self,
        pixel: np.ndarray,
        world: np.ndarray,
        frame: np.ndarray | None = None,
        letter: str | None = None,
        robot_interface: SO101Interface | None = None,
        kinematics: RobotKinematics | None = None,
    ) -> None:
        if letter is not None:
            self.letter = letter
        self.origins_buffer.clear()
        self.directions_buffer.clear()

        if frame is None:
            if self.cap is not None:
                frame = read_frame(self.cap, error_message="Camera stream ended or returned no frame.")
            elif self.initial_frame_gray is not None:
                self.last_frame = self.initial_frame_gray.copy()
                self.current_pixel = np.asarray(pixel, dtype=np.float32).reshape(2)
                self.last_estimate = np.asarray(world, dtype=float).reshape(3)
                return
            else:
                return

        if frame.ndim == 2:
            current_gray = frame.copy()
        else:
            current_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if self.initial_frame_gray is None:
            self.last_frame = current_gray
            self.current_pixel = np.asarray(pixel, dtype=np.float32).reshape(2)
            self.last_estimate = np.asarray(world, dtype=float).reshape(3)
            return

        if robot_interface is not None and getattr(robot_interface, "robot", None) is not None and kinematics is not None:
            joints = read_joints(robot_interface.robot)
            T_WG = kinematics.forward_kinematics(joints)
        else:
            T_WG = np.eye(4)
        T_WC = T_WG @ T_GC

        self.current_pixel = self._track_pixel_on_frame(
            pixel=np.asarray(pixel, dtype=np.float32).reshape(2),
            letter=self.letter,
            prev_gray=self.initial_frame_gray,
            current_gray=current_gray,
        )
        self.last_estimate = self._estimate_world_from_pixel(
            pixel=self.current_pixel,
            T_WC=T_WC,
            origins_buffer=self.origins_buffer,
            directions_buffer=self.directions_buffer,
        )

        preview = cv.cvtColor(current_gray, cv.COLOR_GRAY2BGR)
        if DEBUG_VIZ:
            cv.circle(preview, tuple(np.round(self.current_pixel).astype(int)), 5, (0, 0, 255), -1)
            cv.putText(
                preview,
                str(self.letter),
                tuple(np.round(self.current_pixel + np.array([7, 15])).astype(int)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv.LINE_AA,
            )
            cv.imshow(WINDOW_NAME, preview)
            cv.waitKey(1)

        self.last_frame = current_gray

    def update(self, i: int, robot_interface: SO101Interface, kinematics: RobotKinematics) -> np.ndarray:
        """
        Maintatins the tracking of a key and refines the 3D world estimate with LS on a sliding window of 
        observations 
        -Reads current frame 
        -Tracks the pixel from previous frame to current using KLT 
        -Estimates the 3D location of the point 
        -Updates the estimate using LS over the sliding window of observations

        args: 
        -robot_interface (SO101Interface): the custom robot Interface for the SO101 robot that serves as the 
         wrapper to read and send actions to the joints
        -kinematics (RobotKinematics): the kinematics used for FK and IK 

        returns: 
        -last_estimate : 3D world estimate of the key
        """   

        if self.cap is None or self.current_pixel is None or self.last_frame is None:
            return self.last_estimate

        # READ CURRENT FRAME
        frame = read_frame(self.cap, error_message="Camera stream ended or returned no frame.")
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # TRACK FORWARD USING KLT
        prev_gray = self.last_frame
        new_pixel, status = trackForward(
            pixel_coord=self.current_pixel,
            prevImg=prev_gray,
            nextImg=gray_frame,
        )
        tracking_status = "tracking"
        if status is None or status[0, 0] == 0 or new_pixel is None:
            template_info = self.templates.get(self.letter)
            if template_info is None:
                print("KLT not able to track through")
                self.last_frame = gray_frame
                show_tracking_view(
                    frame,
                    self.current_pixel,
                    letter=self.letter,
                    last_estimate=self.last_estimate,
                    estimator_status="holding last estimate",
                    tracking_status="KLT lost",
                    color=(0, 0, 255),
                    window_name=WINDOW_NAME,
                )
                return self.last_estimate
            new_pixel = template_match(
                template_info=template_info,
                current_gray=gray_frame,
                current_pixel=self.current_pixel,
                matching_roi=self.matching_roi,
            )
            tracking_status = "template fallback"
        else:
            new_pixel = new_pixel[0]

        # READ JOINTS AND COMPUTE FK
        if robot_interface.robot is not None:
            joints = read_joints(robot_interface.robot)
            T_WG = kinematics.forward_kinematics(joints)
            if i %30 ==0:
                print(f"Current joint positions: {joints}")
                print("Current position")
                print(T_WG[:3,3])
        else:
            T_WG = np.eye(4)

        T_WC = T_WG @ T_GC
        
        # RAY COMPUTATION
        x_threed = self._estimate_world_from_pixel(
            pixel=new_pixel,
            T_WC=T_WC,
            origins_buffer=self.origins_buffer,
            directions_buffer=self.directions_buffer,
        )
        estimator_status = (
            f"least-squares ({self.ray_buffer_size})"
            if len(self.origins_buffer) == self.ray_buffer_size
            else f"plane-bootstrap ({len(self.origins_buffer)}/{self.ray_buffer_size})"
        )
        
        self.current_pixel = new_pixel
        if x_threed is not None:
            self.last_estimate = np.asarray(x_threed, dtype=float).reshape(3)

        self.last_frame = gray_frame

        show_tracking_view(
            frame,
            new_pixel,
            letter=self.letter,
            last_estimate=self.last_estimate,
            estimator_status=estimator_status,
            tracking_status=tracking_status,
            color=(0, 0, 255),
            window_name=WINDOW_NAME,
        )
        return self.last_estimate

    def close(self) -> None:
        """
        Closing routine to stop the frame capture and destroy cv2 windows
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv.destroyAllWindows()
