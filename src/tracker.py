from __future__ import annotations

import cv2 as cv
import numpy as np
from collections import deque

try:
    from .kinematics import RobotKinematics

except ImportError:
    from kinematics import RobotKinematics

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
    from .ocr_keyboard_localizer import localize_multiple_with_easyocr
except ImportError:
    from gemini_keyboard_localizer import (
        call_gemini,
        classical_validation,
        localize_with_gemini,
        parse_gemini_response,
        point_from_result,
    )
    from ocr_keyboard_localizer import localize_multiple_with_easyocr

try:
    from .utils.tracking_utils import (
        activate_letter_buffers,
        build_key_templates,
        draw_visual_track_pixels,
        estimate_visual_targets,
        estimate_world_from_pixel,
        read_robot_joints,
        save_initial_pixel_overlay,
        store_target_state,
        trackForward, 
        update_visual_track_pixels,
        show_initial_localizations,
        show_tracking_view,
    )
except ImportError:
    from utils.tracking_utils import (
        activate_letter_buffers,
        build_key_templates,
        draw_visual_track_pixels,
        estimate_visual_targets,
        estimate_world_from_pixel,
        read_robot_joints,
        save_initial_pixel_overlay,
        store_target_state,
        trackForward, 
        update_visual_track_pixels,
        show_initial_localizations,
        show_tracking_view,
    )

try: 
    from .controller import SO101Interface
except ImportError: 
    from controller import SO101Interface


HOMOGRAPHY_PATH = "camera_calib/calibrations/homography_pixel_to_world.npy"
RIGID_T_PATH = "camera_calib/calibrations/rigid_nonlinear_refined.npy"
CAMERA_NO = 5
WINDOW_NAME = "track to world"
DEFAULT_LIVE_MODEL = "gpt-5.5"
# Use gpt-5.4-mini when you want lower latency/cost.
RAY_BUFFER_SIZE = 25

DEBUG_VIZ = True

T_GC = np.load(RIGID_T_PATH)
T_GC[0,3] = 0.00

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
        provider: str = "openai",
        gemini_backend: str = "standard",
        project: str | None = None,
        location: str = "global",
        keyboard_height: float = KEYBOARD_HEIGHT,
        backend: str = "auto",
        frame_width: int = 640,
        frame_height: int = 480,
        ray_buffer_size: int = RAY_BUFFER_SIZE,
        matching_roi: int = 100,
        use_ocr: bool = False,
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
        self.provider = provider
        self.gemini_backend = gemini_backend
        self.project = project
        self.location = location
        self.keyboard_height = keyboard_height
        self.backend = backend
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.ray_buffer_size = ray_buffer_size
        self.plane_n = PLANE_N
        self.camera_transform = T_GC
        self.use_ocr = use_ocr

        #ADJUSTED KEYBOARD HEIGHT FOR PLANE INTERSECTION
        self.keyboard_p0 = PLANE_P0.copy()
        self.keyboard_p0[2] += self.keyboard_height 
        
        self.cap: cv.VideoCapture | None = None
        self.current_pixel: np.ndarray | None = None
        self.last_frame: np.ndarray | None = None
        self.initial_frame_gray: np.ndarray | None = None
        self.last_estimate: np.ndarray | None = None
        self.targets_by_letter: dict[str, dict] = {}
        self.visual_track_pixels: dict[str, np.ndarray] = {}
        self.origins_buffers_by_letter: dict[str, deque] = {}
        self.directions_buffers_by_letter: dict[str, deque] = {}
        self.active_cluster_letters: set[str] = set()
        self.localization_start_time_s: float | None = None
        self.templates = {}
        self.matching_roi = matching_roi
        self.origins_buffer = deque(maxlen=self.ray_buffer_size)
        self.directions_buffer = deque(maxlen=self.ray_buffer_size)

        print(f"Localizing keys by ray intersection")
        print()
        print(f"Handeye transformation being used: {T_GC}")
        print()
        print(f"Table plane height being used: {PLANE_P0[2]}")
    
    def start(self, robot_interface: SO101Interface, kinematics: RobotKinematics) -> None:
        """
        Bootstraps the tracking and world estimation module of the pipeline.

        Opens the camera preview, asks Gemini for the initial key pixels, and
        initializes per-key pixel/world tracking state by intersecting camera
        rays with the keyboard plane.

        Parameters:
        - robot_interface: hardware interface used to read the current SO-101 joints.
        - kinematics: kinematics model used to compute the camera pose from the joints.

        Updates:
        - self.last_estimate: first 3D world estimate.
        - self.targets_by_letter: per-letter pixel/world tracking state.
        """

        # CAPTURING FRAME FROM CV2
        self.cap = cv.VideoCapture(self.camera, resolve_capture_backend(self.backend))
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera} with backend `{self.backend}`.")
        
        
        initial_frame, self.localization_start_time_s = capture_initial_frame_with_preview(
            self.cap,
            self.letter,
        )
        print(read_robot_joints(robot_interface.robot))
        if initial_frame is None:
            raise RuntimeError("Key world tracking cancelled before Gemini localization.")
        
        show_gemini_busy_frame(initial_frame, self.letter)
        
        #LOCALIZATION
        if self.use_ocr:
            print("looking for the letters locally with OCR...")
            initial_results = localize_multiple_with_easyocr(initial_frame, self.letters)
            
            for result in initial_results:
                if not result.found or result.center is None:
                    raise RuntimeError(f"EasyOCR found no match for `{result.target_letter}`.")
                validation = classical_validation(initial_frame, result)
                print(
                    f"Initial localization ({result.target_letter}): "
                    f"center=({result.center['x']}, {result.center['y']}), "
                    f"cv_check={'PASS' if validation.passed else 'FAIL'}"
                )
        else:
            print(f"Looking for letters on the cloud with {self.provider} VLM...")
            if len(self.letters) == 1:
                initial_results = [
                    localize_with_gemini(
                        initial_frame,
                        letter=self.letters[0],
                        model=self.model,
                        fallback_models=self.fallback_models,
                        provider=self.provider,
                        gemini_backend=self.gemini_backend,
                        project=self.project,
                        location=self.location,
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
                    provider=self.provider,
                    gemini_backend=self.gemini_backend,
                    project=self.project,
                    location=self.location,
                )
                initial_results = parse_gemini_response(
                    gemini_call.response_text,
                    image_width=image_width,
                    image_height=image_height,
                    expected_letters=self.letters,
                )
                for result in initial_results:
                    if not result.found or result.center is None:
                        raise RuntimeError(f"Gemini found no match for `{result.target_letter}`.")
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
        output_path = save_initial_pixel_overlay(initial_frame, initial_results, current_pixels)
        print(f"Saved initial Gemini pixels image: {output_path}")
        show_initial_localizations(
            initial_frame,
            initial_results,
            current_pixels,
            window_name=WINDOW_NAME,
        )

        self.initial_frame_gray = cv.cvtColor(initial_frame, cv.COLOR_BGR2GRAY)
        self.templates = build_key_templates(self.initial_frame_gray, initial_results, current_pixels)

        self.last_frame = self.initial_frame_gray.copy()

        # Fill the dictionary for all the letters in the sentence/word for the clustered
        # approach
        self.origins_buffers_by_letter = {
            result.target_letter: deque(maxlen=self.ray_buffer_size)
            for result in initial_results
        }
        self.directions_buffers_by_letter = {
            result.target_letter: deque(maxlen=self.ray_buffer_size)
            for result in initial_results
        }

        # READ JOINTS AND COMPUTE FK FOR RAY INTERSECTION AND LOGGING
        joints = read_robot_joints(robot_interface.robot) #degrees
        T_WG = kinematics.forward_kinematics(joints) #expects degrees

        T_WC = T_WG @ T_GC
        key_positions = []

        for index, (result, current_pixel) in enumerate(zip(initial_results, current_pixels)):
            origins_buffer = self.origins_buffers_by_letter[result.target_letter]
            directions_buffer = self.directions_buffers_by_letter[result.target_letter]
            if index == 0:
                self.origins_buffer = origins_buffer
                self.directions_buffer = directions_buffer

            x_threed = estimate_world_from_pixel(
                pixel=current_pixel,
                T_WC=T_WC,
                origins_buffer=origins_buffer,
                directions_buffer=directions_buffer,
                ray_buffer_size=self.ray_buffer_size,
                keyboard_p0=self.keyboard_p0,
                plane_n=self.plane_n,
            )

            key_positions.append(np.asarray(x_threed, dtype=float).reshape(3))
            print(
                f"Initial Gemini world estimate ({result.target_letter}): "
                f"({key_positions[-1][0]:.4f}, {key_positions[-1][1]:.4f}, {key_positions[-1][2]:.4f})"
            )

        self.targets_by_letter = {
            result.target_letter: {
                "index": index,
                "letter": result.target_letter,
                "initial_pixel": current_pixel.copy(),
                "pixel": current_pixel.copy(),
                "world": key_position.copy(),
            }
            for index, (result, current_pixel, key_position) in enumerate(zip(initial_results, current_pixels, key_positions))
        }

        self.visual_track_pixels = {
            letter: target["pixel"].copy()
            for letter, target in self.targets_by_letter.items()
        }

        self.last_estimate = key_positions[0].copy()


    def set_target(
        self,
        *,
        letter: str,
        robot_interface: SO101Interface,
        kinematics: RobotKinematics,
    ) -> None:
        """
        Select the next letter as the active tracking target.

        This reads one fresh camera frame, advances the selected letter from
        its initial pixel with KLT, updates its ray/world estimate, and also
        refreshes the maintained estimates for the currently active tracking
        cluster. It assumes `start()` has already created per-letter buffers
        and `targets_by_letter`.
        """
        self.letter = letter
        target = self.targets_by_letter[self.letter]
        self.origins_buffer, self.directions_buffer = activate_letter_buffers(
            self.origins_buffers_by_letter,
            self.directions_buffers_by_letter,
            self.letter,
            ray_buffer_size=self.ray_buffer_size,
        )

        frame = read_frame(self.cap, error_message="Camera stream ended or returned no frame.")
        current_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        joints = read_robot_joints(robot_interface.robot)
        T_WG = kinematics.forward_kinematics(joints)
        T_WC = T_WG @ T_GC

        self.current_pixel = np.asarray(
            target.get("initial_pixel", target["pixel"]),
            dtype=np.float32,
        ).reshape(2).copy()
        prev_target_gray = self.initial_frame_gray if self.initial_frame_gray is not None else self.last_frame
        new_pixel, status = trackForward(
            pixel_coord=self.current_pixel,
            prevImg=prev_target_gray,
            nextImg=current_gray,
        )
        if status is not None and status[0, 0] != 0 and new_pixel is not None:
            self.current_pixel = np.asarray(new_pixel[0], dtype=np.float32).reshape(2)

        update_visual_track_pixels(
            visual_track_pixels=self.visual_track_pixels,
            active_cluster_letters=self.active_cluster_letters,
            prev_gray=self.last_frame,
            current_gray=current_gray,
        )
        self.visual_track_pixels[self.letter] = self.current_pixel.copy()

        self.last_estimate = estimate_world_from_pixel(
            pixel=self.current_pixel,
            T_WC=T_WC,
            origins_buffer=self.origins_buffer,
            directions_buffer=self.directions_buffer,
            ray_buffer_size=self.ray_buffer_size,
            keyboard_p0=self.keyboard_p0,
            plane_n=self.plane_n,
        )
        store_target_state(self.targets_by_letter, self.letter, pixel=self.current_pixel, world=self.last_estimate)
        estimate_visual_targets(
            visual_track_pixels=self.visual_track_pixels,
            active_cluster_letters=self.active_cluster_letters,
            origins_buffers_by_letter=self.origins_buffers_by_letter,
            directions_buffers_by_letter=self.directions_buffers_by_letter,
            targets_by_letter=self.targets_by_letter,
            T_WC=T_WC,
            frame=current_gray,
            ray_buffer_size=self.ray_buffer_size,
            keyboard_p0=self.keyboard_p0,
            plane_n=self.plane_n,
            skip_letter=self.letter,
        )

        preview = cv.cvtColor(current_gray, cv.COLOR_GRAY2BGR)
        if DEBUG_VIZ:
            draw_visual_track_pixels(preview, self.visual_track_pixels)
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
        else:
            new_pixel = new_pixel[0]

        update_visual_track_pixels(
            visual_track_pixels=self.visual_track_pixels,
            active_cluster_letters=self.active_cluster_letters,
            prev_gray=prev_gray,
            current_gray=gray_frame,
        )

        # READ JOINTS AND COMPUTE FK
        joints = read_robot_joints(robot_interface.robot)
        T_WG = kinematics.forward_kinematics(joints)
        #if i %30 ==0:
            # print(f"Current joint positions: {joints}")
            # print("Current position")
            # print(T_WG[:3,3])

        T_WC = T_WG @ T_GC
        
        # RAY COMPUTATION
        x_threed = estimate_world_from_pixel(
            pixel=new_pixel,
            T_WC=T_WC,
            origins_buffer=self.origins_buffer,
            directions_buffer=self.directions_buffer,
            ray_buffer_size=self.ray_buffer_size,
            keyboard_p0=self.keyboard_p0,
            plane_n=self.plane_n,
        )
        estimator_status = (
            f"least-squares ({self.ray_buffer_size})"
            if len(self.origins_buffer) == self.ray_buffer_size
            else f"plane-bootstrap ({len(self.origins_buffer)}/{self.ray_buffer_size})"
        )
        
        self.current_pixel = new_pixel
        self.visual_track_pixels[self.letter] = self.current_pixel.copy()
        self.last_estimate = np.asarray(x_threed, dtype=float).reshape(3)
        store_target_state(self.targets_by_letter, self.letter, pixel=self.current_pixel, world=self.last_estimate)
        estimate_visual_targets(
            visual_track_pixels=self.visual_track_pixels,
            active_cluster_letters=self.active_cluster_letters,
            origins_buffers_by_letter=self.origins_buffers_by_letter,
            directions_buffers_by_letter=self.directions_buffers_by_letter,
            targets_by_letter=self.targets_by_letter,
            T_WC=T_WC,
            frame=gray_frame,
            ray_buffer_size=self.ray_buffer_size,
            keyboard_p0=self.keyboard_p0,
            plane_n=self.plane_n,
            skip_letter=self.letter,
        )

        self.last_frame = gray_frame

        draw_visual_track_pixels(frame, self.visual_track_pixels)
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
