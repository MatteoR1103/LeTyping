from __future__ import annotations

import cv2 as cv
import numpy as np
from collections import deque

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
        put_status_lines
    )
except ImportError:
    from utils.general_utils import (
        resolve_capture_backend,
        capture_initial_frame_with_preview,
        show_gemini_busy_frame,
        read_frame, 
        put_status_lines
    )

try:
    from .gemini_keyboard_localizer import (
        localize_with_gemini,
        parse_single_letter,
        point_from_result,
    )
except ImportError:
    from gemini_keyboard_localizer import (
        localize_with_gemini,
        parse_single_letter,
        point_from_result,
    )

try:
    from .utils.tracking_utils import (
        convert_to_ray, 
        find_intersection, 
        trackForward, 
        update_LS, 
        homography, 
        point_to_ray_distance
    )
except ImportError:
    from utils.tracking_utils import (
        convert_to_ray, 
        find_intersection, 
        trackForward, 
        update_LS, 
        homography, 
        point_to_ray_distance
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

HOMOGRAPHY_PATH = "camera_calib/calibrations/homography_pixel_to_world.npy"
RIGID_T_PATH = "camera_calib/calibrations/rigid_nonlinear_refined.npy"
CAMERA_NO = 5
WINDOW_NAME = "track to world"
DEFAULT_LIVE_MODEL = "gemini-3-flash-preview"
RAY_BUFFER_SIZE = 50
HOMOGRAPHY_RAY_WARNING_DISTANCE_M = 0.02
HOMOGRAPHY_RAY_MAX_DISTANCE_M = 0.03

#Camera-to-gripper extrinsics - MEASURED
tilting_angle = 41.25
tilting_angle = np.deg2rad(tilting_angle)

c_theta = np.cos(tilting_angle)
s_theta = np.sin(tilting_angle)

R_GC = np.array([[-1.0 , 0,       0],
                 [0, -c_theta, -s_theta],
                 [0, -s_theta, c_theta]] ,
                dtype=np.float64)

t_GC = np.array([-0.005, 0.052, -0.043])


T_GC = np.load(RIGID_T_PATH)
#T_GC[:3,:3]=R_GC

#PLANE INFO
PLANE_N = np.array([0.0, 0.0, 1.0])
PLANE_P0 = np.array([0.0, 0.0, -0.032459])


print(f"Plane height being used: {PLANE_P0[2]}")
KEYBOARD_HEIGHT = 0.02

#HOMOGRAPHY
H = np.load(HOMOGRAPHY_PATH)


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
        project: str | None = None,
        location: str = "global",
        keyboard_height: float = KEYBOARD_HEIGHT,
        backend: str = "auto",
        frame_width: int = 640,
        frame_height: int = 480,
        ray_buffer_size: int = RAY_BUFFER_SIZE,
        localization_mode: str = "homography"
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

        #ADJUSTED KEYBOARD HEIGHT FOR PLANE INTERSECTION
        self.keyboard_p0 = PLANE_P0.copy()
        self.keyboard_p0[2] += self.keyboard_height 
        
        self.cap: cv.VideoCapture | None = None
        self.current_pixel: np.ndarray | None = None
        self.last_frame: np.ndarray | None = None
        self.last_estimate: np.ndarray | None = None
        self.localization_mode = localization_mode
        self.origins_buffer = deque(maxlen=self.ray_buffer_size)
        self.directions_buffer = deque(maxlen=self.ray_buffer_size)
        
        if self.localization_mode == "ray": 
            print(f"Localization mode: {self.localization_mode}")
            print(f"Handeye transformation being used: {T_GC}")
        elif self.localization_mode == "homography": 
            print(f"Localization mode: {self.localization_mode}")
            print(f"Homography being used: {H}")
        else: 
            raise ValueError(f"Localization mode {self.localization_mode} is unknown")
        
        print(f"Table plane height being used: {PLANE_P0[2]}")
    
    def start(self, robot_interface: SO101Interface, kinematics: RobotKinematics) -> tuple[np.ndarray, np.ndarray]:
        """
        Bootstraps the tracking and world estimation module of the pipeline.
        -Starts the video capture with cv2 
        -Opens preview and prompts gemini-flash-3 for the key location 
        -Finds the first world coordinate by intersecting the ray with the keyboard plane or by using 
         an estimated shomography

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
        
        
        initial_frame = capture_initial_frame_with_preview(self.cap, self.letter)
        print(read_joints(robot_interface.robot))
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

        # READ JOINTS AND COMPUTE FK FOR RAY INTERSECTION AND LOGGING
        if robot_interface.robot is not None:
            joints = read_joints(robot_interface.robot) #degrees 
            print()
            print(f"Initial joints: {joints}")
            T_WG = kinematics.forward_kinematics(joints) #expects degrees
            print()
            print(f"Initial transform:{T_WG}")
        else:
            T_WG = np.eye(4)
        
        T_WC = T_WG @ T_GC
        # RAY COMPUTATION
        ray_o, ray_d = convert_to_ray(self.current_pixel, T_WC=T_WC)
        
        # FILLING BUFFER
        self.origins_buffer.append(ray_o)
        self.directions_buffer.append(ray_d)
        
        if self.localization_mode == "ray":
            
            # 3D ESTIMATE BY INTERSECTING
            x_threed, _, estimator_status = find_intersection(
                plane_n=self.plane_n,
                plane_p0=self.keyboard_p0,
                ray_o=ray_o,
                ray_d=ray_d,
            )
            
            if x_threed is None:
                raise RuntimeError(f"Initial Gemini ray-plane estimate failed: {estimator_status}.")
            
        elif self.localization_mode == "homography":
            print(f"Pixel used by homography: {self.current_pixel}")
            x_threed = homography(H=H, 
                                  pixel_coord=self.current_pixel,
                                  keyboard_height=self.keyboard_p0[2]
                                  )
            T_WC = T_WG @ T_GC

            ray_distance = point_to_ray_distance(x_threed, ray_o, ray_d)
            if ray_distance > HOMOGRAPHY_RAY_MAX_DISTANCE_M:
                print(
                    "Homography estimate is TOO FAR from initial camera ray: "
                    f"{ray_distance:.4f} m"
                )
            elif ray_distance > HOMOGRAPHY_RAY_WARNING_DISTANCE_M and ray_distance < HOMOGRAPHY_RAY_MAX_DISTANCE_M:
                print(
                    "WARNING: homography estimate is far from initial camera ray: "
                    f"{ray_distance:.4f} m"
                )
            
            
        else: 
            raise ValueError("Localization mode is unknown, world location has failed")
        
        #LOG THE LAST LOCATION FOUND BY THE LOCALIZATION MODULE
        self.last_estimate = np.asarray(x_threed, dtype=float).reshape(3)
        print(
            "Initial Gemini world estimate: "
            f"({self.last_estimate[0]:.4f}, {self.last_estimate[1]:.4f}, {self.last_estimate[2]:.4f})"
        )
        return self.last_estimate, joints


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
        new_pixel, status = trackForward(
            pixel_coord=self.current_pixel,
            prevImg=self.last_frame,
            nextImg=gray_frame,
        )
        if status[0, 0] == 0:
            print("KLT not able to track through")
            return self.last_estimate
        
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
        ray_o, ray_d = convert_to_ray(new_pixel, T_WC=T_WC)
        if self.last_estimate is not None:
            ray_distance = point_to_ray_distance(self.last_estimate, ray_o, ray_d)
            if ray_distance > HOMOGRAPHY_RAY_MAX_DISTANCE_M:
                print(
                    "WARNING: tracked ray is far from current world estimate: "
                    f"{ray_distance:.4f} m"
                )
        self.origins_buffer.append(ray_o)
        self.directions_buffer.append(ray_d)

        # UPDATE LS WHEN BUFFER IS FULL
        if len(self.origins_buffer) == self.ray_buffer_size:
            x_threed = update_LS(origins=list(self.origins_buffer), 
                              directions=list(self.directions_buffer), 
                              height=self.keyboard_p0[2],
                              )
            
            estimator_status = f"least-squares ({self.ray_buffer_size})"
        
        # 3D ESTIMATE BY INTERSECTING  
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

        cv.circle(frame, tuple(np.round(new_pixel).astype(int)), 2, (0, 0, 255), -1)
        
        if self.last_estimate is not None:
            put_status_lines(
                frame,
                [f"Letter: {self.letter}",
                f"Pixel: ({new_pixel[0]:.1f}, {new_pixel[1]:.1f})",
                ("World: "
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
        """
        Closing routine to stop the frame capture and destroy cv2 windows
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv.destroyAllWindows()
