import numpy as np
import cv2 as cv
import time

try:
    from .general_utils import put_status_lines, read_frame
except ImportError:
    from utils.general_utils import put_status_lines, read_frame

CAMERA_CALIB_PATH = "camera_calib/calibrations/camera_calibration.npz"
camera_intrinsics = np.load(CAMERA_CALIB_PATH)
K = camera_intrinsics["camera_matrix"]
dist = camera_intrinsics["dist_coeffs"]
DEFAULT_TRACKING_WINDOW_NAME = "track to world"

KLT_PARAMS = dict(
    winSize=(55, 55),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 50, 0.0005),
)

def convert_to_ray(
    pixel: np.ndarray,
    T_WC: np.ndarray,
    K: np.ndarray = K,
    dist: np.ndarray = dist,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts a pixel location into a ray in world coordinates starting from the camera optical axis
    args: 
    - pixel (np.ndarray): pixel location (2,)
    - T_WC (np.ndarray): camera pose as a Homogeneous matrix (4,4)
    - K (np.ndarray): camera intrinsics (3,3)
    - dist(np.ndarray): camera distortion coefficients (5,)
    returns:
    - t_WC (np.ndarray): origin of the ray - position of the camera in the world frame
    - ray_w (np.ndarray): ray direction in world coordinates   
    """

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
    """
    Finds the intersection between a ray in the world coordinates and a plane, which is ultimately the estimate of
    the key location in world coordinates
    args: 
    - plane_n (np.ndarray): normal direction to the plane (3,)
    - plane_p0 (np.ndarray): point on the plane - sets the height of the plane (3,)
    - ray_o (np.ndarray): ray origin (3,)
    - ray_d (np.ndarray): ray direction (3,)
    returns:
    - x (np.ndarray): intersection in the world frame
    - t (np.ndarray): scale
    - "hit" (str): info about the ray intersection    
    """
    
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
    """
    KLT tracker to track a pixel coordinate in consecutive frames 
    args: 
    - pixel_coord (np.ndarray): pixel coordinate
    - prevImg (np.ndarray): previous image
    - nextImg (np.ndarray): next image
    returns: 
    - next_pt: location of the pixel in the next frame
    - status: KLT status - 1 (tracking did not fail) or 0 (tracking failed)
    """
    klt_pixel_coord = pixel_coord[None, :].astype(np.float32)
    next_pt, status, _ = cv.calcOpticalFlowPyrLK(
        prevImg=prevImg,
        nextImg=nextImg,
        prevPts=klt_pixel_coord,
        nextPts=None,
        **KLT_PARAMS,
    )
    return next_pt, status

def template_match(
    template_info: dict,
    current_gray: np.ndarray,
    current_pixel: np.ndarray,
    matching_roi: int,
    threshold: float = 0.6
) -> np.ndarray:
    """Refine a KLT pixel with local template matching and preserve its anchor offset."""
    current_pixel = np.asarray(current_pixel, dtype=np.float32).reshape(2)
    template = template_info.get("template")
    anchor_offset = np.asarray(
        template_info.get("anchor_offset", np.zeros(2)),
        dtype=np.float32,
    ).reshape(2)

    if template is None or template.size == 0:
        return current_pixel.copy()

    if current_gray.ndim == 3:
        current_gray = cv.cvtColor(current_gray, cv.COLOR_BGR2GRAY)
    if template.ndim == 3:
        template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    th, tw = template.shape[:2]
    roi_half = max(int(matching_roi) // 2, th // 2, tw // 2)
    x0 = max(0, int(current_pixel[0] - roi_half))
    y0 = max(0, int(current_pixel[1] - roi_half))
    x1 = min(current_gray.shape[1], int(current_pixel[0] + roi_half))
    y1 = min(current_gray.shape[0], int(current_pixel[1] + roi_half))
    roi = current_gray[y0:y1, x0:x1]
    if roi.shape[0] < th or roi.shape[1] < tw:
        return current_pixel.copy()

    _, max_val, _, max_loc = cv.minMaxLoc(
        cv.matchTemplate(roi, template, cv.TM_CCOEFF_NORMED)
    )
    print("##########MATCHING VALUE##############")
    print(max_val)
    print()
    if max_val < threshold: 
        return current_pixel.copy()
    
    return np.array([x0 + max_loc[0], y0 + max_loc[1]], dtype=np.float32) + anchor_offset

def update_LS(origins: list[np.ndarray], directions: list[np.ndarray], height: float) -> np.ndarray:
    """
    Finds a LS estimate of the world location of the key using a buffer of ray directions and origins, 
    fixing the z position to the height of the plane
    args: 
    - origins (list): buffer of the origins of rays accumulated over the sliding window
    - directions (list): buffer of the directions of rays accumulated over the sliding window
    - height (float): height of the plane
    returns: 
    - x_threed (np.ndarray): 3D location of the key found by LS
    """
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

def point_to_ray_distance(point: np.ndarray, ray_o: np.ndarray, ray_d: np.ndarray) -> float:
    """
    Finds the distance from a world point to a camera ray for sanity checking when tracking.

    args: 
    -point (np.ndarray): 3D point estimate
    -ray_o (np.ndarray): ray origin estimated by camera ray
    -ray_d (np.ndarray): ray direction estimated by camera ray

    returns: 
    distance: point-to-line distance from camera ray to the 3D point estimate
    """
    delta = np.asarray(point, dtype=float).reshape(3) - np.asarray(ray_o, dtype=float).reshape(3)
    direction = np.asarray(ray_d, dtype=float).reshape(3)
    direction /= np.linalg.norm(direction)
    return float(np.linalg.norm(delta - np.dot(delta, direction) * direction))


def show_initial_localizations(
    frame: np.ndarray,
    initial_results: list,
    current_pixels: list[np.ndarray],
    *,
    window_name: str = DEFAULT_TRACKING_WINDOW_NAME,
    duration_s: float = 1.25,
) -> None:
    preview = frame.copy()
    for result, pixel in zip(initial_results, current_pixels):
        center = tuple(np.round(pixel).astype(int))
        cv.circle(preview, center, 4, (0, 0, 255), -1)
        cv.putText(
            preview,
            str(result.target_letter),
            (center[0] + 7, center[1] - 7),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv.LINE_AA,
        )

    put_status_lines(
        preview,
        ["Gemini localized pixels", "Tracking will start next"],
        color=(0, 220, 0),
    )
    cv.imshow(window_name, preview)
    end_time = time.perf_counter() + duration_s
    while time.perf_counter() < end_time:
        cv.waitKey(50)


def draw_tracking_view(
    frame: np.ndarray,
    pixel: np.ndarray | None,
    *,
    letter: str,
    last_estimate: np.ndarray | None,
    estimator_status: str,
    tracking_status: str = "tracking",
    color: tuple[int, int, int] = (0, 0, 255),
) -> None:
    if pixel is not None:
        center = tuple(np.round(pixel).astype(int))
        cv.circle(frame, center, 4, color, -1)

    lines = [
        f"Letter: {letter}",
        f"Tracker: {tracking_status}",
    ]
    if pixel is not None:
        lines.append(f"Pixel: ({pixel[0]:.1f}, {pixel[1]:.1f})")
    if last_estimate is not None:
        lines.append(
            "World: "
            f"({last_estimate[0]:.3f}, {last_estimate[1]:.3f}, "
            f"{last_estimate[2]:.3f})"
        )
    lines.append(f"Estimator: {estimator_status}")

    put_status_lines(frame, lines, color=(0, 220, 0))


def show_tracking_view(
    frame: np.ndarray,
    pixel: np.ndarray | None,
    *,
    letter: str,
    last_estimate: np.ndarray | None,
    estimator_status: str,
    tracking_status: str = "tracking",
    color: tuple[int, int, int] = (0, 0, 255),
    window_name: str = DEFAULT_TRACKING_WINDOW_NAME,
) -> None:
    draw_tracking_view(
        frame,
        pixel,
        letter=letter,
        last_estimate=last_estimate,
        estimator_status=estimator_status,
        tracking_status=tracking_status,
        color=color,
    )
    cv.imshow(window_name, frame)
    cv.waitKey(1)


def show_tracker_current_frame(
    tracker,
    *,
    tracking_status: str = "display only",
    window_name: str = DEFAULT_TRACKING_WINDOW_NAME,
) -> np.ndarray | None:
    """
    Show what the robot currently sees without running KLT or changing the
    tracked pixel/world estimate.
    """
    if tracker.cap is None:
        return tracker.last_estimate

    frame = read_frame(tracker.cap, error_message="Camera stream ended or returned no frame.")
    show_tracking_view(
        frame,
        tracker.current_pixel,
        letter=tracker.letter,
        last_estimate=tracker.last_estimate,
        estimator_status="position not updated",
        tracking_status=tracking_status,
        color=(0, 0, 255),
        window_name=window_name,
    )
    return tracker.last_estimate


def update_tracker_for_duration(
    tracker,
    duration_s: float,
    robot_interface,
    kinematics,
    *,
    interval_s: float = 0.05,
) -> np.ndarray | None:
    """
    Keep the live camera/tracking window updating while the robot is in a
    blocking wait, such as settling at home after a single joint command.
    """
    if tracker.cap is None or tracker.current_pixel is None or tracker.last_frame is None:
        return tracker.last_estimate

    end_time = time.perf_counter() + max(0.0, duration_s)
    last_estimate = tracker.last_estimate
    while time.perf_counter() < end_time:
        update_count = getattr(tracker, "_update_count", 0)
        last_estimate = tracker.update(
            update_count,
            robot_interface=robot_interface,
            kinematics=kinematics,
        )
        tracker._update_count = update_count + 1
        time.sleep(interval_s)
    return last_estimate
