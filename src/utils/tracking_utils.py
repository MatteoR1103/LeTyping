import numpy as np
import cv2 as cv
import time
from collections import deque
from pathlib import Path

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


def read_robot_joints(robot) -> np.ndarray:
    """Return the robot joint positions from a lerobot-style observation dict."""
    obs = robot.get_observation()
    return np.array(
        [float(value) for key, value in obs.items() if key.endswith(".pos")],
        dtype=float,
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

def homography(H: np.ndarray, pixel_coord: np.ndarray, keyboard_height: float)->np.ndarray:
        """
        Return the world coordinate of a point using a Homography transform
        """
        pixel_h = np.array([pixel_coord[0], pixel_coord[1], 1.0])
        print(H)
        world_loc = H @ pixel_h
        world_loc /= world_loc[2]
        return np.array([world_loc[0],world_loc[1], keyboard_height])


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


def build_key_templates(
    initial_frame_gray: np.ndarray,
    initial_results: list,
    current_pixels: list[np.ndarray],
) -> dict:
    """
    Crop one template per localized key and store the template anchor offset.

    The anchor offset keeps the tracked point aligned with Gemini's selected
    center/point even when the template crop includes extra context.
    """
    templates = {}
    image_h, image_w = initial_frame_gray.shape[:2]
    for result, current_pixel in zip(initial_results, current_pixels):
        xmin, ymin, xmax, ymax = result.bounding_box
        xmin -= 10
        ymin -= 10
        xmax += 10
        ymax += 10
        xmin = max(0, min(image_w - 1, xmin))
        xmax = max(0, min(image_w - 1, xmax))
        ymin = max(0, min(image_h - 1, ymin))
        ymax = max(0, min(image_h - 1, ymax))
        if result.target_letter == "SPACE":
            centerx = (xmin + xmax) // 2
            centery = (ymin + ymax) // 2
            len_x = (xmax - xmin) // 2
            len_y = (ymax - ymin) // 2
            template = initial_frame_gray[centery - len_x:centery + len_x + 1, centerx - len_y:centerx + len_y]
        else:
            template = initial_frame_gray[ymin:ymax + 1, xmin:xmax + 1]
        anchor_offset = np.asarray(current_pixel, dtype=np.float32).reshape(2) - np.array([xmin, ymin], dtype=np.float32)
        templates[result.target_letter] = {
            "template": template,
            "anchor_offset": anchor_offset.astype(np.float32),
        }
    return templates


def save_initial_pixel_overlay(
    initial_frame: np.ndarray,
    initial_results: list,
    current_pixels: list[np.ndarray],
    *,
    output_dir: str | Path = "camera",
) -> Path:
    """Save a debug image showing all initial Gemini-localized key pixels."""
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
    output_path = Path(output_dir) / f"initial_gemini_pixels_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(output_path), annotated_initial)
    return output_path


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


def estimate_world_from_pixel(
    *,
    pixel: np.ndarray,
    T_WC: np.ndarray,
    origins_buffer: deque,
    directions_buffer: deque,
    ray_buffer_size: int,
    keyboard_p0: np.ndarray,
    plane_n: np.ndarray,
) -> np.ndarray:
    """
    Convert a tracked pixel into a world key estimate.

    Each call appends the current camera ray to the supplied per-letter buffers.
    Until the buffers are full, the estimate is a ray-plane intersection; once
    full, it uses least squares over the buffered rays with fixed keyboard
    height.
    """
    ray_o, ray_d = convert_to_ray(pixel, T_WC=T_WC)
    origins_buffer.append(ray_o)
    directions_buffer.append(ray_d)

    if len(origins_buffer) == ray_buffer_size:
        x_threed = update_LS(
            origins=list(origins_buffer),
            directions=list(directions_buffer),
            height=keyboard_p0[2],
        )
    else:
        x_threed, _, estimator_status = find_intersection(
            plane_n=plane_n,
            plane_p0=keyboard_p0,
            ray_o=ray_o,
            ray_d=ray_d,
        )
        if x_threed is None:
            raise RuntimeError(f"Ray-plane estimate failed: {estimator_status}.")

    return np.asarray(x_threed, dtype=float).reshape(3)


def store_target_state(
    targets_by_letter: dict[str, dict],
    letter: str,
    *,
    pixel: np.ndarray | None = None,
    world: np.ndarray | None = None,
) -> None:
    """Update the stored pixel/world estimate for one target letter."""
    target = targets_by_letter[letter]
    if pixel is not None:
        target["pixel"] = np.asarray(pixel, dtype=np.float32).reshape(2).copy()
    if world is not None:
        target["world"] = np.asarray(world, dtype=float).reshape(3).copy()


def activate_letter_buffers(
    origins_buffers_by_letter: dict[str, deque],
    directions_buffers_by_letter: dict[str, deque],
    letter: str,
    *,
    ray_buffer_size: int,
) -> tuple[deque, deque]:
    """Return the per-letter ray buffers, creating them if needed."""
    origins_buffer = origins_buffers_by_letter.setdefault(
        letter,
        deque(maxlen=ray_buffer_size),
    )
    directions_buffer = directions_buffers_by_letter.setdefault(
        letter,
        deque(maxlen=ray_buffer_size),
    )
    return origins_buffer, directions_buffer


def active_visual_letters(
    visual_track_pixels: dict[str, np.ndarray],
    active_cluster_letters: set[str],
) -> list[str]:
    """Return the letters that should be visually tracked on this frame."""
    if active_cluster_letters:
        return [
            letter
            for letter in visual_track_pixels
            if letter in active_cluster_letters
        ]
    return list(visual_track_pixels)


def pixel_inside_frame(pixel: np.ndarray, frame: np.ndarray) -> bool:
    """Check whether a 2-D pixel lies inside an image frame."""
    x, y = np.asarray(pixel, dtype=np.float32).reshape(2)
    return 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]


def track_pixel_on_frame(
    *,
    pixel: np.ndarray,
    prev_gray: np.ndarray,
    current_gray: np.ndarray,
) -> np.ndarray:
    """Track one letter pixel from the previous frame to the current frame with KLT."""
    tracked_pixel = np.asarray(pixel, dtype=np.float32).reshape(2)
    new_pixel, status = trackForward(
        pixel_coord=tracked_pixel,
        prevImg=prev_gray,
        nextImg=current_gray,
    )
    if status is not None and status[0, 0] != 0 and new_pixel is not None:
        return np.asarray(new_pixel[0], dtype=np.float32).reshape(2)

    return tracked_pixel.copy()


def update_visual_track_pixels(
    *,
    visual_track_pixels: dict[str, np.ndarray],
    active_cluster_letters: set[str],
    prev_gray: np.ndarray,
    current_gray: np.ndarray,
) -> None:
    """
    Advance the tracked pixels for all active visual letters.

    This updates pixels only; world estimates are updated separately by
    `estimate_visual_targets` after the current camera pose is known.
    """
    for letter in active_visual_letters(visual_track_pixels, active_cluster_letters):
        pixel = visual_track_pixels[letter]
        if not pixel_inside_frame(pixel, prev_gray):
            continue
        visual_track_pixels[letter] = track_pixel_on_frame(
            pixel=pixel,
            prev_gray=prev_gray,
            current_gray=current_gray,
        )


def estimate_visual_targets(
    *,
    visual_track_pixels: dict[str, np.ndarray],
    active_cluster_letters: set[str],
    origins_buffers_by_letter: dict[str, deque],
    directions_buffers_by_letter: dict[str, deque],
    targets_by_letter: dict[str, dict],
    T_WC: np.ndarray,
    frame: np.ndarray,
    ray_buffer_size: int,
    keyboard_p0: np.ndarray,
    plane_n: np.ndarray,
    skip_letter: str | None = None,
) -> None:
    """Update world estimates for active visual letters other than `skip_letter`."""
    for letter in active_visual_letters(visual_track_pixels, active_cluster_letters):
        pixel = visual_track_pixels[letter]
        if letter == skip_letter or not pixel_inside_frame(pixel, frame):
            continue
        origins_buffer, directions_buffer = activate_letter_buffers(
            origins_buffers_by_letter,
            directions_buffers_by_letter,
            letter,
            ray_buffer_size=ray_buffer_size,
        )
        world = estimate_world_from_pixel(
            pixel=pixel,
            T_WC=T_WC,
            origins_buffer=origins_buffer,
            directions_buffer=directions_buffer,
            ray_buffer_size=ray_buffer_size,
            keyboard_p0=keyboard_p0,
            plane_n=plane_n,
        )
        store_target_state(targets_by_letter, letter, pixel=pixel, world=world)


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


def draw_visual_track_pixels(frame: np.ndarray, visual_track_pixels: dict[str, np.ndarray]) -> None:
    """Draw all currently maintained visual key pixels on a preview frame."""
    for letter, pixel in visual_track_pixels.items():
        if not pixel_inside_frame(pixel, frame):
            continue
        center = tuple(np.round(pixel).astype(int))
        cv.circle(frame, center, 4, (255, 0, 0), -1)
        cv.putText(
            frame,
            str(letter),
            (center[0] + 7, center[1] - 7),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv.LINE_AA,
        )


def activate_maintained_target_state(tracker, letter: str, *, world: np.ndarray | None = None) -> None:
    """
    Make a previously maintained letter the active tracker target.

    This is used when the robot is already at a hover position or when a
    retracked cluster has just been built. It switches the tracker's active
    buffers/pixel and optionally pins the world estimate to the commanded hover
    position.
    """
    target = tracker.targets_by_letter[letter]
    tracker.letter = letter
    tracker.origins_buffer, tracker.directions_buffer = activate_letter_buffers(
        tracker.origins_buffers_by_letter,
        tracker.directions_buffers_by_letter,
        letter,
        ray_buffer_size=tracker.ray_buffer_size,
    )
    pixel = tracker.visual_track_pixels.get(letter, target.get("pixel"))
    if world is None:
        world = target.get("world")
    if pixel is not None:
        tracker.current_pixel = np.asarray(pixel, dtype=np.float32).reshape(2).copy()
    if world is not None:
        tracker.last_estimate = np.asarray(world, dtype=float).reshape(3).copy()
        store_target_state(tracker.targets_by_letter, letter, world=tracker.last_estimate)


def retrack_targets_from_current_frame(
    tracker,
    letters: list[str],
    *,
    robot_interface,
    kinematics,
    window_name: str = DEFAULT_TRACKING_WINDOW_NAME,
    debug_viz: bool = True,
) -> None:
    """
    Re-localize a group of remaining letters from the current home-view frame.

    Each letter is refreshed from the initial home pixel with KLT, then local
    template matching, receives fresh ray buffers, and gets a new world estimate
    from the current camera pose. This resets the maintained state used before
    building the next cluster.
    """
    frame = read_frame(tracker.cap, error_message="Camera stream ended or returned no frame.")
    current_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    joints = read_robot_joints(robot_interface.robot)
    T_WG = kinematics.forward_kinematics(joints)
    T_WC = T_WG @ tracker.camera_transform

    for letter in dict.fromkeys(letters):
        template_info = tracker.templates[letter]
        target = tracker.targets_by_letter[letter]
        current_pixel = np.asarray(
            target.get("initial_pixel", target["pixel"]),
            dtype=np.float32,
        ).reshape(2).copy()

        prev_target_gray = tracker.initial_frame_gray if tracker.initial_frame_gray is not None else tracker.last_frame
        new_pixel, status = trackForward(
            pixel_coord=current_pixel,
            prevImg=prev_target_gray,
            nextImg=current_gray,
        )
        if status is not None and status[0, 0] != 0 and new_pixel is not None:
            current_pixel = np.asarray(new_pixel[0], dtype=np.float32).reshape(2)

        refreshed_pixel = template_match(
            template_info=template_info,
            current_gray=current_gray,
            current_pixel=current_pixel,
            matching_roi=tracker.matching_roi,
        )
        origins_buffer = deque(maxlen=tracker.ray_buffer_size)
        directions_buffer = deque(maxlen=tracker.ray_buffer_size)
        refreshed_world = estimate_world_from_pixel(
            pixel=refreshed_pixel,
            T_WC=T_WC,
            origins_buffer=origins_buffer,
            directions_buffer=directions_buffer,
            ray_buffer_size=tracker.ray_buffer_size,
            keyboard_p0=tracker.keyboard_p0,
            plane_n=tracker.plane_n,
        )

        tracker.visual_track_pixels[letter] = refreshed_pixel.copy()
        tracker.origins_buffers_by_letter[letter] = origins_buffer
        tracker.directions_buffers_by_letter[letter] = directions_buffer
        store_target_state(tracker.targets_by_letter, letter, pixel=refreshed_pixel, world=refreshed_world)
        # print(
        #     f"Retracked {letter} from home: "
        #     f"pixel=({refreshed_pixel[0]:.1f}, {refreshed_pixel[1]:.1f}), "
        #     f"world=({refreshed_world[0]:.4f}, {refreshed_world[1]:.4f}, {refreshed_world[2]:.4f})"
        # )

    tracker.last_frame = current_gray

    if debug_viz:
        preview = frame.copy()
        draw_visual_track_pixels(preview, tracker.visual_track_pixels)
        cv.imshow(window_name, preview)
        cv.waitKey(1)


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
