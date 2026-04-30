import numpy as np
import cv2 as cv

CAMERA_CALIB_PATH = "camera_calib/calibrations/camera_calibration.npz"
camera_intrinsics = np.load(CAMERA_CALIB_PATH)
K = camera_intrinsics["camera_matrix"]
dist = camera_intrinsics["dist_coeffs"]

KLT_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.001),
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

def homography(self, H: np.ndarray, pixel_coord: np.ndarray, keyboard_height: float)->np.ndarray:
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