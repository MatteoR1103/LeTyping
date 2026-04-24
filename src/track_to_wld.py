import numpy as np
import cv2 as cv
import glob

#CALIBRATION PATHS 
RIGID_T_PATH = "camera_calib/rigid_transform.npy"
CAMERA_CALIB_PATH = "camera_calib/camera_calibration.npz"

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
PLANE_P0 = np.array([0.0, 0.0, 0.0])

def convert_to_ray(pixel: np.ndarray, T_WC: np.ndarray, K: np.ndarray = K) -> tuple[np.ndarray, np.ndarray]: 
  
  pixel_h = np.array([pixel[0],pixel[1], 1.0])
  K_inv = np.linalg.inv(K)
  R_WC = T_WC [:3,:3]
  t_WC = T_WC [:3,3]

  ray_c = K_inv @ pixel_h
  ray_w = R_WC @ ray_c 

  ray_w /= np.linalg.norm(ray_w)
  return t_WC, ray_w 

def find_intersection(plane_n: np.ndarray, plane_p0: np.ndarray, ray_o: np.ndarray, ray_d: np.ndarray, eps: float = 1e-5) -> np.ndarray: 
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

def trackForward(pixel_coord: np.ndarray, prevImg: np.ndarray, nextImg: np.ndarray) -> np.ndarray:
  """
  Module 2-Tracking across frames: track keypoint from previous frame to current one with KLT optical flow.

  Args:
      pixel_coord (np.ndarray): (2,) keypoint in previous frame
      img_prev (np.ndarray): (H,W) previous image
      img_curr (np.ndarray): (H,W) current image

  Returns:
      new_pixel_coord (np.ndarray): (2,) tracked keypoint in current frame
  """

  #Track keypoint from previous frame to next one

  klt_pixel_coord = pixel_coord[None, :].astype(np.float32)
  nextPt, status, _ = cv.calcOpticalFlowPyrLK(prevImg=prevImg,nextImg=nextImg, prevPts=klt_pixel_coord, nextPts=None, **KLT_PARAMS)
  
  return nextPt, status

def forward_kinematics()-> None: 
  raise NotImplementedError

def main()->None: 
  current_pixel = np.array([320, 240])
  plane_n = PLANE_N
  plane_p0 = PLANE_P0
  
  cap = cv.VideoCapture(CAMERA_NO)
  if not cap.isOpened():
    raise RuntimeError(f"Could not open camera {CAMERA_NO}")

  last_frame = None

  #VISUALIZATION LOOP
  while True:
    #READ CURRENT FRAME
    ok, frame = cap.read()
    if not ok:
      break
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    #TRACKING AND PLANE-INTERSECTION LOOP
    if last_frame is not None: 
      #TRACK PIXEL FORWARD
      new_pixel, status = trackForward(pixel_coord=current_pixel, prevImg=last_frame, nextImg=gray_frame)
      
      if status[0, 0] == 0: 
        print("KLT not able to track through")
        break 
      new_pixel = new_pixel[0]
      
      # FIND THE CORRESPONDING 3D POSITION
      # NEED FORWARD KINEMATICS HERE 
      # TAKE THE POSIITON OF THE GRIPPER AT THE CURRENT INSTANT AND CONCATENATE IT WITH T_GC

      #TODO: Implement the forward kinematics to get T_WG
      T_WC = T_WG @ T_GC
      ray_o, ray_d = convert_to_ray(new_pixel, T_WC=T_WC)
      x_threed, _, _ = find_intersection(plane_n=plane_n, plane_p0=plane_p0, ray_o=ray_o, ray_d=ray_d)
      #print(f"3D coordinates of the letter: {x_threed}")
      
      current_pixel = new_pixel
      cv.circle(frame, tuple(new_pixel.astype(int)), 2, (0, 0, 255), -1)
      cv.imshow("triangulation", frame)
      
    
    last_frame = gray_frame

    if cv.waitKey(1) & 0xFF == ord("q"):
      break

  cap.release()
  cv.destroyAllWindows()

if __name__ == "__main__":
  main()
