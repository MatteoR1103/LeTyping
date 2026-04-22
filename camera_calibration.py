import numpy as np
import cv2 as cv
import glob

K = np.array([
              [686.8923, 0,        955.3188],
              [0,       685.0062, 542.2074],
              [0,        0,        1]
            ])

def convert_to_ray(pixel: np.ndarray, T_WC: np.ndarray, K: np.ndarray = K) -> tuple[np.ndarray, np.ndarray]: 
  
  pixel_h = np.array([pixel[0],pixel[1], 1.0])
  K_inv = np.linalg.inv(K)
  R_WC = T_WC [:3,:3]
  t_WC = T_WC [:3,3]

  ray_c = K_inv @ pixel_h
  print("Ray in camera frame")
  print(ray_c/np.linalg.norm(ray_c))
  ray_w = R_WC @ ray_c 

  ray_w /= np.linalg.norm(ray_w)
  print("Ray in world frame")
  print(ray_w)
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

def main()->None: 
  pixel = np.array([1119, 113])
  plane_n = np.array([0,0,1.0])
  plane_p0 = np.array([0.0, 0.0, 0.0])
  
  T_WC = np.array([ [1.0, 0.0, 0.0, 0.0], 
                    [0.0, -1.0, 0.0, 0.0], 
                    [0.0, 0.0, -1.0, 21.0], 
                    [0.0, 0.0, 0.0, 1.0]])
  
  ray_o, ray_d = convert_to_ray(pixel, T_WC=T_WC)
  x_threed, _, _ = find_intersection(plane_n=plane_n, plane_p0=plane_p0, ray_o=ray_o, ray_d=ray_d)
  
  print(f"3D coordinates of the letter: {x_threed}")

if __name__ == "__main__":
  main()

