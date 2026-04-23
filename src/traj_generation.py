import sys
from pathlib import Path
sys.path.append("/home/rubinim/Desktop/final_group_project/lerobot/src")
from scipy.interpolate import CubicSpline, QuinticSpline
from lerobot.model import RobotModel

import numpy as np

kin = RobotKinematics("../cfg/arm_model/so101.urdf")
# maybe define the number of jonits

def create_pose(xyz):
    """Create a 4x4 homogeneous transformation matrix for a given position."""
    pose = np.eye(4)
    pose[:3, 3] = xyz
    return pose

# solve the imports problem
q_current = read_current_joint_positions()  # shape (n_joints,)


# we ignore the orientation because we just care about hovering there and going down, then back up
q_hover   = kin.inverse_kinematics(q_current, make_pose(p_hover),
                                  position_weight=1.0, orientation_weight=0.0)

q_press = kin.inverse_kinematics(q_hover, make_pose(p_press),
                                  position_weight=1.0, orientation_weight=0.0)

t_waypoints = np.array([0.0, 0.4, 0.7])
q_waypoints  = np.array([q_hover, q_press, q_hover])  # shape (3, n_joints)

# One spline per joint
splines = [CubicSpline(t_waypoints, q_waypoints[:, j],
                       bc_type=((1, 0.0), (1, 0.0)))   # constraint zero velocity at endpoints
           for j in range(n_joints)]

# Sample at control frequency 
dt = 0.02 # 
t_exec = np.arange(0, t_waypoints[-1], dt)

# Evaluate the splines to get the joint trajectories
q_traj   = np.stack([s(t_exec)   for s in splines], axis=1)  
dq_traj  = np.stack([s(t_exec, 1) for s in splines], axis=1) 