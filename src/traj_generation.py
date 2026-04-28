"""
Trajectory generation to press a key.
Pipeline:
  1. Given a 3-D keyboard-key position (robot world frame), compute a
     "hover" pose directly above the key and a "press" pose at key level.
  2. Solve IK for both configurations, ignoring orientation. 
  3. Interpolate current → hover → press → hover with a cubic spline
     whose endpoint velocities are zero so the arm stops smoothly.
  4. Return (q_traj, dq_traj, t_exec) ready for the PD + gravity-
     compensation controller


# NOTE: Cubic Splines might be overkill, a much easier approach would be that of using straight lines in joint space with a trapezoidal velocity profile. 
We will have to test this out and then consider switching to a simpler approach if the cubic spline interpolation is not satisfactory.
"""
#MERGING


from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

try:
    import pinocchio as pin
except ImportError as exc:
    raise SystemExit(
        "pinocchio is required. Install it with: conda install pinocchio -c conda-forge"
    ) from exc

try:
    from lerobot.model.kinematics import RobotKinematics as _LerobotKinematics
    _LEROBOT_AVAILABLE = True
except ImportError:
    _LerobotKinematics = None  # type: ignore[assignment,misc]
    _LEROBOT_AVAILABLE = False

# urdf path:
URDF_PATH = "cfg/arm_model/so101_new_calib.urdf"

# Joint names 
ARM_JOINT_NAMES: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    
]
ALL_JOINT_NAMES: list[str] = ARM_JOINT_NAMES + ["wrist_roll", "gripper"]
DEFAULT_EE_FRAME = "gripper_frame_link"
DEBUG_PLOT_TRAJECTORY = True # set to true if you want to see debug plots of the generated trajectories 

# ---------------------------------------------------------------------------
# RobotKinematics
# ---------------------------------------------------------------------------
class RobotKinematics:
    """Kinematics / dynamics wrapper for the SO-101.
    * **FK and IK** are delegated to lerobot's ``RobotKinematics`` when available, which gives a robust iterative IK solver.
    * **Gravity torques** are computed by pinocchio, which lerobot/placo does not provide.
    Parameters
    ----------
    urdf_path:
        Path to the SO-101 URDF. If *None* the class searches the default
        candidate paths defined at module level.
    ee_frame:
        Name of the end-effector frame in the URDF (used by pinocchio and
        passed to lerobot as ``target_frame_name``).
    arm_dof:
        Number of arm joints used for IK (gripper excluded). 
    """

    def __init__(
        self,
        urdf_path: str | Path | None = None,
        ee_frame: str = DEFAULT_EE_FRAME,
        arm_dof: int = len(ARM_JOINT_NAMES),
    ) -> None:
        urdf_path = self._resolve_urdf(urdf_path)
        self.model: pin.Model = pin.buildModelFromUrdf(str(urdf_path))
        self.data: pin.Data = self.model.createData()
        self.arm_dof = arm_dof
        self.n_joints = self.model.nq  # full DOF including gripper

        # Resolve end-effector frame id (pinocchio, used for gravity)
        if self.model.existFrame(ee_frame):
            self.ee_frame_id: int = self.model.getFrameId(ee_frame)
        else:
            self.ee_frame_id = self.model.nframes - 1
            print(
                f"[RobotKinematics] Frame '{ee_frame}' not found; "
                f"using frame id {self.ee_frame_id} instead."
            )

        # Use all joints for FK so wrist roll / gripper affect the measured pose,
        # but solve IK only over the arm joints.
        if _LEROBOT_AVAILABLE:
            self._fk = _LerobotKinematics(
                urdf_path=str(urdf_path),
                target_frame_name=ee_frame,
                joint_names=ALL_JOINT_NAMES,
            )
            self._ik = _LerobotKinematics(
                urdf_path=str(urdf_path),
                target_frame_name=ee_frame,
                joint_names=ARM_JOINT_NAMES,  # gripper excluded from IK
            )
        else:
            self._fk = None
            self._ik = None
            print(
                "[RobotKinematics] lerobot not available - "
                "forward_kinematics / inverse_kinematics will raise."
            )

    def neutral_configuration(self) -> np.ndarray:
        """Return the pinocchio neutral configuration (zeros for revolute)."""
        return pin.neutral(self.model)

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Return end-effector pose as a 4x4 matrix for configuration *q* (deg)."""
        if self._fk is None or self._ik is None:
            raise RuntimeError("lerobot is required for forward_kinematics.")

        q = np.asarray(q, dtype=float)
        if len(q) == len(ARM_JOINT_NAMES):
            return self._ik.forward_kinematics(q)
        return self._fk.forward_kinematics(q)

    def ee_position(self, q: np.ndarray) -> np.ndarray:
        """Return end-effector position (3,) for configuration *q* (rad)."""
        return self.forward_kinematics(q)[:3, 3].copy()

    def inverse_kinematics(
        self,
        q_init: np.ndarray,
        target_pos: np.ndarray,
        position_weight: float = 100.0,
        orientation_weight: float = 0.01,
        tol : float = 1e-3,
        max_iters: float = 20
    ) -> np.ndarray:
        """Position-only IK via lerobot's placo solver.
        Parameters
        ----------
        q_init:
            Initial joint configuration in **degrees** (n_joints,).
        target_pos:
            Desired end-effector position (3,) in metres.
        position_weight:
            Weight for the position constraint in placo.
        orientation_weight:
            Weight for the orientation constraint (0 = position-only).
        Returns
        -------
        q:
            Solution joint configuration in **degrees** (n_joints,).
        """
        if self._ik is None:
            raise RuntimeError("lerobot is required for inverse_kinematics.")

        # lerobot expects degrees; build a 4×4 target pose
        
        T_init = self.forward_kinematics(q_init) #expect degrees
        print(f"Initial end-effector position: {T_init[:3,3]}")
        downward_orientation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])  # gripper pointing down

        T_target = make_pose(target_pos, downward_orientation)
        q_sol_deg = q_init.copy()
        
        for _ in range(max_iters): 
            # print(f"IK iteration {_+1}/{max_iters}...")
            # print(f"Current solution (deg): {q_sol_deg}")
            q_sol_deg = self._ik.inverse_kinematics(
                q_sol_deg, T_target,
                position_weight = position_weight,
                orientation_weight = orientation_weight,
            )
            ee_sol_pos = self.forward_kinematics(q_sol_deg)[:3,3]
            err = np.linalg.norm(ee_sol_pos-T_target[:3,3])
            
            if err<tol: 
                print("IK converged")
                print(f"Final end-effector position: {ee_sol_pos}, error: {err:.4f} m")
                print(f"Target joints", q_sol_deg)
                break

        return q_sol_deg

    def gravity_torques(self, q: np.ndarray) -> np.ndarray:
        """Return the (n_joints,) gravity-compensation torque vector g(q).
        Uses pinocchio, which provides full rigid-body dynamics unlike placo.
        """
        return pin.computeGeneralizedGravity(self.model, self.data, q).copy()

    @staticmethod
    def _resolve_urdf(urdf_path: str | Path | None) -> Path:
        if urdf_path is not None:
            p = Path(urdf_path)
            if not p.is_file():
                raise FileNotFoundError(f"URDF not found: {p}")
            return p
        if URDF_PATH.is_file():
            return URDF_PATH
        raise FileNotFoundError(
            "Could not locate the SO-101 URDF. Place it at "
            f"{URDF_PATH} or pass urdf_path explicitly."
        )

def make_pose(xyz: np.ndarray, rot: np.ndarray | None = None) -> np.ndarray:
    """Build a 4x4 homogeneous transformation from a position (and optionally
    a 3x3 rotation matrix).  If *rot* is None the identity rotation is used."""
    T = np.eye(4)
    T[:3, 3] = xyz
    if rot is not None:
        T[:3, :3] = rot
    return T


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

def generate_travel_spline(
    q_start: np.ndarray, 
    q_end: np.ndarray, 
    v_start: np.ndarray, 
    v_end: np.ndarray, 
    duration: float, 
    dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a cubic spline between two points with specific boundary velocities.
    q_start, q_end: (n_joints,) start and end joint positions in radians
    v_start, v_end: (n_joints,) start and end joint velocities in radians/s
    duration: total time to execute the segment in seconds
    dt: time step for the output trajectory in seconds

    Returns: (q_traj, dq_traj, t_exec)
    """

    t_waypoints = np.array([0.0, duration])
    q_waypoints = np.array([q_start, q_end])
    n_joints = len(q_start)
    
    splines = [
        CubicSpline(
            t_waypoints,
            q_waypoints[:, j],
            bc_type=((1, v_start[j]), (1, v_end[j]))
        )
        for j in range(n_joints)
    ]
    
    t_exec = np.arange(0.0, duration + dt/2, dt)
    q_traj = np.stack([s(t_exec) for s in splines], axis=1)
    dq_traj = np.stack([s(t_exec, 1) for s in splines], axis=1)
    
    return q_traj, dq_traj, t_exec

def generate_press_trajectory(
    q_hover: np.ndarray, 
    q_press: np.ndarray, 
    duration: float, 
    dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a half-sine wave press motion.
    q_hover: (n_joints,) joint configuration for the hover pose in radians
    q_press: (n_joints,) joint configuration for the press pose in radians
    duration: total time to execute the press in seconds
    dt: time step for the output trajectory in seconds
    Returns: (q_traj, dq_traj, t_exec, v_strike)
    Where v_strike is the constant velocity needed to enter and exit the pressing smoothly.
    """
    t_exec = np.arange(0.0, duration + dt/2, dt)
    delta_q = q_press - q_hover
    
    # Position: q_hover + delta_q * sin(pi * t / duration)
    phase = (t_exec / duration) * np.pi
    q_traj = q_hover + np.outer(np.sin(phase), delta_q)    
    dq_traj = np.outer(np.cos(phase), delta_q) * (np.pi / duration)
    
    v_strike = delta_q * (np.pi / duration)
    
    return q_traj, dq_traj, t_exec, v_strike

def generate_typing_trajectory(
    key_positions: np.ndarray | list,
    q_current: np.ndarray,
    kinematics: RobotKinematics,
    hover_height: float = 0.05, # dummy value, will need to be tuned based on the actual keyboard geometry 
    press_depth: float = 0.005, # dummy value, will need to be tuned based on the actual key travel distance of the keyboard
    travel_duration: float = 0.8, # dummy value, will need to be tuned based on the actual travel speed of the robot between keys (should be made variable)
    press_duration: float = 0.3, # dummy value, will need to be tuned based on the actual key pressing speed of the robot
    dt: float = 0.02,
    ik_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a continuous, non-stop trajectory to type an entire word.
    key_positions: (n_keys, 3) array of target key positions in world coordinates
    q_current: (n_joints,) current joint configuration in radians
    kinematics: RobotKinematics instance for FK/IK
    hover_height: height above the key to hover before/after pressing (m)
    press_depth: depth below the key to press (m)
    travel_duration: time to move between keys (s)
    press_duration: time to execute the press motion for each key (s)
    dt: time step for the output trajectory (s)
    ik_kwargs: additional keyword arguments to pass to the IK solver (e.g. weights)
    Returns: (q_traj, dq_traj, t_exec) for the entire word, ready for execution.
    """
    ik_kwargs = ik_kwargs or {}
    key_positions = np.atleast_2d(key_positions) 
    
    q_hovers = []
    q_presses = []
    
    # lock the orientation joints (wrist_roll, gripper) to their current values
    q_arm_current = q_current[:4]
    orientation_joints = q_current[4:]
    
    print(f"\n[Planner] Pre-computing IK for {len(key_positions)} keys...")
    
    # Pre-calculate ALL Inverse Kinematics points before moving
    q_arm_seed = q_arm_current
    for key_pos in key_positions:
        p_hover = key_pos + np.array([0.0, 0.0, hover_height])
        p_press = key_pos - np.array([0.0, 0.0, press_depth])
        
        q_arm_h = kinematics.inverse_kinematics(q_arm_seed, p_hover, **ik_kwargs)
        q_arm_p = kinematics.inverse_kinematics(q_arm_h, p_press, **ik_kwargs)
        
        # Stitch orientation joints back on
        q_hovers.append(np.deg2rad(np.concatenate((q_arm_h, orientation_joints))))
        q_presses.append(np.deg2rad(np.concatenate((q_arm_p, orientation_joints))))        
        q_arm_seed = q_arm_h 

    # Avoids duplicating the overlapping timestamps
    q_all, dq_all, t_all = [], [], []
    def append_segment(q, dq, t):
        if not t_all:
            q_all.append(q)
            dq_all.append(dq)
            t_all.append(t)
        else:
            t_offset = t_all[-1][-1]
            q_all.append(q[1:])    # Drop the first frame to avoid a 0.0dt duplicate
            dq_all.append(dq[1:])
            t_all.append(t[1:] + t_offset)

    print("[Planner] Assembling continuous swooping splines...")
    
    # Pre-calculate the strike velocities for every key
    v_strikes = []
    for q_h, q_p in zip(q_hovers, q_presses):
        _, _, _, v_strike = generate_press_trajectory(q_h, q_p, press_duration, dt)
        v_strikes.append(v_strike)
        
    q_curr_rad = np.deg2rad(q_current)

    # Build the Trajectory
    for i in range(len(key_positions)):
        # TRAVEL PHASE
        if i == 0:
            # First key starts with no speed
            q_tr, dq_tr, t_tr = generate_travel_spline(
                q_curr_rad, q_hovers[i], np.zeros_like(q_curr_rad), v_strikes[i], travel_duration, dt
            )
        else:
            # Subsequent keys start with the strike velocity of the previous key
            q_tr, dq_tr, t_tr = generate_travel_spline(
                q_hovers[i-1], q_hovers[i], -v_strikes[i-1], v_strikes[i], travel_duration, dt
            )
        append_segment(q_tr, dq_tr, t_tr)
        # PRESS PHASE
        q_pk, dq_pk, t_pk, _ = generate_press_trajectory(q_hovers[i], q_presses[i], press_duration, dt)
        append_segment(q_pk, dq_pk, t_pk)

    # Final Braking Phase (Stop the bouncing)
    q_brk, dq_brk, t_brk = generate_travel_spline(
        q_hovers[-1], q_hovers[-1], -v_strikes[-1], np.zeros_like(q_curr_rad), 0.4, dt
    )
    append_segment(q_brk, dq_brk, t_brk)

    print("[Planner] Complete.")
    return np.concatenate(q_all), np.concatenate(dq_all), np.concatenate(t_all)


# if __name__ == "__main__":
# only needed for testing, but leaving here for now to avoid deleting code that might be useful later
