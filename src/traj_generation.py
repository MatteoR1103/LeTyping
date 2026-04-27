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
            #print(f"Current end-effector position: {ee_sol_pos}, error: {err:.4f} m")
            if err<tol: 
                print("IK converged")
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
def generate_key_press_trajectory(
    key_pos: np.ndarray,
    q_current: np.ndarray,
    kinematics: RobotKinematics,
    hover_height: float = 0.05,
    press_depth: float = 0.005,
    hover_duration: float = 0.5,
    press_duration: float = 0.3,
    dt: float = 0.02,
    ik_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a joint-space trajectory for a single key-press action.
    The motion has four waypoints (all with zero velocity):
        q_current → q_hover → q_press → q_hover
    Where:
    * ``q_hover``  is the IK solution for the point *hover_height* above the key.
    * ``q_press``  is the IK solution for the key surface (shifted down by
      *press_depth* to account for key travel).
    Parameters
    ----------
    key_pos:
        3-D position of the key centre in the robot world frame (metres).
    q_current:
        Current joint configuration (n_joints,).
    kinematics:
        RobotKinematics instance loaded with the robot URDF.
    hover_height:
        Height in metres above the key for the hover pose.
    press_depth:
        Additional downward offset (metres) for the press pose.
    hover_duration:
        Duration (s) for each of the three motion segments between waypoints.
        If two floats are needed (approach vs. retract) extend as required.
    press_duration:
        Duration (s) of the press segment (hover → press → hover).
    dt:
        Control timestep in seconds.
    ik_kwargs:
        Extra keyword arguments forwarded to ``kinematics.inverse_kinematics``
        (e.g. ``position_weight``, ``orientation_weight``).
    Returns
    -------
    q_traj : np.ndarray, shape (T, n_joints)
        Joint-position trajectory.
    dq_traj : np.ndarray, shape (T, n_joints)
        Joint-velocity trajectory (first derivative of spline).
    t_exec : np.ndarray, shape (T,)
        Time stamps for each sample.
    """
    ik_kwargs = ik_kwargs or {}
    key_pos = np.asarray(key_pos, dtype=float)

    p_hover = key_pos + np.array([0.0, 0.0, hover_height])
    p_press = key_pos - np.array([0.0, 0.0, press_depth])
    
    q_arm_current = q_current[:4]
    orientation_joints = q_current[4:]

    q_arm_hover = kinematics.inverse_kinematics(q_arm_current, p_hover, **ik_kwargs) #degrees
    q_arm_press = kinematics.inverse_kinematics(q_arm_hover,   p_press, **ik_kwargs) #degrees


    # Segments: approach (current→hover) | press (hover→press) | retract (press→hover)
    t_approach = hover_duration
    t_press    = t_approach + press_duration
    t_retract  = t_press + hover_duration

    print(f"Current arm joints (deg): {q_arm_current}")
    print(f"Hover arm joints (deg): {q_arm_hover}")
    q_hover = np.concatenate((q_arm_hover, orientation_joints))
    q_press = np.concatenate((q_arm_press, orientation_joints))
    
    t_waypoints = np.array([0.0, t_approach, t_press, t_retract])
    q_waypoints = np.array([q_current, q_hover, q_press, q_hover])  # (4, n_joints)

    q_waypoints = np.deg2rad(q_waypoints) # convert to radians for spline

    n_joints = q_current.shape[0]
    splines = [
        CubicSpline(
            t_waypoints,
            q_waypoints[:, j],
            bc_type=((1, 0.0), (1, 0.0)),  # zero velocity at start and end
        )
        for j in range(n_joints)
    ]

    t_exec = np.arange(0.0, t_waypoints[-1] + dt * 0.5, dt)
    q_traj  = np.stack([s(t_exec)      for s in splines], axis=1)  # (T, n_joints)
    dq_traj = np.stack([s(t_exec, 1)   for s in splines], axis=1)  # (T, n_joints)
    
    return q_traj, dq_traj, t_exec

def debug_plot_trajectory(
    q_traj: np.ndarray, 
    dq_traj: np.ndarray, 
    t_exec: np.ndarray, 
    joint_names: list[str] = ALL_JOINT_NAMES
) -> None:
    """
    Plots the joint positions and velocities for debugging.
    Only executes if the global DEBUG_PLOT_TRAJECTORY flag is True.
    """
    if not DEBUG_PLOT_TRAJECTORY:
        return

    print("[Debug] Plotting trajectory... Close the window to continue execution.")
    
    n_joints = q_traj.shape[1]
    
    # Create a plot with 2 rows and 1 column
    _, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot Positions
    for j in range(n_joints):
        name = joint_names[j] if j < len(joint_names) else f"Joint {j}"
        axs[0].plot(t_exec, q_traj[:, j], label=name, linewidth=2)
        
    axs[0].set_ylabel("Position [rad]", fontsize=12)
    axs[0].set_title("Trajectory Debug: Joint Positions", fontsize=14)
    axs[0].grid(True, linestyle="--", alpha=0.7)
    # Put the legend outside the plot so it doesn't block the lines
    axs[0].legend(loc='center left', bbox_to_anchor=(1.0, 0.5)) 

    # Plot Velocities
    for j in range(n_joints):
        name = joint_names[j] if j < len(joint_names) else f"Joint {j}"
        axs[1].plot(t_exec, dq_traj[:, j], label=name, linewidth=2)
        
    axs[1].set_ylabel("Velocity [rad/s]", fontsize=12)
    axs[1].set_xlabel("Time [s]", fontsize=12)
    axs[1].set_title("Trajectory Debug: Joint Velocities", fontsize=14)
    axs[1].grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()


# if __name__ == "__main__":
# only needed for testing, but leaving here for now to avoid deleting code that might be useful later
