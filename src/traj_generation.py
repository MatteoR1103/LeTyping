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
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from scipy.interpolate import CubicSpline


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

try:
    from .utils.tracking_utils import show_tracker_current_frame
except ImportError:
    from utils.tracking_utils import show_tracker_current_frame

if TYPE_CHECKING:
    try:
        from .controller import SO101Interface, execute_joint_trajectory
        from .tracker import KeyWorldTracker
        from main_pipeline import DEFAULT_URDF_PATH
    except ImportError:
        from controller import SO101Interface, execute_joint_trajectory
        from tracker import KeyWorldTracker
        from main_pipeline import DEFAULT_URDF_PATH

# Joint names 
ARM_JOINT_NAMES: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    
]
ALL_JOINT_NAMES: list[str] = ARM_JOINT_NAMES + ["wrist_roll", "gripper"]
DEFAULT_EE_FRAME = "gripper_frame_link"

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
        orientation_weight: float = 0.15,
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
        #print(f"Initial end-effector position: {T_init[:3,3]}")
        downward_orientation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])  # gripper pointing down

        T_target = make_pose(target_pos, downward_orientation)
        q_sol_deg = q_init.copy()
        
        for _ in range(max_iters): 

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
        if DEFAULT_URDF_PATH.is_file():
            return DEFAULT_URDF_PATH
        raise FileNotFoundError(
            "Could not locate the SO-101 URDF. Place it at "
            f"{DEFAULT_URDF_PATH} or pass urdf_path explicitly."
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

def generate_point_to_point_trajectory(
    target_pos: np.ndarray,
    q_current: np.ndarray,
    kinematics: RobotKinematics,
    duration: float,
    dt: float = 0.02,
    position_weight: float = 100.0,
    orientation_weight: float = 0.15,
    q_target: np.ndarray = np.zeros(6),
    override_pos: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """q_current is in degrees for IK; returned trajectory is in radians."""
    q_arm_current = q_current[:4]
    orientation_joints = q_current[4:]

    q_arm_target = kinematics.inverse_kinematics(
                                                q_init = q_arm_current,
                                                target_pos = target_pos,
                                                position_weight = position_weight,
                                                orientation_weight = orientation_weight
                                                )
    q_current_rad = np.deg2rad(q_current)
    
    # Used to go to an explicit joint configuration without passing through IK.
    if override_pos:
        q_target_rad = np.deg2rad(np.asarray(q_target, dtype=float).reshape(-1))
    else:
        q_target_rad = np.deg2rad(np.concatenate((q_arm_target, orientation_joints)))
    zero_velocity = np.zeros_like(q_current_rad)

    return generate_travel_spline(
        q_current_rad,
        q_target_rad,
        zero_velocity,
        zero_velocity,
        duration,
        dt,
    )

def deliver_typing_trajectory(
    key_position: np.ndarray,
    tracker: KeyWorldTracker,
    robot_interface: SO101Interface,
    q_current: np.ndarray,
    kinematics: RobotKinematics,
    hover_height: float = 0.03,
    press_depth: float = 0.01, 
    travel_duration: float = 0.8, # dummy value, will need to be tuned based on the actual travel speed of the robot between keys (should be made variable)
    press_duration: float = 0.3, # dummy value, will need to be tuned based on the actual key pressing speed of the robot
    dt: float = 0.02,
    position_weight: float = 100.0,
    orientation_weight: float = 0.15,
    start_from_hover: bool = False,
    q_final_config: np.ndarray | Callable[[], np.ndarray | None] | None = None,
) -> None:
    """
    High-level function to generate and execute a full trajectory for typing a key, consisting of:
    1. Hovering above the key
    2. Pressing down on the key
    3. Coming back up to the hover position
    Parameters:
    - key_position: (3,) position of the key in world coordinates
    - robot_interface: instance of SO101Interface to send commands to the robot
    - q_current: (n_joints,) current joint configuration in degrees
    - kinematics: instance of RobotKinematics for FK/IK computations
    - hover_height: height above the key to hover before and after pressing (in metres)
    - press_depth: depth to press down below the key plane (in metres)
    - travel_duration: duration of the hover → press and press → hover segments (in seconds)
    - press_duration: duration of the hover → press segment (in seconds)
    - dt: time step for the generated trajectory (in seconds)
    - position_weight: weight for the position constraint in IK
    - orientation_weight: weight for the orientation constraint in IK
    """
    # local import to prevent circular dependencies with main_pipeline
    try:
        from .controller import execute_joint_trajectory
    except ImportError:
        from controller import execute_joint_trajectory


    p_hover = key_position + np.array([0.0, 0.0, hover_height])

    def update_tracker(i) -> None:
        updated_key_pos = tracker.update(i, robot_interface=robot_interface, kinematics=kinematics)
        if i % 10 == 0:
            print(f"Tracked key_pos in world by LS: {updated_key_pos}")

    def show_tracker_frame(_: int) -> None:
        show_tracker_current_frame(tracker, tracking_status="holding")

    if not start_from_hover:
        q_traj, dq_traj, t_exec = generate_point_to_point_trajectory(
            target_pos=p_hover,
            q_current=q_current,
            kinematics=kinematics,
            duration=travel_duration,
            dt=dt,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
        ) #in radians
        print(f"Generated hover trajectory length: {len(t_exec)} samples")
        print("Starting hover trajectory execution.")
        execute_joint_trajectory(
            robot_interface=robot_interface,
            q_traj=q_traj, #radians
            dq_traj=dq_traj, #radians/s
            t_exec=t_exec,
            kinematics=kinematics,
            key_pos=p_hover,
            step_callback=update_tracker,
            hold_callback=show_tracker_frame,
        )


    #-------------------PREPRESS TRAJECTORY-------------------#
    q_current = np.rad2deg(robot_interface.read_joints()[0])

    if tracker.last_estimate is not None:
        key_position = tracker.last_estimate.copy()

    # press_depth=0.0 means descend exactly to the estimated key position.
    p_pre_press = key_position 
    q_traj, dq_traj, t_exec = generate_point_to_point_trajectory(
        target_pos=p_pre_press,
        q_current=q_current,
        kinematics=kinematics,
        duration=press_duration,
        dt=dt,
        position_weight=position_weight,
        orientation_weight=orientation_weight,
    ) #in radians

    print(f"Generated pre-press trajectory length: {len(t_exec)} samples")
    print("Starting pre-press trajectory execution.")
    execute_joint_trajectory(
        robot_interface=robot_interface,
        q_traj=q_traj, #radians
        dq_traj=dq_traj, #radians/s
        t_exec=t_exec,
        kinematics=kinematics,
        key_pos=p_pre_press,
        step_callback=update_tracker,
        hold_callback=show_tracker_frame,
        hold_time = 0.1
    )
    
    #-------------------PRESS TRAJECTORY-------------------#
    q_current = np.rad2deg(robot_interface.read_joints()[0])

    if tracker.last_estimate is not None:
        key_position = tracker.last_estimate.copy()

    # press_depth=0.0 means descend exactly to the estimated key position.
    p_press = key_position - np.array([0.0, 0.0, press_depth])
    q_traj, dq_traj, t_exec = generate_point_to_point_trajectory(
        target_pos=p_press,
        q_current=q_current,
        kinematics=kinematics,
        duration=press_duration,
        dt=dt,
        position_weight=position_weight,
        orientation_weight=orientation_weight,
    ) #in radians

    print(f"Generated descent trajectory length: {len(t_exec)} samples")
    print("Starting descent trajectory execution.")
    execute_joint_trajectory(
        robot_interface=robot_interface,
        q_traj=q_traj, #radians
        dq_traj=dq_traj, #radians/s
        t_exec=t_exec,
        kinematics=kinematics,
        key_pos=p_press,
        step_callback=update_tracker,
        hold_callback=show_tracker_frame,
        hold_time = 0.1
    )
    
    
    #-------------------FINAL TRAJECTORY-------------------#
    q_current = np.rad2deg(robot_interface.read_joints()[0])
    resolved_final_config = q_final_config() if callable(q_final_config) else q_final_config
    if resolved_final_config is None:
        q_traj, dq_traj, t_exec = generate_point_to_point_trajectory(
            target_pos=p_hover,
            q_current=q_current,
            kinematics=kinematics,
            duration=travel_duration,
            dt=dt,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
        ) #in radians
        trajectory_key_pos = p_hover
        hold_time = 0.2
    else:
        q_traj, dq_traj, t_exec = generate_point_to_point_trajectory(
            target_pos=np.zeros(3),
            q_current=q_current,
            kinematics=kinematics,
            duration=travel_duration,
            dt=dt,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
            q_target=resolved_final_config,
            override_pos=True,
        ) #in radians
        trajectory_key_pos = np.zeros(3)
        hold_time = 0.5

    execute_joint_trajectory(
        robot_interface=robot_interface,
        q_traj=q_traj, #radians
        dq_traj=dq_traj, #radians/s
        t_exec=t_exec,
        kinematics=kinematics,
        key_pos=trajectory_key_pos,
        step_callback=update_tracker,
        hold_callback=show_tracker_frame,
        hold_time=hold_time,
    )
