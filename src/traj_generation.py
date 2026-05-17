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
from typing import TYPE_CHECKING
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
        from .controller import SO101Interface
        from .tracker import KeyWorldTracker
        from main_pipeline import DEFAULT_URDF_PATH
    except ImportError:
        from controller import SO101Interface
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
DEFAULT_PRESS_EE_FRAME = "key_contact_frame_link"
DEFAULT_URDF_PATH = Path("cfg/arm_model/so101_new_calib.urdf")

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
                # print("IK converged")
                # print(f"Final end-effector position: {ee_sol_pos}, error: {err:.4f} m")
                # print(f"Target joints", q_sol_deg)
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

def duration_from_cartesian_distance(
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    *,
    speed: float,
    min_duration: float,
    max_duration: float | None = None,
) -> float:
    """
    Choose a segment duration from Cartesian distance and speed limits.
    Inputs:
    - start_pos: (3,) start position in metres
    - target_pos: (3,) target position in metres
    - speed: Cartesian speed in m/s
    - min_duration: minimum duration for the segment in seconds
    - max_duration: maximum duration for the segment in seconds (optional)
    Returns:
    - duration: segment duration in seconds"""
    if speed <= 0.0:
        raise ValueError("speed must be positive.")
    if min_duration <= 0.0:
        raise ValueError("min_duration must be positive.")

    distance = float(np.linalg.norm(np.asarray(target_pos, dtype=float) - np.asarray(start_pos, dtype=float)))
    duration = max(min_duration, distance / speed)
    if max_duration is not None:
        duration = min(duration, max(max_duration, min_duration))
    return duration


def current_robot_state(
    robot_interface: SO101Interface,
    kinematics: RobotKinematics,
) -> tuple[np.ndarray, np.ndarray]:
    """Read current joints in degrees and end-effector position in world frame."""
    q_current = np.rad2deg(robot_interface.read_joints()[0])
    ee_position = np.asarray(kinematics.forward_kinematics(q_current)[:3, 3], dtype=float).reshape(3)
    return q_current, ee_position


def execute_segment(
    *,
    label: str,
    target_pos: np.ndarray,
    robot_interface: SO101Interface,
    kinematics: RobotKinematics,
    segment_speed: float,
    max_duration: float,
    min_segment_duration: float,
    dt: float,
    position_weight: float,
    orientation_weight: float,
    hold_time: float = 0.1,
    step_callback=None,
    hold_callback=None,
    override_q_target: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate and execute one spline segment from the current robot state.
    Inputs: 
    - label: name for logging the segment
    - target_pos: (3,) target position for the end-effector in world frame (ignored if override_q_target is provided)
    - robot_interface: instance of SO101Interface to send commands to the robot
    - kinematics: instance of RobotKinematics for FK/IK computations
    - segment_speed: Cartesian speed in m/s used to choose segment duration from distance
    - max_duration: maximum duration for the segment in seconds
    - min_segment_duration: minimum duration for the segment in seconds to ensure smoothness
    - dt: time step for the generated trajectory in seconds
    - position_weight: weight for the position constraint in IK
    - orientation_weight: weight for the orientation constraint in IK
    - hold_time: time to hold at the end of the segment after reaching the target (in seconds)
    - step_callback: optional function to call at each control step during execution, with signature step_callback(step_index: int) -> None
    - hold_callback: optional function to call repeatedly during the hold time after reaching the target, with signature hold_callback(step_index: int) -> None
    - override_q_target: if provided, a (n_joints,) target joint configuration in degrees to go to instead of using IK to reach target_pos. If this is not None,
    """
    try:
        from .controller import execute_trajectory
    except ImportError:
        from controller import execute_trajectory

    q_now, ee_now = current_robot_state(robot_interface, kinematics)
    if override_q_target is None:
        target_for_duration = np.asarray(target_pos, dtype=float).reshape(3)
        duration = duration_from_cartesian_distance(
            ee_now,
            target_for_duration,
            speed=segment_speed,
            min_duration=min_segment_duration,
            max_duration=max_duration,
        )
        q_traj, dq_traj, t_exec = generate_point_to_point_trajectory(
            target_pos=target_for_duration,
            q_current=q_now,
            kinematics=kinematics,
            duration=duration,
            dt=dt,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
        )
        key_pos_for_log = target_for_duration
    else:
        q_target = np.asarray(override_q_target, dtype=float).reshape(-1)
        q_target_rad = np.deg2rad(q_target)
        current_rad = np.deg2rad(q_now)
        joint_distance = float(np.max(np.abs(q_target_rad - current_rad)))
        duration = min(
            max(max_duration, min_segment_duration),
            max(min_segment_duration, joint_distance / 0.9),
        )
        q_traj, dq_traj, t_exec = generate_point_to_point_trajectory(
            target_pos=np.zeros(3),
            q_current=q_now,
            kinematics=kinematics,
            duration=duration,
            dt=dt,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
            q_target=q_target,
            override_pos=True,
        )
        key_pos_for_log = np.zeros(3)

    # print(f"Generated {label} trajectory length: {len(t_exec)} samples, duration={duration:.3f}s")
    print(f"Starting {label} trajectory execution.")
    if label == "descent":
        print(f"Target {label} position: {key_pos_for_log}")
    execute_trajectory(
        robot_interface=robot_interface,
        q_traj=q_traj,
        dq_traj=dq_traj,
        t_exec=t_exec,
        kinematics=kinematics,
        key_pos=key_pos_for_log,
        step_callback=step_callback,
        hold_callback=hold_callback,
        hold_time=hold_time,
        label=label,
    )
    return current_robot_state(robot_interface, kinematics)


def deliver_typing_trajectory(
    key_position: np.ndarray,
    tracker: KeyWorldTracker,
    robot_interface: SO101Interface,
    kinematics: RobotKinematics,
    tracking_kinematics: RobotKinematics | None = None,
    hover_height: float = 0.04,
    press_depth: float = 0.014, 
    travel_duration: float = 0.8,
    press_duration: float = 0.3, 
    dt: float = 0.02,
    position_weight: float = 100.0,
    orientation_weight: float = 0.15,
    q_final_config: np.ndarray | None = None,
    track_during_hover: bool = True,
    lock_key_position: bool = False,
    approach_speed: float = 0.06,
    press_speed: float = 0.04,
    min_segment_duration_default: float = 0.4,
    max_refine_steps: int = 3,
    refine_xy_threshold: float = 0.002,
    estimate_stability_threshold: float = 0.002,
    estimate_stability_window: int = 3,
    default_hold_time: float = 0.1,
    shorter_segment_duration: float = 0.1,
) -> np.ndarray:
    """
    High-level function to generate and execute a full trajectory for typing a key, consisting of:
    1. Hovering above the key
    2. Pressing down on the key
    3. Coming back up to the hover position
    Parameters:
    - key_position: (3,) position of the key in world coordinates
    - robot_interface: instance of SO101Interface to send commands to the robot
    - kinematics: RobotKinematics instance for pressing/contact FK and IK
    - tracking_kinematics: RobotKinematics instance for the calibrated camera frame;
      defaults to kinematics for backward compatibility
    - hover_height: height above the key to hover before and after pressing (in metres)
    - press_depth: depth to press down below the key plane (in metres)
    - travel_duration: duration of the hover → press and press → hover segments (in seconds)
    - press_duration: duration of the hover → press segment (in seconds)
    - dt: time step for the generated trajectory (in seconds)
    - position_weight: weight for the position constraint in IK
    - orientation_weight: weight for the orientation constraint in IK
    - track_during_hover: if True, update the tracker during hover approach
      and refinement. If False, keep using the supplied maintained estimate.
    - lock_key_position: if True, reuse the supplied key position for all phases
      instead of updating it from the tracker between phases
    - approach_speed/press_speed: Cartesian speeds used to choose segment
      duration from distance; travel_duration and press_duration are retained
      as maximum durations for the corresponding segment types
    - min_segment_duration_default: minimum duration for any segment to ensure smoothness
    - shorter_segment_duration: a shorter minimum duration to use for hover refinement segments after the first one, since they should be shorter 
    - max_refine_steps: maximum number of hover → hover refinement iterations
    - refine_xy_threshold: if the end-effector is within this distance of the key in XY and the estimate is stable, stop refining and proceed to press
    - estimate_stability_threshold: if the recent estimates are within this distance of their median, consider the estimate stable
    - estimate_stability_window: number of recent estimates to consider for stability checking
    """
    tracking_kinematics = tracking_kinematics or kinematics

    def update_tracker(i) -> None:
        updated_key_pos = tracker.update(i, robot_interface=robot_interface, kinematics=tracking_kinematics)
        if i % 50 == 0:
            print(f"Tracked key_pos in world by LS: {updated_key_pos}")

    def show_tracker_frame(_: int) -> None:
        show_tracker_current_frame(tracker, tracking_status="holding")

    def log_maintained_world_positions() -> None:
        active_letters = getattr(tracker, "active_cluster_letters", set())
        if not active_letters:
            return

        print("Maintained tracker world positions at hover for active cluster:")
        for letter in sorted(active_letters):
            target = tracker.targets_by_letter.get(letter)
            world = None if target is None else target.get("world")
            if world is None:
                print(f"  {letter}: unavailable")
                continue

            world = np.asarray(world, dtype=float).reshape(3)
            print(f"  {letter}: ({world[0]:.4f}, {world[1]:.4f}, {world[2]:.4f})")

    step_callback = update_tracker if track_during_hover else None

    def maybe_update_key_position(current_key_position: np.ndarray) -> np.ndarray:
        if track_during_hover and not lock_key_position and tracker.last_estimate is not None:
            return np.asarray(tracker.last_estimate, dtype=float).reshape(3).copy()
        return np.asarray(current_key_position, dtype=float).reshape(3).copy()

    #-------------------ADAPTIVE APPROACH / HOVER REFINEMENT-------------------#
    estimate_history: list[np.ndarray] = []
    first_target = True
    max_refine_steps = 1 if (lock_key_position or not track_during_hover) else max(1, int(max_refine_steps))

    for refine_index in range(max_refine_steps):
        key_position = maybe_update_key_position(key_position)
        estimate_history.append(key_position.copy())
        if len(estimate_history) > estimate_stability_window:
            estimate_history.pop(0)

        hover_scale = 1.5 if first_target and track_during_hover else 1.0
        target_hover = key_position + np.array([0.0, 0.0, hover_scale * hover_height])
        _, ee_position = current_robot_state(robot_interface, kinematics)
        xy_error = float(np.linalg.norm(ee_position[:2] - key_position[:2]))
        estimate_stable = False
        stability_samples = min(max(2, estimate_stability_window), max_refine_steps)
        if len(estimate_history) >= stability_samples:
            recent_xy = np.array([estimate[:2] for estimate in estimate_history])
            estimate_stable = float(np.max(np.linalg.norm(recent_xy - np.median(recent_xy, axis=0), axis=1))) <= estimate_stability_threshold

        if not first_target and xy_error <= refine_xy_threshold and estimate_stable:
            print(
                "Hover refinement converged: "
                f"xy_error={xy_error:.4f}m, stable={estimate_stable}."
            )
            break

        label = "pre-hover" if first_target else f"hover-refine-{refine_index}"
        min_segment_duration = min_segment_duration_default if first_target else shorter_segment_duration 
        execute_segment(
            label=label,
            target_pos=target_hover,
            robot_interface=robot_interface,
            kinematics=kinematics,
            segment_speed=approach_speed,
            max_duration=travel_duration,
            min_segment_duration=min_segment_duration,
            dt=dt,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
            hold_time=default_hold_time,
            step_callback=step_callback,
            hold_callback=show_tracker_frame,
        )
        first_target = False

    log_maintained_world_positions()

    # Freeze the refined estimate before contact phases. 
    key_position = maybe_update_key_position(key_position)
    frozen_press_key_position = key_position.copy()

    #-------------------PREPRESS TRAJECTORY-------------------#
    key_position = frozen_press_key_position

    # press_depth=0.0 means descend exactly to the estimated key position.
    p_pre_press = key_position + np.array([0.0, 0.0, press_depth/2])
    execute_segment(
        label="pre-press",
        target_pos=p_pre_press,
        robot_interface=robot_interface,
        kinematics=kinematics,
        segment_speed=press_speed,
        max_duration=press_duration,
        min_segment_duration=min_segment_duration_default,
        dt=dt,
        position_weight=position_weight,
        orientation_weight=orientation_weight,
        hold_time=default_hold_time,
        hold_callback=show_tracker_frame,
    )
    
    #-------------------PRESS TRAJECTORY-------------------#
    key_position = frozen_press_key_position

    # press_depth=0.0 means descend exactly to the estimated key position.
    press_key_position = key_position.copy()
    p_press = key_position - np.array([0.0, 0.0, press_depth])
    execute_segment(
        label="descent",
        target_pos=p_press,
        robot_interface=robot_interface,
        kinematics=kinematics,
        segment_speed=press_speed,
        max_duration=press_duration,
        min_segment_duration=min_segment_duration_default,
        dt=dt,
        position_weight=position_weight,
        orientation_weight=orientation_weight,
        hold_time=default_hold_time,
        hold_callback=show_tracker_frame,
    )
    
    
    #-------------------FINAL TRAJECTORY-------------------#
    p_hover_back = press_key_position + np.array([0.0, 0.0, hover_height])
    final_hold_time = 5*default_hold_time if q_final_config is not None else 2*default_hold_time
    if q_final_config is None:
        execute_segment(
            label="hover-back",
            target_pos=p_hover_back,
            robot_interface=robot_interface,
            kinematics=kinematics,
            segment_speed=approach_speed,
            max_duration=travel_duration,
            min_segment_duration=min_segment_duration_default,
            dt=dt,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
            hold_time=final_hold_time,
            hold_callback=show_tracker_frame,
        )
    else:
        execute_segment(
            label="final",
            target_pos=np.zeros(3),
            robot_interface=robot_interface,
            kinematics=kinematics,
            segment_speed=approach_speed,
            max_duration=travel_duration,
            min_segment_duration=min_segment_duration_default,
            dt=dt,
            position_weight=position_weight,
            orientation_weight=orientation_weight,
            hold_time=final_hold_time,
            hold_callback=show_tracker_frame,
            override_q_target=q_final_config,
        )
    return press_key_position.copy()


def go_home(
    robot_interface: SO101Interface,
    kinematics: RobotKinematics,
    q_home_rad: np.ndarray,
    dt: float = 0.02,
    minimum_duration: float = 0.5, # heuristic
    joint_speed: float = 0.9, # heuristic
) -> None:
    """
    Move the robot to the provided home configuration with a smooth spline trajectory.
    robot_interface: instance of SO101Interface to send commands to the robot
    kinematics: instance of RobotKinematics for FK/IK computations
    q_home_rad: target home configuration in radians
    dt: time step for the generated trajectory (in seconds)
    minimum_duration: minimum duration for the trajectory to ensure smoothness (in seconds)
    joint_speed: approximate speed in radians/s to choose trajectory duration from distance (heuristic)
    """
    try:
        from .controller import execute_trajectory
    except ImportError:
        from controller import execute_trajectory

    q_current, _ = current_robot_state(robot_interface, kinematics)
    q_current_rad = np.deg2rad(q_current)
    joint_distance = float(np.max(np.abs(q_home_rad - q_current_rad)))
    duration = max(minimum_duration, joint_distance / joint_speed)
    q_traj, dq_traj, t_exec = generate_travel_spline(
        q_start=q_current_rad,
        q_end=q_home_rad,
        v_start=np.zeros_like(q_current_rad),
        v_end=np.zeros_like(q_home_rad),
        duration=duration,
        dt=dt,
    )
    execute_trajectory(
        robot_interface=robot_interface,
        q_traj=q_traj,
        dq_traj=dq_traj,
        t_exec=t_exec,
        kinematics=kinematics,
        key_pos=np.zeros(3),
        step_callback=None,
        hold_callback=None,
    )
