"""
PD + gravity-compensation controller for the SO-101 arm.

Control law
-----------
    τ(t) = g(q) + Kp · (q_des − q) + Kd · (dq_des − dq)

where:
* g(q)  – gravity-torque vector computed by pinocchio
* Kp    – diagonal position-gain matrix (n_joints × n_joints)
* Kd    – diagonal velocity-gain matrix (n_joints × n_joints)

Because the Feetech STS3215 servos used in the SO-101 are position-controlled, the controller DOES NOT send raw torques to the hardware.
Instead it implements a feed-forward gravity-compensated reference that is expressed as a corrected position set-point:

    q_cmd = q_des + Kp^{-1} · [g(q) + Kd · (dq_des − dq)]
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Sequence
import numpy as np

from traj_generation import RobotKinematics

try:
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    _LEROBOT_AVAILABLE = True
except ImportError:
    _LEROBOT_AVAILABLE = False
    print("WARNING: lerobot hardware modules not found. Interface will default to simulation.")


_DEFAULT_KP = np.array([80.0, 80.0, 80.0, 60.0, 40.0, 20.0])  # N·m / rad
_DEFAULT_KD = np.array([ 8.0,  8.0,  8.0,  6.0,  4.0,  2.0])  # N·m·s / rad

_ARM_JOINT_NAMES: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
#
_DEG2RAD = np.pi / 180.0
_RAD2DEG = 180.0 / np.pi


# ---------------------------------------------------------------------------
# PDGravityController
# ---------------------------------------------------------------------------

class PDGravityController:
    """Outer-loop PD controller with gravity feed-forward."""

    def __init__(
        self,
        kinematics: RobotKinematics, # defined in traj_generation.py
        Kp: np.ndarray | float | None = None,
        Kd: np.ndarray | float | None = None,
    ) -> None:
        self.kin = kinematics
        n = kinematics.n_joints

        self.Kp = _broadcast_gains(Kp if Kp is not None else _DEFAULT_KP[:n], n)
        self.Kd = _broadcast_gains(Kd if Kd is not None else _DEFAULT_KD[:n], n)

    def compute_torque(self, q: np.ndarray, dq: np.ndarray, q_des: np.ndarray, dq_des: np.ndarray) -> np.ndarray:
        """Compute the full control torque τ = g(q) + Kp·e_q + Kd·e_dq."""
        g   = self.kin.gravity_torques(q)
        tau = g + self.Kp * (q_des - q) + self.Kd * (dq_des - dq)
        return tau

    def compute_position_command(self, q: np.ndarray, dq: np.ndarray, q_des: np.ndarray, dq_des: np.ndarray) -> np.ndarray:
        """Convert the torque command to a corrected position set-point."""
        g       = self.kin.gravity_torques(q)
        ff      = g + self.Kd * (dq_des - dq)
        # Divide only where Kp is non-zero (safety check but should not happen with valid gains)
        q_cmd   = q_des + np.where(self.Kp != 0.0, ff / self.Kp, 0.0)
        return q_cmd

    def execute_trajectory(
        self,
        q_traj: np.ndarray,
        dq_traj: np.ndarray,
        t_exec: np.ndarray,
        robot_interface: "SO101Interface",
        step_callback: Callable[[int, np.ndarray, np.ndarray], None] | None = None,
    ) -> None:
        """Execute a pre-computed joint-space trajectory in real-time."""
        T  = len(t_exec)
        errors: list[float] = []
        robot_interface.robot.connect()
        print("[PDGravityController] Starting trajectory execution...")
        for i in range(T):
            q, dq = robot_interface.read_joints()
            q_cmd = self.compute_position_command(q, dq, q_traj[i], dq_traj[i])

            robot_interface.write_joints(q_cmd)
            if step_callback is not None:
                step_callback(i, q, dq)
            time.sleep(0.02)
            # Error tracking
            err = float(np.linalg.norm(q_traj[i] - q))
            errors.append(err)
        time.sleep(3.0)  # Hold final position for a moment
        print(f"[PDGravityController] Trajectory execution complete. Final position error: {errors[-1]:.4f} rad")
        robot_interface.robot.disconnect()


# ---------------------------------------------------------------------------
# SO101Interface – lerobot hardware wrapper
# ---------------------------------------------------------------------------

class SO101Interface:
    """Hardware interface for the SO-101 follower arm"""

    _DEFAULT_CALIB = Path("cfg/arms_calibration/follower/zi_padrone.json")

    def __init__(
        self,
        port: str = "/dev/ttyACM0", # pay attention to the default here, it might switch to ACM1
        joint_names: Sequence[str] = _ARM_JOINT_NAMES,
        calibration_path: str | Path | None = None,
        velocity_alpha: float = 0.3, # exponential smoothing factor for velocity estimation (kill the derivative kick from noisy measurements)
    ) -> None:
        self.port        = port
        self.joint_names = list(joint_names)
        self.n_joints    = len(self.joint_names)
        self.alpha       = velocity_alpha 

        config = SOFollowerRobotConfig(port=port, id = "zi_padrone")
        self.robot = SOFollower(config)
        self.robot.connect()
        calib_path = Path(calibration_path) if calibration_path else self._DEFAULT_CALIB
        self._calib_path = calib_path

        self._use_lerobot = False

        if _LEROBOT_AVAILABLE:
            try:
                print(f"[SO101Interface] Connected to {port}.")
                self._use_lerobot = True

            except Exception as exc:
                print(
                    f"[SO101Interface] WARNING – could not connect to robot on {port}: {exc}\n"
                    "Running in SIMULATION mode."
                )

        self._q_prev:  np.ndarray | None = None
        self._t_prev:  float | None      = None
        self._dq_filt: np.ndarray        = np.zeros(self.n_joints)


    def read_joints(self) -> tuple[np.ndarray, np.ndarray]:
        """Read current joint positions and velocities."""
        now = time.perf_counter()

        if self._use_lerobot is not None:
            obs = self.robot.get_observation()
            q_list = []
            for name in self.joint_names:
                val = obs.get(f"{name}.pos", 0.0)
                q_list.append(float(val) * _DEG2RAD)
            q = np.array(q_list, dtype=float)
            # print(f"[SO101Interface] Read joints: {q}")
        else:
            # print("Robot not found")
            q = np.zeros(self.n_joints)

        # Finite-difference velocity, with exponential smoothing to reduce noise and avoid derivative kick
        if self._q_prev is not None and self._t_prev is not None:
            dt_meas = now - self._t_prev
            if dt_meas > 1e-6:
                dq_raw = (q - self._q_prev) / dt_meas
                self._dq_filt = self.alpha * dq_raw + (1.0 - self.alpha) * self._dq_filt

        self._q_prev = q.copy()
        self._t_prev = now

        return q, self._dq_filt.copy()

    def write_joints(self, q_cmd: np.ndarray) -> None:
        """Send a joint-position command."""
        action = {
            "shoulder_pan.pos": q_cmd[0] * _RAD2DEG,
            "shoulder_lift.pos": q_cmd[1] * _RAD2DEG,
            "elbow_flex.pos": q_cmd[2] * _RAD2DEG,
            "wrist_flex.pos": q_cmd[3] * _RAD2DEG,
            "wrist_roll.pos": q_cmd[4] * _RAD2DEG,
            "gripper.pos": q_cmd[5] * _RAD2DEG,
        }
        self.robot.send_action(action)

    def close(self) -> None:
        """Disconnect from the motor bus."""
        self.robot.disconnect()
        

    def __enter__(self) -> "SO101Interface":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def execute_joint_trajectory(
    robot_interface: SO101Interface,
    q_traj: np.ndarray,
    dq_traj: np.ndarray,
    t_exec: np.ndarray,
    kinematics: RobotKinematics,
    step_callback: Callable[[int, np.ndarray, np.ndarray], None] | None = None,
) -> None:
    """Execute a precomputed joint trajectory with the existing PD controller."""
    controller = PDGravityController(kinematics)
    controller.execute_trajectory(q_traj, dq_traj, t_exec, robot_interface, step_callback)


def _broadcast_gains(gains: np.ndarray | float, n: int) -> np.ndarray:
    """Expand scalar or vector gains to shape (n,)."""
    g = np.asarray(gains, dtype=float)
    if g.ndim == 0:
        return np.full(n, float(g))
    if g.shape == (n,):
        return g
    raise ValueError(f"Gains must be a scalar or shape ({n},), got {g.shape}.")
