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

import json
import time
from pathlib import Path
from typing import Sequence
import numpy as np

from traj_generation import RobotKinematics

try:
    from lerobot.motors.feetech.feetech import FeetechMotorsBus
    from lerobot.motors.motors_bus import Motor, MotorCalibration
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
        # Divide only where Kp is non-zero
        q_cmd   = q_des + np.where(self.Kp != 0.0, ff / self.Kp, 0.0)
        return q_cmd

    def execute_trajectory(
        self,
        q_traj: np.ndarray,
        dq_traj: np.ndarray,
        t_exec: np.ndarray,
        robot_interface: "SO101Interface",
        verbose: bool = False,
    ) -> None:
        """Execute a pre-computed joint-space trajectory in real-time."""
        dt = float(t_exec[1] - t_exec[0]) if len(t_exec) > 1 else 0.02
        T  = len(t_exec)
        errors: list[float] = []

        for i in range(T):
            t_step_start = time.perf_counter()

            q, dq = robot_interface.read_joints()
            q_cmd = self.compute_position_command(q, dq, q_traj[i], dq_traj[i])

            robot_interface.write_joints(q_cmd)

            # Error tracking
            err = float(np.linalg.norm(q_traj[i] - q))
            errors.append(err)
            if verbose and (i % 50 == 0):
                print(f"[ctrl] step {i:4d}/{T}  t={t_exec[i]:.3f}s  |e_q|={err:.4f} rad")

            while (time.perf_counter() - t_step_start) < dt:
                pass

        if verbose:
            errors_arr = np.array(errors)
            print(f"[ctrl] Done. Mean |e_q|={errors_arr.mean():.4f} rad  Max |e_q|={errors_arr.max():.4f} rad")


# ---------------------------------------------------------------------------
# SO101Interface – lerobot hardware wrapper
# ---------------------------------------------------------------------------

class SO101Interface:
    """Hardware interface for the SO-101 follower arm via lerobot v0.5.2."""

    _DEFAULT_CALIB = (
        Path(__file__).resolve().parent.parent
        / "cfg" / "arms_calibration" / "follower" / "zi_padrone.json"
    )

    def __init__(
        self,
        port: str = "/dev/ttyACM0", # pay attention to the default here, it might switch to ACM1
        joint_names: Sequence[str] = _ARM_JOINT_NAMES,
        calibration_path: str | Path | None = None,
        velocity_alpha: float = 0.3, # exponential smoothing factor for velocity estimation
    ) -> None:
        self.port        = port
        self.joint_names = list(joint_names)
        self.n_joints    = len(self.joint_names)
        self.alpha       = velocity_alpha 

        calib_path = Path(calibration_path) if calibration_path else self._DEFAULT_CALIB
        self._calib_path = calib_path

        self._bus = None
        self._use_lerobot = False

        if _LEROBOT_AVAILABLE:
            try:
                calib_dict, motors_dict = self._build_motor_config()
                self._bus = FeetechMotorsBus(
                    port=self.port, 
                    motors=motors_dict, 
                    calibration=calib_dict
                )
                self._bus.connect()
                print(f"[SO101Interface] Connected to {port}.")

                # Auto-enable torque on startup
                for motor_name in self.joint_names:
                    self._bus.write(
                        data_name="Torque_Enable", 
                        motor=motor_name, 
                        value=1, 
                        normalize=False
                    )
                print("[SO101Interface] Torque enabled.")
                
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

        if self._use_lerobot and self._bus is not None:
            # Create an empty array to hold our readings
            pos_deg = np.zeros(self.n_joints)    

            for i, motor_name in enumerate(self.joint_names):
                pos_deg[i] = self._bus.read(
                    data_name="Present_Position", 
                    motor=motor_name
                )
            
            q = pos_deg * _DEG2RAD
        else:
            q = np.zeros(self.n_joints)

        # Finite-difference velocity
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
        if self._use_lerobot and self._bus is not None:
            pos_deg = q_cmd * _RAD2DEG
            for i, motor_name in enumerate(self.joint_names):
                self._bus.write(
                    data_name="Goal_Position",
                    motor=motor_name,
                    value=pos_deg[i],
                    normalize=False
                )

    def close(self) -> None:
        """Disconnect from the motor bus."""
        if self._use_lerobot and self._bus is not None:
            self._bus.disconnect()
            print("[SO101Interface] Disconnected.")

    def __enter__(self) -> "SO101Interface":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _build_motor_config(self) -> tuple[dict, dict]:
        """Load JSON and build motor objects."""
        ids: dict[str, int] = {}
        calib_dict: dict[str, MotorCalibration] = {}
        
        if self._calib_path.is_file():
            with self._calib_path.open() as fh:
                raw_calib = json.load(fh)
            for name, info in raw_calib.items():
                ids[name] = info.get("id", len(ids) + 1)
                # Convert raw JSON dict to a MotorCalibration object
                calib_dict[name] = MotorCalibration(**info)

        motors_dict: dict[str, Motor] = {}
        for i, name in enumerate(self.joint_names, start=1):
            motor_id = ids.get(name, i)
            # Create a Motor object using "identity" norm_mode
            motors_dict[name] = Motor(motor_id, "sts3215", "identity")
            
        return calib_dict, motors_dict


def _broadcast_gains(gains: np.ndarray | float, n: int) -> np.ndarray:
    """Expand scalar or vector gains to shape (n,)."""
    g = np.asarray(gains, dtype=float)
    if g.ndim == 0:
        return np.full(n, float(g))
    if g.shape == (n,):
        return g
    raise ValueError(f"Gains must be a scalar or shape ({n},), got {g.shape}.")