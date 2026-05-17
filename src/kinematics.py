from __future__ import annotations

from pathlib import Path

import numpy as np

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


class RobotKinematics:
    """Kinematics / dynamics wrapper for the SO-101.

    * FK and IK are delegated to lerobot's RobotKinematics when available.
    * Gravity torques are computed by pinocchio.
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
        self.n_joints = self.model.nq

        if self.model.existFrame(ee_frame):
            self.ee_frame_id: int = self.model.getFrameId(ee_frame)
        else:
            self.ee_frame_id = self.model.nframes - 1
            print(
                f"[RobotKinematics] Frame '{ee_frame}' not found; "
                f"using frame id {self.ee_frame_id} instead."
            )

        if _LEROBOT_AVAILABLE:
            self._fk = _LerobotKinematics(
                urdf_path=str(urdf_path),
                target_frame_name=ee_frame,
                joint_names=ALL_JOINT_NAMES,
            )
            self._ik = _LerobotKinematics(
                urdf_path=str(urdf_path),
                target_frame_name=ee_frame,
                joint_names=ARM_JOINT_NAMES,
            )
        else:
            self._fk = None
            self._ik = None
            print(
                "[RobotKinematics] lerobot not available - "
                "forward_kinematics / inverse_kinematics will raise."
            )

    def neutral_configuration(self) -> np.ndarray:
        """Return the pinocchio neutral configuration."""
        return pin.neutral(self.model)

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Return end-effector pose as a 4x4 matrix for configuration q in degrees."""
        if self._fk is None or self._ik is None:
            raise RuntimeError("lerobot is required for forward_kinematics.")

        q = np.asarray(q, dtype=float)
        if len(q) == len(ARM_JOINT_NAMES):
            return self._ik.forward_kinematics(q)
        return self._fk.forward_kinematics(q)

    def ee_position(self, q: np.ndarray) -> np.ndarray:
        """Return end-effector position (3,) for configuration q."""
        return self.forward_kinematics(q)[:3, 3].copy()

    def inverse_kinematics(
        self,
        q_init: np.ndarray,
        target_pos: np.ndarray,
        position_weight: float = 100.0,
        orientation_weight: float = 0.15,
        tol: float = 1e-3,
        max_iters: float = 20,
    ) -> np.ndarray:
        """Position-only IK via lerobot's placo solver.

        q_init and the returned solution are in degrees.
        """
        if self._ik is None:
            raise RuntimeError("lerobot is required for inverse_kinematics.")

        downward_orientation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        T_target = make_pose(target_pos, downward_orientation)
        q_sol_deg = q_init.copy()

        for _ in range(max_iters):
            q_sol_deg = self._ik.inverse_kinematics(
                q_sol_deg,
                T_target,
                position_weight=position_weight,
                orientation_weight=orientation_weight,
            )
            ee_sol_pos = self.forward_kinematics(q_sol_deg)[:3, 3]
            err = np.linalg.norm(ee_sol_pos - T_target[:3, 3])

            if err < tol:
                break

        return q_sol_deg

    def gravity_torques(self, q: np.ndarray) -> np.ndarray:
        """Return the gravity-compensation torque vector g(q)."""
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
    """Build a 4x4 homogeneous transform from a position and optional rotation."""
    T = np.eye(4)
    T[:3, 3] = xyz
    if rot is not None:
        T[:3, :3] = rot
    return T
