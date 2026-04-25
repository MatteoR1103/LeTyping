#!/usr/bin/env python

import argparse
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

CALIBRATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CALIBRATION_DIR.parent


def add_lerobot_src_to_path():
    for candidate in (
        PROJECT_ROOT / "lerobot" / "src",
        PROJECT_ROOT.parent / "lerobot" / "src",
    ):
        if (candidate / "lerobot").is_dir():
            sys.path.insert(0, str(candidate))
            return


add_lerobot_src_to_path()

from lerobot.model.kinematics import RobotKinematics

DEFAULT_INPUT_DIR = CALIBRATION_DIR / "raw_calib_data" / "2026-04-25_12-27-21"
DEFAULT_OUTPUT_DIR = CALIBRATION_DIR / "calib_poses_data" / "handeye_samples_poses_2504"
DEFAULT_TARGET_FRAME = "gripper_frame_link"
DEFAULT_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute gripper poses from saved hand-eye calibration joint samples."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing samples.json and images. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where the pose-enriched samples will be written. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--urdf-path",
        type=Path,
        required=True,
        help="Path to the robot URDF used for forward kinematics.",
    )
    parser.add_argument(
        "--target-frame",
        default=DEFAULT_TARGET_FRAME,
        help=f"URDF frame name for the gripper pose. Default: {DEFAULT_TARGET_FRAME}",
    )
    parser.add_argument(
        "--joint-names",
        nargs="+",
        default=DEFAULT_JOINT_NAMES,
        help="Joint names in the order expected by the URDF / kinematics solver.",
    )
    return parser.parse_args()


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    """Convert a 3x3 rotation matrix to quaternion [x, y, z, w]."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    return [float(x), float(y), float(z), float(w)]


def rotation_matrix_to_rotvec(rotation: np.ndarray) -> list[float]:
    """Convert a 3x3 rotation matrix to axis-angle rotation vector."""
    cos_theta = (np.trace(rotation) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))

    if theta < 1e-12:
        return [0.0, 0.0, 0.0]

    sin_theta = float(np.sin(theta))
    if abs(sin_theta) < 1e-8:
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        axis = axis / np.linalg.norm(axis)
        return [float(v) for v in axis * theta]

    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=float,
    ) / (2.0 * sin_theta)
    return [float(v) for v in axis * theta]


def build_pose_dict(transform: np.ndarray, gripper_pos: float | None) -> dict:
    position = [float(v) for v in transform[:3, 3]]
    rotation = transform[:3, :3]
    rotvec = rotation_matrix_to_rotvec(rotation)
    quaternion_xyzw = rotation_matrix_to_quaternion_xyzw(rotation)

    pose = {
        "position_m": position,
        "rotation_matrix": [[float(v) for v in row] for row in rotation],
        "quaternion_xyzw": quaternion_xyzw,
        "rotvec": rotvec,
        "transform_matrix": [[float(v) for v in row] for row in transform],
    }
    if gripper_pos is not None:
        pose["gripper_pos"] = float(gripper_pos)
    return pose


def load_samples(samples_path: Path) -> list[dict]:
    if not samples_path.exists():
        raise FileNotFoundError(f"Could not find samples file: {samples_path}")
    with samples_path.open() as f:
        return json.load(f)


def extract_joint_vector(sample: dict, joint_names: list[str]) -> np.ndarray:
    joint_state = sample.get("joint_state")
    if not isinstance(joint_state, dict):
        raise ValueError(f"Sample {sample.get('sample_idx')} has no valid joint_state dictionary")

    missing = [name for name in joint_names if f"{name}.pos" not in joint_state]
    if missing:
        raise ValueError(f"Sample {sample.get('sample_idx')} is missing joints: {missing}")

    return np.array([joint_state[f"{name}.pos"] for name in joint_names], dtype=float)


def copy_image(image_path: Path, src_root: Path, dst_root: Path) -> str:
    if image_path.is_absolute():
        src_image = image_path
    else:
        src_image = src_root / image_path
        if not src_image.exists():
            src_image = Path.cwd() / image_path

    if not src_image.exists():
        raise FileNotFoundError(f"Could not find image referenced by sample: {src_image}")

    relative_name = src_image.name
    dst_image = dst_root / "images" / relative_name
    dst_image.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_image, dst_image)
    return str(dst_image.relative_to(dst_root))


def main():
    args = parse_args()
    urdf_path = args.urdf_path.resolve()
    assets_dir = urdf_path.parent / "assets"

    if not urdf_path.exists():
        raise SystemExit(f"URDF not found: {urdf_path}")

    if not assets_dir.exists():
        raise SystemExit(
            "The URDF references mesh files in a sibling 'assets/' directory, but it was not found at "
            f"{assets_dir}. Use the original SO101 URDF in its full folder, or copy the matching assets folder "
            "next to the URDF."
        )

    try:
        with working_directory(urdf_path.parent):
            kinematics = RobotKinematics(
                urdf_path=str(urdf_path),
                target_frame_name=args.target_frame,
                joint_names=args.joint_names,
            )
    except ImportError as exc:
        raise SystemExit(
            "Failed to import the kinematics backend. Install the optional placo dependency first, "
            "for example: `pip install -e '.[placo-dep]'`."
        ) from exc
    except Exception as exc:
        raise SystemExit(
            "Failed to initialize kinematics. Make sure the URDF is used together with its matching "
            f"'assets/' directory. Original error: {exc}"
        ) from exc

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(input_dir / "samples.json")
    converted_samples = []

    for sample in samples:
        joint_vector = extract_joint_vector(sample, args.joint_names)
        transform = kinematics.forward_kinematics(joint_vector)

        image_path = sample.get("image_path")
        copied_image_path = None
        if image_path is not None:
            copied_image_path = copy_image(Path(image_path), input_dir, output_dir)

        pose = build_pose_dict(transform, sample["joint_state"].get("gripper.pos"))
        rotvec = pose["rotvec"]
        position = pose["position_m"]

        converted_sample = {
            **sample,
            "image_path": copied_image_path if copied_image_path is not None else image_path,
            "gripper_pose": pose,
            "ee.x": position[0],
            "ee.y": position[1],
            "ee.z": position[2],
            "ee.wx": rotvec[0],
            "ee.wy": rotvec[1],
            "ee.wz": rotvec[2],
        }
        if "gripper_pos" in pose:
            converted_sample["ee.gripper_pos"] = pose["gripper_pos"]
        converted_samples.append(converted_sample)

    with (output_dir / "samples.json").open("w") as f:
        json.dump(converted_samples, f, indent=2)

    manifest = {
        "source_samples": str((input_dir / "samples.json").resolve()),
        "urdf_path": str(urdf_path),
        "target_frame": args.target_frame,
        "joint_names": args.joint_names,
        "num_samples": len(converted_samples),
    }
    with (output_dir / "metadata.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(converted_samples)} pose samples to {output_dir / 'samples.json'}")


if __name__ == "__main__":
    main()
