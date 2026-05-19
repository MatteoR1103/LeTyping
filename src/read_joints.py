#!/usr/bin/env python3
"""Read SO-101 joint positions each time ENTER is pressed.

Run from the repository root:

    python3 src/read_joints.py --port /dev/ttyACM0 --camera 5

Press ENTER to print the current joint positions. Type q, quit, or exit and
press ENTER to disconnect.
"""

from __future__ import annotations

import argparse
import select
import sys
from datetime import datetime

import cv2 as cv
import numpy as np

try:
    from .utils.general_utils import put_status_lines, read_frame, resolve_capture_backend
except ImportError:
    from utils.general_utils import put_status_lines, read_frame, resolve_capture_backend

try:
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower
except ImportError as exc:
    raise SystemExit(
        "lerobot hardware modules are required for this script. "
        "Activate the project environment first."
    ) from exc


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read SO-101 follower arm joints whenever ENTER is pressed."
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyACM0",
        help="Serial port for the SO-101 follower arm. Default: /dev/ttyACM0.",
    )
    parser.add_argument(
        "--robot-id",
        default="zi_padrone",
        help="Robot id used by lerobot for calibration lookup. Default: zi_padrone.",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=2, # for Piro it is 5, for Rub is 2
        help="OpenCV camera index. Default: 2.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "dshow", "msmf", "any"],
        default="auto",
        help="OpenCV camera backend. Default: auto.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=640,
        help="Requested camera frame width. Default: 640.",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=480,
        help="Requested camera frame height. Default: 480.",
    )
    return parser.parse_args()


def read_joint_degrees(robot: SOFollower, joint_names: list[str]) -> np.ndarray:
    obs = robot.get_observation()
    missing = [name for name in joint_names if f"{name}.pos" not in obs]
    if missing:
        available = ", ".join(sorted(obs.keys()))
        raise KeyError(
            "Observation is missing joint position keys for "
            f"{', '.join(missing)}. Available keys: {available}"
        )

    return np.array([float(obs[f"{name}.pos"]) for name in joint_names], dtype=float)


def print_joint_table(joint_names: list[str], q_deg: np.ndarray) -> None:
    q_rad = np.deg2rad(q_deg)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{stamp}] Current joints")
    print(f"{'joint':<16} {'deg':>12} {'rad':>12}")
    print("-" * 42)
    for name, deg, rad in zip(joint_names, q_deg, q_rad):
        print(f"{name:<16} {deg:12.6f} {rad:12.6f}")
    print()
    print("deg array:", np.array2string(q_deg, precision=6, separator=", "))
    print("rad array:", np.array2string(q_rad, precision=6, separator=", "))


def open_camera(args: argparse.Namespace) -> cv.VideoCapture:
    cap = cv.VideoCapture(args.camera, resolve_capture_backend(args.backend))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.frame_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.frame_height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera} with backend `{args.backend}`.")
    return cap


def terminal_command_ready() -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not ready:
        return None
    line = sys.stdin.readline()
    if line == "":
        return None
    return line.strip().lower()


def main() -> None:
    args = parse_args()
    joint_names = list(JOINT_NAMES)

    cap = open_camera(args)
    config = SOFollowerRobotConfig(port=args.port, id=args.robot_id)
    robot = SOFollower(config)

    window_name = "SO-101 camera feedback"
    last_read_text = "No joint read yet"
    robot_connected = False

    print(f"Connecting to SO-101 follower on {args.port}...")
    try:
        robot.connect()
        robot_connected = True
        robot.bus.disable_torque()
        print("Connected. Camera preview is open.")
        print("Press ENTER in the terminal or camera window to read joints.")
        print("Press q in the terminal or camera window to quit.")

        while True:
            frame = read_frame(cap, error_message="Camera stream ended or returned no frame.")
            preview = frame.copy()
            put_status_lines(
                preview,
                [
                    "SO-101 camera feedback",
                    "ENTER: read joints",
                    "q: quit",
                    last_read_text,
                ],
            )
            cv.imshow(window_name, preview)

            key = cv.waitKey(1) & 0xFF
            command = terminal_command_ready()

            should_quit = key == ord("q") or command in {"q", "quit", "exit"}
            should_read = key in (10, 13) or command == ""

            if should_quit:
                break
            if should_read:
                q_deg = read_joint_degrees(robot, joint_names)
                print_joint_table(joint_names, q_deg)
                last_read_text = f"Last read: {datetime.now().strftime('%H:%M:%S')}"
    finally:
        cap.release()
        cv.destroyAllWindows()
        if robot_connected:
            robot.bus.enable_torque()
            robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()