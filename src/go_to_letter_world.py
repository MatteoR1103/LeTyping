from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import controller as controller_module
from controller import PDGravityController, SO101Interface
from traj_generation import RobotKinematics, generate_typing_trajectory


DEFAULT_LETTERS_PATH = Path("camera_calib/calibrations/letters_world.json")
DEFAULT_URDF_PATH = Path("cfg/arm_model/so101_new_calib.urdf")
DEFAULT_ROBOT_PORT = "/dev/ttyACM0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move the SO-101 to the recorded world position of a keyboard letter, "
            "without using camera tracking."
        ),
        epilog=(
            "Examples:\n"
            "  python3 src/go_to_letter_world.py M --dry-run\n"
            "  python3 src/go_to_letter_world.py M --travel-duration 5.0 --press-duration 1.5\n\n"
            "You must pass the desired letter as the first argument."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("letter", help="Keyboard letter to reach, for example A or M.")
    parser.add_argument(
        "--letters-path",
        type=Path,
        default=DEFAULT_LETTERS_PATH,
        help=f"Path to recorded letter positions. Default: {DEFAULT_LETTERS_PATH}.",
    )
    parser.add_argument(
        "--urdf-path",
        type=Path,
        default=DEFAULT_URDF_PATH,
        help=f"Path to the SO-101 URDF. Default: {DEFAULT_URDF_PATH}.",
    )
    parser.add_argument(
        "--robot-port",
        default=DEFAULT_ROBOT_PORT,
        help=f"Serial port for the SO follower arm. Default: {DEFAULT_ROBOT_PORT}.",
    )
    parser.add_argument(
        "--hover-height",
        type=float,
        default=0.0,
        help="Height above the recorded letter position, in metres. Default: 0.0.",
    )
    parser.add_argument(
        "--press-depth",
        type=float,
        default=0.0,
        help="Depth below the recorded letter position, in metres. Default: 0.0.",
    )
    parser.add_argument(
        "--travel-duration",
        type=float,
        default=3.0,
        help="Travel duration in seconds. Increase this for slower motion. Default: 3.0.",
    )
    parser.add_argument(
        "--press-duration",
        type=float,
        default=0.8,
        help="Press duration in seconds. Default: 0.8.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Controller/planner timestep in seconds. Default: 0.02.",
    )
    parser.add_argument(
        "--hold-time",
        type=float,
        default=2.0,
        help="Seconds to keep commanding the final target after the trajectory. Default: 2.0.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and print diagnostics without connecting to or moving the robot.",
    )
    parser.add_argument(
        "--plot-controller",
        action="store_true",
        help="Show controller telemetry plots after execution.",
    )
    if len(sys.argv) == 1:
        parser.print_help()
        raise SystemExit(
            "\nMissing required letter. Example: python3 src/go_to_letter_world.py M --dry-run"
        )

    return parser.parse_args()


def load_letter_position(letters_path: Path, letter: str) -> np.ndarray:
    with letters_path.open("r", encoding="utf-8") as f:
        samples = json.load(f)

    requested = letter.strip().upper()
    matches = [sample for sample in samples if str(sample.get("key", "")).upper() == requested]
    if not matches:
        available = sorted({str(sample.get("key", "")).upper() for sample in samples})
        raise ValueError(
            f"Letter {requested!r} not found in {letters_path}. "
            f"Available letters: {', '.join(available)}"
        )

    sample = matches[-1]
    try:
        position = sample["gripper_pose"]["position_m"]
    except KeyError as exc:
        raise KeyError(
            f"Letter {requested!r} exists, but it has no gripper_pose.position_m field."
        ) from exc

    return np.asarray(position, dtype=float)


def hold_final_position(
    robot_interface: SO101Interface,
    controller: PDGravityController,
    q_target: np.ndarray,
    hold_time: float,
    dt: float,
) -> None:
    if hold_time <= 0.0:
        return

    dq_target = np.zeros_like(q_target)
    deadline = time.perf_counter() + hold_time
    last_time = time.perf_counter()
    while time.perf_counter() < deadline:
        now = time.perf_counter()
        control_dt = max(now - last_time, 1e-3)
        last_time = now

        q, dq = robot_interface.read_joints()
        q_cmd = controller.compute_position_command(q, dq, q_target, dq_target, control_dt)
        robot_interface.write_joints(q_cmd)
        time.sleep(dt)


def main() -> None:
    args = parse_args()
    print(
        "Usage reminder: insert the desired keyboard letter as the first argument. "
        "Use --dry-run to test without moving the robot."
    )
    target_pos = load_letter_position(args.letters_path, args.letter)
    print(f"Target letter: {args.letter.upper()}")
    print(f"Target world position [m]: {target_pos}")

    kinematics = RobotKinematics(urdf_path=args.urdf_path)

    if args.dry_run:
        q_current = kinematics.neutral_configuration()
        print("Dry run: using neutral configuration as q_current.")
    else:
        robot_interface = SO101Interface(port=args.robot_port)
        q_current, _ = robot_interface.read_joints()
        print(f"Current joints [deg]: {np.rad2deg(q_current)}")

    q_traj, dq_traj, t_exec = generate_typing_trajectory(
        key_positions=[target_pos],
        q_current=q_current,
        kinematics=kinematics,
        hover_height=args.hover_height,
        press_depth=args.press_depth,
        travel_duration=args.travel_duration,
        press_duration=args.press_duration,
        dt=args.dt,
    )
    

    final_fk = kinematics.forward_kinematics(np.rad2deg(q_traj[-1]))
    final_pos = final_fk[:3, 3]
    print(f"Generated {len(t_exec)} trajectory samples over {t_exec[-1]:.2f} s.")
    print(f"Planned final end-effector position [m]: {final_pos}")
    print(f"Planner final position error [m]: {np.linalg.norm(final_pos - target_pos):.4f}")

    if args.dry_run:
        return

    controller_module.DEBUG_PLOT_CONTROLLER = args.plot_controller
    controller = PDGravityController(kinematics)

    try:
        print("Starting robot motion.")
        controller.execute_trajectory(q_traj, dq_traj, t_exec, robot_interface)
        hold_final_position(
            robot_interface=robot_interface,
            controller=controller,
            q_target=q_traj[-1],
            hold_time=args.hold_time,
            dt=args.dt,
        )
        q_final, _ = robot_interface.read_joints()
        final_actual = kinematics.forward_kinematics(np.rad2deg(q_final))[:3, 3]
        print(f"Actual final joints [deg]: {np.rad2deg(q_final)}")
        print(f"Actual final end-effector position [m]: {final_actual}")
        print(f"Actual final position error [m]: {np.linalg.norm(final_actual - target_pos):.4f}")
    finally:
        input("Press ENTER to disconnect the robot...")
        robot_interface.close()


if __name__ == "__main__":
    main()
