from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

try:
    from .controller import SO101Interface
    from .tracker import KeyWorldTracker
    from .traj_generation import (
        DEFAULT_PRESS_EE_FRAME,
        RobotKinematics,
        deliver_typing_trajectory,
        go_home,
    )
    from .utils.general_utils import build_typing_runs
    from .utils.tracking_utils import (
        activate_maintained_target_state,
        build_tracking_cluster,
        retrack_targets_from_current_frame,
        update_tracker_for_duration,
    )
except ImportError:
    from controller import SO101Interface
    from tracker import KeyWorldTracker
    from traj_generation import (
        DEFAULT_PRESS_EE_FRAME,
        RobotKinematics,
        deliver_typing_trajectory,
        go_home,
    )
    from utils.general_utils import build_typing_runs
    from utils.tracking_utils import (
        activate_maintained_target_state,
        build_tracking_cluster,
        retrack_targets_from_current_frame,
        update_tracker_for_duration,
    )


DEFAULT_URDF_PATH = "cfg/arm_model/so101_new_calib.urdf"
ROBOT_PORT = "/dev/ttyACM0"
TASK1_TARGETS = ["SPACE", "ENTER", "R", "L"]
DEFAULT_LIVE_MODEL = "gemini-3-flash-preview"
DEFAULT_HOME_POSITION = np.deg2rad(
    np.array([3.07692308, -33.14285714, 41.18681319, 61.8021978, -89.62637363, 50.0])
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate keyboard keys in world coordinates and press them with the SO-101."
    )

    run_source = parser.add_mutually_exclusive_group(required=True)
    run_source.add_argument(
        "--word",
        nargs="+",
        type=str,
        help='The word, letters, or sentence to type. For example: CAT, C A T, or "RUB IS GOAT".',
    )
    run_source.add_argument(
        "--task-1",
        action="store_true",
        help="Run predefined task 1: presses SPACE, ENTER, R, L in order.",
    )
    run_source.add_argument(
        "--list-path",
        type=Path,
        help="Path to a text file with one word or sentence per row.",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=5, # for Piro it is either 4 or 5, for RUb it is 2
        help="OpenCV camera index. Default: 5.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LIVE_MODEL,
        help=f"Gemini model used for initial localization. Default: {DEFAULT_LIVE_MODEL}.",
    )
    parser.add_argument(
        "--gemini-backend",
        choices=["standard", "priority", "provisioned"],
        default="standard",
        help=(
            "Vertex AI Gemini request mode: standard PayGo, Priority PayGo, "
            "or Provisioned Throughput. Default: standard."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "dshow", "msmf", "any"],
        default="auto",
        help="OpenCV camera backend. Default: auto.",
    )
    parser.add_argument(
        "--project",
        default=os.getenv("GOOGLE_CLOUD_PROJECT"),
        help="Google Cloud project for Vertex AI. Defaults to GOOGLE_CLOUD_PROJECT.",
    )
    parser.add_argument(
        "--location",
        default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
        help="Google Cloud location for Vertex AI. Defaults to GOOGLE_CLOUD_LOCATION or global.",
    )
    parser.add_argument(
        "--urdf-path",
        default=DEFAULT_URDF_PATH,
        help="Path to the SO-101 URDF.",
    )
    parser.add_argument(
        "--press-ee-frame",
        default=DEFAULT_PRESS_EE_FRAME,
        help=(
            "URDF frame used as the physical key-contact point for pressing "
            f"trajectories. Default: {DEFAULT_PRESS_EE_FRAME}."
        ),
    )
    parser.add_argument(
        "--robot-port",
        default=ROBOT_PORT,
        help="Serial port for the SO follower arm, for example /dev/ttyACM0.",
    )
    parser.add_argument(
        "--calibration-path",
        default="cfg/calibration/follower/zi_padrone.json",
        help="Optional calibration file path forwarded to the SO101 interface.",
    )
    parser.add_argument(
        "--keyboard-height",
        type=float,
        default=0.02,
        help="Keyboard plane height in world coordinates, in metres. Default: 0.02.",
    )
    parser.add_argument(
        "--hover-height",
        type=float,
        default=0.04,
        help="Hover height above the key, in metres. Default: 0.04.",
    )
    parser.add_argument(
        "--press-depth",
        type=float,
        default=0.014,
        help="Press depth below the key plane, in metres. Default: 0.014.",
    )
    parser.add_argument(
        "--travel-duration",
        dest="travel_duration",
        type=float,
        default=0.8,
        help="Maximum duration cap for approach/final travel spline segments. Default: 0.8.",
    )
    parser.add_argument(
        "--press-duration",
        dest="press_duration",
        type=float,
        default=0.3,
        help="Maximum duration cap for pre-press/descent spline segments. Default: 0.3.",
    )
    parser.add_argument(
        "--approach-speed",
        type=float,
        default=0.06,
        help="Approximate Cartesian speed for approach/refinement moves in m/s. Default: 0.07.",
    )
    parser.add_argument(
        "--press-speed",
        type=float,
        default=0.04,
        help="Approximate Cartesian speed for pre-press/descent moves in m/s. Default: 0.04.",
    )
    parser.add_argument(
        "--min-segment-duration_default",
        type=float,
        default=0.4,
        help="Minimum default duration for any generated spline segment in seconds. Default: 0.4.",
    )
    parser.add_argument(
        "--max-refine-steps",
        type=int,
        default=3,
        help="Maximum adaptive hover refinement moves before pressing a key. Default: 3.",
    )
    parser.add_argument(
        "--refine-xy-threshold",
        type=float,
        default=0.002,
        help="Stop hover refinement once end-effector/key xy error is below this many metres. Default: 0.002.",
    )
    parser.add_argument(
        "--estimate-stability-threshold",
        type=float,
        default=0.002,
        help="Stop hover refinement only when recent key xy estimates vary less than this many metres. Default: 0.002.",
    )
    parser.add_argument(
        "--tracking-cluster-radius",
        type=float,
        default=0.02,
        help="World radius in metres used to group nearby letters for continuous tracking. Default: 0.02.",
    )

    parser.add_argument(
        "--shorter-segment-duration",
        type=float,
        default=0.1,
        help="A shorter minimum duration to use for hover refinement segments after the first one, since they should be shorter. Default: 0.1.",
    )

    return parser.parse_args()


def make_cluster_world_positions_coherent(
    active_cluster: set[str],
    frozen_world_by_letter: dict[str, np.ndarray],
    anchor_letter: str,
    min_dist_m: float = 0.01,
) -> None:
    if anchor_letter not in frozen_world_by_letter:
        return

    anchor_pos = np.asarray(frozen_world_by_letter[anchor_letter], dtype=float).copy()
    for letter in sorted(active_cluster):
        if letter == anchor_letter or letter not in frozen_world_by_letter:
            continue

        pos = np.asarray(frozen_world_by_letter[letter], dtype=float).copy()
        delta = pos[:3] - anchor_pos[:3]
        dist = float(np.linalg.norm(delta))

        if dist >= min_dist_m:
            continue

        direction = np.array([1.0, 0.0, 0.0]) if dist < 1e-9 else delta / dist
        pos[:3] = anchor_pos[:3] + min_dist_m * direction
        frozen_world_by_letter[letter] = pos
        print(
            f"[WARNING] Corrected collapsed key positions {anchor_letter}-{letter}: "
            f"distance was {dist * 1000:.2f} mm, enforced {min_dist_m * 1000:.1f} mm."
        )


def main() -> np.ndarray | None:
    args = parse_args()
    typing_runs = build_typing_runs(args, task1_targets=TASK1_TARGETS)

    tracking_kinematics = RobotKinematics(urdf_path=args.urdf_path)
    pressing_kinematics = RobotKinematics(urdf_path=args.urdf_path, ee_frame=args.press_ee_frame)
    print("Tracking/camera kinematics frame: gripper_frame_link")
    print(f"Pressing/contact kinematics frame: {args.press_ee_frame}")

    robot_interface = SO101Interface(
        port=args.robot_port,
        calibration_path=args.calibration_path,
    )
    print("Robot is now connected")
    print("Changing PID coefficients of internal motors...")

    robot_interface.robot.bus.enable_torque()
    for motor in robot_interface.robot.bus.motors:
        robot_interface.robot.bus.write("P_Coefficient", motor, 20)
        robot_interface.robot.bus.write("I_Coefficient", motor, 1)
        robot_interface.robot.bus.write("D_Coefficient", motor, 16)

    go_home(robot_interface, tracking_kinematics, q_home_rad=DEFAULT_HOME_POSITION)
    time.sleep(2.0)

    tracker: KeyWorldTracker | None = None
    try:
        for run_index, (run_label, letters) in enumerate(typing_runs, start=1):
            if not letters:
                raise ValueError(f"No supported typing targets found for run `{run_label}`.")

            print(f"Starting typing run {run_index}/{len(typing_runs)}: {run_label}")
            tracker = KeyWorldTracker(
                letter=",".join(letters),
                camera=args.camera,
                model=args.model,
                gemini_backend=args.gemini_backend,
                project=args.project,
                location=args.location,
                keyboard_height=args.keyboard_height,
                backend=args.backend,
            )

            try:
                print("Main operation loop starting ...")
                tracker.start(robot_interface=robot_interface, kinematics=tracking_kinematics)

                runtime_targets = [dict(tracker.targets_by_letter[letter]) for letter in letters]
                q_home_config = np.rad2deg(DEFAULT_HOME_POSITION)
                active_cluster: set[str] = set()
                frozen_world_by_letter: dict[str, np.ndarray] = {}
                retrack_from_home = True

                for index, target in enumerate(runtime_targets):
                    immediate_next = runtime_targets[index + 1] if index + 1 < len(runtime_targets) else None
                    current_letter = target["letter"]
                    is_space_target = current_letter == "SPACE"

                    unrefined_remaining_letters = []
                    for future_target in runtime_targets[index:]:
                        letter = future_target["letter"]
                        if letter not in frozen_world_by_letter and letter != "SPACE":
                            unrefined_remaining_letters.append(letter)

                    cluster_candidates = ["SPACE"] if is_space_target else unrefined_remaining_letters

                    if is_space_target:
                        active_cluster = set()
                        retrack_from_home = current_letter not in frozen_world_by_letter
                    elif current_letter in frozen_world_by_letter:
                        retrack_from_home = False

                    if retrack_from_home:
                        retrack_targets_from_current_frame(
                            tracker,
                            cluster_candidates,
                            robot_interface=robot_interface,
                            kinematics=tracking_kinematics,
                        )
                        active_cluster = set(
                            build_tracking_cluster(
                                tracker.targets_by_letter,
                                current_letter,
                                cluster_candidates,
                                radius=args.tracking_cluster_radius,
                            )
                        )
                        tracker.active_cluster_letters = set(active_cluster)

                        activate_maintained_target_state(tracker, current_letter)
                        retrack_from_home = False

                        if tracker.last_estimate is not None:
                            target["world"] = tracker.last_estimate.copy()
                    else:
                        if current_letter in frozen_world_by_letter:
                            active_cluster = set()
                        tracker.active_cluster_letters = set(active_cluster)

                        if current_letter in frozen_world_by_letter:
                            frozen_world = frozen_world_by_letter[current_letter]
                            activate_maintained_target_state(
                                tracker,
                                current_letter,
                                world=frozen_world,
                            )
                            target["world"] = frozen_world.copy()
                        else:
                            tracker.set_target(
                                letter=current_letter,
                                robot_interface=robot_interface,
                                kinematics=tracking_kinematics,
                            )
                            if tracker.last_estimate is not None:
                                target["world"] = tracker.last_estimate.copy()

                    next_requires_retrack = False
                    if immediate_next is not None:
                        immediate_next_letter = immediate_next["letter"]
                        next_is_ready = (
                            immediate_next_letter in active_cluster
                            or immediate_next_letter in frozen_world_by_letter
                        )
                        next_requires_retrack = not next_is_ready
                        if next_is_ready:
                            print(f"Using previous estimate for {immediate_next_letter}")
                        else:
                            print(
                                f"Leaving cluster before {immediate_next_letter}; "
                                "returning home before rebuilding the next tracking cluster."
                            )

                    key_position = np.asarray(target["world"], dtype=float).reshape(3)
                    track_during_hover = bool(active_cluster)
                    lock_key_position = not track_during_hover

                    print(f"Commanded key position for letter {current_letter}: {key_position}")
                    pressed_key_position = deliver_typing_trajectory(
                        key_position=key_position,
                        tracker=tracker,
                        robot_interface=robot_interface,
                        hover_height=args.hover_height,
                        press_depth=args.press_depth,
                        kinematics=pressing_kinematics,
                        tracking_kinematics=tracking_kinematics,
                        travel_duration=args.travel_duration,
                        press_duration=args.press_duration,
                        q_final_config=q_home_config if (next_requires_retrack or immediate_next is None) else None,
                        track_during_hover=track_during_hover,
                        lock_key_position=lock_key_position,
                        approach_speed=args.approach_speed,
                        press_speed=args.press_speed,
                        min_segment_duration_default=args.min_segment_duration_default,
                        max_refine_steps=args.max_refine_steps,
                        refine_xy_threshold=args.refine_xy_threshold,
                        estimate_stability_threshold=args.estimate_stability_threshold,
                        shorter_segment_duration=args.shorter_segment_duration,
                    )

                    if active_cluster:
                        frozen_world_by_letter[current_letter] = np.asarray(
                            pressed_key_position,
                            dtype=float,
                        ).reshape(3).copy()
                        for letter in active_cluster:
                            if letter not in frozen_world_by_letter:
                                frozen_world_by_letter[letter] = np.asarray(
                                    tracker.targets_by_letter[letter]["world"],
                                    dtype=float,
                                ).reshape(3).copy()

                        make_cluster_world_positions_coherent(
                            active_cluster,
                            frozen_world_by_letter,
                            current_letter,
                            min_dist_m=0.015,
                        )

                    if next_requires_retrack:
                        active_cluster = set()
                        retrack_from_home = True
            finally:
                go_home(robot_interface, tracking_kinematics, q_home_rad=DEFAULT_HOME_POSITION)
                if tracker.cap is not None:
                    update_tracker_for_duration(
                        tracker=tracker,
                        duration_s=1.0,
                        robot_interface=robot_interface,
                        kinematics=tracking_kinematics,
                    )
                else:
                    time.sleep(1.0)
                tracker.close()
                tracker = None
    finally:
        if tracker is not None:
            tracker.close()
        input("Press ENTER when the robot is back at the home position to disconnect...")
        robot_interface.close()


if __name__ == "__main__":
    main()
