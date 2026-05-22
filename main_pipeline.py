from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from src.controller import SO101Interface
from src.keyboard_cluster import KeyboardClusterManager
from src.kinematics import DEFAULT_PRESS_EE_FRAME, RobotKinematics
from src.tracker import KeyWorldTracker
from src.traj_generation import deliver_typing_trajectory, go_home
from src.utils.general_utils import build_typing_runs, config_value, load_pipeline_config
from src.utils.tracking_utils import update_tracker_for_duration


DEFAULT_CONFIG_PATH = Path("cfg/main_pipeline.yaml")


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value: true or false.")


def arg_was_passed(flag: str, argv: list[str]) -> bool:
    return flag in argv or any(arg.startswith(f"{flag}=") for arg in argv)


def task_config(config: dict, task: int | None) -> dict:
    if task is None:
        return {}
    tasks = config.get("tasks", {})
    if not isinstance(tasks, dict):
        return {}
    value = tasks.get(task, tasks.get(str(task), {}))
    return value if isinstance(value, dict) else {}


def normalize_key_list(value, fallback: list[str]) -> list[str]:
    if value is None:
        return fallback.copy()
    if isinstance(value, str):
        return [value.strip().upper()] if value.strip() else []
    return [str(item).strip().upper() for item in value if str(item).strip()]


def normalize_string_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def parse_xy_offset(value) -> tuple[float, float]:
    if value is None:
        return (0.01, 0.0)
    if len(value) != 2:
        raise argparse.ArgumentTypeError("Expected two values: X Y.")
    return (float(value[0]), float(value[1]))


def parse_args() -> argparse.Namespace:
    raw_argv = sys.argv[1:]
    config_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the main pipeline YAML config. Default: {DEFAULT_CONFIG_PATH}.",
    )
    config_args, _ = config_parser.parse_known_args()
    config = load_pipeline_config(config_args.config)
    home_position_default = config_value(
        config,
        "home_position_deg",
        None,
    )
    model_default = config_value(config, "gemini.model", "gpt-5.5")
    provider_default = config_value(config, "gemini.provider", "openai")
    project_default = config_value(config, "gemini.project", os.getenv("GOOGLE_CLOUD_PROJECT"))
    location_default = config_value(config, "gemini.location", os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
    capture_screens_default = str_to_bool(config_value(config, "capture_screens", False))
    disabled_klt_default = normalize_key_list(config_value(config, "tracking.disable_klt_for"), ["SPACE"])
    cluster_excluded_default = normalize_key_list(config_value(config, "cluster.excluded_letters"), ["SPACE"])

    parser = argparse.ArgumentParser(
        description="Estimate keyboard keys in world coordinates and press them with the SO-101.",
        parents=[config_parser],
        allow_abbrev=False,
    )

    parser.add_argument(
        "--task",
        type=int,
        choices=[1, 2, 3],
        help="Competition task number. Defaults to the list_path configured under tasks.<n>.",
    )
    run_source = parser.add_mutually_exclusive_group(required=False)
    run_source.add_argument(
        "--word",
        nargs="+",
        type=str,
        help='The word, letters, or sentence to type. For example: CAT, C A T, or "RUB IS GOAT".',
    )
    run_source.add_argument(
        "--task-1",
        action="store_true",
        help="Run task 1 using tasks.1.list_path from the YAML config.",
    )
    run_source.add_argument(
        "--list-path",
        type=Path,
        help="Path to a text file with one word or sentence per row.",
    )
    parser.add_argument(
        "--home-position-deg",
        nargs=6,
        type=float,
        default=home_position_default,
        metavar="DEG",
        help="Home joint configuration in degrees. Defaults to home_position_deg in the YAML config.",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=config_value(config, "camera.index", 5),
        help="OpenCV camera index. Defaults to camera.index in the YAML config.",
    )
    parser.add_argument(
        "--model",
        default=model_default,
        help=f"Vision-language model used for initial localization. Default: {model_default}.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini"],
        default=provider_default,
        help=f"Localization provider. Default: {provider_default}.",
    )
    parser.add_argument(
        "--gemini-backend",
        choices=["standard", "priority", "provisioned"],
        default=config_value(config, "gemini.backend", "standard"),
        help=(
            "Vertex AI Gemini request mode: standard PayGo, Priority PayGo, "
            "or Provisioned Throughput. Ignored by OpenAI."
        ),
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Use local EasyOCR for keyboard localization instead of Gemini API.",
    )
    parser.add_argument(
        "--capture-screens",
        nargs="?",
        const=True,
        type=str_to_bool,
        default=capture_screens_default,
        metavar="BOOL",
        help="Save localization/debug images when true. Defaults to capture_screens in the YAML config.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "dshow", "msmf", "any"],
        default=config_value(config, "camera.backend", "auto"),
        help="OpenCV camera backend. Defaults to camera.backend in the YAML config.",
    )
    parser.add_argument(
        "--project",
        default=project_default,
        help="Google Cloud project for Vertex AI Gemini. Defaults to gemini.project or GOOGLE_CLOUD_PROJECT.",
    )
    parser.add_argument(
        "--location",
        default=location_default,
        help="Google Cloud location for Vertex AI Gemini. Defaults to gemini.location or GOOGLE_CLOUD_LOCATION.",
    )
    parser.add_argument(
        "--urdf-path",
        default=config_value(config, "kinematics.urdf_path", "cfg/arm_model/so101_new_calib.urdf"),
        help="Path to the SO-101 URDF. Defaults to kinematics.urdf_path in the YAML config.",
    )
    parser.add_argument(
        "--press-ee-frame",
        default=config_value(config, "kinematics.press_ee_frame", DEFAULT_PRESS_EE_FRAME),
        help=(
            "URDF frame used as the physical key-contact point for pressing "
            "trajectories. Defaults to kinematics.press_ee_frame in the YAML config."
        ),
    )
    parser.add_argument(
        "--robot-port",
        default=config_value(config, "robot.port", "/dev/ttyACM0"),
        help="Serial port for the SO follower arm. Defaults to robot.port in the YAML config.",
    )
    parser.add_argument(
        "--calibration-path",
        default=config_value(config, "robot.calibration_path", "cfg/calibration/follower/<your_follower_name>.json"),
        help=(
            "Follower calibration path. The filename stem is used as the SO follower id "
            "(for example <your_follower_name>.json -> <your_follower_name>). Defaults to robot.calibration_path "
            "in the YAML config."
        ),
    )
    parser.add_argument(
        "--keyboard-height",
        type=float,
        default=config_value(config, "camera.keyboard_height", 0.018),
        help="Keyboard plane height in world coordinates, in metres. Defaults to camera.keyboard_height in the YAML config.",
    )
    parser.add_argument(
        "--hover-height",
        type=float,
        default=config_value(config, "trajectory.hover_height", 0.04),
        help="Hover height above the key, in metres. Defaults to trajectory.hover_height in the YAML config.",
    )
    parser.add_argument(
        "--press-depth",
        type=float,
        default=config_value(config, "trajectory.press_depth", 0.014),
        help="Press depth below the key plane, in metres. Defaults to trajectory.press_depth in the YAML config.",
    )
    parser.add_argument(
        "--travel-duration",
        dest="travel_duration",
        type=float,
        default=config_value(config, "trajectory.travel_duration", 0.8),
        help="Maximum duration cap for approach/final travel spline segments. Defaults to trajectory.travel_duration in the YAML config.",
    )
    parser.add_argument(
        "--press-duration",
        dest="press_duration",
        type=float,
        default=config_value(config, "trajectory.press_duration", 0.4),
        help="Fixed duration for the descent/key press spline segment. Defaults to trajectory.press_duration in the YAML config.",
    )
    parser.add_argument(
        "--approach-speed",
        type=float,
        default=config_value(config, "trajectory.approach_speed", 0.065),
        help="Approximate Cartesian speed for approach/refinement moves in m/s. Defaults to trajectory.approach_speed in the YAML config.",
    )
    parser.add_argument(
        "--min-segment-duration-default",
        type=float,
        default=config_value(config, "trajectory.min_segment_duration_default", 0.4),
        help="Minimum default duration for any generated spline segment in seconds. Defaults to trajectory.min_segment_duration_default in the YAML config.",
    )
    parser.add_argument(
        "--max-refine-steps",
        type=int,
        default=config_value(config, "trajectory.max_refine_steps", 3),
        help="Maximum adaptive hover refinement moves before pressing a key. Defaults to trajectory.max_refine_steps in the YAML config.",
    )
    parser.add_argument(
        "--refine-xy-threshold",
        type=float,
        default=config_value(config, "trajectory.refine_xy_threshold", 0.002),
        help="Stop hover refinement once end-effector/key xy error is below this many metres. Defaults to trajectory.refine_xy_threshold in the YAML config.",
    )
    parser.add_argument(
        "--estimate-stability-threshold",
        type=float,
        default=config_value(config, "trajectory.estimate_stability_threshold", 0.002),
        help="Stop hover refinement only when recent key xy estimates vary less than this many metres. Defaults to trajectory.estimate_stability_threshold in the YAML config.",
    )
    parser.add_argument(
        "--tracking-cluster-radius",
        type=float,
        default=config_value(config, "cluster.tracking_radius", 0.02),
        help="World radius in metres used to group nearby letters for continuous tracking. Defaults to cluster.tracking_radius in the YAML config.",
    )
    parser.add_argument(
        "--cluster-min-distance",
        type=float,
        default=config_value(config, "cluster.min_distance", 0.015),
        help="Minimum world distance in metres enforced between frozen clustered key positions. Defaults to cluster.min_distance in the YAML config.",
    )
    parser.add_argument(
        "--cluster-max-horizontal-delta",
        type=float,
        default=config_value(config, "cluster.max_horizontal_delta", 0.022),
        help="Maximum x-axis offset in metres allowed between a clustered key and its anchor. Defaults to cluster.max_horizontal_delta in the YAML config.",
    )
    parser.add_argument(
        "--cluster-max-vertical-delta",
        type=float,
        default=config_value(config, "cluster.max_vertical_delta", 0.014),
        help="Maximum y-axis offset in metres allowed between a clustered key and its anchor. Defaults to cluster.max_vertical_delta in the YAML config.",
    )
    parser.add_argument(
        "--cluster-excluded-letters",
        nargs="+",
        default=cluster_excluded_default,
        help="Keys handled alone instead of grouped into tracking clusters. Defaults to cluster.excluded_letters in the YAML config.",
    )
    parser.add_argument(
        "--disable-klt-for",
        nargs="+",
        default=disabled_klt_default,
        help="Keys whose estimate is held after initial localization instead of tracked with KLT. Defaults to tracking.disable_klt_for in the YAML config.",
    )

    parser.add_argument(
        "--shorter-segment-duration",
        type=float,
        default=config_value(config, "trajectory.shorter_segment_duration", 0.1),
        help="A shorter minimum duration to use for hover refinement segments after the first one. Defaults to trajectory.shorter_segment_duration in the YAML config.",
    )
    parser.add_argument(
        "--hover-offset-xy",
        nargs=2,
        type=float,
        default=parse_xy_offset(config_value(config, "trajectory.hover_offset_xy", [0.01, 0.0])),
        metavar=("X", "Y"),
        help="XY hover offset in metres added before pressing. Defaults to trajectory.hover_offset_xy in the YAML config.",
    )
    parser.add_argument(
        "--first-hover-height-scale",
        type=float,
        default=config_value(config, "trajectory.first_hover_height_scale", 1.5),
        help="Height multiplier for the first tracked hover approach. Defaults to trajectory.first_hover_height_scale in the YAML config.",
    )
    parser.add_argument(
        "--locked-refine-steps",
        type=int,
        default=config_value(config, "trajectory.locked_refine_steps", 2),
        help="Maximum refinement moves when tracking is locked or disabled. Defaults to trajectory.locked_refine_steps in the YAML config.",
    )
    parser.add_argument(
        "--final-home-hold-multiplier",
        type=float,
        default=config_value(config, "trajectory.final_home_hold_multiplier", 5.0),
        help="Hold-time multiplier when returning home after pressing. Defaults to trajectory.final_home_hold_multiplier in the YAML config.",
    )
    parser.add_argument(
        "--final-hover-hold-multiplier",
        type=float,
        default=config_value(config, "trajectory.final_hover_hold_multiplier", 2.0),
        help="Hold-time multiplier when returning to hover after pressing. Defaults to trajectory.final_hover_hold_multiplier in the YAML config.",
    )

    parser.add_argument(
        "--internal-p-coefficient",
        type=int,
        default=config_value(config, "internal_controller.p_coefficient", 20),
        help="Internal motor P coefficient. Defaults to internal_controller.p_coefficient in the YAML config.",
    )
    parser.add_argument(
        "--internal-i-coefficient",
        type=int,
        default=config_value(config, "internal_controller.i_coefficient", 1),
        help="Internal motor I coefficient. Defaults to internal_controller.i_coefficient in the YAML config.",
    )
    parser.add_argument(
        "--internal-d-coefficient",
        type=int,
        default=config_value(config, "internal_controller.d_coefficient", 16),
        help="Internal motor D coefficient. Defaults to internal_controller.d_coefficient in the YAML config.",
    )
    parser.add_argument(
        "--initial-home-sleep-s",
        type=float,
        default=config_value(config, "timing.initial_home_sleep_s", 2.0),
        help="Seconds to wait after the initial go-home. Defaults to timing.initial_home_sleep_s in the YAML config.",
    )
    parser.add_argument(
        "--post-run-tracker-update-s",
        type=float,
        default=config_value(config, "timing.post_run_tracker_update_s", 1.0),
        help="Seconds to keep updating the tracker after returning home. Defaults to timing.post_run_tracker_update_s in the YAML config.",
    )

    args = parser.parse_args()

    if args.task_1:
        if args.task is not None and args.task != 1:
            parser.error("--task-1 cannot be combined with --task 2 or --task 3.")
        args.task = 1
    current_task_config = task_config(config, args.task)
    if args.task is not None:
        if args.word is None and args.list_path is None and current_task_config.get("list_path"):
            args.list_path = Path(current_task_config["list_path"])
        if not arg_was_passed("--provider", raw_argv) and current_task_config.get("provider"):
            args.provider = current_task_config["provider"]
        if not arg_was_passed("--model", raw_argv) and current_task_config.get("model"):
            args.model = current_task_config["model"]
        if not arg_was_passed("--gemini-backend", raw_argv) and current_task_config.get("gemini_backend"):
            args.gemini_backend = current_task_config["gemini_backend"]
    if args.task is None and args.word is None and args.list_path is None:
        parser.error("Pass --word, --task-1, --task 1, or --list-path.")

    if args.provider == "gemini" and not arg_was_passed("--model", raw_argv) and args.model == model_default and model_default == "gpt-5.5":
        args.model = "gemini-3-flash-preview"
    args.prompt_context = current_task_config.get("prompt_context")
    args.prompt_instructions = normalize_string_list(current_task_config.get("prompt_instructions"))
    args.cluster_excluded_letters = normalize_key_list(args.cluster_excluded_letters, [])
    args.disable_klt_for = normalize_key_list(args.disable_klt_for, [])
    args.hover_offset_xy = parse_xy_offset(args.hover_offset_xy)
    try:
        home_position_deg = np.asarray(args.home_position_deg, dtype=float)
    except (TypeError, ValueError):
        parser.error(
            "home_position_deg must contain six numeric joint angles in degrees. "
            "Replace the placeholders in cfg/main_pipeline.yaml with a safe home "
            "pose for your own robot/keyboard setup, or pass six values with "
            "--home-position-deg."
        )
    if home_position_deg.shape != (6,):
        parser.error("home_position_deg / --home-position-deg must contain exactly 6 joint values.")
    args.home_position_rad = np.deg2rad(home_position_deg)
    return args

def main() -> np.ndarray | None:
    args = parse_args()
    typing_runs = build_typing_runs(args)

    tracking_kinematics = RobotKinematics(urdf_path=args.urdf_path)
    pressing_kinematics = RobotKinematics(urdf_path=args.urdf_path, ee_frame=args.press_ee_frame)

    robot_interface = SO101Interface(
        port=args.robot_port,
        calibration_path=args.calibration_path,
    )
    print("Robot is now connected")

    print("Changing PID coefficients of internal motors...")
    robot_interface.initialize_internal_controller(
        p_coefficient=args.internal_p_coefficient,
        i_coefficient=args.internal_i_coefficient,
        d_coefficient=args.internal_d_coefficient,
    )

    go_home(robot_interface, tracking_kinematics, q_home_rad=args.home_position_rad)
    time.sleep(args.initial_home_sleep_s)

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
                provider=args.provider,
                gemini_backend=args.gemini_backend,
                project=args.project,
                location=args.location,
                keyboard_height=args.keyboard_height,
                backend=args.backend,
                use_ocr=args.ocr,
                capture_screens=args.capture_screens,
                disable_klt_for=args.disable_klt_for,
                prompt_context=args.prompt_context,
                prompt_instructions=args.prompt_instructions,
            )

            try:
                print("Main operation loop starting ...")
                tracker.start(robot_interface=robot_interface, kinematics=tracking_kinematics)

                cluster_manager = KeyboardClusterManager.from_tracker(
                    tracker,
                    letters,
                    tracking_radius=args.tracking_cluster_radius,
                    min_distance=args.cluster_min_distance,
                    max_horizontal_delta=args.cluster_max_horizontal_delta,
                    max_vertical_delta=args.cluster_max_vertical_delta,
                    excluded_letters=args.cluster_excluded_letters,
                )
                q_home_config = np.rad2deg(args.home_position_rad)
                last_target_index = len(cluster_manager.runtime_targets) - 1

                for index, target in cluster_manager.indexed_targets():
                    cluster_plan = cluster_manager.prepare_target(
                        index,
                        target,
                        tracker,
                        robot_interface=robot_interface,
                        kinematics=tracking_kinematics,
                    )

                    pressed_key_position = deliver_typing_trajectory(
                        key_position=cluster_plan.key_position,
                        tracker=tracker,
                        robot_interface=robot_interface,
                        hover_height=args.hover_height,
                        press_depth=args.press_depth,
                        kinematics=pressing_kinematics,
                        tracking_kinematics=tracking_kinematics,
                        travel_duration=args.travel_duration,
                        press_duration=args.press_duration,
                        q_final_config=q_home_config if cluster_plan.should_go_home_after_press else None,
                        track_during_hover=cluster_plan.track_during_hover,
                        lock_key_position=cluster_plan.lock_key_position,
                        approach_speed=args.approach_speed,
                        min_segment_duration_default=args.min_segment_duration_default,
                        max_refine_steps=args.max_refine_steps,
                        refine_xy_threshold=args.refine_xy_threshold,
                        estimate_stability_threshold=args.estimate_stability_threshold,
                        shorter_segment_duration=args.shorter_segment_duration,
                        hover_offset_xy=args.hover_offset_xy,
                        first_hover_height_scale=args.first_hover_height_scale,
                        locked_refine_steps=args.locked_refine_steps,
                        final_home_hold_multiplier=args.final_home_hold_multiplier,
                        final_hover_hold_multiplier=args.final_hover_hold_multiplier,
                    )
                    if index == last_target_index and tracker.localization_start_time_s is not None:
                        elapsed_s = time.perf_counter() - tracker.localization_start_time_s
                        print(
                            "Elapsed time from ENTER localization trigger to "
                            f"last trajectory point for run `{run_label}`: "
                            f"{elapsed_s:.3f} s"
                        )

                    cluster_manager.finish_target(
                        cluster_plan,
                        pressed_key_position,
                        tracker,
                    )
            finally:

                go_home(robot_interface, tracking_kinematics, q_home_rad=args.home_position_rad)
                if tracker.cap is not None:
                    update_tracker_for_duration(
                        tracker=tracker,
                        duration_s=args.post_run_tracker_update_s,
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
