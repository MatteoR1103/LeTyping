from __future__ import annotations

import argparse
import os
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


def fallback_provider_for_task(task: int) -> str:
    return "gemini" if task == 2 else "openai"


def fallback_model_for_provider(provider: str) -> str:
    return "gemini-3-flash-preview" if provider == "gemini" else "gpt-5.5"


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    config_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the main pipeline YAML config. Default: {DEFAULT_CONFIG_PATH}.",
    )
    config_args, _ = config_parser.parse_known_args()
    config = load_pipeline_config(config_args.config)
    task1_targets_default = config_value(config, "task1_targets", ["SPACE", "ENTER", "R", "L"])
    home_position_default = config_value(
        config,
        "home_position_deg",
        [3.07692308, -33.14285714, 41.18681319, 61.8021978, -89.62637363, 50.0],
    )
    model_default = config_value(config, "gemini.model", "gpt-5.5")
    provider_default = config_value(config, "gemini.provider", "openai")
    project_default = config_value(config, "gemini.project", os.getenv("GOOGLE_CLOUD_PROJECT"))
    location_default = config_value(config, "gemini.location", os.getenv("GOOGLE_CLOUD_LOCATION", "global"))
    capture_screens_default = str_to_bool(config_value(config, "capture_screens", False))

    parser = argparse.ArgumentParser(
        description="Estimate keyboard keys in world coordinates and press them with the SO-101.",
        parents=[config_parser],
        allow_abbrev=False,
    )

    parser.add_argument(
        "--task",
        type=int,
        choices=[1, 2, 3],
        help=(
            "Competition task number. Task 1 uses the predefined targets; "
            "tasks 2 and 3 require --word or --list-path."
        ),
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
        help="Run predefined task 1: presses SPACE, ENTER, R, L in order.",
    )
    run_source.add_argument(
        "--list-path",
        type=Path,
        help="Path to a text file with one word or sentence per row.",
    )
    parser.add_argument(
        "--task-1-targets",
        nargs="+",
        default=task1_targets_default,
        help="Targets used by --task-1. Defaults to task1_targets in the YAML config.",
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
        "--task3-poil-pixel-x-bias",
        type=float,
        default=config_value(config, "camera.task3_poil_pixel_x_bias", 4.0),
        help="Task 3 only: shift P/O/I/L localized pixels to the right by this many pixels.",
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
        default=config_value(config, "robot.calibration_path", "cfg/calibration/follower/zi_padrone.json"),
        help=(
            "Follower calibration path. The filename stem is used as the SO follower id "
            "(for example zi_padrone.json -> zi_padrone). Defaults to robot.calibration_path "
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
        "--task3-space-extra-press-depth",
        type=float,
        default=config_value(config, "trajectory.task3_space_extra_press_depth", 0.002),
        help="Task 3 only: extra press depth for SPACE, in metres.",
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
        "--shorter-segment-duration",
        type=float,
        default=config_value(config, "trajectory.shorter_segment_duration", 0.1),
        help="A shorter minimum duration to use for hover refinement segments after the first one. Defaults to trajectory.shorter_segment_duration in the YAML config.",
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
    if args.task == 1 and (args.word is not None or args.list_path is not None):
        parser.error("--task 1 uses --task-1-targets and does not accept --word or --list-path.")
    if args.task in {2, 3} and args.word is None and args.list_path is None:
        parser.error(f"--task {args.task} requires --word or --list-path.")
    if args.task is None and args.word is None and args.list_path is None:
        parser.error("Pass --word, --task-1, --task 1, or --list-path.")

    if args.ocr and args.task is not None:
        args.provider = fallback_provider_for_task(args.task)
        if args.provider == "gemini" and args.model.startswith("gpt-"):
            args.model = fallback_model_for_provider(args.provider)
        elif args.provider == "openai" and args.model.startswith("gemini-"):
            args.model = fallback_model_for_provider(args.provider)
    if args.provider == "gemini" and args.model == model_default and model_default == "gpt-5.5":
        args.model = "gemini-3-flash-preview"
    if args.task == 3:
        args.keyboard_height = config_value(config, "task3_parameters.keyboard_height", args.keyboard_height)
        args.hover_height = config_value(config, "task3_parameters.hover_height", args.hover_height)
        args.press_depth = config_value(config, "task3_parameters.press_depth", args.press_depth)
    args.task1_targets = [str(target).upper() for target in args.task_1_targets]
    home_position_deg = np.asarray(args.home_position_deg, dtype=float)
    if home_position_deg.shape != (6,):
        parser.error("home_position_deg / --home-position-deg must contain exactly 6 joint values.")
    args.home_position_rad = np.deg2rad(home_position_deg)
    return args

def main() -> np.ndarray | None:
    args = parse_args()
    typing_runs = build_typing_runs(args, task1_targets=args.task1_targets)

    # ------------- Class initialization ------------- #
    tracking_kinematics = RobotKinematics(urdf_path=args.urdf_path)
    pressing_kinematics = RobotKinematics(urdf_path=args.urdf_path, ee_frame=args.press_ee_frame)
    # print("Tracking/camera kinematics frame: gripper_frame_link")
    # print(f"Pressing/contact kinematics frame: {args.press_ee_frame}")

    robot_interface = SO101Interface(
        port=args.robot_port,
        calibration_path=args.calibration_path,
    )
    print("Robot is now connected")

    # ------------- Set internal controller parameters ------------- #
    print("Changing PID coefficients of internal motors...")
    robot_interface.initialize_internal_controller(
        p_coefficient=args.internal_p_coefficient,
        i_coefficient=args.internal_i_coefficient,
        d_coefficient=args.internal_d_coefficient,
    )

    # ------------- Initial go-home ------------- #
    go_home(robot_interface, tracking_kinematics, q_home_rad=args.home_position_rad)
    time.sleep(args.initial_home_sleep_s)

    # ------------- Tracker initialization ------------- #
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
                task3_poil_pixel_x_bias=args.task3_poil_pixel_x_bias if args.task == 3 else 0.0,
            )
            
            # ------------- Main operation loop ------------- #
            try:
                print("Main operation loop starting ...")
                tracker.start(robot_interface=robot_interface, kinematics=tracking_kinematics)

                # ------------- Initialize cluster manager for this run's targets ------------- #
                cluster_manager = KeyboardClusterManager.from_tracker(
                    tracker,
                    letters,
                    tracking_radius=args.tracking_cluster_radius,
                    min_distance=args.cluster_min_distance,
                    max_horizontal_delta=args.cluster_max_horizontal_delta,
                    max_vertical_delta=args.cluster_max_vertical_delta,
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

                    print(
                        f"Commanded key position for letter {cluster_plan.current_letter}: "
                        f"{cluster_plan.key_position}"
                    )

                    # ------------- Deliver trajectory and press key ------------- #
                    press_depth_for_key = args.press_depth
                    if args.task == 3 and cluster_plan.current_letter == "SPACE":
                        press_depth_for_key += args.task3_space_extra_press_depth
                        # print(
                        #     "Task 3 SPACE extra press depth: "
                        #     f"+{args.task3_space_extra_press_depth:.4f}m -> {press_depth_for_key:.4f}m"
                        # )

                    pressed_key_position = deliver_typing_trajectory(
                        key_position=cluster_plan.key_position,
                        tracker=tracker,
                        robot_interface=robot_interface,
                        hover_height=args.hover_height,
                        press_depth=press_depth_for_key,
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
                    )
                    if index == last_target_index and tracker.localization_start_time_s is not None:
                        elapsed_s = time.perf_counter() - tracker.localization_start_time_s
                        print(
                            "Elapsed time from ENTER localization trigger to "
                            f"last trajectory point for run `{run_label}`: "
                            f"{elapsed_s:.3f} s"
                        )

                    # ------------- Update cluster manager with press result and decide next steps ------------- #
                    cluster_manager.finish_target(
                        cluster_plan,
                        pressed_key_position,
                        tracker,
                    )
            finally:

                # ------------- Return home and keep updating tracker for a bit ------------- #
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
