from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from track_to_wld import DEFAULT_LIVE_MODEL, KeyWorldTracker, parse_fallback_models


DEFAULT_URDF_PATH = "cfg/arm_model/so101_new_calib.urdf"
ROBOT_PORT = "/dev/ttyACM0"

DEFAULT_HOME_POSITION = np.array(np.deg2rad([0.0, -30.0, 30.0, 40.0, 0.0, 0.0]))  # in radians

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate a keyboard key in world coordinates and press it with the SO-101."
    )
    parser.add_argument("--letter", required=True, help="Single keyboard letter to press, for example X.")
    parser.add_argument("--camera", type=int, default=1, help="OpenCV camera index. Default: 1.")
    parser.add_argument(
        "--model",
        default=DEFAULT_LIVE_MODEL,
        help=f"Gemini model used for initial localization. Default: {DEFAULT_LIVE_MODEL}.",
    )
    parser.add_argument(
        "--fallback-models",
        default="",
        help="Optional comma-separated fallback Gemini models. Default: none.",
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
    parser.add_argument("--urdf-path", default=DEFAULT_URDF_PATH, help="Path to the SO-101 URDF.")
    parser.add_argument(
        "--robot-port",
        default=ROBOT_PORT,
        help="Serial port for the SO follower arm, for example /dev/ttyACM0.",
    )
    parser.add_argument("--no-robot", action="store_true", help="Estimate and plan without motor commands.")
    parser.add_argument(
        "--keyboard-height",
        type=float,
        default=0.02,
        help="Keyboard plane height in world coordinates, in metres. Default: 0.0.",
    )
    parser.add_argument(
        "--hover-height",
        type=float,
        default=0.0,
        help="Hover height above the key, in metres. Default: 0.05.",
    )
    parser.add_argument(
        "--press-depth",
        type=float,
        default=0.0,
        help="Press depth below the key plane, in metres. Default: 0.005.",
    )
    return parser.parse_args()


def resolve_urdf_path(path: str) -> Path:
    urdf_path = Path(path)
    if not urdf_path.is_absolute():
        urdf_path = Path(__file__).resolve().parents[1] / urdf_path
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    return urdf_path


def try_plan_no_robot(
    *,
    key_pos: np.ndarray,
    urdf_path: Path,
    hover_height: float,
    press_depth: float,
) -> None:
    try:
        from traj_generation import RobotKinematics, generate_key_press_trajectory

        kinematics = RobotKinematics(urdf_path=urdf_path)
        q_current = kinematics.neutral_configuration()
        q_traj, dq_traj, t_exec = generate_key_press_trajectory(
            key_pos,
            q_current,
            kinematics,
            hover_height=hover_height,
            press_depth=press_depth,
        )
    except (ImportError, RuntimeError, FileNotFoundError, SystemExit) as exc:
        print(f"Trajectory generation skipped in --no-robot mode: {exc}")
        return

    print(f"Generated trajectory length: {len(t_exec)} samples")
    print("Execution skipped because --no-robot is set.")


def main() -> None:
    """
    Pipeline main function: instantiates the tracker, reads joints, computes a trajectory and executes it
    """

    #PARSE ARGUMENTS
    args = parse_args()
    #FALLBACK MODEL FOR GEMINI LOCALIZATION
    fallback_models = parse_fallback_models(args.fallback_models)
    #URDF PATH FOR FK
    urdf_path = resolve_urdf_path(args.urdf_path) if not args.no_robot else None

    #INSTANTIATE THE TRACKER TO TRACK POINTS WITH KLT DURING OPERATION

    tracker = KeyWorldTracker(
        letter=args.letter,
        camera=args.camera,
        model=args.model,
        fallback_models=fallback_models,
        project=args.project,
        location=args.location,
        keyboard_height=args.keyboard_height,
        backend=args.backend,
    )

    #NO ROBOT PATH FOR VERIFICATION 
    if args.no_robot:
        try:
            print("Running in no-robot mode: using a fixed T_WG = I pose for testing.")
            key_pos = tracker.start(np.eye(4))
            print(f"Estimated key_pos world: {key_pos}")
            try:
                urdf_path = resolve_urdf_path(args.urdf_path)
            except FileNotFoundError as exc:
                print(f"Trajectory generation skipped in --no-robot mode: {exc}")
                print("Execution skipped because --no-robot is set.")
                return
            try_plan_no_robot(
                key_pos=key_pos,
                urdf_path=urdf_path,
                hover_height=args.hover_height,
                press_depth=args.press_depth,
            )
        finally:
            tracker.close()
        return


    from controller import SO101Interface, execute_joint_trajectory
    from traj_generation import RobotKinematics, generate_key_press_trajectory
    from lerobot.model.kinematics import RobotKinematics as RK
    #KINEMATICS CLASS FOR FK AND IK FOR TRAJECTORY GENERATION AND POSE ESTIMATION 
    kinematics = RobotKinematics(urdf_path=DEFAULT_URDF_PATH) # expects rads
    #kinematics_tracking = RK(urdf_path=DEFAULT_URDF_PATH, target_frame_name="gripper_frame_link")

    #ROBOT INTERFACE TO READ AND WRITE JOINTS
    robot_interface = SO101Interface(port=args.robot_port)
    print("Robot is now connected")
    execution_completed = False
    #MAIN OPERATION LOOP
    print("Main operation loop starting ...")
    try:
        robot_interface.robot.bus.enable_torque()
        robot_interface.robot.write_joints(DEFAULT_HOME_POSITION)  # Move to a home position to start
        # INITIALIZE THE WORLD KEYPOINT LOCATION AND THE CURRENT JOINTS in DEGREES
        key_pos, q_current = tracker.start(robot_interface=robot_interface, kinematics=kinematics)
        print(f"Estimated key_pos world: {key_pos}")
        
        #GENERATE THE TRAJECTORY AT STARTUP
        q_traj, dq_traj, t_exec = generate_key_press_trajectory(
            key_pos,
            q_current,
            kinematics,
            hover_height=args.hover_height,
            press_depth=args.press_depth,
        ) #in radians 
    

        print(f"Generated trajectory length: {len(t_exec)} samples")
        print("Starting trajectory execution.")

        def update_tracker(i) -> None:
            updated_key_pos = tracker.update(robot_interface=robot_interface, kinematics=kinematics)
            if i % 10 == 0:
                print(f"Tracked key_pos world: {updated_key_pos}")

        execute_joint_trajectory(
            robot_interface,
            q_traj, #radians
            dq_traj, #radians/s
            t_exec,
            kinematics,
            step_callback=update_tracker,
        )

        execution_completed = True
    finally:
        tracker.close()
        if not execution_completed:
            robot_interface.robot.write_joints(DEFAULT_HOME_POSITION)  # Move back to home position on early exit
            robot_interface.close()


if __name__ == "__main__":
    main()
