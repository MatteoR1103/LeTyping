from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np

try:
    from .tracker import KeyWorldTracker
    from .tracking_script import DEFAULT_LIVE_MODEL
    from controller import SO101Interface, execute_joint_trajectory
    from traj_generation import RobotKinematics, generate_typing_trajectory
except ImportError:
    from tracker import KeyWorldTracker
    from tracking_script import DEFAULT_LIVE_MODEL
    from controller import SO101Interface, execute_joint_trajectory
    from traj_generation import RobotKinematics, generate_typing_trajectory


DEFAULT_URDF_PATH = "cfg/arm_model/so101_new_calib.urdf"
ROBOT_PORT = "/dev/ttyACM0"

DEFAULT_HOME_POSITION = np.array(np.deg2rad([0.0, -30.0, 30.0, 60.0, -90.0, 0.0]))  # in radians

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate a keyboard key in world coordinates and press it with the SO-101."
    )
    
    parser.add_argument(
        "--word", 
        required=True, 
        type=str, 
        help="The word to type. For example, 'HELLO'."
    )
    
    parser.add_argument(
        "--camera",
        type=int,
        default=1, 
        help="OpenCV camera index. Default: 1."
    )
    
    parser.add_argument(
        "--model",
        default=DEFAULT_LIVE_MODEL,
        help=f"Gemini model used for initial localization. Default: {DEFAULT_LIVE_MODEL}.",
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
        help="Path to the SO-101 URDF."
    )
    
    parser.add_argument(
        "--robot-port",
        default=ROBOT_PORT,
        help="Serial port for the SO follower arm, for example /dev/ttyACM0.",
    )

    parser.add_argument(
        "--no-robot", 
        action="store_true", 
        help="Estimate and plan without motor commands."
    )
    
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

    parser.add_argument(
        "--travel_duration",
        type=float,
        default=0.8,
        help="Duration of the travel phase in seconds. Default: 0.8.",
    )

    parser.add_argument(
        "--press_duration",
        type=float,
        default=0.3,
        help="Duration of the press phase in seconds. Default: 0.3.",
    )

    return parser.parse_args()


def try_plan_no_robot(
    *,
    key_pos: np.ndarray,
    urdf_path: Path,
    hover_height: float,
    press_depth: float,
    travel_duration: float,
    press_duration: float,
) -> None:
    try:
        kinematics = RobotKinematics(urdf_path=urdf_path)
        q_current = kinematics.neutral_configuration()
        _, _, t_exec = generate_typing_trajectory(
            key_positions=[key_pos],
            q_current=q_current,
            kinematics=kinematics,
            hover_height=hover_height,
            press_depth=press_depth,
            travel_duration=travel_duration,
            press_duration=press_duration,
            dt=0.02,
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
    #URDF PATH FOR FK
    urdf_path = args.urdf_path if not args.no_robot else None

    #INSTANTIATE THE TRACKER TO TRACK POINTS WITH KLT DURING OPERATION
    tracker = KeyWorldTracker(
        letter=args.word,
        camera=args.camera,
        model=args.model,
        project=args.project,
        location=args.location,
        keyboard_height=args.keyboard_height,
        backend=args.backend,
    )

    #NO ROBOT PATH FOR VERIFICATION 
    if args.no_robot:
        try:
            print("Running in no-robot mode: using a fixed T_WG = I pose for testing.")
            key_pos, _ = tracker.start(np.eye(4))
            print(f"Estimated key_pos world: {key_pos}")
            try:
                urdf_path = args.urdf_path
            except FileNotFoundError as exc:
                print(f"Trajectory generation skipped in --no-robot mode: {exc}")
                print("Execution skipped because --no-robot is set.")
                return
            try_plan_no_robot(
                key_pos=key_pos,
                urdf_path=urdf_path,
                hover_height=args.hover_height,
                press_depth=args.press_depth,
                travel_duration=args.travel_duration,
                press_duration=args.press_duration,
            )
        finally:
            if 'robot_interface' in locals() and robot_interface is not None:
                robot_interface.write_joints(DEFAULT_HOME_POSITION)
                input("Press ENTER when the robot is back at the home position to disconnect...")
                robot_interface.close()
            tracker.close()
        return

    #KINEMATICS CLASS FOR FK AND IK FOR TRAJECTORY GENERATION AND POSE ESTIMATION 
    kinematics = RobotKinematics(urdf_path=urdf_path) # expects rads

    #ROBOT INTERFACE TO READ AND WRITE JOINTS
    robot_interface = SO101Interface(port=args.robot_port)
    print("Robot is now connected")
    print("Changing PID coefficients of internal motors...")
    
    for motor in robot_interface.robot.bus.motors:
        # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
        robot_interface.robot.bus.write("P_Coefficient", motor, 16)
        # Set I_Coefficient and D_Coefficient to default value 0 and 32
        robot_interface.robot.bus.write("I_Coefficient", motor, 5)
        robot_interface.robot.bus.write("D_Coefficient", motor, 16)
    
    #MAIN OPERATION LOOP
    print("Main operation loop starting ...")
    try:

        robot_interface.write_joints(DEFAULT_HOME_POSITION)  # Move to a home position to start
        # INITIALIZE THE WORLD KEYPOINT LOCATION AND THE CURRENT JOINTS in DEGREES
        key_pos, q_current = tracker.start(robot_interface=robot_interface, kinematics=kinematics)
        key_pos = np.array([[ 0.27428768,  0.07951523, -0.0095034 ]])
        print(f"Estimated key_pos world: {key_pos}")
        
        #GENERATE THE TRAJECTORY AT STARTUP
        q_traj, dq_traj, t_exec = generate_typing_trajectory(
            key_positions=[key_pos],
            q_current=q_current,
            kinematics=kinematics,
            hover_height=args.hover_height,
            press_depth=args.press_depth,
            travel_duration=args.travel_duration,
            press_duration=args.press_duration,
            dt=0.02,
        ) #in radians 
    

        print(f"Generated trajectory length: {len(t_exec)} samples")
        print("Starting trajectory execution.")

        def update_tracker(i) -> None:
            _ = tracker.update(i, robot_interface=robot_interface, kinematics=kinematics)
            #if i % 10 == 0:
                #print(f"Tracked key_pos world: {updated_key_pos}")

        execute_joint_trajectory(
            robot_interface=robot_interface,
            q_traj=q_traj, #radians
            dq_traj=dq_traj, #radians/s
            t_exec=t_exec,
            kinematics=kinematics,
            step_callback=update_tracker,
        )
        
    finally:
        #robot_interface.write_joints(DEFAULT_HOME_POSITION)
        input("Press ENTER when the robot is back at the home position to disconnect...")
        tracker.close()
        # Move back to home position on early exit
        robot_interface.close()


if __name__ == "__main__":
    main()
