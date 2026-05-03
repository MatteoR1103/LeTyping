from __future__ import annotations

import argparse
import os
import time
import numpy as np

try:
    from .tracker import KeyWorldTracker
    from .tracking_script import DEFAULT_LIVE_MODEL
    from controller import SO101Interface 
    from traj_generation import RobotKinematics, deliver_typing_trajectory
except ImportError:
    from tracker import KeyWorldTracker
    from tracking_script import DEFAULT_LIVE_MODEL
    from controller import SO101Interface
    from traj_generation import RobotKinematics, deliver_typing_trajectory


DEFAULT_URDF_PATH = "cfg/arm_model/so101_new_calib.urdf"
ROBOT_PORT = "/dev/ttyACM1"

DEFAULT_HOME_POSITION =np.array(np.deg2rad([3.07692308, -33.14285714,  41.18681319,  61.8021978,  -89.62637363, 0.0]))  # in degrees

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate a keyboard key in world coordinates and press it with the SO-101."
    )
    
    parser.add_argument(
        "--word", 
        required=True, 
        nargs="+",
        type=str,
        help="The word or letters to type. For example, 'CAT' or C A T."
    )
    
    parser.add_argument(
        "--camera",
        type=int,
        default=5, # for Piro, for Rub the camera index is 2
        help="OpenCV camera index. Default: 5."
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
        "--keyboard-height",
        type=float,
        default=0.02,
        help="Keyboard plane height in world coordinates, in metres. Default: 0.0.",
    )
    parser.add_argument(
        "--hover-height",
        type=float,
        default=0.03,
        help="Hover height above the key, in metres. Default: 0.05.",
    )
    parser.add_argument(
        "--press-depth",
        type=float,
        default=0.01,
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
    
    parser.add_argument(
        "--localization_mode",
        type=str,
        default="ray",
        help="Localization mode of the pipeline - available modes: [homography, ray]",
    )

    return parser.parse_args()


def main() -> np.ndarray | None:
    """
    Pipeline main function: instantiates the tracker, reads joints, computes a trajectory and executes it
    """

    #PARSE ARGUMENTS
    args = parse_args()
    letters = [
        letter
        for token in args.word
        for letter in token.replace(",", "").upper()
        if letter.strip()
    ]
    if not letters:
        raise ValueError("At least one target letter is required.")

    #URDF PATH FOR FK
    urdf_path = args.urdf_path

    #INSTANTIATE THE TRACKER TO TRACK POINTS WITH KLT DURING OPERATION
    tracker = KeyWorldTracker(
        letter=",".join(letters),
        camera=args.camera,
        model=args.model,
        project=args.project,
        location=args.location,
        keyboard_height=args.keyboard_height,
        backend=args.backend,
        localization_mode=args.localization_mode
    )

    #KINEMATICS CLASS FOR FK AND IK FOR TRAJECTORY GENERATION AND POSE ESTIMATION 
    kinematics = RobotKinematics(urdf_path=urdf_path) # expects rads

    #ROBOT INTERFACE TO READ AND WRITE JOINTS
    robot_interface = SO101Interface(port=args.robot_port)
    print("Robot is now connected")
    print("Changing PID coefficients of internal motors...")
    
    
    #MAIN OPERATION LOOP
    print("Main operation loop starting ...")
    try:

        robot_interface.write_joints(DEFAULT_HOME_POSITION)  # Move to a home position to start
        
        # INITIALIZE THE WORLD KEYPOINT LOCATIONS AND THE CURRENT JOINTS in DEGREES
        key_pos, q_current = tracker.start(robot_interface=robot_interface, kinematics=kinematics)
        print(f"Estimated key_pos world: {key_pos}")

        
        robot_interface.robot.bus.enable_torque()
        for motor in robot_interface.robot.bus.motors:
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            robot_interface.robot.bus.write("P_Coefficient", motor, 20)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            robot_interface.robot.bus.write("I_Coefficient", motor, 5)
            robot_interface.robot.bus.write("D_Coefficient", motor, 16)
        
#--------------------------- GENERATING AND EXECUTING A TRAJECTORY FOR EACH LETTER ---------------------------#
        for target in tracker.targets:
            q_current = np.rad2deg(robot_interface.read_joints()[0])
            tracker.set_target(
                pixel=target["pixel"],
                world=target["world"],
                letter=target["letter"],
            )
            deliver_typing_trajectory(
                key_position=target["world"],
                tracker=tracker,
                robot_interface=robot_interface,
                hover_height=args.hover_height,
                press_depth=args.press_depth,
                q_current=q_current,
                kinematics=kinematics,
                travel_duration=args.travel_duration,
                press_duration=args.press_duration
            )
            robot_interface.write_joints(DEFAULT_HOME_POSITION)
            tracker.update_for_duration(
                1.5,
                robot_interface=robot_interface,
                kinematics=kinematics,
            )
    finally:
        robot_interface.write_joints(DEFAULT_HOME_POSITION)  # Move to a home position just for the sake of it
        if tracker.cap is not None:
            tracker.update_for_duration(
                1.0,
                robot_interface=robot_interface,
                kinematics=kinematics,
            )
        else:
            time.sleep(1.0)  # wait for the robot to reach home before closing connection and ending the program
        input("Press ENTER when the robot is back at the home position to disconnect...")
        tracker.close()
        robot_interface.close()


if __name__ == "__main__":
    main()
