from __future__ import annotations

import argparse
import os
import time
import numpy as np

try:
    from .tracker import KeyWorldTracker
    from controller import SO101Interface 
    from traj_generation import RobotKinematics, deliver_typing_trajectory
except ImportError:
    from tracker import KeyWorldTracker
    from controller import SO101Interface
    from traj_generation import RobotKinematics, deliver_typing_trajectory

try:
    from .utils.tracking_utils import update_tracker_for_duration
except ImportError:
    from utils.tracking_utils import update_tracker_for_duration


DEFAULT_URDF_PATH = "cfg/arm_model/so101_new_calib.urdf"
ROBOT_PORT = "/dev/ttyACM0"
TASK1_TARGETS = ["SPACE", "ENTER", "R", "L"]
DEFAULT_LIVE_MODEL = "gemini-3-flash-preview"

DEFAULT_HOME_POSITION =np.array(np.deg2rad([3.07692308, -33.14285714,  41.18681319,  61.8021978,  -89.62637363, 50.0]))  # in degrees

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate a keyboard key in world coordinates and press it with the SO-101."
    )
    
    parser.add_argument(
        "--word", 
        required=False,
        nargs="+",
        type=str,
        help="The word, letters, or sentence to type. For example, CAT, C A T, or \"RUB IS GOAT\"."
    )

    parser.add_argument(
        "--task",
        choices=["1"],
        help="Run a predefined task. Task 1 presses SPACE, ENTER, R, L in order.",
    )
    
    parser.add_argument(
        "--camera",
        type=int,
        default=4, # for Piro, for Rub the camera index is 2
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
        "--calibration-path",
        default="cfg/calibration/follower/zi_padrone.json",
        help="Optional calibration file path forwarded to the SO101 interface.",
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


def parse_typing_targets(word_args: list[str]) -> list[str]:
    text = " ".join(word_args)
    targets: list[str] = []
    for char in text:
        if char.isspace():
            targets.append("SPACE")
        elif char.isalpha():
            targets.append(char.upper())
    return targets


def main() -> np.ndarray | None:
    """
    Pipeline main function: instantiates the tracker, reads joints, computes a trajectory and executes it
    """

    #PARSE ARGUMENTS
    args = parse_args()
    if args.task == "1":
        letters = TASK1_TARGETS.copy()
    elif args.word is not None:
        letters = parse_typing_targets(args.word)
    else:
        raise ValueError("Pass --word or --task 1.")
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
    robot_interface = SO101Interface(
        port=args.robot_port,
        calibration_path=args.calibration_path,
    )
    print("Robot is now connected")
    print("Changing PID coefficients of internal motors...")
    

    #MAIN OPERATION LOOP
    print("Main operation loop starting ...")
    try:

        robot_interface.write_joints(DEFAULT_HOME_POSITION)  # Move to a home position to start
        time.sleep(1.0)
        robot_interface.robot.bus.enable_torque()
        for motor in robot_interface.robot.bus.motors:
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            robot_interface.robot.bus.write("P_Coefficient", motor, 20)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            robot_interface.robot.bus.write("I_Coefficient", motor, 5)
            robot_interface.robot.bus.write("D_Coefficient", motor, 16)
        
        # INITIALIZE THE WORLD KEYPOINT LOCATIONS AND THE CURRENT JOINTS in DEGREES
        key_pos, q_current = tracker.start(robot_interface=robot_interface, kinematics=kinematics)
        print(f"Estimated key_pos world: {key_pos}")

    
#--------------------------- GENERATING AND EXECUTING A TRAJECTORY FOR EACH LETTER ---------------------------#
        runtime_targets = [dict(tracker.targets_by_letter[letter]) for letter in letters]
        q_home_config = np.rad2deg(DEFAULT_HOME_POSITION)
        robot_at_hover = False
        previous_letter: str | None = None
        previous_commanded_key_position: np.ndarray | None = None

        for index, target in enumerate(runtime_targets):
            immediate_next = runtime_targets[index + 1] if index + 1 < len(runtime_targets) else None
            repeat_current = robot_at_hover and previous_letter == target["letter"]
            repeat_next = immediate_next is not None and immediate_next["letter"] == target["letter"]

            q_current = np.rad2deg(robot_interface.read_joints()[0])
            if not repeat_current:
                tracker.set_target(
                    pixel=target["pixel"],
                    world=target["world"],
                    letter=target["letter"],
                    robot_interface=robot_interface,
                    kinematics=kinematics,
                )
                if tracker.current_pixel is not None:
                    target["pixel"] = tracker.current_pixel.copy()
                if tracker.last_estimate is not None:
                    target["world"] = tracker.last_estimate.copy()

            key_position = target["world"]
            if repeat_current and previous_commanded_key_position is not None:
                key_position = previous_commanded_key_position.copy()

            pressed_key_position = deliver_typing_trajectory(
                key_position=key_position,
                tracker=tracker,
                robot_interface=robot_interface,
                hover_height=args.hover_height,
                press_depth=args.press_depth,
                q_current=q_current,
                kinematics=kinematics,
                travel_duration=args.travel_duration,
                press_duration=args.press_duration,
                start_from_hover=repeat_current,
                q_final_config=None if repeat_next else q_home_config,
                lock_key_position=repeat_current,
                freeze_hover_to_press_point=repeat_current or repeat_next,
            )

            previous_commanded_key_position = pressed_key_position.copy()
            robot_at_hover = repeat_next
            previous_letter = target["letter"]
    finally:
        robot_interface.write_joints(DEFAULT_HOME_POSITION)  # Move to a home position just for the sake of it
        if tracker.cap is not None:
            update_tracker_for_duration(
                tracker=tracker,
                duration_s=1.0,
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
