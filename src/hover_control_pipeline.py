from __future__ import annotations

import argparse
import json
import time
import os
from pathlib import Path
import numpy as np

try:
    from .tracking_script import DEFAULT_LIVE_MODEL
    from controller import SO101Interface, execute_joint_trajectory
    from traj_generation import RobotKinematics, generate_point_to_point_trajectory
except ImportError:
    from tracking_script import DEFAULT_LIVE_MODEL
    from controller import SO101Interface, execute_joint_trajectory
    from traj_generation import RobotKinematics, generate_point_to_point_trajectory


DEFAULT_URDF_PATH = "cfg/arm_model/so101_new_calib.urdf"
DEFAULT_SAMPLES_JSON = "camera_calib/data/calib_poses_data/2026-04-30_14-41-28/samples.json"

ROBOT_PORT = "/dev/ttyACM0"

DEFAULT_HOME_POSITION = np.array(np.deg2rad([3.07692308, -33.14285714,  41.18681319,  61.8021978,  -89.62637363, 0.0]))  # in degrees

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
        default=5, 
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
        "--samples-json",
        type=Path,
        default=Path(DEFAULT_SAMPLES_JSON),
        help="Path to samples.json containing key gripper poses.",
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
    
    parser.add_argument(
        "--localization_mode",
        type=str,
        default="homography",
        help="Localization mode of the pipeline - available modes: [homography, ray]",
    )


    return parser.parse_args()


def load_key_positions(samples_json: Path) -> dict[str, np.ndarray]:
    with samples_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"] if isinstance(data, dict) and "samples" in data else data
    key_positions = {}
    for sample in samples:
        key = str(sample["key"]).strip().upper()
        key_positions[key] = np.asarray(sample["gripper_pose"]["position_m"], dtype=float)
    return key_positions


def read_current_joint_degrees(robot_interface: SO101Interface) -> np.ndarray:
    obs = robot_interface.robot.get_observation()
    return np.array(
        [float(obs[f"{name}.pos"]) for name in robot_interface.joint_names],
        dtype=float,
    )


def main() -> None:
    """
    Pipeline main function: instantiates the tracker, reads joints, computes a trajectory and executes it
    """

    #PARSE ARGUMENTS
    args = parse_args()
    #URDF PATH FOR FK
    urdf_path = args.urdf_path
    key_positions_by_letter = load_key_positions(args.samples_json)
    letters = [letter for letter in args.word.upper() if letter.strip()]
    missing_letters = [letter for letter in letters if letter not in key_positions_by_letter]
    if missing_letters:
        raise ValueError(f"Missing letters in {args.samples_json}: {missing_letters}")

    #KINEMATICS CLASS FOR FK AND IK FOR TRAJECTORY GENERATION AND POSE ESTIMATION 
    kinematics = RobotKinematics(urdf_path=urdf_path) # expects rads

    #ROBOT INTERFACE TO READ AND WRITE JOINTS
    robot_interface = SO101Interface(port=args.robot_port)
    print("Robot is now connected")
    print("Changing PID coefficients of internal motors...")
    
    for motor in robot_interface.robot.bus.motors:
        # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
        robot_interface.robot.bus.write("P_Coefficient", motor, 20)
        # Set I_Coefficient and D_Coefficient to default value 0 and 32
        robot_interface.robot.bus.write("I_Coefficient", motor, 5)
        robot_interface.robot.bus.write("D_Coefficient", motor, 16)
    
    #MAIN OPERATION LOOP
    print("Main operation loop starting ...")
    
    errors_x = []
    errors_y = []
    errors_z = []
    try:
        #robot_interface.robot.bus.disable_torque()
        # Move to a home position to start
        #robot_interface.robot.bus.enable_torque()
        robot_interface.write_joints(DEFAULT_HOME_POSITION) 
        time.sleep(1)
        print(f"Loaded key poses from: {args.samples_json}")

        for letter in letters:
             
            key_pos = key_positions_by_letter[letter]
            q_current = read_current_joint_degrees(robot_interface)
            print("################ TRAJECTORY GENERATION #####################")
            print(f"Generating trajectory for {letter}: {key_pos}")
            
            p_hover = key_pos + np.array([0.0, 0.0, args.hover_height])
            q_traj, dq_traj, t_exec = generate_point_to_point_trajectory(
                target_pos=p_hover,
                q_current=q_current,
                kinematics=kinematics,
                duration=args.travel_duration,
                dt=0.02,
            ) #in radians 
            print(f"Generated hover trajectory length: {len(t_exec)} samples")
            print("################ TRAJECTORY GENERATION ENDED #####################")
            print()

            print("################ TRAJECTORY EXECUTION #####################")
            print(f"Starting hover trajectory execution for {letter}.")
            execute_joint_trajectory(
                robot_interface=robot_interface,
                q_traj=q_traj, #radians
                dq_traj=dq_traj, #radians/s
                t_exec=t_exec,
                kinematics=kinematics,
                key_pos = p_hover
            )

            q_current = read_current_joint_degrees(robot_interface)
            # press_depth=0.0 means descend exactly to the sampled key position.
            p_press = key_pos - np.array([0.0, 0.0, args.press_depth])
            q_traj, dq_traj, t_exec = generate_point_to_point_trajectory(
                target_pos=p_press,
                q_current=q_current,
                kinematics=kinematics,
                duration=args.press_duration,
                dt=0.02,
            ) #in radians
            print(f"Generated descent trajectory length: {len(t_exec)} samples")
            print(f"Starting descent trajectory execution for {letter}.")
            error_x, error_y, error_z = execute_joint_trajectory(
                robot_interface=robot_interface,
                q_traj=q_traj, #radians
                dq_traj=dq_traj, #radians/s
                t_exec=t_exec,
                kinematics=kinematics,
                key_pos = p_press
            )
            errors_x.append(error_x)
            errors_y.append(error_y)
            errors_z.append(error_z)

            print("################ TRAJECTORY EXECTUTION ENDED #####################")
            
            q_current = read_current_joint_degrees(robot_interface)
            q_traj, dq_traj, t_exec = generate_point_to_point_trajectory(
                target_pos=p_hover,
                q_current=q_current,
                kinematics=kinematics,
                duration=args.travel_duration,
                dt=0.02,
            ) #in radians 
            execute_joint_trajectory(
                robot_interface=robot_interface,
                q_traj=q_traj, #radians
                dq_traj=dq_traj, #radians/s
                t_exec=t_exec,
                kinematics=kinematics,
                key_pos = p_hover
            )

            robot_interface.write_joints(DEFAULT_HOME_POSITION) 
            time.sleep(1)
        
    finally:
        input("Press ENTER when the robot is back at the home position to disconnect...")
        errors_x = np.stack(errors_x)
        errors_y = np.stack(errors_y)
        errors_z = np.stack(errors_z)
        mean_error_x = np.mean(error_x)
        mean_error_y = np.mean(error_y)
        mean_error_z = np.mean(error_z)
        print(f"Mean error x: {mean_error_x}")
        print(f"Mean error y: {mean_error_y}")
        print(f"Mean error z: {mean_error_z}") 
        robot_interface.close()


if __name__ == "__main__":
    main()
