from __future__ import annotations

import argparse
import os
import numpy as np

try:
    from .tracker import KeyWorldTracker
    from .tracking_script import DEFAULT_LIVE_MODEL
    from controller import SO101Interface, execute_joint_trajectory
    from traj_generation import RobotKinematics, generate_point_to_point_trajectory
except ImportError:
    from tracker import KeyWorldTracker
    from tracking_script import DEFAULT_LIVE_MODEL
    from controller import SO101Interface, execute_joint_trajectory
    from traj_generation import RobotKinematics, generate_point_to_point_trajectory


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


def main() -> None:
    """
    Pipeline main function: instantiates the tracker, reads joints, computes a trajectory and executes it
    """

    #PARSE ARGUMENTS
    args = parse_args()
    #URDF PATH FOR FK
    urdf_path = args.urdf_path

    #INSTANTIATE THE TRACKER TO TRACK POINTS WITH KLT DURING OPERATION
    tracker = KeyWorldTracker(
        letter=args.word,
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
        # INITIALIZE THE WORLD KEYPOINT LOCATION AND THE CURRENT JOINTS in DEGREES
        
        key_pos, q_current = tracker.start(robot_interface=robot_interface, kinematics=kinematics)
        print(f"Estimated key_pos world: {key_pos}")
        
        robot_interface.robot.bus.enable_torque()
        for motor in robot_interface.robot.bus.motors:
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            robot_interface.robot.bus.write("P_Coefficient", motor, 20)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            robot_interface.robot.bus.write("I_Coefficient", motor, 5)
            robot_interface.robot.bus.write("D_Coefficient", motor, 16)
        
        #GENERATE AND EXECUTE HOVER TRAJECTORY
        p_hover = key_pos + np.array([0.0, 0.0, args.hover_height])
        q_traj, dq_traj, t_exec = generate_point_to_point_trajectory(
            target_pos=p_hover,
            q_current=q_current,
            kinematics=kinematics,
            duration=args.travel_duration,
            dt=0.02,
        ) #in radians 
    

        print(f"Generated hover trajectory length: {len(t_exec)} samples")

        def update_tracker(i) -> None:
            updated_key_pos = tracker.update(i, robot_interface=robot_interface, kinematics=kinematics)
            if i % 10 == 0:
                print(f"Tracked key_pos in world by LS: {updated_key_pos}")

        print("Starting hover trajectory execution.")
        execute_joint_trajectory(
            robot_interface=robot_interface,
            q_traj=q_traj, #radians
            dq_traj=dq_traj, #radians/s
            t_exec=t_exec,
            kinematics=kinematics,
            key_pos=p_hover,
            step_callback=update_tracker,
        )

        #GENERATE AND EXECUTE DESCENT TRAJECTORY FROM THE REAL POST-HOVER STATE
        q_current = np.rad2deg(robot_interface.read_joints()[0])
        if tracker.last_estimate is not None:
            key_pos = tracker.last_estimate.copy()
        # press_depth=0.0 means descend exactly to the estimated key position.
        p_press = key_pos - np.array([0.0, 0.0, args.press_depth])
        q_traj, dq_traj, t_exec = generate_point_to_point_trajectory(
            target_pos=p_press,
            q_current=q_current,
            kinematics=kinematics,
            duration=args.press_duration,
            dt=0.02,
        ) #in radians

        print(f"Generated descent trajectory length: {len(t_exec)} samples")
        print("Starting descent trajectory execution.")
        execute_joint_trajectory(
            robot_interface=robot_interface,
            q_traj=q_traj, #radians
            dq_traj=dq_traj, #radians/s
            t_exec=t_exec,
            kinematics=kinematics,
            key_pos=p_press,
            step_callback=update_tracker,
        )

        q_current = np.rad2deg(robot_interface.read_joints()[0])
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
        tracker.close()
        robot_interface.close()


if __name__ == "__main__":
    main()
