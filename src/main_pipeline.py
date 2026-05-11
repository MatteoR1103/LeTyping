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

try:
    from .utils.tracking_utils import (
        activate_maintained_target_state,
        build_tracking_cluster,
        retrack_targets_from_current_frame,
    )
except ImportError:
    from utils.tracking_utils import (
        activate_maintained_target_state,
        build_tracking_cluster,
        retrack_targets_from_current_frame,
    )


DEFAULT_URDF_PATH = "cfg/arm_model/so101_new_calib.urdf"
ROBOT_PORT = "/dev/ttyACM0"
TASK1_TARGETS = ["SPACE", "ENTER", "R", "L"]
DEFAULT_LIVE_MODEL = "gemini-3-flash-preview"

DEFAULT_HOME_POSITION =np.array(np.deg2rad([3.07692308, -33.14285714,  41.18681319,  61.8021978,  -89.62637363, 40.0]))  # in degrees

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
        "--tracking-cluster-radius",
        type=float,
        default=0.02,
        help="World radius in metres used to group nearby letters for continuous tracking. Default: 0.03.",
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

        #SET INTERNAL BUS PID GAINS
        for motor in robot_interface.robot.bus.motors:
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            robot_interface.robot.bus.write("P_Coefficient", motor, 20)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            robot_interface.robot.bus.write("I_Coefficient", motor, 5)
            robot_interface.robot.bus.write("D_Coefficient", motor, 16)

        #BLOCK MOTORS TO AVOID SHAKING
        robot_interface.robot.bus.enable_torque()

        # INITIALIZE THE WORLD KEYPOINT LOCATIONS AND THE CURRENT JOINTS in DEGREES
        tracker.start(robot_interface=robot_interface, kinematics=kinematics)


#--------------------------- GENERATING AND EXECUTING A TRAJECTORY FOR EACH LETTER ---------------------------#
        runtime_targets = [dict(tracker.targets_by_letter[letter]) for letter in letters]
        q_home_config = np.rad2deg(DEFAULT_HOME_POSITION)
        hover_letter: str | None = None
        hover_key_position: np.ndarray | None = None
        active_cluster: set[str] = set()
        refined_letters: set[str] = set()
        frozen_world_by_letter: dict[str, np.ndarray] = {}
        retrack_from_home = False

        for index, target in enumerate(runtime_targets):
            immediate_next = runtime_targets[index + 1] if index + 1 < len(runtime_targets) else None
            is_space_target = target["letter"] == "SPACE"
            at_target_hover = (hover_letter == target["letter"]) and not is_space_target
            next_hover_letter = None
            next_hover_key_position = None
            remaining_letters = [future_target["letter"] for future_target in runtime_targets[index:]]
            unrefined_remaining_letters = [
                letter
                for letter in remaining_letters
                if letter not in refined_letters and letter != "SPACE"
            ]
            cluster_candidates = ["SPACE"] if is_space_target else unrefined_remaining_letters
            target_activated = False

            q_current = np.rad2deg(robot_interface.read_joints()[0])
            if is_space_target:
                active_cluster = set()
                retrack_from_home = True
            elif target["letter"] in frozen_world_by_letter:
                retrack_from_home = False

            # This path is intended for when the next letter is not in a refined cluster,
            # thus, for robustness, the robot goes back to its homing position to retrack the letter
            # and approach it from there, refining the position in that way
            if retrack_from_home:
                retrack_targets_from_current_frame(
                    tracker,
                    cluster_candidates,
                    robot_interface=robot_interface,
                    kinematics=kinematics,
                )

                # CREATE CLUSTER AROUND THE CURRENT TARGET LETTER
                active_cluster = set(
                    build_tracking_cluster(
                        tracker.targets_by_letter,
                        target["letter"],
                        cluster_candidates,
                        radius=args.tracking_cluster_radius,
                    )
                )
                #SET THE TRACKING CLUSTER
                tracker.active_cluster_letters = set(active_cluster)

                activate_maintained_target_state(tracker, target["letter"])
                target_activated = True
                retrack_from_home = False

                # Update the world position for robot control
                if tracker.last_estimate is not None:
                    target["world"] = tracker.last_estimate.copy()
            else:
                if target["letter"] in refined_letters:
                    active_cluster = set()
                elif not active_cluster or target["letter"] not in active_cluster:
                    active_cluster = set(
                        build_tracking_cluster(
                            tracker.targets_by_letter,
                            target["letter"],
                            cluster_candidates,
                            radius=args.tracking_cluster_radius,
                        )
                    )
                tracker.active_cluster_letters = set(active_cluster)

            if at_target_hover:
                activate_maintained_target_state(
                    tracker,
                    target["letter"],
                    world=hover_key_position,
                )
                target["world"] = hover_key_position.copy()
                target_activated = True

            if not target_activated and target["letter"] in frozen_world_by_letter:
                frozen_world = frozen_world_by_letter[target["letter"]]
                activate_maintained_target_state(
                    tracker,
                    target["letter"],
                    world=frozen_world,
                )
                target["world"] = frozen_world.copy()
                target_activated = True

            if not target_activated:
                tracker.set_target(
                    letter=target["letter"],
                    robot_interface=robot_interface,
                    kinematics=kinematics,
                )
                if tracker.last_estimate is not None:
                    target["world"] = tracker.last_estimate.copy()

            # FOR THE NEXT LETTERS IN THE WORD

            # The idea is to not go back home if the next letter is in a refined cluster
            # So we look if it's in the active one (the one around the letter that is currently being typed)
            # or in the set of past refined letters
            # Think of the sequence PALEKS: P,L,K belong to a cluster, while A,E,S belong to another,
            # but the single consecutive keys are spaced out. When going towards P, L and K can also be tracked
            # and their world position refined. When pressing P, they are in the active cluster,
            # after pressing P, they get inserted in the refined cluster.
            # Since A is not in the active cluster, nor in the refined, we can't set the trajectory
            # to go to A's hover location after pressing P, so we need to go back home to retrack
            # with template matching. When going towards A, also E and S can get refined, so they end up in
            # the refined cluster. The next target letter is L, but since it's in the refined cluster
            # we can set it as the next hover location for the trajectory. After that, no need to go back home
            # cause all the locations have already been refined

            if immediate_next is not None:
                immediate_next_letter = immediate_next["letter"]

                if is_space_target or immediate_next_letter == "SPACE":
                    print(
                        f"Leaving cluster before {immediate_next_letter}; "
                        "returning home before rebuilding the next tracking cluster."
                    )
                elif (immediate_next_letter in active_cluster) or (immediate_next_letter in refined_letters):
                    next_hover_letter = immediate_next_letter
                    if immediate_next_letter in frozen_world_by_letter:
                        next_hover_key_position = frozen_world_by_letter[immediate_next_letter].copy()
                    else:
                        next_hover_key_position = np.asarray(
                            tracker.targets_by_letter[immediate_next_letter]["world"],
                            dtype=float,
                        ).reshape(3).copy()
                    print(f"Using previous estimate for {immediate_next_letter}")
                else:
                    print(
                        f"Leaving cluster before {immediate_next_letter}; "
                        "returning home before rebuilding the next tracking cluster."
                    )

            key_position = target["world"]
            track_during_hover = bool(active_cluster)

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
                start_from_hover=at_target_hover,
                q_final_config=None if next_hover_key_position is not None else q_home_config,
                final_hover_letter=next_hover_letter,
                final_hover_key_position=next_hover_key_position,
                track_during_hover=track_during_hover,
            )

            if active_cluster:
                for letter in active_cluster:
                    if letter not in frozen_world_by_letter:
                        frozen_world_by_letter[letter] = np.asarray(
                            tracker.targets_by_letter[letter]["world"],
                            dtype=float,
                        ).reshape(3).copy()
                refined_letters.update(active_cluster)

            if next_hover_letter is not None:
                hover_letter = next_hover_letter
                if next_hover_letter in frozen_world_by_letter:
                    hover_key_position = frozen_world_by_letter[next_hover_letter].copy()
                else:
                    hover_key_position = np.asarray(
                        tracker.targets_by_letter[next_hover_letter]["world"],
                        dtype=float,
                    ).reshape(3).copy()
            else:
                hover_letter = None
                hover_key_position = None
                if immediate_next is not None:
                    active_cluster = set()
                    retrack_from_home = True
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
