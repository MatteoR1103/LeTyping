from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
import numpy as np

try:
    from .tracker import KeyWorldTracker
    from controller import SO101Interface 
    from traj_generation import RobotKinematics, deliver_typing_trajectory, go_home
except ImportError:
    from tracker import KeyWorldTracker
    from controller import SO101Interface
    from traj_generation import RobotKinematics, deliver_typing_trajectory, go_home

try:
    from .utils.tracking_utils import update_tracker_for_duration
    from .utils.general_utils import build_typing_runs
except ImportError:
    from utils.tracking_utils import update_tracker_for_duration
    from utils.general_utils import build_typing_runs

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
    
    run_source = parser.add_mutually_exclusive_group(required=True)
    run_source.add_argument(
        "--word", 
        required=False,
        nargs="+",
        type=str,
        help="The word, letters, or sentence to type. For example, CAT, C A T, or \"RUB IS GOAT\"."
    )

    run_source.add_argument(
        "--task",
        choices=["1"],
        help="Run a predefined task. Task 1 presses SPACE, ENTER, R, L in order.",
    )

    run_source.add_argument(
        "--list-path",
        type=Path,
        help="Path to a text file with one word or sentence per row.",
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

        direction = delta / dist
        pos[:3] = anchor_pos[:3] + min_dist_m * direction
        frozen_world_by_letter[letter] = pos
        print(
            f"[WARNING] Corrected collapsed key positions {anchor_letter}-{letter}: "
            f"distance was {dist * 1000:.2f} mm, enforced {min_dist_m * 1000:.1f} mm."
        )


def main() -> np.ndarray | None:
    """
    Pipeline main function: instantiates the tracker, reads joints, computes a trajectory and executes it
    """
    #PARSE ARGUMENTS
    args = parse_args()
    typing_runs = build_typing_runs(args, task1_targets=TASK1_TARGETS)

    for run_index, (run_label, letters) in enumerate(typing_runs, start=1):
        if not letters:
            raise ValueError(f"No supported typing targets found for run `{run_label}`.")
        print(f"Starting typing run {run_index}/{len(typing_runs)}: {run_label}")
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

            go_home(robot_interface, kinematics, DEFAULT_HOME_POSITION)
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
            active_cluster: set[str] = set()
            frozen_world_by_letter: dict[str, np.ndarray] = {}
            retrack_from_home = True

            for index, target in enumerate(runtime_targets):
                immediate_next = runtime_targets[index + 1] if index + 1 < len(runtime_targets) else None
                is_space_target = target["letter"] == "SPACE"

                ##########################CLUSTER PLANNING##########################
                # This part of the code builds the next targets and clusters and 
                # detects whether the letters have already been refined in any cluster
                #####################################################################

                # Find the remaining letters in the word/sentence and see if already refined 
                unrefined_remaining_letters = []
                
                for future_target in runtime_targets[index:]:
                    l = future_target["letter"]
                    if l not in frozen_world_by_letter and l != "SPACE":
                        unrefined_remaining_letters.append(l)
                
                cluster_candidates = ["SPACE"] if is_space_target else unrefined_remaining_letters

                if is_space_target:
                    active_cluster = set()
                    retrack_from_home = target["letter"] not in frozen_world_by_letter
                elif target["letter"] in frozen_world_by_letter:
                    retrack_from_home = False

                # This path is intended for when the next letter is not in a refined cluster,
                # thus, for robustness, the robot goes back to its homing position to retrack the letter
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
                    retrack_from_home = False

                    # Update the world position for robot control
                    if tracker.last_estimate is not None:
                        target["world"] = tracker.last_estimate.copy()
                else:  
                    # If the current target already has a frozen position,
                    # no active cluster is needed. Otherwise, keep the cluster
                    # that was already built when entering this target.
                
                    if target["letter"] in frozen_world_by_letter:
                        active_cluster = set()
                    tracker.active_cluster_letters = set(active_cluster)

                    if target["letter"] in frozen_world_by_letter:
                        frozen_world = frozen_world_by_letter[target["letter"]]
                        activate_maintained_target_state(
                            tracker,
                            target["letter"],
                            world=frozen_world,
                        )
                        target["world"] = frozen_world.copy()
                    else:
                        tracker.set_target(
                            letter=target["letter"],
                            robot_interface=robot_interface,
                            kinematics=kinematics,
                        )
                        if tracker.last_estimate is not None:
                            target["world"] = tracker.last_estimate.copy()

                # If the next key is not covered by the current cluster or a frozen
                # position, return home after this press so the next loop can retrack.
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

                key_position = target["world"]
                track_during_hover = bool(active_cluster)
                
                q_current = np.rad2deg(robot_interface.read_joints()[0])
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
                    q_final_config=q_home_config if (next_requires_retrack or immediate_next is None) else None,
                    track_during_hover=track_during_hover,
                )

                if active_cluster:
                    for letter in active_cluster:
                        if letter not in frozen_world_by_letter:
                            frozen_world_by_letter[letter] = np.asarray(
                                tracker.targets_by_letter[letter]["world"],
                                dtype=float,
                            ).reshape(3).copy()

                    make_cluster_world_positions_coherent(
                        active_cluster,
                        frozen_world_by_letter,
                        target["letter"],
                        min_dist_m=0.015,
                    )

                if next_requires_retrack:
                    active_cluster = set()
                    retrack_from_home = True
        finally:
            go_home(robot_interface, kinematics, DEFAULT_HOME_POSITION)
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
