import argparse
import time
import sys
from pathlib import Path
import numpy as np

# Ensure local src modules are importable when launching this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from traj_generation import RobotKinematics, generate_key_press_trajectory, debug_plot_trajectory
from controller import PDGravityController, SO101Interface

URDF_PATH = str(REPO_ROOT / "cfg/arm_model/so101_new_calib.urdf")
PORT = "/dev/ttyACM0"

def press_target_key(target_pos: np.ndarray) -> None:
    """Main function to generate and execute a trajectory to press a target key."""
    # Initialize kinematics and controller

    print(f"TARGET WORLD COORDINATE: [X: {target_pos[0]}, Y: {target_pos[1]}, Z: {target_pos[2]}]")

    print("Initializing kinematics and controller...")
    kin = RobotKinematics(URDF_PATH)
    controller = PDGravityController(kin)

    with SO101Interface(PORT) as robot_interface:
        time.sleep(2.0)  # Wait for the connection to stabilize
        q_init, _ = robot_interface.read_joints()
        print(f"Current joint positions: {q_init*180/np.pi}")  # Print in degrees for readability

    ik_kwargs = {
        "position_weight": 1.0,
        "orientation_weight": 0.0,  # We only care about position for pressing the key
    }

    print("Generating trajectory to press the target key.")
    # Generate the trajectory to press the target key
    q_traj, dq_traj, t_exec = generate_key_press_trajectory(
        kinematics=kin,
        q_current=q_init,
        ik_kwargs=ik_kwargs,
        key_pos=target_pos,
        hover_height=0.0,
        press_depth=0.0,
        dt=0.02
    )

    # debug_plot_trajectory(t_exec=t_exec, q_traj = q_traj, dq_traj=dq_traj)

    print("Executing trajectory...")
    controller.execute_trajectory(q_traj=q_traj, dq_traj=dq_traj, t_exec=t_exec, robot_interface=robot_interface)
    print("Trajectory execution complete.")
    time.sleep(1.0)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script to move the SO-101 arm to press a target key.")
    parser.add_argument("-t", "--target", type=float, nargs=3, required=True, 
                        help="Target [X Y Z] coordinates in meters. Example: -t 0.35 0.0 0.12")
    args = parser.parse_args()
    target_array = np.array(args.target)
    
    try:
        press_target_key(target_array)
    except KeyboardInterrupt:
        print("\n Execution cancelled by user.")
    except Exception as e:
        print(f"\n FATAL ERROR: {e}")