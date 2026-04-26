import time
import json
import abc
import traceback  
from lerobot.motors.motors_bus import Motor, MotorCalibration
from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


PORT = "/dev/ttyACM0" 
CALIB_PATH = "cfg/arms_calibration/follower/zi_padrone.json"


# RUB_GUESS = [-12, -45, 55, 15, 0, 0]
TARGET_JOINTS = [-12, -15, 0, 0, 0, 0] 
FOLLOWER_ID = "zi_padrone"

robot = SO101Follower(SO101FollowerConfig(port = PORT, id = FOLLOWER_ID))


def move_to_hardcoded_pos():
    robot.connect()
    print("Connected. Moving to hardcoded position...")
    # with open(CALIB_PATH, "r") as f:
    #     raw_calib_data = json.load(f)

    # calibration = {
    #     name: MotorCalibration(**values) 
    #     for name, values in raw_calib_data.items()
    # }

    # motors = {
    #     "shoulder_pan": Motor(1, "sts3215", "identity"),
    #     "shoulder_lift": Motor(2, "sts3215", "identity"),
    #     "elbow_flex": Motor(3, "sts3215", "identity"),
    #     "wrist_flex": Motor(4, "sts3215", "identity"),
    #     "wrist_roll": Motor(5, "sts3215", "identity"),
    #     "gripper": Motor(6, "sts3215", "identity"),
    # }

    # bus = FeetechMotorsBus(port=PORT, motors=motors, calibration=calibration)

    action = {
        "shoulder_pan.pos": TARGET_JOINTS[0],
        "shoulder_lift.pos": TARGET_JOINTS[1],
        "elbow_flex.pos": TARGET_JOINTS[2],
        "wrist_flex.pos": TARGET_JOINTS[3],
        "wrist_roll.pos": TARGET_JOINTS[4],
        "gripper.pos": TARGET_JOINTS[5],
    }

    robot.send_action(action)
    print("Action sent.")

    # try:
    #     bus.connect()
    #     print("Connected. Enabling torque...")
        
    #     for motor_name in motors.keys():
    #         # Use normalize=False to skip internal scaling math
    #         bus.write("Torque_Enable", motor_name, 1, normalize=False)
        
    #     print("Torque enabled. Moving...")

    #     for i, motor_name in enumerate(motors.keys()):
    #         print(f"Writing to {motor_name}...")
    #         bus.write(
    #             data_name="Goal_Position", 
    #             motor=motor_name, 
    #             value=TARGET_JOINTS[i],
    #             normalize=False
    #         )

    #     time.sleep(3)
    #     print("Target reached.")

    # except Exception:
    #     traceback.print_exc()
    
    # finally:
    #     bus.disconnect()
    #     print("Disconnected.")
    time.sleep(3)
    robot.disconnect()
    print("Disconnected.")

if __name__ == "__main__":
    move_to_hardcoded_pos()