import cv2
import json
import select
import sys
import termios
import time
import tty
from datetime import datetime
from contextlib import contextmanager
from pathlib import Path

CALIBRATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CALIBRATION_DIR.parent


def add_lerobot_src_to_path():
    for candidate in (
        PROJECT_ROOT / "lerobot" / "src",
        PROJECT_ROOT.parent / "lerobot" / "src",
    ):
        if (candidate / "lerobot").is_dir():
            sys.path.insert(0, str(candidate))
            return


add_lerobot_src_to_path()

from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

FOLLOWER_PORT = "/dev/ttyACM0"
FOLLOWER_ID = "zi_padrone"

LEADER_PORT = "/dev/ttyACM1"
LEADER_ID = "caesar_salad"

CAMERA_INDEX = 5
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
RAW_CALIB_DATA_DIR = CALIBRATION_DIR / "raw_calib_data"
WINDOW_NAME = "collect_data_calib"


def to_jsonable(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    try:
        json.dumps(x)
        return x
    except TypeError:
        return str(x)


def extract_joint_state(obs):
    if "observation.state" in obs:
        return obs["observation.state"]

    joint_like = {}
    for k, v in obs.items():
        if ".pos" in k or "joint" in k.lower():
            joint_like[k] = v
    return joint_like if joint_like else None


@contextmanager
def cbreak_stdin():
    if not sys.stdin.isatty():
        yield False
        return

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def read_key():
    if not sys.stdin.isatty():
        return None

    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return None

    return sys.stdin.read(1)


def has_opencv_gui():
    try:
        info = cv2.getBuildInformation()
    except Exception:
        return False

    for line in info.splitlines():
        if line.strip().startswith("GUI:"):
            return "NONE" not in line.upper()
    return False


def main():
    run_dir = RAW_CALIB_DATA_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    images_dir = run_dir / "images"
    images_dir.mkdir(exist_ok=True)

    robot = SO101Follower(
        SO101FollowerConfig(
            port=FOLLOWER_PORT,
            id=FOLLOWER_ID,
        )
    )
    teleop = SO101Leader(
        SO101LeaderConfig(
            port=LEADER_PORT,
            id=LEADER_ID,
        )
    )

    robot.connect()
    teleop.connect()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    use_opencv_gui = has_opencv_gui()

    samples = []
    sample_idx = 0
    printed_keys = False

    print("Teleop running")
    print(f"Saving samples to: {run_dir}")
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")
    if use_opencv_gui:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        print("OpenCV GUI detected.")
        print("Focus the camera window, then press 's', SPACE or ENTER to save a sample, 'q' to quit")
    else:
        print("OpenCV GUI not available, using terminal controls.")
        print("Press 's', SPACE or ENTER to save a sample, 'q' to quit")

    try:
        with cbreak_stdin() as stdin_ready:
            if not use_opencv_gui and not stdin_ready:
                print("Warning: stdin is not a TTY, so on-demand key capture is unavailable.")
                print("Run this script from a terminal to save samples interactively.")

            while True:
                observation = robot.get_observation()
                action = teleop.get_action()
                robot.send_action(action)

                ret, frame = cap.read()
                if not ret:
                    continue
                if use_opencv_gui:
                    cv2.imshow(WINDOW_NAME, frame)

                if not printed_keys:
                    print("Observation keys:")
                    for k in observation.keys():
                        print(" ", k)
                    printed_keys = True

                if use_opencv_gui:
                    key_code = cv2.waitKey(1) & 0xFF
                    key = chr(key_code) if key_code not in (255, 0xFF) else None
                    if key_code in (13, 10):
                        key = "\n"
                else:
                    key = read_key()

                if key == "q":
                    break
                if key not in ("s", " ", "\n", "\r"):
                    continue

                ts = time.time()
                img_name = f"sample_{sample_idx:03d}.png"
                img_path = images_dir / img_name
                cv2.imwrite(str(img_path), frame)

                sample = {
                    "sample_idx": sample_idx,
                    "timestamp": ts,
                    "image_path": str(img_path.relative_to(run_dir)),
                    "joint_state": to_jsonable(extract_joint_state(observation)),
                    "observation": to_jsonable(observation),
                }

                samples.append(sample)

                with open(run_dir / "samples.json", "w") as f:
                    json.dump(samples, f, indent=2)

                print(f"Saved sample {sample_idx}: {img_path}")
                sample_idx += 1

    finally:
        cap.release()
        if use_opencv_gui:
            cv2.destroyAllWindows()
        try:
            teleop.disconnect()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
