# 🤖 SO-101 Keyboard Typing Robot

![Build](https://img.shields.io/badge/build-manual%20hardware%20run-lightgrey?style=flat-square)
![Python](https://img.shields.io/badge/python-3.12-yellow?style=flat-square)
![Status](https://img.shields.io/badge/status-research%20prototype-orange?style=flat-square)

> A robotic system for visually localizing QWERTY keyboard keys and pressing them with an **SO-101** robotic arm, combining computer vision, VLM/OCR-based localization, inverse kinematics, spline trajectory generation, and gravity-compensated control.

## General Description (About)

**SO-101 Keyboard Typing Robot** is an applied robotics project designed to automate physical keyboard interaction using a serial robotic manipulator. The system captures images from a camera, localizes target keys, estimates their 3D position in the robot world frame, and generates smooth trajectories to reach and press each selected key.

The project provides an end-to-end pipeline for:

- initial key localization using cloud vision-language models, such as **OpenAI** or **Gemini**, or local OCR;
- visual tracking of keyboard targets while the robot is moving;
- 3D key position estimation through camera-ray and keyboard-plane intersection;
- forward and inverse kinematics based on the SO-101 URDF model;
- cubic-spline trajectory generation for `home -> hover -> press -> home` motions;
- outer-loop PID control with gravity feed-forward compensation;
- execution of predefined tasks, individual words, or text files containing multiple typing runs.

The repository is intended as a publishable research prototype for robot learning, manipulator control, visual servoing, and human-interface automation experiments on real hardware.

##  System Architecture

### Hardware

| Component | Description | Notes |
| --- | --- | --- |
| Robotic arm | SO-101 follower arm | Controlled through LeRobot/Feetech |
| Actuators | Feetech STS3215 or compatible servos | Position-controlled motors |
| RGB camera | USB/OpenCV camera | Camera index configured in `cfg/main_pipeline.yaml` |
| Keyboard | Physical QWERTY keyboard | Modeled as a planar surface in world coordinates |
| Workstation | Linux/WSL recommended | Micromamba environment: `rl-project` |
| Calibration | Robot calibration, camera intrinsics, and camera-to-robot transform | Must be generated for the specific hardware setup |

>  Before running on the real robot, verify the serial port, clear the workspace, calibrate the connected robot, set a safe home pose for the local keyboard placement, and confirm that the camera intrinsics plus hand-eye/nonlinear refinement belong to that exact camera/gripper setup.

###  Software

| Module | File/Directory | Responsibility |
| --- | --- | --- |
| Main pipeline | `main_pipeline.py` | Task parsing, robot initialization, tracking, and key pressing |
| Configuration | `cfg/main_pipeline.yaml` | Camera, robot, VLM, trajectory, and clustering parameters |
| Kinematics | `src/kinematics.py` | LeRobot FK/IK and Pinocchio dynamics |
| Control | `src/controller.py` | SO-101 hardware interface and gravity-compensated PID |
| Trajectory generation | `src/traj_generation.py` | Cubic splines, hover/press motions, and home return |
| 3D tracking | `src/tracker.py` | Pixel localization, visual tracking, and world-frame estimation |
| VLM/OCR localization | `src/gemini_keyboard_localizer.py`, `src/ocr_keyboard_localizer.py` | OpenAI/Gemini/EasyOCR localization backends |
| Target clustering | `src/keyboard_cluster.py` | Nearby-key tracking, freezing, and retracking logic |
| Calibration | `camera_calib/` | Camera and hand-eye calibration scripts/results |

Simplified pipeline:

```text
Camera frame
   ↓
VLM/OCR key localization
   ↓
KLT/template visual tracking
   ↓
Pixel ray + keyboard plane intersection
   ↓
World key position
   ↓
IK + cubic spline trajectory
   ↓
SO-101 controller
   ↓
Physical key press
```

##  Prerequisites and Dependencies

### System Requirements

- Ubuntu/Linux or WSL2
- Python 3.12
- Micromamba or Conda
- Git
- USB camera accessible through OpenCV
- Serial permissions for real robot control

```bash
sudo apt-get update
sudo apt-get install -y git build-essential ffmpeg
sudo usermod -a -G dialout $USER
```

After adding the user to the `dialout` group, log out and back in, or restart the session.

### Main Dependencies

| Category | Packages |
| --- | --- |
| Robotics | `lerobot`, `pinocchio` |
| Computer vision | `opencv-python` |
| Scientific computing | `numpy`, `scipy`, `matplotlib` |
| VLM/OCR | `openai`, `google-genai`, optional `easyocr` |
| Configuration | `pyyaml` |

### API Keys

To use cloud-based localization, configure at least one provider:

```bash
export OPENAI_API_KEY="<your_openai_api_key>"
export GOOGLE_CLOUD_PROJECT="<your_google_cloud_project>"
export GOOGLE_CLOUD_LOCATION="global"
```

For Gemini, Google Cloud application-default credentials may also be required:

```bash
gcloud auth application-default login
```

## Installation and Configuration

### 1. Clone the Repository

```bash
git clone <repository_url>
cd robot_learning_group_task
```

### 2. Create the Environment

The recommended installation path is through `setup/environment.yml`, which defines the project environment and Python dependencies.

```bash
micromamba env create -f setup/environment.yml
micromamba activate rl-project
```

If the environment already exists:

```bash
micromamba env update -f setup/environment.yml --prune
micromamba activate rl-project
```

Quick verification:

```bash
python -c "import cv2, pinocchio, lerobot, openai; print('Environment OK')"
```

### 3. Configure the Robot

Review and update `cfg/main_pipeline.yaml` before running:

```yaml
tasks:
  1:
    provider: openai
    model: gpt-5.5
    list_path: key_sequence/task_1.txt
  2:
    provider: gemini
    model: gemini-3-flash-preview
    list_path: key_sequence/task_2.txt
  3:
    provider: openai
    model: gpt-5.5
    list_path: key_sequence/task_3.txt

robot:
  port: /dev/ttyACM0
  calibration_path: cfg/<your_follower_name>.json  # or cfg/calibration/follower/<robot_id>.json

camera:
  index: 5
  backend: auto
  keyboard_height: 0.02

kinematics:
  urdf_path: cfg/arm_model/so101_new_calib.urdf
  press_ee_frame: key_contact_frame_link

tracking:
  disable_klt_for: [SPACE]

cluster:
  excluded_letters: [SPACE]
```

Make sure that:

- the serial port matches the connected robot;
- `robot.calibration_path` points to the calibration file for the connected SO-101 follower;
- `home_position_deg` is set for the local setup: use `src/utils/read_joints.py` to read a safe pose that keeps the wrist-mounted camera looking at the keyboard and leaves the full keyboard area reachable;
- the URDF file is available under `cfg/arm_model/`;
- `camera.index`, `camera.backend`, and `camera.keyboard_height` match the local camera and keyboard placement;
- camera intrinsics and the refined camera-to-robot transform in `camera_calib/calibrations/` were produced for this exact camera/gripper calibration, including the nonlinear refinement step;
- the keyboard is placed in the calibrated workspace.

### 4. Validate the Python Workspace

This repository is currently structured as a Python-first robotics project. No compilation step is required for the main pipeline.

```bash
python -m compileall main_pipeline.py src camera_calib
```

##  Usage

### Run on Real Hardware

Activate the environment:

```bash
micromamba activate rl-project
```

Run task 1 from the file configured in `cfg/main_pipeline.yaml`:

```bash
python main_pipeline.py --config cfg/main_pipeline.yaml --task-1
```

Task key files live under `key_sequence/`:

```text
key_sequence/task_1.txt
key_sequence/task_2.txt
key_sequence/task_3.txt
```

Type a word or a sequence of letters:

```bash
python main_pipeline.py \
  --config cfg/main_pipeline.yaml \
  --word C A T \
  --camera 5 \
  --robot-port /dev/ttyACM0 \
  --provider openai \
  --model gpt-5.5
```

Run a task from a text file:

```bash
python main_pipeline.py \
  --task 2 \
  --list-path key_sequence/task_2.txt \
  --provider gemini \
  --model gemini-3-flash-preview
```

Numbered tasks default to their configured files under `key_sequence/`.
The provider/model defaults for each task come from `cfg/main_pipeline.yaml`; command-line flags still override them.

Use local OCR when available:

```bash
python main_pipeline.py \
  --word HELLO \
  --ocr \
  --camera 5
```

### Useful Parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `--config` | Pipeline YAML configuration file | `cfg/main_pipeline.yaml` |
| `--camera` | OpenCV camera index | `5` |
| `--robot-port` | SO-101 serial port | `/dev/ttyACM0` |
| `--provider` | Cloud localization provider | `openai` |
| `--model` | Vision-language model | `gpt-5.5` |
| `--urdf-path` | Robot URDF path | `cfg/arm_model/so101_new_calib.urdf` |
| `--hover-height` | Offset above the target key | Configured in YAML |
| `--press-depth` | Key press depth | Configured in YAML |
| `--disable-klt-for` | Keys held after initial localization instead of KLT tracking | `tracking.disable_klt_for` |
| `--cluster-excluded-letters` | Keys handled alone instead of grouped into clusters | `cluster.excluded_letters` |
| `--hover-offset-xy` | XY hover offset before pressing | `trajectory.hover_offset_xy` |
| `--task-1` | Shortcut for `--task 1` | `tasks.1.list_path` |

## Calibration and Debugging

The example calibration values in this repository are not portable across
robots. Each hardware setup must provide its own robot calibration, camera
intrinsics, hand-eye transform, nonlinear hand-eye refinement, keyboard height,
and `home_position_deg` before running the eval scripts.

See `camera_calib/CALIBRATION_SETUP.md` for the step-by-step calibration
checklist.

The scripts under `camera_calib/` support camera calibration and hand-eye
transform refinement:

```bash
python camera_calib/camera_calibration.py
python camera_calib/hand_eye_calibration.py
python camera_calib/refine_handeye_from_keyboard.py
```

Relevant files:

| File | Purpose |
| --- | --- |
| `cfg/calibration/follower/<robot_id>.json` | SO-101 follower calibration for the connected robot |
| `camera_calib/calibrations/camera_calibration.npz` | Camera intrinsics for the mounted camera |
| `camera_calib/calibrations/rigid_nonlinear_refined.npy` | Refined camera-to-robot transform for that same camera/gripper setup |
| `camera_calib/stats/nonlinear_handeye_report.txt` | Calibration report |

## Contributing

Contributions are welcome. To propose a change:

1. Fork the repository.
2. Create a dedicated branch.
3. Implement the change while keeping the existing structure and style.
4. Test on hardware when applicable.
5. Open a Pull Request with a clear description, test results, and safety notes.

```bash
git checkout -b feature/<feature-name>
git commit -m "Add <description>"
git push origin feature/<feature-name>
```
