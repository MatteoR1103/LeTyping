# SO-101 Keyboard Typing Robot Setup

This guide sets up the current pipeline in this repository: camera preview,
VLM/OCR key localization, visual tracking, hand-eye calibrated 3D key
estimation, and SO-101 key pressing through `main_pipeline.py`.

## 1. System Packages

Use Linux or WSL2 with USB access to the robot and camera.

```bash
sudo apt-get update
sudo apt-get install -y git build-essential ffmpeg v4l-utils
sudo usermod -a -G dialout $USER
```

Log out and back in after adding yourself to `dialout`.

## 2. Python Environment

Create or update the Micromamba environment from the repository root:

```bash
micromamba env create -f setup/environment.yml
micromamba activate rl-project
```

If the environment already exists:

```bash
micromamba env update -f setup/environment.yml --prune
micromamba activate rl-project
```

Quick import check:

```bash
python -c "import cv2, pinocchio, lerobot, openai, yaml; print('Environment OK')"
```

## 3. API Credentials

OpenAI localization needs:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

Gemini localization needs:

```bash
export GOOGLE_CLOUD_PROJECT="your_google_cloud_project"
export GOOGLE_CLOUD_LOCATION="global"
gcloud auth application-default login
```

Only the provider selected in `cfg/main_pipeline.yaml` or via `--provider` is
needed for a given run. Local OCR can be enabled with `--ocr`, but cloud
localization is still used as a fallback when OCR cannot build a keyboard map.

## 4. Robot Files

The default URDF path is:

```text
cfg/arm_model/so101_new_calib.urdf
```

If it is missing, copy the SO-101 model from the SO-ARM100 repository:

```bash
git clone --filter=blob:none --sparse https://github.com/TheRobotStudio/SO-ARM100.git
cd SO-ARM100
git sparse-checkout set Simulation/SO101

mkdir -p ../cfg/arm_model
cp -r Simulation/SO101/assets ../cfg/arm_model/
cp Simulation/SO101/so101_new_calib.urdf ../cfg/arm_model/
cd ..
rm -rf SO-ARM100
```

Set the follower calibration file and serial port in `cfg/main_pipeline.yaml`:

```yaml
robot:
  port: /dev/ttyACM0
  calibration_path: cfg/calibration/follower/<your_follower_name>.json
```

Check the serial device with:

```bash
ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
```

## 5. Camera Setup

List cameras:

```bash
v4l2-ctl --list-devices
ls /dev/video*
```

Test candidate OpenCV indices:

```bash
python - <<'PY'
import cv2
for i in range(10):
    cap = cv2.VideoCapture(i)
    ok, frame = cap.read()
    print(i, ok, None if frame is None else frame.shape)
    cap.release()
PY
```

Update the selected camera in `cfg/main_pipeline.yaml`:

```yaml
camera:
  index: 4
  backend: auto
  keyboard_height: 0.02
```

The camera preview opens before localization. Press `ENTER` or `SPACE` in the
preview window to localize, or `q` to cancel.

## 6. Calibration Files

For the full hardware-specific calibration workflow, see
`camera_calib/CALIBRATION_SETUP.md`.

The live tracker expects these calibration artifacts:

```text
camera_calib/calibrations/rigid_nonlinear_refined.npy
camera_calib/calibrations/camera_calibration.npz
```

Generate or refresh intrinsics:

```bash
python camera_calib/camera_calibration.py
```

Collect calibration poses when needed:

```bash
python camera_calib/collect_data_calib.py
```

Run hand-eye calibration/refinement when calibration changes:

```bash
python camera_calib/hand_eye_calibration.py
python camera_calib/refine_handeye_from_keyboard.py
```

Do not run the real robot until the camera image, hand-eye transform, keyboard
height, and home pose are all checked.

## 7. Pipeline Configuration

Main runtime configuration is in `cfg/main_pipeline.yaml`.

Important sections:

```yaml
home_position_deg:
  - <shoulder_pan_deg>
  - <shoulder_lift_deg>
  - <elbow_flex_deg>
  - <wrist_flex_deg>
  - <wrist_roll_deg>
  - <gripper_deg>

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

tracking:
  disable_klt_for: [SPACE]

cluster:
  excluded_letters: [SPACE]

trajectory:
  hover_height: 0.03
  press_depth: 0.01
  hover_offset_xy: [0.01, 0.0]
  first_hover_height_scale: 1.5
```

Replace the `home_position_deg` placeholders before running. The pose must be
safe, keep the mounted camera looking at the keyboard, and leave all requested
keys reachable.

Task files live in:

```text
key_sequence/task_1.txt
key_sequence/task_2.txt
key_sequence/task_3.txt
```

## 8. Run Commands

Task 1 from its configured key-sequence file:

```bash
python main_pipeline.py --config cfg/main_pipeline.yaml --task-1
```

Task 2 or 3 from configured files:

```bash
python main_pipeline.py --config cfg/main_pipeline.yaml --task 2
python main_pipeline.py --config cfg/main_pipeline.yaml --task 3
```

Custom word:

```bash
python main_pipeline.py --config cfg/main_pipeline.yaml --word C A T
```

Custom file:

```bash
python main_pipeline.py --config cfg/main_pipeline.yaml --list-path key_sequence/task_2.txt
```

Read joints/camera preview helper:

```bash
python src/utils/read_joints.py --camera 4 --port /dev/ttyACM0
```

Before each real run, confirm the robot starts at a safe home pose, the keyboard
is fixed in the calibrated workspace, and the arm path is clear.
