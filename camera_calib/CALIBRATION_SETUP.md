# Camera And Robot Calibration Setup

This pipeline is hardware-specific. Before running any eval task, each user must
calibrate their own SO-101 robot, camera, camera-to-gripper transform, keyboard
height, and home pose. The calibration files from another setup should be treated
only as examples.

## Required Outputs

The live pipeline expects:

```text
cfg/calibration/follower/<your_follower_name>.json
camera_calib/calibrations/camera_calibration.npz
camera_calib/calibrations/rigid_nonlinear_refined.npy
```

Then update `cfg/main_pipeline.yaml`:

```yaml
robot:
  port: /dev/ttyACM0
  calibration_path: cfg/calibration/follower/<your_follower_name>.json

camera:
  index: <your_opencv_camera_index>
  backend: auto
  keyboard_height: <keyboard_plane_height_in_metres>

home_position_deg:
  - <shoulder_pan_deg>
  - <shoulder_lift_deg>
  - <elbow_flex_deg>
  - <wrist_flex_deg>
  - <wrist_roll_deg>
  - <gripper_deg>
```

The home pose must be safe, keep the wrist-mounted camera looking at the
keyboard, and leave the full keyboard area reachable.

## 1. Robot Calibration

Calibrate the SO-101 follower with LeRobot/Feetech for the connected hardware.
Save the follower calibration JSON under:

```text
cfg/calibration/follower/<your_follower_name>.json
```

Set `robot.port` and `robot.calibration_path` in `cfg/main_pipeline.yaml`.

## 2. Camera Index

List connected cameras:

```bash
v4l2-ctl --list-devices
ls /dev/video*
```

Test OpenCV indices:

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

Put the selected index in `camera.index`.

## 3. Camera Intrinsics

Collect checkerboard images with the same camera resolution and focus used by the
pipeline. Then run:

```bash
python camera_calib/camera_calibration.py \
  --images camera_calib/data/calib_poses_data/<your_run>/images \
  --output-prefix camera_calib/calibrations/camera_calibration
```

Use the correct checkerboard `--rows`, `--cols`, and `--square-size` for your
printed board. The output must include:

```text
camera_calib/calibrations/camera_calibration.npz
```

## 4. Hand-Eye Calibration

Collect synchronized robot/camera calibration poses:

```bash
python camera_calib/collect_data_calib.py
```

Before using that script, check the constants at the top of the file for your
ports, robot IDs, camera index, and URDF path. Then run:

```bash
python camera_calib/hand_eye_calibration.py
```

This estimates the camera-to-gripper transform for your physical camera mount.

## 5. Nonlinear Refinement

The final tracker uses the nonlinear refined transform, so refine the hand-eye
estimate for the same camera/gripper setup:

```bash
python camera_calib/refine_handeye_from_keyboard.py \
  --camera-calib camera_calib/calibrations/camera_calibration.npz \
  --output camera_calib/calibrations/rigid_nonlinear_refined.npy
```

When `--output` ends in `.npy`, the script writes the 4x4 transform used by the
live tracker and a `.json` sidecar with refinement metadata. Use keyboard/world
annotations and the gripper pose from your own setup. Do not reuse
`rigid_nonlinear_refined.npy` from another camera mount.

## 6. Home Pose

Move the robot to a safe pose where:

- the camera sees the keyboard clearly;
- all requested keys are reachable;
- the arm can descend without colliding with the keyboard or table.

Read the joint values:

```bash
python src/utils/read_joints.py --camera <your_camera_index> --port <your_robot_port>
```

Copy the six joint angles into `home_position_deg` in `cfg/main_pipeline.yaml`.
Leaving the placeholder values there will intentionally stop the pipeline with a
clear configuration error.

## 7. Final Checks

Before running `./scripts/run_eval_*.sh`, confirm:

- `camera.index` opens the mounted camera;
- `robot.calibration_path` points to the connected follower calibration;
- `camera_calib/calibrations/camera_calibration.npz` is from that camera;
- `camera_calib/calibrations/rigid_nonlinear_refined.npy` is from that camera/gripper setup;
- `camera.keyboard_height` matches the keyboard plane;
- `home_position_deg` has six numeric values, not placeholders.
