# Evaluation Quick Start

This is the short run guide for evaluators. The full project notes are in
`README.md` and the hardware setup checklist is in `setup/SETUP.md`.

## What The Scripts Do

Use one script per task:

```bash
./scripts/run_eval_1.sh
./scripts/run_eval_2.sh
./scripts/run_eval_3.sh
```

Each script installs or updates the `rl-project` environment from
`setup/environment.yml`, then runs:

```bash
python main_pipeline.py --config cfg/main_pipeline.yaml --task X
```

Task inputs are read from:

```text
key_sequence/task_1.txt
key_sequence/task_2.txt
key_sequence/task_3.txt
```

The selected provider/model, camera index, robot port, calibration file,
keyboard height, trajectory settings, and clustering settings are all configured
in `cfg/main_pipeline.yaml`.

## Before Running

Check these values in `cfg/main_pipeline.yaml`:

```yaml
robot:
  port: /dev/ttyACM0
  calibration_path: cfg/calibration/follower/<your_follower_name>.json

camera:
  index: 4
  backend: auto
  keyboard_height: 0.02
```

Also set `home_position_deg` for the local setup. The home pose must be safe for
the robot, keep the mounted camera looking at the keyboard, and leave the whole
keyboard area reachable for the planned key presses. Use
`src/utils/read_joints.py` to read the six joint values for that pose.

For the full calibration checklist, see `camera_calib/CALIBRATION_SETUP.md`.

Required calibration files:

```text
camera_calib/calibrations/camera_calibration.npz
camera_calib/calibrations/rigid_nonlinear_refined.npy
```

Cloud localization also needs the API key for the configured provider, usually:

```bash
export OPENAI_API_KEY="..."
```

For Gemini tasks, also configure Google Cloud credentials as described in
`setup/SETUP.md`.

## Useful Overrides

Skip environment installation after the first run:

```bash
INSTALL_ENV=0 ./scripts/run_eval_1.sh
```

Pass any extra `main_pipeline.py` argument after the script name:

```bash
INSTALL_ENV=0 ./scripts/run_eval_2.sh --camera 2 --robot-port /dev/ttyACM1
```

Use a different config:

```bash
CONFIG_PATH=cfg/main_pipeline.yaml ./scripts/run_eval_3.sh
```
