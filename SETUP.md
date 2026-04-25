# SO-101 Robotic Arm Project - Installation Guide

This repository contains our custom code for the **SO-101** robotic arm project.
On Linux/WSL we use **Micromamba** and the shared environment definition in
[environment.yml](C:/Users/angel/robot_learning_group_task/environment.yml).

The environment is designed to include:
- `lerobot` with `placo`, `feetech`, `aloha`, and `pusht`
- `google-genai`
- `opencv`
- the rest of the project dependencies

## 1. System Prerequisites (Linux/WSL)

Install the base system tools first:

```bash
sudo apt-get update
sudo apt-get install -y git build-essential ffmpeg
```

If you plan to connect the real robot over USB, also add your user to the
`dialout` group:

```bash
sudo usermod -a -G dialout $USER
```

You may need to log out and log back in after this change.

## 2. Create the Project Environment with Micromamba

Create the environment directly from `environment.yml`:

```bash
micromamba env create -f environment.yml
micromamba activate rl-project
```

If the environment already exists and you want to refresh it:

```bash
micromamba env update -f environment.yml --prune
micromamba activate rl-project
```
This IS needed to use servos, if u built the environment before 25/04

Notes:
- The environment name is `rl-project`.
- `environment.yml` is the source of truth for this repository.
- You do not need to clone the Hugging Face `lerobot` repository just to run this project, because `lerobot` is installed as a package through the environment file.
- If you are developing inside a separate local `lerobot` checkout, run the editable install from that checkout instead:
  `pip install -e ".[placo-dep,feetech,aloha,pusht]"`.

## 3. Verify the Environment

Check that the key packages import correctly:

```bash
python -c "import cv2, placo, lerobot; from google import genai; print('Environment OK')"
```

If this succeeds, the environment is ready for:
- LeRobot kinematics
- Gemini / Vertex AI
- OpenCV-based tracking

## 4. Optional: Download the SO-101 URDF and Assets

Some scripts, such as `src/track_to_wld.py`, need the SO-101 URDF and its
assets. The easiest way is to copy only the required folder from the
`SO-ARM100` repository:

```bash
git clone --filter=blob:none --sparse https://github.com/TheRobotStudio/SO-ARM100.git
cd SO-ARM100
git sparse-checkout set Simulation/SO101

mkdir -p ../SO101
cp -r Simulation/SO101/assets ../SO101/
cp Simulation/SO101/so101_new_calib.urdf ../SO101/
cd ..
```

After that, the default project path will be

```text
robot_learning_group_task/
├── SO101/
│   ├── assets/
│   └── so101_new_calib.urdf
└── src/
```

## 5. Example Commands

Activate the environment first:

```bash
micromamba activate rl-project
```

Run the Gemini localizer on a saved image:

```bash
python src/gemini_keyboard_localizer.py --letter X --image camera/WIN_20260422_12_48_55_Pro.jpeg --model gemini-3-flash-preview
```

Run `track_to_wld.py` without the real robot, for visual testing only:

```bash
python src/track_to_wld.py --letter X --model gemini-3-flash-preview --no-robot --camera 0
```

## Configure Gemini API via Vertex AI on Linux/WSL

The Gemini keyboard localizer uses the Google Gen AI SDK through Vertex AI.
These steps configure Google Cloud credentials locally and tell the Python
script which project and location to use.

Project: `quixotic-skill-424213-h6`  (ID for login)
Location: `global`

Run these commands from a Linux/WSL terminal. Use normal double hyphens (`--`),
not typographic dashes copied from rich text.

### 1. Install Google Cloud CLI

On Ubuntu/Linux systems with Snap support:

```bash
sudo snap install google-cloud-cli --classic
```

Verify that `gcloud` is available:

```bash
gcloud --version
```

### 2. Authenticate Google Cloud

Login for normal `gcloud` CLI commands:

```bash
gcloud auth login
```

Login for Python client libraries through Application Default Credentials
(ADC):

```bash
gcloud auth application-default login
```

Attach the quota/billing project to the ADC credentials:

```bash
gcloud auth application-default set-quota-project quixotic-skill-424213-h6
```

Optional verification. This prints a private access token, so do not share it:

```bash
gcloud auth application-default print-access-token
```

### 3. Enable Vertex AI

```bash
gcloud services enable aiplatform.googleapis.com --project=quixotic-skill-424213-h6
```

Verify that the Vertex AI API is enabled:

```bash
gcloud services list --enabled --filter="name:aiplatform.googleapis.com" --project=quixotic-skill-424213-h6
```

### 4. Activate the project environment

Activate the project environment first:

```bash
micromamba activate rl-project
```

Then set the environment variables in the same terminal where you will run the
Python script:

```bash
export GOOGLE_CLOUD_PROJECT="quixotic-skill-424213-h6"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="true"
```

### 5. Run `track_to_wld.py`

The main goal is to run `src/track_to_wld.py`.

For a first visual test without the real robot connected, use `--no-robot`:

```bash
python src/track_to_wld.py --letter X --model gemini-3-flash-preview --no-robot --camera 0
```

This mode:
- uses Gemini to initialize the tracked keypoint
- tracks it with KLT
- runs the world-point estimation with a fixed camera pose
- does not require the robot serial port

If you want to run with the real robot connected, pass the robot port and make
sure the SO-101 URDF is available:

```bash
export ROBOT_PORT=/dev/ttyACM0
python src/track_to_wld.py --letter X --model gemini-3-flash-preview --urdf-path ./SO101/so101_new_calib.urdf
```
