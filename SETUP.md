# SO-101 Robotic Arm Project - Installation Guide

This repository contains our custom code for the **SO-101** robotic arm project
On Linux/WSL we use **Micromamba** and the shared environment definition in
[environment.yml](C:/Users/angel/robot_learning_group_task/environment.yml).

The environment is designed to include:
- `lerobot` with `placo`, `feetech`, `aloha`, and `pusht`
- `openai`
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
This is needed to use servos, if u built the environment before 25/04

Notes:
- The environment name is `rl-project`.
- `environment.yml` is the source of truth for this repository.
- You do not need to clone the Hugging Face `lerobot` repository just to run this project, because `lerobot` is installed as a package through the environment file.
- If you are developing inside a separate local `lerobot` checkout, run the editable install from that checkout instead:
  `pip install -e ".[placo-dep,feetech,aloha,pusht]"`.

## 3. Verify the Environment

Check that the key packages import correctly:

```bash
python -c "import cv2, placo, lerobot, openai; print('Environment OK')"
```

If this succeeds, the environment is ready for:
- LeRobot kinematics
- OpenAI API localization
- OpenCV-based tracking

## 4. Optional: Download the SO-101 URDF and Assets

Some scripts, such as `src/track_to_wld.py`, need the SO-101 URDF and its
assets. The easiest way is to copy only the required folder from the
`SO-ARM100` repository:

```bash
git clone --filter=blob:none --sparse https://github.com/TheRobotStudio/SO-ARM100.git
cd SO-ARM100
git sparse-checkout set Simulation/SO101

mkdir -p ../cfg/arm_model
cp -r Simulation/SO101/assets ../cfg/arm_model
cp Simulation/SO101/so101_new_calib.urdf ../cfg/arm_model
cd ..
rm -rf SO-ARM100
```

After that, the default project path will be

```text
robot_learning_group_task/
├── cfg/
|   | arm_model/
│       ├── assets/
│       └── so101_new_calib.urdf
└── src/
```

## 5. Example Commands

Activate the environment first:

```bash
micromamba activate rl-project
```

Run the localizer on a saved image:

```bash
python src/gemini_keyboard_cli.py --provider openai --letter X --image camera/WIN_20260422_12_48_55_Pro.jpeg --model gpt-5.5
python src/gemini_keyboard_cli.py --provider gemini --letter X --image camera/WIN_20260422_12_48_55_Pro.jpeg
```

Run `track_to_wld.py` without the real robot, for visual testing only:

```bash
python src/track_to_wld.py --letter X --model gpt-5.5 --no-robot --camera 0
```

## Configure API keys

Activate the project environment first:

```bash
micromamba activate rl-project
```

Then set the key for the provider you want to use in the same terminal where
you will run the Python script:

```bash
export OPENAI_API_KEY="your_api_key_here"
export GOOGLE_CLOUD_PROJECT="your_project_id"
export GOOGLE_CLOUD_LOCATION="global"
```

For Gemini, authenticate with Google Cloud application-default credentials, for
example with `gcloud auth application-default login`.

## Optional
If you want to make everything easier, you can set up the key when you activate the environment as follows (assuming you have bash):
```bash
nano ~/.bashrc
```

Then at the bottom of the file, paste this:
```bash
rl-project() {
    micromamba activate rl-project
    export OPENAI_API_KEY="your_api_key_here"
    echo "Environment activated and OPENAI_API_KEY exported."
}
```
Close the file (Ctrl+X and then save, of course), then source to apply and use these changes:
```bash
source ~/.bashrc
```

Then, you can try and type the following command to set up everything:
```bash
rl-project
```


### 5. Run `track_to_wld.py`

The main goal is to run `src/track_to_wld.py`.

```bash
python src/track_to_wld.py --letter X --model gpt-5.5  --camera your_camera_ID
```

# NOTE this has to be fixed: no robot mode was removed long ago
This mode:
- uses OpenAI to initialize the tracked keypoint
- tracks it with KLT
- runs the world-point estimation with a fixed camera pose
- does not require the robot serial port

If you want to run with the real robot connected, pass the robot port and make
sure the SO-101 URDF is available:

```bash
export ROBOT_PORT=/dev/ttyACM0 # has to be always double checked cause ports might randomly change
python src/track_to_wld.py --letter X --model gpt-5.5 --urdf-path ./SO101/so101_new_calib.urdf
```
