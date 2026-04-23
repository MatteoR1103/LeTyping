# SO-101 Robotic Arm Project - Installation Guide

This repository contains the custom code for our group project using the **SO-101** robotic arm. To ensure environment consistency across the team, we use **Micromamba** and **Python 3.12**.

---

## 1. Prerequisites (Linux System)
Before setting up the Python environment, install the necessary system dependencies for video processing and hardware communication:

```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential ffmpeg
```

---

## 2. Create Micromamba Environment
Create a new isolated Python 3.12 environment using Micromamba:

```bash
# Create the environment with Python 3.12
micromamba create -n lerobot python=3.12 -c conda-forge -y

# Activate the environment
micromamba activate lerobot
```

---

## 3. Install FFmpeg in Environment
Install FFmpeg within the conda environment for video processing support:

```bash
micromamba install ffmpeg -c conda-forge -y
```

---

## 4. Clone and Install LeRobot Repository
Clone the official Hugging Face LeRobot repository and install it with hardware and simulation support:

```bash
# Clone the official Hugging Face LeRobot repository, I suggest doing it outside of this folder, or add the directory to the .gitignore (I would not recommend it, still)
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# Install with hardware support (Feetech) and simulations (Aloha/PushT)
pip install -e ".[feetech,aloha,pusht]"
```

Suggested folder tree structure:

/your-parent-folder/
├── lerobot/               # The official library (cloned earlier)
└── robot_learning_group_task/   # THIS repository

---

## 5. Configure Hardware Permissions
Grant user access to serial ports for hardware communication with the SO-101 arm:

```bash
sudo usermod -a -G dialout $USER
```

**Note:** You may need to log out and log back in for this change to take effect.

---

## 6. Verification
Verify that LeRobot has been successfully installed:

```bash
micromamba activate lerobot
python -c "import lerobot; print('✅ LeRobot successfully installed!')"
```

---

## Next Steps
After completing all setup steps, you're ready to:
- Configure the SO-101 robotic arm connection
- Run training pipelines
- Collect demonstration data

## Configure Gemini API via Vertex AI on Linux/WSL

The Gemini keyboard localizer uses the Google Gen AI SDK through Vertex AI.
These steps configure Google Cloud credentials locally and tell the Python
script which project and location to use.

Project: `quixotic-skill-424213-h6`  (ID)
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

If you use conda or micromamba, activate the project environment first:

```bash
conda activate rl-project
```

Then set the environment variables in the same terminal where you will run the
Python script:

```bash
export GOOGLE_CLOUD_PROJECT="quixotic-skill-424213-h6"
export GOOGLE_CLOUD_LOCATION="global"
export GOOGLE_GENAI_USE_VERTEXAI="true"
```

### 5. Run the localizer

```bash
python gemini_keyboard_localizer.py --letter X --image camera/WIN_20260422_12_48_55_Pro.jpeg --model gemini-3-flash-preview
```
