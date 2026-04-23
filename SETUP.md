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

## Configure Gemini API via Vertex AI

Use these steps to configure Gemini through Vertex AI with Google Application Default Credentials (ADC).

Project: `quixotic-skill-424213-h6`  
Location: `global`

### 1. Install Google Cloud CLI

Run from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -c "iex (irm https://storage.googleapis.com/cloud-samples-data/adc/setup_adc.ps1)"
```

### 2. Verify the installation

```cmd
gcloud.cmd --version
```

If `gcloud.cmd` is not found, use the full path:

```powershell
& "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd" --version
```

### 3. Configure Google Cloud project and login

```cmd
gcloud.cmd config set project quixotic-skill-424213-h6
gcloud.cmd auth application-default login
gcloud.cmd auth application-default set-quota-project quixotic-skill-424213-h6
```

Optional verification. Do not share the printed token:

```cmd
gcloud.cmd auth application-default print-access-token
```

### 4. Enable Vertex AI API

```cmd
gcloud.cmd services enable aiplatform.googleapis.com --project=quixotic-skill-424213-h6
```

Verify that Vertex AI is enabled:

```cmd
gcloud.cmd services list --enabled --filter="name:aiplatform.googleapis.com" --project=quixotic-skill-424213-h6
```

### 5. Set environment variables in the active conda terminal

```cmd
set "GOOGLE_CLOUD_PROJECT=quixotic-skill-424213-h6"
set "GOOGLE_CLOUD_LOCATION=global"
set "GOOGLE_GENAI_USE_VERTEXAI=true"
```

### 6. Run a test

```cmd
python test_gemini.py
```
