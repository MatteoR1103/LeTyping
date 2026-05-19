# RL task 

CHECK THIS BEFORE RUNNING: 

```bash

python -c "import sys, importlib.util; spec=importlib.util.find_spec('lerobot.model.kinematics'); print(sys.executable); print(spec.origin if spec else None)"
```
OUTPUT NEEDS TO BE 

```bash
/home/team14/micromamba/envs/rl-project/bin/python
/home/team14/Desktop/rl_proj/lerobot/src/lerobot/model/kinematics.py
```
OTHERWISE 
(OUTPUT LOOKS LIKE:
```bash
/home/team14/micromamba/envs/rl-project/bin/python
/home/team14/micromamba/envs/rl-project/lib/python3.12/site-packages/lerobot/model/kinematics.py
```
)
RUN FROM DESKTOP FOLDER

```bash
micromamba activate rl-project
cd rl_proj/lerobot
python -m pip install -e ".[placo-dep,feetech,aloha,pusht]"
```
