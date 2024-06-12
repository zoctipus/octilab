# ORBIT Installation Guide

This guide provides step-by-step instructions use octilab on top of IsaacLab

## Cloning ORBIT

First, clone the Isaac Lab repository from NVIDIA-Omniverse:

- For SSH:
`git@github.com:isaac-sim/IsaacLab.git`

- For HTTPS:
`https://github.com/isaac-sim/IsaacLab.git`


After cloning, navigate to the `orbit` directory:

`cd IsaacLab`


## Environment Setup

Follow the installation guide from the [official Isaac Lab installation page](https://isaac-sim.github.io/IsaacLab/source/setup/installation/index.html) or use the recommended setup below.

### Recommended Setup

1. Link Isaac Sim Package:
    `ln -s ~/.local/share/ov/pkg/isaac_sim-4.0.0/ _isaac_sim`

2. Initialize Conda Environment:
   `./isaaclab.sh --conda IsaacLab`

3. Install Dependencies:
    `sudo apt install cmake build-essential`

4. Install ORBIT:
    `./isaaclab.sh --install`

5. Install Extra Packages:
    `./isaaclab.sh --extra rsl_rl`


6. Setup VSCode for Autocomplete:
Open VSCode in the repository folder, press `Ctrl + Shift + P`, type "run task", select "setup_python_task", and run it.

### Clone octilab Repository

- For SSH: `git clone git@github.com:zoctipus/octilab.git`
- For HTTPS: `git clone https://github.com/zoctipus/octilab.git`


## Running and Training

- Run Inference: 
    -   `python play.py --task Isaac-Chop-Ball-Hebi-v0 --num_envs 3 --checkpoint models/cube_model_1600.pt`


- Training:
    - `python train.py --task Isaac-Chop-Ball-Hebi-v0 --headless`