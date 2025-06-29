# Efficient Topological and Learning-Based Navigation

This repository contains an experimental navigation framework that uses RGB-D feature-based pose estimation and a learned navigation policy. It is designed to run inside the [Habitat Lab](https://github.com/facebookresearch/habitat-lab) simulator.

## Overview
The entry point is `main.py` which performs the following steps:

1. Loads experiment parameters from a YAML configuration file.
2. Initializes a Habitat environment and sets the starting pose of the robot.
3. Creates the modules required for navigation including:
   - **FeatureBasedPointCloudRegistration** for RGB‑D pose estimation.
   - **RGBDSimilarity** to compute visual similarity between frames.
   - **Navigator** which predicts high level actions using a trained MLP model.
   - **NavigationPolicy** for keyboard control and Integrated IBVS (Image Based Visual Servoing).
4. Builds a visual path from a set of key frames.
5. Repeatedly estimates pose, selects an action, and executes it within Habitat.
6. Logs the run and optionally stores visual memory information for future training.

The system follows the methodology presented in the following paper:

```
@article{rodriguez2025efficient,
  title={Efficient Learning-Based Robotic Navigation Using Feature-Based RGB-D Pose Estimation and Topological Maps},
  author={Rodr{\'\i}guez-Mart{\'\i}nez, Eder A and El{\'\i}as Miranda-Vega, Jes{\'\u} s and Achakir, Farouk and Sergiyenko, Oleg and Rodr{\'\i}guez-Qui{\~n}onez, Julio C and Hern{\'\a}ndez Balbuena, Daniel and Flores-Fuentes, Wendy},
  journal={Entropy},
  volume={27},
  number={6},
  pages={641},
  year={2025},
  publisher={MDPI}
}
```

## Installation
1. Install Python 3.8 or later.
2. Install [Habitat Lab](https://github.com/facebookresearch/habitat-lab) and its prerequisites by following the instructions in the Habitat repository.
3. Install other Python dependencies used in the modules, for example:
   ```
   pip install numpy opencv-python torch scikit-learn scikit-image scikit-fuzzy pysift lightglue
   ```

## Running the Demo
1. Update the paths inside `main.py` to point to your local Habitat datasets, configuration files and model checkpoints.
2. Run the main script:
   ```bash
   python main.py
   ```
   Press the keys listed in `NavigationPolicy.KeyBindings` to interact with the simulation.

## Notes
- The provided paths in the code are placeholders used for our experiments. Adjust them according to your environment.
- The navigation model expects RGB‑D inputs and was trained using the method described in the referenced paper.

## Examples
The `examples` directory contains small scripts that demonstrate how to use key
components of this project:

- `feature_based_point_cloud_registration_demo.py` shows how to estimate a
  relative pose between two RGB-D frames using
  `FeatureBasedPointCloudRegistration`.
- `rgbd_similarity_demo.py` computes similarity between two RGB-D observations
  with `RGBDSimilarity`.
- `navigator_demo.py` loads a trained classifier and predicts a navigation
  action with `Navigator`.
- `navigation_policy_demo.py` demonstrates invoking the keyboard-controlled
  `NavigationPolicy`.
