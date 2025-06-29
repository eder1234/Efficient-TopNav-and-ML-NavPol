import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.config import read_write
from habitat.config.default import get_agent_config

import cv2
import numpy as np
import yaml
import time
import pickle
import json
import torch

# Custom modules
from modules.navigation_policy import NavigationPolicy
from modules.image_processor import ImageProcessor
from modules.localization import Localization
from sensors.range_sensor import SimulatedRangeSensor
from modules.scenes import Scene
from modules.feature_based_point_cloud_registration import FeatureBasedPointCloudRegistration
from modules.rgbd_similarity import RGBDSimilarity
from modules.navigator import Navigator
from modules.log_manager import LogManager
from modules.feature_matcher import FeatureMatcher
from modules.vm_manager import VMManager
from modules.ibvs_controller import IBVSController
from modules.pose_config import PoseConfig

import quaternion

def demo():
    # Load experiment configuration
    conf_exp_path = "/home/rodriguez/Documents/GitHub/habitat/habitat-lab/exp_config/"
    with open(conf_exp_path + 'config.yaml', 'r') as file:
        config_exp = yaml.safe_load(file)
    print(config_exp)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize scene and environment
    scene = Scene(config_exp['config']['scene'])
    config_path = "/home/rodriguez/Documents/GitHub/habitat/habitat-lab/habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_habitat_test-0.yaml"
    overrides = [
        "habitat.environment.max_episode_steps=10000",
        "habitat.simulator.forward_step_size=0.1",
        "habitat.simulator.turn_angle=2"
    ]
    config = habitat.get_config(config_path, overrides=overrides)
    with read_write(config):
        agent_config = get_agent_config(sim_config=config.habitat.simulator)
        agent_config.sim_sensors.depth_sensor.normalize_depth = False
    env = habitat.Env(config=config)
    env.episodes[0].scene_id = scene.get_path()
    observations = env.reset()

    print("Environment created successfully")
    print(f"Scene ID: {env.current_episode.scene_id}, Episode ID: {env.current_episode.episode_id}")

    # Experiment parameters
    nav_pol_train = config_exp['config']['nav_pol_train']
    manual_operation = config_exp['config']['manual_operation']
    autonomous_nav = config_exp['config']['autonomous_nav']
    id_run = config_exp['config']['id_run']
    id_vm = config_exp['config']['id_vm']
    feature_nav_conf = config_exp['config']['feature_nav']
    img2goal_mode = config_exp['config']['img2goal_mode']
    feature_mode = 'mnn'
    path_name = config_exp['config']['path_name']

    # Load goal image information
    goal_path = '/home/rodriguez/Documents/GitHub/habitat/habitat-lab/goal.json'
    with open(goal_path, 'r') as f:
        goal_dict = json.load(f)
    g_vm_id, g_frame = goal_dict['vm'], goal_dict['frame']

    # Determine key visual memory ID
    k_vm_id = config_exp['config']['k_vm_id'] if img2goal_mode else id_vm

    # Load VM IDs by config
    vm_ids_conf = config_exp['config']['vm_ids']
    vm_ids = {
        'castle': [70, 71],
        'rep': [93, 94, 95, 96, 97],
        'train-0': [63],
        'train-1': [64]
    }.get(vm_ids_conf, [])

    if not vm_ids:
        print('Invalid vm_ids')

    print("VM ids:", vm_ids)

    # Set agent's initial pose
    start_position = PoseConfig.poses[path_name]['position']
    rotation_quaternion = PoseConfig.poses[path_name]['quaternion']
    env.sim.set_agent_state(start_position, rotation_quaternion)
    observations = env.step(env.action_space.sample())

    # Constants and thresholds
    max_steps = 1200
    th_rmse_nav = 1.0
    th_simi = 0.6

    # Paths for data and models
    vms_root_path = '/home/rodriguez/Documents/GitHub/habitat/habitat-lab/selected_frames/'
    model_rgbd_path = '/home/rodriguez/Documents/GitHub/habitat/habitat-lab/nav_pol_models/model_20241029_065808.joblib'
    scaler_rgbd_path = '/home/rodriguez/Documents/GitHub/habitat/habitat-lab/nav_pol_models/scaler_20241029_065808.joblib'

    # Initialize components
    range_sensor = SimulatedRangeSensor(threshold_distance=0.1)
    rgbd_similarity = RGBDSimilarity(device=device)
    log_manager = LogManager()
    feature_registration = FeatureBasedPointCloudRegistration(config=config, device=device, id_run=id_run,
                                                              feature_nav_conf=feature_nav_conf,
                                                              feature_mode=feature_mode,
                                                              topological_map=None,
                                                              manual_operation=manual_operation)
    localization_instance = Localization(data_df=None, feature_registration=feature_registration,
                                         similarity=rgbd_similarity, th_rmse_nav=th_rmse_nav,
                                         tm_path=None, vm_path=None, th_vis=th_simi)
    ibvs = IBVSController(lambda_gain=0.5, assumed_depth=1.0)
    feature_matcher = FeatureMatcher(config={}, device=device)
    feature_matcher.set_threshold(0.1)
    feature_matcher.set_feature(feature_nav_conf)
    feature_matcher.set_mode(feature_mode)
    img_processor = ImageProcessor()
    vm_manager = VMManager(vms_root_path, vm_ids, rgbd_similarity, feature_registration, th_vis=0.8, sim_mode='rgbd')
    navigator = Navigator(model_path=model_rgbd_path, scaler_path=scaler_rgbd_path, rgbd=True)
    nav = NavigationPolicy(config=config, vm_len=1)  # Updated later when visual path is ready

    # Load goal image
    g_colors = np.load(vms_root_path + f'path_{g_vm_id}/selected_rgbs.npy')
    g_depths = np.load(vms_root_path + f'path_{g_vm_id}/selected_depths.npy')
    g_color, g_depth = g_colors[g_frame], g_depths[g_frame]

    # Initialize visual memory or perform localization if bot is lost
    if img2goal_mode and not manual_operation:
        vm_manager.set_o_frame(observations["rgb"], observations["depth"])
        vm_manager.set_g_frame(g_color, g_depth)
        k_colors, k_depths = vm_manager.get_img2goal_vm('v3', save=True)
        visual_path, orig_vm = vm_manager.get_visual_path()
        vm_manager.save_key_vm(k_vm_id)
        print("Visual path initialized")
    else:
        k_colors = np.load(vms_root_path + f'path_{id_vm}/selected_rgbs.npy')
        k_depths = np.load(vms_root_path + f'path_{id_vm}/selected_depths.npy')
        visual_path = list(range(k_colors.shape[0]))

    path_len = len(visual_path)
    nav = NavigationPolicy(config=config, vm_len=path_len)

    # Logging and loop state variables
    vm_image_index = 0
    count_steps = 0
    bot_lost = False
    bot_lost_counter = 0
    vp_empty = False
    o_color_list, o_depth_list = [], []
    agent_state_list, suggested_action_list, desired_velocity_list = [], [], []

    print("Starting main loop")

    while not env.episode_over:
        loop_start_time = time.time()

        # Relocalize if lost
        if img2goal_mode and bot_lost and not manual_operation:
            print("Relocalizing...")
            vm_manager.set_o_frame(observations["rgb"], observations["depth"])
            vm_manager.set_g_frame(g_color, g_depth)
            k_colors, k_depths = vm_manager.get_img2goal_vm('v3', save=True)
            visual_path = list(range(k_colors.shape[0]))
            path_len = len(visual_path)
            bot_lost_counter += 1
            if path_len == 0 or bot_lost_counter > 5:
                print("Relocalization failed. Exiting...")
                break
            bot_lost = False

        # Agent state and display
        agent_state = env.sim.get_agent_state()
        agent_state_list.append(agent_state)
        k_color, k_depth = k_colors[vm_image_index], k_depths[vm_image_index]
        o_color, o_depth = observations["rgb"], observations["depth"]
        img_processor.display_rgbd_data(o_color, k_color, o_depth, k_depth)

        if manual_operation or autonomous_nav:
            o_color_list.append(o_color)
            o_depth_list.append(o_depth)

        # Obstacle detection
        obstacles = range_sensor.detect_obstacles(o_depth)

        # Visual similarity
        if not manual_operation:
            sim_score = rgbd_similarity.compute_image_similarity(o_color, o_depth, k_color, k_depth)
            if sim_score < th_simi:
                print(f"Similarity too low ({sim_score}). Marking as lost.")
                bot_lost = True
                continue

        # Pose estimation
        if not manual_operation:
            bot_lost, est_quaternion, rmse, est_t_source_target, _ = feature_registration.compute_relative_pose(
                o_color, o_depth, k_color, k_depth)

        # Navigation decision
        if manual_operation:
            suggested_action = "forward"
        else:
            suggested_action = navigator.suggest_action_rgbd(rmse, est_t_source_target, est_quaternion, sim_score)

        nav.set_suggested_action(suggested_action)

        # IBVS for fine motion control
        current_keypoints, desired_keypoints = feature_matcher.filtered_matched_points_with_lightglue(o_color, k_color)
        desired_velocity = ibvs.compute_desired_velocity(np.array(current_keypoints), np.array(desired_keypoints))
        ibvs_action = ibvs.suggest_action_ibvs(desired_velocity)
        nav.set_ibvs_action(ibvs_action)

        # Obstacle avoidance
        if suggested_action == "forward" and not manual_operation and obstacles["center"]:
            print("Front obstacle detected. Marking bot as lost.")
            bot_lost = True
            continue

        # Execute action
        keystroke = cv2.waitKey(0)
        vm_image_index, action, _ = nav.handle_keystroke(keystroke, vm_image_index)
        print(f"Action: {action}")
        suggested_action_list.append(action)
        desired_velocity_list.append(desired_velocity)

        if action == "finish":
            break
        elif action:
            observations = env.step(action)
            count_steps += 1
        if count_steps >= max_steps:
            break

        print("-" * 50)

    # Save logs and outputs
    ibvs_dict = {
        'suggested_action': suggested_action_list,
        'desired_velocity': desired_velocity_list
    }
    with open("ibvs_dict_train-2.pkl", "wb") as f:
        pickle.dump(ibvs_dict, f)

    if nav_pol_train or autonomous_nav or manual_operation:
        log_manager.save_logs("dataset", id_run=id_run, id_vm=k_vm_id, feature_nav_conf=feature_nav_conf)
        log_manager.save_visual_path(visual_path, "manual_operation/all_frames", id_run=id_run, id_vm=k_vm_id,
                                     feature_nav_conf=feature_nav_conf)
    if manual_operation or autonomous_nav:
        np.save(f"manual_operation/all_frames/source_color_{id_run}_{k_vm_id}_{feature_nav_conf}.npy", o_color_list)
        np.save(f"manual_operation/all_frames/source_depth_{id_run}_{k_vm_id}_{feature_nav_conf}.npy", o_depth_list)
    with open(f"manual_operation/all_poses/agent_state_{id_run}_{k_vm_id}_{feature_nav_conf}.pkl", "wb") as f:
        pickle.dump(agent_state_list, f)

    log_manager.print_summary()


if __name__ == "__main__":
    demo()
