# Give priority to goal frame
import pandas as pd
import numpy as np
import pickle as pkl
import os

class VMManager:
    def __init__(self, vm_path, vm_ids, similarity, fb_pc_reg, th_vis=0.8, sim_mode='rgbd'):
        self.vm_paths = vm_path
        self.vm_ids = vm_ids
        self.th_vis = th_vis
        self.similarity = similarity
        self.fb_pc_reg = fb_pc_reg
        self.sim_mode = sim_mode
        self.o_color = None
        self.o_depth = None
        self.o_pose = None
        self.g_color = None
        self.g_depth = None
        self.g_pose = None
        self.key_vm_id = None  # To store the ID of the key VM
        self.init_candidate = None  # To store the best init candidate
        self.goal_candidate = None  # To store the best goal candidate
        self.best_init_candidates_df = None  # To store the DataFrame of best init candidates
        self.best_goal_candidates_df = None  # To store the DataFrame of best goal candidates
        self.meta_info = None  # To store meta-information for the key VM

    def set_o_frame(self, o_color, o_depth, o_pose=None):
        self.o_color = o_color
        self.o_depth = o_depth
        self.o_pose = o_pose

    def set_g_frame(self, g_color, g_depth, g_pose=None):
        self.g_color = g_color
        self.g_depth = g_depth
        self.g_pose = g_pose

    def get_k_frames(self, vm_path, vm_id=None):
        k_colors = np.load(vm_path + 'selected_rgbs.npy')
        k_depths = np.load(vm_path + 'selected_depths.npy')
        if vm_id is None:
            k_poses = None
        else:
            k_poses = pkl.load(open(vm_path + f'selected_poses_{vm_id}.pkl', 'rb'))
        return k_colors, k_depths, k_poses

    def create_vm_df(self, mode='obs'):
        sim_list, steps_list, rmse_list, poses_list, vm_list, frame_ids = [], [], [], [], [], []
        if mode == 'obs':
            print('vm mode = obs')
            t_color, t_depth = self.o_color, self.o_depth
        elif mode == 'goal':
            print('vm mode = goal')
            t_color, t_depth = self.g_color, self.g_depth
        else:
            raise ValueError("Invalid mode. Choose 'obs' or 'goal'.")
        for vm_id in self.vm_ids:
            VM_ID = vm_id
            vm_path = self.vm_paths + f'path_{vm_id}/'
            k_colors, k_depths, k_poses = self.get_k_frames(vm_path, vm_id)
            for frame_id, (k_pose, k_color, k_depth) in enumerate(zip(k_poses, k_colors, k_depths)):
                rmse, steps, t, R = self.fb_pc_reg.steps_from_source_to_target(
                    t_color, t_depth, k_color, k_depth, min_matches=4
                )
                if not self.fb_pc_reg.nav_eval(t,R): # geometric criteria
                    steps = np.inf
                sim_score = self.similarity.compute_image_similarity(
                    t_color, t_depth, k_color, k_depth
                )
                sim_list.append(sim_score)
                steps_list.append(steps)
                rmse_list.append(rmse)
                poses_list.append(k_pose)
                vm_list.append(VM_ID)
                frame_ids.append(frame_id)
        df = pd.DataFrame({
            'sim_score': sim_list,
            'steps': steps_list,
            'rmse': rmse_list,
            'frame_id': frame_ids,
            'vm_id': vm_list,
            'pose': poses_list,
        })

        # Store the DataFrame for later use
        if mode == 'obs':
            self.best_init_candidates_df = df
        elif mode == 'goal':
            self.best_goal_candidates_df = df
        else:
            raise ValueError("Invalid mode. Choose 'obs' or 'goal'.")

        return df

    def save_best_df(self, path, mode='obs'):
        """
        Saves the DataFrame of best candidates to a CSV file.

        Parameters:
        path (str): The file path where the DataFrame will be saved.
        mode (str): 'obs' to save the best init candidates,
                    'goal' to save the best goal candidates.
        """
        if mode == 'obs':
            df = self.best_init_candidates_df
        elif mode == 'goal':
            df = self.best_goal_candidates_df
        else:
            raise ValueError("Invalid mode. Choose 'obs' or 'goal'.")

        if df is not None:
            df.to_csv(path, index=False)
            print(f"Best candidates DataFrame saved to {path}")
        else:
            print(f"No DataFrame found for mode '{mode}'. Please run create_vm_df('{mode}') first.")

    def nav_path(self, best_init_candidates_df, best_goal_candidates_df):
        vm_scores = {}
        vm_paths = {}

        # Get the list of VMs present in both init and goal candidates
        vm_ids = np.intersect1d(
            best_init_candidates_df['vm_id'].unique(),
            best_goal_candidates_df['vm_id'].unique()
        )
        
        if len(vm_ids) == 0:
            print("No VMs have both valid init and goal candidates.")
            return None, None

        for vm_id in vm_ids:
            # Get init candidates in this VM
            init_candidates_vm = best_init_candidates_df[best_init_candidates_df['vm_id'] == vm_id]
            # Get goal candidates in this VM
            goal_candidates_vm = best_goal_candidates_df[best_goal_candidates_df['vm_id'] == vm_id]
            
            if init_candidates_vm.empty or goal_candidates_vm.empty:
                continue
            
            # Find best init candidate in this VM
            best_init_candidate = init_candidates_vm.loc[init_candidates_vm['sim_score'].idxmax()]
            init_frame_id = best_init_candidate['frame_id']

            # Filter goal candidates to those with frame_id greater than init_frame_id
            valid_goal_candidates_vm = goal_candidates_vm[goal_candidates_vm['frame_id'] > init_frame_id]
            
            if valid_goal_candidates_vm.empty:
                # No valid goal candidates after init candidate in this VM
                continue

            # Find best goal candidate among valid ones
            best_goal_candidate = valid_goal_candidates_vm.loc[valid_goal_candidates_vm['sim_score'].idxmax()]
            goal_frame_id = best_goal_candidate['frame_id']

            # Compute combined score (e.g., sum of sim_scores)
            combined_score = best_init_candidate['sim_score'] + best_goal_candidate['sim_score']
            
            # Save the data
            vm_scores[vm_id] = combined_score
            vm_paths[vm_id] = {
                'init_candidate': best_init_candidate,
                'goal_candidate': best_goal_candidate
            }
            
        if not vm_scores:
            print("No valid VMs found with acceptable init and goal candidates.")
            return None, None

        # Select the VM with the highest combined score
        best_vm_id = max(vm_scores, key=vm_scores.get)
        best_vm_data = vm_paths[best_vm_id]
        init_candidate = best_vm_data['init_candidate']
        goal_candidate = best_vm_data['goal_candidate']

        # Store the key VM ID and candidates for later use
        self.key_vm_id = best_vm_id
        self.init_candidate = init_candidate
        self.goal_candidate = goal_candidate

        # Extract the frames from init_frame_id to goal_frame_id within this VM
        vm_path = self.vm_paths + f'path_{best_vm_id}/'
        k_colors, k_depths, _ = self.get_k_frames(vm_path)
        with open(vm_path + f'selected_poses_{best_vm_id}.pkl', 'rb') as f:
            k_poses = pkl.load(f)

        # Extract the sequence from init_frame_id to goal_frame_id (inclusive)
        init_frame_id = init_candidate['frame_id']
        goal_frame_id = goal_candidate['frame_id']
        path_colors = k_colors[init_frame_id:goal_frame_id+1]
        path_depths = k_depths[init_frame_id:goal_frame_id+1]
        path_poses = k_poses[init_frame_id:goal_frame_id+1]

        # Append the actual goal frame and its pose (if available)
        path_colors = np.concatenate((path_colors, [self.g_color]), axis=0)
        path_depths = np.concatenate((path_depths, [self.g_depth]), axis=0)
        if self.g_pose is not None:
            path_poses.append(self.g_pose)
        else:
            path_poses.append(None)

        # Prepare meta-information
        meta_info = pd.DataFrame({
            'frame_id': range(len(path_colors)),
            'original_frame_id': list(range(init_frame_id, goal_frame_id+1)) + [None],
            'vm_id': [best_vm_id] * (goal_frame_id - init_frame_id + 1) + [None],
            'sim_score': [None] * (goal_frame_id - init_frame_id + 1) + [None],
            'rmse': [None] * (goal_frame_id - init_frame_id + 1) + [None],
            # Add other meta-information as needed
        })

        # Store meta-information for saving later
        self.meta_info = meta_info

        return path_colors, path_depths

    def nav_path_v2(self, best_init_candidates_df, best_goal_candidates_df):
        """
        Similar to nav_path_v2 but without using a combined score of init and goal candidates.
        Instead, this version prioritizes the VM with the highest goal candidate similarity score.

        Steps:
        1. Identify VMs present in both init and goal candidate sets.
        2. For each VM, select the best goal candidate (highest goal sim_score).
        3. From that VM's init candidates, choose an init candidate that comes before the goal candidate.
        If no suitable init candidate is found, skip this VM.
        4. Among all VMs that have suitable pairs (init before goal), select the VM whose best goal candidate
        has the highest sim_score. 
        
        Returns:
        -------
        path_colors : np.ndarray
            A sequence of RGB frames from the chosen init candidate up to the chosen goal candidate.
        path_depths : np.ndarray
            A sequence of depth frames corresponding to the chosen path.
        """

        # Dictionary to store the best goal candidate and corresponding init candidate per VM
        vm_results = {}

        # Find VMs that have both init and goal candidates
        vm_ids = np.intersect1d(
            best_goal_candidates_df['vm_id'].unique(),
            best_init_candidates_df['vm_id'].unique()
        )

        if len(vm_ids) == 0:
            print("No VMs have both valid init and goal candidates.")
            return None, None

        for vm_id in vm_ids:
            # Filter candidates for the current VM
            goal_candidates_vm = best_goal_candidates_df[best_goal_candidates_df['vm_id'] == vm_id]
            init_candidates_vm = best_init_candidates_df[best_init_candidates_df['vm_id'] == vm_id]

            if goal_candidates_vm.empty or init_candidates_vm.empty:
                continue

            # Find the best goal candidate (highest sim_score)
            best_goal_candidate = goal_candidates_vm.loc[goal_candidates_vm['sim_score'].idxmax()]
            goal_frame_id = best_goal_candidate['frame_id']

            # Look for init candidates that come before this goal candidate
            valid_init_candidates_vm = init_candidates_vm[init_candidates_vm['frame_id'] < goal_frame_id]

            if valid_init_candidates_vm.empty:
                # No suitable init candidate before the chosen goal candidate
                continue

            # Choose the init candidate with the highest sim_score
            best_init_candidate = valid_init_candidates_vm.loc[valid_init_candidates_vm['sim_score'].idxmax()]

            # Store results for this VM
            vm_results[vm_id] = {
                'goal_candidate': best_goal_candidate,
                'init_candidate': best_init_candidate
            }

        if not vm_results:
            print("No valid VMs found with acceptable init and goal candidates.")
            return None, None

        # Select the VM based solely on the highest goal candidate sim_score
        best_vm_id = max(vm_results.keys(), key=lambda vid: vm_results[vid]['goal_candidate']['sim_score'])
        best_vm_data = vm_results[best_vm_id]

        init_candidate = best_vm_data['init_candidate']
        goal_candidate = best_vm_data['goal_candidate']

        # Store the key VM ID and chosen candidates
        self.key_vm_id = best_vm_id
        self.init_candidate = init_candidate
        self.goal_candidate = goal_candidate

        # Load the frames and poses from the chosen VM
        vm_path = self.vm_paths + f'path_{best_vm_id}/'
        k_colors, k_depths, _ = self.get_k_frames(vm_path)
        with open(vm_path + f'selected_poses_{best_vm_id}.pkl', 'rb') as f:
            k_poses = pkl.load(f)

        # Extract the sequence from init_frame_id to goal_frame_id
        init_frame_id = init_candidate['frame_id']
        goal_frame_id = goal_candidate['frame_id']
        path_colors = k_colors[init_frame_id:goal_frame_id+1]
        path_depths = k_depths[init_frame_id:goal_frame_id+1]
        path_poses = k_poses[init_frame_id:goal_frame_id+1]

        # Append the actual goal frame and its pose (if available)
        path_colors = np.concatenate((path_colors, [self.g_color]), axis=0)
        path_depths = np.concatenate((path_depths, [self.g_depth]), axis=0)
        if self.g_pose is not None:
            path_poses.append(self.g_pose)
        else:
            path_poses.append(None)

        # Prepare meta-information DataFrame
        meta_info = pd.DataFrame({
            'frame_id': range(len(path_colors)),
            'original_frame_id': list(range(init_frame_id, goal_frame_id+1)) + [None],
            'vm_id': [best_vm_id] * (goal_frame_id - init_frame_id + 1) + [None],
            'sim_score': [None] * (goal_frame_id - init_frame_id + 1) + [None],
            'rmse': [None] * (goal_frame_id - init_frame_id + 1) + [None],
            # Additional fields can be added as needed
        })

        # Store meta-information for later use
        self.meta_info = meta_info

        return path_colors, path_depths

    def nav_path_v3(self, best_init_candidates_df, best_goal_candidates_df):
        """
        Selects the best candidate (init or goal) based on the highest sim score and finds a path from that candidate
        to the other within the same VM.

        Steps:
        1. Identify VMs present in both init and goal candidate sets.
        2. For each VM, select the best init candidate and the best goal candidate.
        3. Among all VMs, select the pair where either the init or goal candidate has the highest sim_score.
        4. Ensure that the init candidate precedes the goal candidate in the frame sequence.
        5. Extract the path from the init candidate to the goal candidate within the selected VM.
        6. Append the actual goal frame and its pose (if available).

        Returns:
        -------
        path_colors : np.ndarray
            A sequence of RGB frames from the chosen init candidate up to the chosen goal candidate.
        path_depths : np.ndarray
            A sequence of depth frames corresponding to the chosen path.
        """

        # Dictionary to store the best init and goal candidates per VM
        vm_candidates = {}

        # Find VMs that have both init and goal candidates
        vm_ids = np.intersect1d(
            best_goal_candidates_df['vm_id'].unique(),
            best_init_candidates_df['vm_id'].unique()
        )

        if len(vm_ids) == 0:
            print("No VMs have both valid init and goal candidates.")
            return None, None

        for vm_id in vm_ids:
            # Filter candidates for the current VM
            goal_candidates_vm = best_goal_candidates_df[best_goal_candidates_df['vm_id'] == vm_id]
            init_candidates_vm = best_init_candidates_df[best_init_candidates_df['vm_id'] == vm_id]

            if goal_candidates_vm.empty or init_candidates_vm.empty:
                continue

            # Find the best init candidate (highest sim_score)
            best_init_candidate = init_candidates_vm.loc[init_candidates_vm['sim_score'].idxmax()]
            # Find the best goal candidate (highest sim_score)
            best_goal_candidate = goal_candidates_vm.loc[goal_candidates_vm['sim_score'].idxmax()]

            # Store candidates for this VM
            vm_candidates[vm_id] = {
                'init_candidate': best_init_candidate,
                'goal_candidate': best_goal_candidate
            }

        if not vm_candidates:
            print("No valid VMs found with acceptable init and goal candidates.")
            return None, None

        # Find the pair with the highest sim_score among all VMs
        best_vm_id = None
        highest_sim_score = -np.inf

        for vm_id, candidates in vm_candidates.items():
            init_sim = candidates['init_candidate']['sim_score']
            goal_sim = candidates['goal_candidate']['sim_score']
            # Determine the higher sim_score between init and goal
            current_max_sim = max(init_sim, goal_sim)
            if current_max_sim > highest_sim_score:
                highest_sim_score = current_max_sim
                best_vm_id = vm_id

        if best_vm_id is None:
            print("No valid VMs found with acceptable candidates.")
            return None, None

        # Retrieve the best candidates for the selected VM
        best_vm_data = vm_candidates[best_vm_id]
        init_candidate = best_vm_data['init_candidate']
        goal_candidate = best_vm_data['goal_candidate']

        # Ensure that init candidate precedes the goal candidate
        if init_candidate['frame_id'] >= goal_candidate['frame_id']:
            print("Init candidate does not precede goal candidate in the selected VM.")
            return None, None

        # Load the frames and poses from the chosen VM
        vm_path = self.vm_paths + f'path_{best_vm_id}/'
        k_colors, k_depths, _ = self.get_k_frames(vm_path)
        with open(vm_path + f'selected_poses_{best_vm_id}.pkl', 'rb') as f:
            k_poses = pkl.load(f)

        # Extract the sequence from init_frame_id to goal_frame_id
        init_frame_id = init_candidate['frame_id']
        goal_frame_id = goal_candidate['frame_id']
        path_colors = k_colors[init_frame_id:goal_frame_id+1]
        path_depths = k_depths[init_frame_id:goal_frame_id+1]
        path_poses = k_poses[init_frame_id:goal_frame_id+1]

        # Append the actual goal frame and its pose (if available)
        path_colors = np.concatenate((path_colors, [self.g_color]), axis=0)
        path_depths = np.concatenate((path_depths, [self.g_depth]), axis=0)
        if self.g_pose is not None:
            path_poses.append(self.g_pose)
        else:
            path_poses.append(None)

        # Prepare meta-information DataFrame
        meta_info = pd.DataFrame({
            'frame_id': range(len(path_colors)),
            'original_frame_id': list(range(init_frame_id, goal_frame_id+1)) + [None],
            'vm_id': [best_vm_id] * (goal_frame_id - init_frame_id + 1) + [None],
            'sim_score': [None] * (goal_frame_id - init_frame_id + 1) + [None],
            'rmse': [None] * (goal_frame_id - init_frame_id + 1) + [None],
            # Additional fields can be added as needed
        })

        # Store meta-information for later use
        self.meta_info = meta_info

        # Store the key VM ID and chosen candidates
        self.key_vm_id = best_vm_id
        self.init_candidate = init_candidate
        self.goal_candidate = goal_candidate

        return path_colors, path_depths

    def get_img2goal_vm(self, version='v1', save=False, save_path="temp/"):
        print(f'visual threshold: {self.th_vis}')
        init_loc_df = self.create_vm_df(mode='obs')
        # Filter best init candidates with th_vis
        best_init_candidates_df = init_loc_df[
            (init_loc_df['sim_score'] >= self.th_vis)
        ]
        
        goal_loc_df = self.create_vm_df(mode='goal')
        # Filter best goal candidates with th_vis
        best_goal_candidates_df = goal_loc_df[
            (goal_loc_df['sim_score'] >= self.th_vis)
        ]
        if save:
            best_init_candidates_df.to_csv(save_path + 'best_init_candidates.csv')
            best_goal_candidates_df.to_csv(save_path + 'best_goal_candidates.csv')

        # Pass the filtered DataFrames to nav_path
        if version == 'v1':
            print('v1')
            k_colors, k_depths = self.nav_path(best_init_candidates_df, best_goal_candidates_df)
        elif  version == 'v2':
            print('v2')
            k_colors, k_depths = self.nav_path_v2(best_init_candidates_df, best_goal_candidates_df)
        elif  version == 'v3':
            print('v3')
            k_colors, k_depths = self.nav_path_v3(best_init_candidates_df, best_goal_candidates_df)
        else:
            print('Invalid version: choose from v1, v2, v3')
            return None, None
        return k_colors, k_depths

    def save_key_vm(self, vm_id):
        """
        Saves the key VM data (colors, depths, poses) for the specified vm_id.
        The key VM is composed of frames from the init candidate to the goal candidate
        within the selected key VM, and appends the goal frame at the end.
        The data is saved to path_{vm_id}/, including a CSV with meta-information.

        Parameters:
        vm_id (int): The ID to assign to the new key VM.
        """
        if self.key_vm_id is None or self.init_candidate is None or self.goal_candidate is None:
            print("No key VM has been determined. Please run get_img2goal_vm() first.")
            return

        # Get the data for the key VM from nav_path
        best_vm_id = self.key_vm_id
        init_candidate = self.init_candidate
        goal_candidate = self.goal_candidate

        # Extract the frames from init_frame_id to goal_frame_id within the key VM
        vm_path = self.vm_paths + f'path_{best_vm_id}/'
        k_colors, k_depths, _ = self.get_k_frames(vm_path)
        with open(vm_path + f'selected_poses_{best_vm_id}.pkl', 'rb') as f:
            k_poses = pkl.load(f)

        # Extract the sequence from init_frame_id to goal_frame_id (inclusive)
        init_frame_id = init_candidate['frame_id']
        goal_frame_id = goal_candidate['frame_id']
        path_colors = k_colors[init_frame_id:goal_frame_id+1]
        path_depths = k_depths[init_frame_id:goal_frame_id+1]
        path_poses = k_poses[init_frame_id:goal_frame_id+1]

        # Append the actual goal frame and its pose (if available)
        path_colors = np.concatenate((path_colors, [self.g_color]), axis=0)
        path_depths = np.concatenate((path_depths, [self.g_depth]), axis=0)
        if self.g_pose is not None:
            path_poses.append(self.g_pose)
        else:
            path_poses.append(None)

        # Define the directory to save the key VM data
        save_dir = self.vm_paths + f'path_{vm_id}/'
        os.makedirs(save_dir, exist_ok=True)

        # Save the colors
        np.save(save_dir + 'selected_rgbs.npy', path_colors)
        print(f"Saved colors to {save_dir + 'selected_rgbs.npy'}")

        # Save the depths
        np.save(save_dir + 'selected_depths.npy', path_depths)
        print(f"Saved depths to {save_dir + 'selected_depths.npy'}")

        # Save the poses
        with open(save_dir + f'selected_poses_{vm_id}.pkl', 'wb') as f:
            pkl.dump(path_poses, f)
        print(f"Saved poses to {save_dir + f'selected_poses_{vm_id}.pkl'}")

        # Update meta-information DataFrame
        self.meta_info['frame_id'] = range(len(path_colors))
        self.meta_info['original_frame_id'] = list(range(init_frame_id, goal_frame_id+1)) + [None]
        self.meta_info['vm_id'] = [best_vm_id] * (goal_frame_id - init_frame_id + 1) + [None]
        # Include sim_score and rmse if available
        # For demonstration, we can set them to None or extract from DataFrames if needed

        # Save the meta-information CSV
        meta_csv_path = save_dir + 'meta_info.csv'
        self.meta_info.to_csv(meta_csv_path, index=False)
        print(f"Saved meta-information to {meta_csv_path}")

        print(f"Key VM {vm_id} data saved successfully to {save_dir}")

    def get_visual_path(self):
        visual_path = self.meta_info['frame_id'].tolist()
        orig_vm = self.meta_info['vm_id'].unique()
        #assert len(orig_vm) == 1, "Multiple vm_id found"
        orig_vm = orig_vm[0]
        return visual_path, orig_vm
