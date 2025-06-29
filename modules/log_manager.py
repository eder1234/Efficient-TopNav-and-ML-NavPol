import pandas as pd
import numpy as np

class LogManager:
    def __init__(self):
        """
        Initialize all log lists within a dictionary for organized data management.
        """
        self.logs = {
            "rmse": [],
            "x": [],
            "y": [],
            "z": [],
            "qw": [],
            "qx": [],
            "qy": [],
            "qz": [],
            "action": [],
            "vm_list": [],
            "sim": [],
            "bot_state":[],
            "topo_update_time": [],
            "image_proc_time": [],
            "obstacle_time": [],
            "registration_time": [],
            "nav_policy_time": [],
            "action_exec_time": [],
        }
    
    def add_entry(self, **kwargs):
        """
        Add a single entry to the logs. Each keyword argument corresponds to a log key.

        Example:
            log_manager.add_entry(
                rmse=0.5,
                x=10,
                y=20,
                z=30,
                qw=0.707,
                qx=0,
                qy=0.707,
                qz=0,
                action="move_forward",
                vm_list=[1, 2, 3],
                sim=True,
                bot_state="FrontObs"
                topo_update_time=0.05,
                image_proc_time=0.1,
                obstacle_time=0.02,
                registration_time=0.03,
                nav_policy_time=0.04,
                action_exec_time=0.06,
            )
        """
        for key, value in kwargs.items():
            if key in self.logs:
                self.logs[key].append(value)
            else:
                raise KeyError(f"Log key '{key}' is not recognized.")
    
    def log_data(self, visual_path, vm_image_index, topological_map, rmse, action, est_t_source_target, 
                 est_quaternion, sim_score, bot_state, topo_update_time, image_proc_time, obstacle_time, 
                 registration_time, nav_policy_time, action_exec_time):#, data_logging_time):
        """
        Log data for a single step in the navigation system.

        Args:
            visual_path (list/array): The visual path data.
            vm_image_index (int): Index to access the visual path.
            topological_map (bool): Flag indicating if topological map is used.
            rmse (float): Root Mean Square Error.
            action (int/str/None): Action taken.
            est_t_source_target (tuple/list): Estimated translation (x, y, z).
            est_quaternion (object): Estimated orientation as a quaternion with attributes w, x, y, z.
            sim_score (float): Simulation score.
            bot_state (str): Bot's state within the main loop ("AutNav","LostVis", "FrontObs")
            topo_update_time (float): Time taken to update the topology.
            image_proc_time (float): Time taken for image processing.
            obstacle_time (float): Time taken for obstacle detection.
            registration_time (float): Time taken for registration.
            nav_policy_time (float): Time taken for navigation policy.
            action_exec_time (float): Time taken to execute the action.
        """
        # Determine the VM list entry
        vm_entry = visual_path[vm_image_index]# if topological_map else 0
        
        # Map action integer to string if necessary
        if isinstance(action, int):
            action_mapping = {1: "forward", 2: "left", 3: "right"}
            mapped_action = action_mapping.get(action, "update" if action is None else f"action_{action}")
        else:
            mapped_action = action  # Assume action is already a string or appropriate representation
        
        # Prepare the entry dictionary
        entry = {
            "vm_list": vm_entry,
            "rmse": rmse,
            "action": mapped_action,
            "x": est_t_source_target[0],
            "y": est_t_source_target[1],
            "z": est_t_source_target[2],
            "qw": est_quaternion.w,
            "qx": est_quaternion.x,
            "qy": est_quaternion.y,
            "qz": est_quaternion.z,
            "sim": sim_score,
            "bot_state": bot_state,
            "topo_update_time": topo_update_time,
            "image_proc_time": image_proc_time,
            "obstacle_time": obstacle_time,
            "registration_time": registration_time,
            "nav_policy_time": nav_policy_time,
            "action_exec_time": action_exec_time,
        }
        
        # Add the entry to the logs
        self.add_entry(**entry)
    
    def print_summary(self):
        """
        Print the number of entries for each log key.
        """
        summary = ", ".join([f"{key}: {len(values)} steps" for key, values in self.logs.items()])
        print(summary)
    
    def reset_logs(self):
        """
        Clear all log lists.
        """
        for key in self.logs:
            self.logs[key].clear()
        print("All logs have been reset.")
    
    def to_dataframe(self):
        """
        Convert the logs dictionary to a pandas DataFrame.
        """
        return pd.DataFrame(self.logs)
    
    def save_logs(self, filepath, id_run, id_vm, feature_nav_conf):
        """
        Save the logs to a CSV file with a specified filename pattern.

        Args:
            filepath (str): The directory path where the CSV will be saved.
            id_run (str/int): Identifier for the run.
            id_vm (str/int): Identifier for the VM.
            feature_nav_conf (str): Configuration identifier.
        """
        df = self.to_dataframe()
        filename = f"{filepath}/logs_{id_run}_{id_vm}_{feature_nav_conf}.csv"
        df.to_csv(filename, index=False)
        print(f"Logs saved to {filename}")
    
    def save_visual_path(self, visual_path, filepath, id_run, id_vm, feature_nav_conf):
        """
        Save the visual path to a NumPy file.

        Args:
            visual_path (list/array): The visual path data to save.
            filepath (str): The directory path where the NumPy file will be saved.
            id_run (str/int): Identifier for the run.
            id_vm (str/int): Identifier for the VM.
            feature_nav_conf (str): Configuration identifier.
        """
        filename = f"{filepath}/visual_path_{id_run}_{id_vm}_{feature_nav_conf}.npy"
        np.save(filename, visual_path)
        print(f"Visual path saved to {filename}")
