import numpy as np

class Localization:
    def __init__(self, data_df, feature_registration, similarity, th_rmse_nav, tm_path, vm_path, th_vis=0.8):
        self.data_df = data_df
        self.feature_registration = feature_registration  # Instance of FeatureBasedPointCloudRegistration
        self.similarity = similarity
        self.th_rmse_nav = th_rmse_nav
        self.tm_path = tm_path
        self.vm_path = vm_path
        self.th_vis = th_vis

    def localization(self, source_color, source_depth):
        list_steps = []
        for i in range(len(self.data_df)):
            _, target_color, target_depth = self.load_vm_arrays(i)
            simi_score = self.similarity.compute_image_similarity(source_color, target_color)
            if simi_score < self.th_vis:
                steps = float('inf')
            else:
                rmse, steps, _, _ = self.feature_registration.steps_from_source_to_target(
                    source_color, source_depth, target_color, target_depth)
                if rmse > self.th_rmse_nav:
                    steps = float('inf')
            list_steps.append(steps)
        close_top_map_id = list_steps.index(min(list_steps))
        print("Close to node:", close_top_map_id)
        return close_top_map_id, list_steps
    
    def localization_rgbd_in_vm(self, source_color, source_depth):
        list_steps = []
        target_colors = np.load(self.vm_path + "selected_rgbs.npy")
        target_depths = np.load(self.vm_path + "selected_depths.npy")
        len_vm = target_colors.shape[0]
        for i in range(len_vm):
            target_color = target_colors[i]
            target_depth = target_depths[i]
            simi_score = self.similarity.compute_image_similarity(source_color, source_depth, target_color, target_depth)
            if simi_score < self.th_vis:
                steps = float('inf')
            else:
                rmse, steps, _, _ = self.feature_registration.steps_from_source_to_target(
                    source_color, source_depth, target_color, target_depth)
            if rmse > self.th_rmse_nav:
                steps = float('inf')
            list_steps.append(steps)
        close_vm_id = list_steps.index(min(list_steps))
        return close_vm_id, list_steps
    
    def geometric_localization(self, source_colors, source_depths, target_color, target_depth):
        list_steps = []
        for i in range(len(source_colors)):
            source_color = source_colors[i]
            source_depth = source_depths[i]
            rmse, steps, _, _ = self.feature_registration.steps_from_source_to_target(
                    source_color, source_depth, target_color, target_depth)
            if rmse > self.th_rmse_nav:
                steps = float('inf')
            list_steps.append(steps)
        close_id = list_steps.index(min(list_steps))
        return close_id, list_steps

    def localization_rgb_in_vm(self, source_color, source_depth):
        list_steps = []
        target_colors = np.load(self.vm_path + "selected_rgbs.npy")
        target_depths = np.load(self.vm_path + "selected_depths.npy")
        len_vm = target_colors.shape[0]
        for i in range(len_vm):
            target_color = target_colors[i]
            target_depth = target_depths[i]
            simi_score = self.similarity.compute_image_similarity(source_color, target_color)
            if simi_score < self.th_vis:
                steps = float('inf')
            else:
                rmse, steps, _, _ = self.feature_registration.steps_from_source_to_target(
                    source_color, source_depth, target_color, target_depth)
            if rmse > self.th_rmse_nav:
                steps = float('inf')
            list_steps.append(steps)
        close_vm_id = list_steps.index(min(list_steps))
        return close_vm_id, list_steps

    def load_vm_arrays(self, vm_image_index):    
        target_depths = np.load(self.vm_path + "selected_depths.npy")
        target_depth = target_depths[vm_image_index]
        target_colors = np.load(self.vm_path + "selected_rgbs.npy")
        target_color = target_colors[vm_image_index]
        vm_len = len(target_depths)
        return vm_len, target_color, target_depth
    
    def image_to_goal_localization(self, goal_color):
        #goal_color = np.load(goal_color_path)
        score_list = []
        for i in range(len(self.data_df)):
            _, target_color, _ = self.load_vm_arrays(i)
            score = self.similarity.compute_image_similarity(goal_color, target_color)
            score_list.append(score)
        max_index = score_list.index(max(score_list))
        print("Most similar to target color:", max_index)
        return score_list
    
    