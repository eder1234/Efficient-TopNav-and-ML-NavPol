import numpy as np
from modules.feature_matcher import FeatureMatcher

class ICPRegistration:
    def __init__(self, config={}, feature_name='AKAZE', th_error=0.5):
        self.config = config
        self.feature = FeatureMatcher(config={}, device='cuda')
        self.feature.set_feature(feature_name)
        # Consider to include a th to the feature's error
        self.th_error = th_error

    def generate_pc_in_cam_ref_frame(self, depth_img, T_cam_world=None):
        #W = H = 256 # fixed for simplifity, but depth.shape could be used
        W, H, _ = depth_img.shape
        hfov = np.pi / 2
        K = np.array([
            [1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., 1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        
        # Now get an approximation for the true world coordinates -- see if they make sense
        # [-1, 1] for x and [1, -1] for y as array indexing is y-down while world is y-up
        xs, ys = np.meshgrid(np.linspace(-1,1,W), np.linspace(1,-1,W))
        depth = depth_img.reshape(1,W,W)
        xs = xs.reshape(1,W,W)
        ys = ys.reshape(1,W,W)

        # Unproject
        # negate depth as the camera looks along -Z
        xys = np.vstack((xs * depth , ys * depth, -depth, np.ones(depth.shape)))
        xys = xys.reshape(4, -1)
        pc_cam_h = np.matmul(np.linalg.inv(K), xys)
        return pc_cam_h
    
    def get_ipc_from_pc(self, pc_cam, kp_cam):
        W = 256
        cam_key_id = [int(kp[1]*W+ kp[0]) for kp in kp_cam]
        ipc_cam_h = pc_cam[:, cam_key_id]
        ipc_cam = ipc_cam_h[:3].T
        return ipc_cam
    
    def transform_point_cloud(self, point_cloud, transformation_matrix):
        # Add a ones column for homogeneous coordinates
        homogeneous_points = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
        # Apply the transformation
        return np.dot(homogeneous_points, transformation_matrix.T)[:, :3]
    
    def execute_SVD_registration(self, source_pc, target_pc):
        # Compute centroids
        centroid_source = np.mean(source_pc, axis=0)
        centroid_target = np.mean(target_pc, axis=0)
        
        # Center the point clouds
        source_centered = source_pc - centroid_source
        target_centered = target_pc - centroid_target
        
        # Cross-covariance
        H = np.dot(source_centered.T, target_centered)
        
        # SVD
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        # Translation
        t = centroid_target - np.dot(R, centroid_source)
        
        # Homogeneous transformation matrix
        T = np.identity(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        # Transform the source point cloud
        transformed_source_pc = self.transform_point_cloud(source_pc, T)
        
        # RMSE calculation
        distances = np.linalg.norm(transformed_source_pc - target_pc, axis=1)
        rmse = np.sqrt(np.mean(distances**2))
        
        return rmse, transformed_source_pc, T
    
    def frontal_obstacle_detection(self, k_dmap, est_T_o_k, distance_threshold=4.0, z_threshold=0.0, angle_threshold=0.5):
        # Generate point cloud from k_dmap
        k_pc = self.generate_pc_in_cam_ref_frame(k_dmap)
        
        # Extract rotation matrix and translation vector from est_T_o_k
        R = est_T_o_k[:3, :3]
        t = est_T_o_k[:3, 3]
        
        # Check if the target pose is in front
        z = t[2]
        # Use the trace of R to check if the rotation is small
        rotation_angle = np.arccos((np.trace(R) - 1) / 2)
        
        print(f"z: {z}, rotation_angle: {rotation_angle}")
        
        if z > z_threshold and abs(rotation_angle) < angle_threshold:
            print("Target pose is in front")
            
            # Get central points of the point cloud
            H, W = k_dmap.shape[:2]
            center_x, center_y = W // 2, H // 2
            center_range = 200  # Adjust this value to change the size of the central region
            
            # Create a mask for the central region
            y, x = np.ogrid[:H, :W]
            mask = ((x - center_x)**2 + (y - center_y)**2 <= center_range**2)
            
            # Apply the mask to get central points
            central_points = k_pc[:, mask.flatten()]
            
            print(f"central_points.shape: {central_points.shape}")
            
            # Transform central points to the bot's frame
            central_points_bot_frame = np.linalg.inv(est_T_o_k) @ central_points
            
            # Check if any transformed points are closer than the threshold
            distances = np.linalg.norm(central_points_bot_frame[:3, :], axis=0)
            
            if np.any(distances < distance_threshold):
                print(f"Obstacle detected. Closest distance to the bot: {np.min(distances)}")
                return True  # Obstacle detected
            else:
                print(f"No obstacle detected. Closest distance to the bot: {np.min(distances)}")
        else:
            print("Target pose is not in front")
        
        return False  # No obstacle detected or target not in front

    def navigability_eval(self, o_cimg, o_dmap, k_cimg, k_dmap, verbose=1, collision_detection=False, lateral_threshold=0.5, rotation_threshold=0.2, lateral_disabled=True, frontal_th=0.0):
        # navigability is either 0 or 1
        o_kp, k_kp = self.feature.compute_matches(o_cimg, k_cimg)
        frontal_obstacle = False
        rmse = np.inf
        if len(o_kp) < 4:
            return 0, rmse
        o_pc = self.generate_pc_in_cam_ref_frame(o_dmap)
        k_pc = self.generate_pc_in_cam_ref_frame(k_dmap)

        o_ipc = self.get_ipc_from_pc(o_pc, o_kp)
        k_ipc = self.get_ipc_from_pc(k_pc, k_kp)
        rmse, _, est_T_o_k = self.execute_SVD_registration(o_ipc, k_ipc)
        if collision_detection:
            frontal_obstacle = self.frontal_obstacle_detection(k_dmap, est_T_o_k)
        if rmse > self.th_error:
            if verbose == 1:
                print(f'rmse too big: {rmse}')
            return 0, rmse
        forward_displacement = est_T_o_k[2, 3]  # Z-axis displacement (corrected)
        if forward_displacement < frontal_th:
            if verbose == 1:
                print("target behind bot")
            return 0, rmse
        if collision_detection and frontal_obstacle:
            #if verbose == 1:
            print("\033[91mfrontal obstacle\033[0m")
            return 0, rmse
        # New condition: Check lateral displacement vs forward displacement
        lateral_displacement = np.sqrt(est_T_o_k[0, 3]**2 + est_T_o_k[1, 3]**2)  # X-Y plane displacement
        # Calculate rotation angle from the rotation matrix
        rotation_angle = np.arccos((np.trace(est_T_o_k[:3, :3]) - 1) / 2)
        print(f'Lateral displacement: {lateral_displacement:.2f}, Forward displacement: {forward_displacement:.2f}, Rotation angle: {rotation_angle:.2f}')
        # Check if lateral displacement is high relative to forward displacement
        # and rotation is small
        if forward_displacement > 0 and (lateral_displacement / forward_displacement) > lateral_threshold and abs(rotation_angle) < rotation_threshold and lateral_disabled: # UPDATE frontal_th
            if verbose == 1:
                print(f"\033[91mHigh lateral displacement ({lateral_displacement:.2f}) compared to forward displacement ({forward_displacement:.2f}) with small rotation ({rotation_angle:.2f})\033[0m")
            return 0, rmse

                
        # Consider to include a th to n_steps (cost)
        return 1, rmse