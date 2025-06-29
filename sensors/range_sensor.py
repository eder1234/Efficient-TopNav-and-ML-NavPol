import numpy as np

class SimulatedRangeSensor:
    def __init__(self, threshold_distance=1.0, debug=False):
        self.threshold_distance = threshold_distance  # Distance threshold to consider an object as an obstacle
        self.debug = debug

    def generate_pc_in_cam_ref_frame(self, depth_img):
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
    
    def detect_obstacles(self, depth_img):
        # Generate point cloud in camera reference frame
        pc_cam_h = self.generate_pc_in_cam_ref_frame(depth_img)

        # Remove homogeneous coordinate
        pc_cam = pc_cam_h[:3, :]
        # Remove points that are too close
        pc_cam = pc_cam[:, np.linalg.norm(pc_cam, axis=0) > 0.0001]

        # Calculate distances
        distances = np.linalg.norm(pc_cam, axis=0)

        # Define regions (left, center, right)
        left_mask = pc_cam[0, :] < -0.33
        right_mask = pc_cam[0, :] > 0.33
        center_mask = ~(left_mask | right_mask)

        # Debug: Print information about the point cloud and masks
        if self.debug:
            print(f"Total points: {pc_cam.shape[1]}")
            print(f"Points in left region: {np.sum(left_mask)}")
            print(f"Points in center region: {np.sum(center_mask)}")
            print(f"Points in right region: {np.sum(right_mask)}")
            print(f"Min distance: {np.min(distances):.2f}")
            print(f"Max distance: {np.max(distances):.2f}")
            print(f"Threshold distance: {self.threshold_distance:.2f}")

        # Check for obstacles in each region
        left_obstacle = np.any(distances[left_mask] < self.threshold_distance)
        center_obstacle = np.any(distances[center_mask] < self.threshold_distance)
        right_obstacle = np.any(distances[right_mask] < self.threshold_distance)
        
        print(f'center min distance: {np.min(distances[center_mask]):.2f}')
        if self.debug:
            # Debug: Print minimum distances in each region
            print(f"Min distance in left region: {np.min(distances[left_mask]):.2f}")
            print(f"Min distance in center region: {np.min(distances[center_mask]):.2f}")
            print(f"Min distance in right region: {np.min(distances[right_mask]):.2f}")

        return {
            "left": left_obstacle,
            "center": center_obstacle,
            "right": right_obstacle
        }
    
    def process_depth_image(self, depth_image):
        # Remove the singleton dimension from the depth image
        depth_image = np.squeeze(depth_image)

        # Now depth_image is a 2D array with shape (256, 256)
        # Continue with your existing processing...
        height, width = depth_image.shape

        # Define regions for front-left, front-center, and front-right detection
        center_width = width // 2
        region_width = width // 3  # Divide the width into three equal parts

        # Define regions of interest (ROIs) in the depth image
        left_roi = depth_image[:, :region_width]
        center_roi = depth_image[:, region_width:2*region_width]
        right_roi = depth_image[:, 2*region_width:]

        # Detect obstacles in each ROI
        obstacle_left = self.detect_obstacle(left_roi)
        obstacle_center = self.detect_obstacle(center_roi)
        obstacle_right = self.detect_obstacle(right_roi)

        return obstacle_left, obstacle_center, obstacle_right

    def detect_obstacle_(self, roi):
        # Check if there are pixels within the threshold distance indicating an obstacle
        obstacle_detected = np.any(roi < self.threshold_distance)
        return obstacle_detected

    def suggest_action(self, obstacle_left, obstacle_center, obstacle_right):
        # Suggest an action based on the detected obstacles
        if obstacle_center:
            if obstacle_left and not obstacle_right:
                return "Turn Right"
            elif obstacle_right and not obstacle_left:
                return "Turn Left"
            else:
                return "Stop"  # Obstacle detected in all directions or directly ahead
        else:
            return "Move Forward"  # No obstacle detected directly ahead
        
    def evaluate_action_reliability(self, obstacle_left, obstacle_center, obstacle_right, pose_estimation):
        """
        Evaluate the reliability of actions based on detected obstacles and pose estimation.
        
        Parameters:
        - obstacle_left (bool): Indicates if there's an obstacle on the left.
        - obstacle_center (bool): Indicates if there's an obstacle in the center.
        - obstacle_right (bool): Indicates if there's an obstacle on the right.
        - pose_estimation: The estimated pose from the point cloud registration module, which might include orientation and position data.
        
        Returns:
        - A dictionary with actions as keys and their reliability as values (True for reliable, False for unreliable).
        """
        # Initialize all actions as reliable
        action_reliability = {"Move Forward": True, "Turn Left": True, "Turn Right": True, "Stop": True}

        # Evaluate reliability based on obstacles
        if obstacle_center:
            action_reliability["Move Forward"] = False  # Not reliable if there's an obstacle directly ahead

        if obstacle_left:
            action_reliability["Turn Left"] = False  # Not reliable if there's an obstacle on the left

        if obstacle_right:
            action_reliability["Turn Right"] = False  # Not reliable if there's an obstacle on the right

        # Further adjust reliability based on pose estimation
        # Example: If the bot's orientation suggests a turn might lead to a collision, mark that action as unreliable
        # This is a placeholder for your pose estimation logic
        # if pose_estimation suggests high risk of collision for "Turn Left":
        #     action_reliability["Turn Left"] = False

        # Similarly for "Turn Right" and "Move Forward" based on pose_estimation

        return action_reliability