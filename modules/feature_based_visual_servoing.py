import numpy as np

class FeatureBasedVisualServoing:
    def __init__(self, image_width=256, image_height=256, hfov=np.pi/2, feature_matcher=None, depth_mode=False):
        self.W = image_width
        self.H = image_height
        self.hfov = hfov
        # Compute focal lengths in pixels
        self.fx = (self.W / 2) / np.tan(self.hfov / 2)
        self.fy = (self.H / 2) / np.tan(self.hfov / 2)
        # Principal points
        self.cx = self.W / 2
        self.cy = self.H / 2
        # Camera intrinsic matrix
        self.K = np.array([
            [self.fx,    0, self.cx],
            [   0,    self.fy, self.cy],
            [   0,       0,     1   ]
        ])
        self.feature_matcher = feature_matcher
        self.depth_mode = depth_mode

    def compute_camera_velocity_from_keypoints(self, kp_current, kp_target):
        """
        Compute the desired camera velocity to minimize the error between current and target keypoints.

        Parameters:
            kp_current (list of lists): Keypoints from the current image [[x1, y1], [x2, y2], ...].
            kp_target (list of lists): Keypoints from the target image [[x1, y1], [x2, y2], ...].

        Returns:
            v (numpy array): Desired camera velocity [vx, vy, vz, wx, wy, wz].
        """
        kp_current = np.array(kp_current, dtype=np.float64)
        kp_target = np.array(kp_target, dtype=np.float64)

        if kp_current.shape != kp_target.shape:
            raise ValueError("Current and target keypoints must have the same shape.")

        N = kp_current.shape[0]  # Number of keypoints

        # Compute the error vector (e)
        error = (kp_current - kp_target).flatten()  # Shape: (2N,)

        # Initialize the interaction matrix (L)
        L = np.zeros((2 * N, 6))

        for i in range(N):
            # Pixel coordinates
            u = kp_current[i, 0]
            v = kp_current[i, 1]

            # Convert pixel coordinates to normalized image coordinates
            xn = (u - self.cx) / self.fx
            yn = (v - self.cy) / self.fy

            # Assume unit depth Z = 1 (or use actual depth if available)
            Z = 1.0

            # Interaction matrix for point (xn, yn)
            Li = np.array([
                [-1/Z,      0,    xn/Z,   xn*yn, -(1 + xn**2),     yn],
                [    0, -1/Z,    yn/Z,  1 + yn**2,     -xn*yn,    -xn]
            ])
            L[2*i:2*i+2, :] = Li

        # Regularization parameter (gain)
        lambda_gain = 0.5

        # Compute camera velocity using least squares
        # v = -lambda * (L^T L + mu*I)^-1 L^T e
        # Adding a small damping factor mu for numerical stability
        mu = 1e-3
        H = L.T @ L + mu * np.eye(6)
        v = -lambda_gain * np.linalg.inv(H) @ L.T @ error

        return v  # Desired camera velocity [vx, vy, vz, wx, wy, wz]

    def compute_camera_velocity_from_images(self, c_img, t_img, dmap_current=None, dmap_target=None):
        kp_current, kp_target = self.feature_matcher.compute_matches(c_img, t_img)
        if self.depth_mode:
            v = self.compute_cam_vel_from_kp_and_depth(kp_current, kp_target, dmap_current, dmap_target)
        else:
            v = self.compute_camera_velocity_from_keypoints(self, kp_current, kp_target)
        return v
    
    def compute_navigation_action(self, v, linear_threshold=0.05, angular_threshold=np.deg2rad(1)):
        """
        Converts camera velocity v to one of the navigation actions.

        Parameters:
            v (numpy array): Desired camera velocity [vx, vy, vz, wx, wy, wz].
            linear_threshold (float): Threshold for linear movement over dt to consider moving forward (e.g., 0.05 m).
            angular_threshold (float): Threshold for angular movement over dt to consider turning (e.g., 1 degree in radians).

        Returns:
            action (str): One of 'Move Forward', 'Turn Left', 'Turn Right', 'Update Memory'.
        """
        vx = v[0]    # Linear velocity in x (forward)
        wz = v[5]    # Angular velocity around z (yaw)

        # Compute desired movements over a time step (assuming dt = 1 second)
        desired_linear_movement = vx * 1.0          # in meters
        desired_angular_movement = wz * 1.0         # in radians

        # Threshold checks
        if abs(desired_linear_movement) < linear_threshold and abs(desired_angular_movement) < angular_threshold:
            return 'Update Memory'  # Robot has reached the target image

        if abs(desired_linear_movement) >= abs(desired_angular_movement):
            if desired_linear_movement > 0:
                return 'Move Forward'
            else:
                return 'Update Memory'  # Cannot move backward; consider updating memory
        else:
            if desired_angular_movement > 0:
                return 'Turn Left'
            elif desired_angular_movement < 0:
                return 'Turn Right'
            else:
                return 'Update Memory'

    def compute_cam_vel_from_kp_and_depth(self, kp_current, kp_target, dmap_current, dmap_target):
        """
        Compute the desired camera velocity using keypoints and depth maps.

        Parameters:
            kp_current (list of lists): Keypoints from the current image [[x1, y1], [x2, y2], ...].
            kp_target (list of lists): Keypoints from the target image [[x1, y1], [x2, y2], ...].
            dmap_current (numpy array): Depth map of the current image.
            dmap_target (numpy array): Depth map of the target image.

        Returns:
            v (numpy array): Desired camera velocity [vx, vy, vz, wx, wy, wz].
        """
        kp_current = np.array(kp_current, dtype=np.float64)
        kp_target = np.array(kp_target, dtype=np.float64)

        if kp_current.shape != kp_target.shape:
            raise ValueError("Current and target keypoints must have the same shape.")

        N = kp_current.shape[0]  # Number of keypoints

        # Compute the error vector (e)
        error = (kp_current - kp_target).flatten()  # Shape: (2N,)

        # Initialize the interaction matrix (L)
        L = np.zeros((2 * N, 6))

        valid_indices = []

        for i in range(N):
            # Pixel coordinates
            u_curr = kp_current[i, 0]
            v_curr = kp_current[i, 1]
            u_tgt = kp_target[i, 0]
            v_tgt = kp_target[i, 1]

            # Depth values from depth maps
            u_int_curr = int(round(u_curr))
            v_int_curr = int(round(v_curr))
            u_int_tgt = int(round(u_tgt))
            v_int_tgt = int(round(v_tgt))

            # Check if keypoints are within image bounds
            if (0 <= u_int_curr < self.W) and (0 <= v_int_curr < self.H) and \
            (0 <= u_int_tgt < self.W) and (0 <= v_int_tgt < self.H):
                Z_curr = dmap_current[v_int_curr, u_int_curr]
                Z_tgt = dmap_target[v_int_tgt, u_int_tgt]
            else:
                continue  # Skip if keypoint is outside the image

            # Validate depth values
            if Z_curr > 0 and Z_tgt > 0:
                # Use the average depth
                Z = (Z_curr + Z_tgt) / 2.0
            else:
                continue  # Skip if depth is invalid

            # Convert pixel coordinates to normalized image coordinates
            xn = (u_curr - self.cx) / self.fx
            yn = (v_curr - self.cy) / self.fy

            # Interaction matrix for point (xn, yn)
            Li = np.array([
                [-1/Z,      0,    xn/Z,   xn*yn, -(1 + xn**2),     yn],
                [    0, -1/Z,    yn/Z,  1 + yn**2,     -xn*yn,    -xn]
            ])
            L[2*i:2*i+2, :] = Li
            valid_indices.append(2*i)
            valid_indices.append(2*i+1)

        # Remove zero rows corresponding to invalid keypoints
        L = L[valid_indices, :]
        error = error[valid_indices]

        if L.shape[0] < 6:
            raise ValueError("Not enough valid keypoints with depth information.")

        # Regularization parameter (gain)
        lambda_gain = 0.5

        # Compute camera velocity using least squares
        # Adding a small damping factor mu for numerical stability
        mu = 1e-3
        H = L.T @ L + mu * np.eye(6)
        v = -lambda_gain * np.linalg.solve(H, L.T @ error)

        return v  # Desired camera velocity [vx, vy, vz, wx, wy, wz]

    def compute_nav_actions_from_images(self, c_img, t_img, dmap_current=None, dmap_target=None):
        v = self.compute_camera_velocity_from_images(self, c_img, t_img, dmap_current=dmap_current, dmap_target=dmap_target)
        nav_action = self.compute_navigation_action(v)
        return nav_action

