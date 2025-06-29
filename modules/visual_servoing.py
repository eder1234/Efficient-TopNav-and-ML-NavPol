import numpy as np
import cv2
from collections import deque

class VisualServoing:
    def __init__(self, method='fbvs', image_width=256, image_height=256, hfov=np.pi/2, feature_matcher=None, depth_mode=False, visual_threshold=10.0, feature_error_threshold=5.0):
        """
        Initializes the VisualServoing class with the specified method.

        Parameters:
            method (str): 'fbvs' for Feature-Based Visual Servoing, 'pvs' for Photometric Visual Servoing.
            image_width (int): Width of the image in pixels.
            image_height (int): Height of the image in pixels.
            hfov (float): Horizontal field of view in radians.
            feature_matcher: An object with a method compute_matches(current_image, target_image) for FBVS.
            depth_mode (bool): Whether to use depth information in FBVS.
        """
        self.method = method
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

        self.action_history = deque(maxlen=4)  # Adjust maxlen as needed

        # Parameters specific to FBVS
        self.feature_matcher = feature_matcher
        self.depth_mode = depth_mode

        # Parameters for navigation action decision
        self.linear_threshold = 0.008  # meters
        self.angular_threshold = 0.002 #np.deg2rad(1)  # radians

        self.visual_threshold = visual_threshold
        self.feature_error_threshold = feature_error_threshold
        
        self.photometric_error = None

        # Validate the method parameter
        if method not in ['fbvs', 'fbvsgm', 'pvs', 'epvs', 'epvsgm', 'dvs', 'il']:
            print(f'Invalid method: {method}. Must be one of: fbvs, pvs, dvs, il')
            raise ValueError("Method must be 'fbvs', 'pvs', 'epvs' or 'dvs'")
        self.method = method

    def detect_oscillation(self):
        """
        Detects if the robot is oscillating between 'left' and 'right' actions.

        Returns:
            oscillation_detected (bool): True if oscillation is detected, False otherwise.
        """
        # Define the minimum number of actions to consider for oscillation detection
        oscillation_length = 4  # Adjust as needed

        # Check if the action history is long enough
        if len(self.action_history) < oscillation_length:
            return False  # Not enough data to detect oscillation

        # Get the last few actions
        recent_actions = list(self.action_history)[-oscillation_length:]

        # Check for oscillation pattern (e.g., alternating 'left' and 'right')
        oscillation_patterns = [
            ['left', 'right'] * (oscillation_length // 2),
            ['right', 'left'] * (oscillation_length // 2)
        ]

        for pattern in oscillation_patterns:
            if recent_actions == pattern:
                return True

        return False

    def compute_navigation_action(self, img_current, img_target, dmap_current=None, dmap_target=None):
        """
        Computes the navigation action based on the current and target images.
        """
        if self.method == 'fbvs':
            # Existing code...
            pass
        elif self.method == 'pvs':
            # Existing code...
            pass
        elif self.method == 'dvs':
            # Existing code...
            pass
        elif self.method == 'epvs':
            # Compute camera velocity using Enhanced PVS
            v, _ = self.compute_camera_velocity_enhanced_pvs(img_current, img_target)
        else:
            raise ValueError("Method must be 'fbvs', 'pvs', 'dvs', 'il', or 'epvs'")
        
        # Compute navigation action based on camera velocity
        action = self.compute_navigation_action_from_velocity(v)
        return action



    def compute_camera_velocity_fbvs(self, img_current, img_target, dmap_current=None, dmap_target=None):
        """
        Compute the desired camera velocity using Feature-Based Visual Servoing.

        Parameters:
            img_current (numpy array): Current image.
            img_target (numpy array): Target image.
            dmap_current (numpy array): Depth map of the current image.
            dmap_target (numpy array): Depth map of the target image.

        Returns:
            v (numpy array): Desired camera velocity [vx, vy, vz, wx, wy, wz].
        """
        # Compute keypoints and matches using the feature matcher
        kp_current, kp_target = self.feature_matcher.compute_matches(img_current, img_target)

        if self.depth_mode:
            v = self.compute_cam_vel_from_kp_and_depth(kp_current, kp_target, dmap_current, dmap_target)
        else:
            v = self.compute_camera_velocity_from_keypoints(kp_current, kp_target)
        return v

    def compute_camera_velocity_pvs(self, img_current, img_target):
        """
        Compute the desired camera velocity using Photometric Visual Servoing.

        Parameters:
            img_current (numpy array): Current image (grayscale or color).
            img_target (numpy array): Target image (grayscale or color).

        Returns:
            v (numpy array): Desired camera velocity [vx, vy, vz, wx, wy, wz].
        """
        # Check if images are color and convert to grayscale if necessary
        if len(img_current.shape) == 3 and img_current.shape[2] == 3:
            img_current_gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
            img_target_gray = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
        else:
            img_current_gray = img_current
            img_target_gray = img_target

        # Compute photometric error
        error_image = img_current_gray.astype(np.float32) - img_target_gray.astype(np.float32)
        photometric_error = np.mean(np.abs(error_image))
        self.photometric_error = photometric_error

        # Compute image gradients
        grad_x = np.gradient(img_current_gray.astype(np.float32), axis=1)
        grad_y = np.gradient(img_current_gray.astype(np.float32), axis=0)

        # Compute interaction matrix and camera velocity
        v = self.compute_camera_velocity_pvs_internal(error_image, grad_x, grad_y)
        return v
    
    def get_photometric_error(self):
        return self.photometric_error

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
        # Adding a small damping factor mu for numerical stability
        mu = 1e-3
        H = L.T @ L + mu * np.eye(6)
        v = -lambda_gain * np.linalg.solve(H, L.T @ error)

        return v  # Desired camera velocity [vx, vy, vz, wx, wy, wz]
    
    def compute_fbvs_rmse_from_images(self, img_current, img_target):
        """
        Compute the desired camera velocity to minimize the error between current and target keypoints.

        Parameters:
            kp_current (list of lists): Keypoints from the current image [[x1, y1], [x2, y2], ...].
            kp_target (list of lists): Keypoints from the target image [[x1, y1], [x2, y2], ...].

        Returns:
            v (numpy array): Desired camera velocity [vx, vy, vz, wx, wy, wz].
        """
        kp_current, kp_target = self.feature_matcher.compute_matches(img_current, img_target)
        kp_current = np.array(kp_current, dtype=np.float64)
        kp_target = np.array(kp_target, dtype=np.float64)

        if kp_current.shape != kp_target.shape:
            raise ValueError("Current and target keypoints must have the same shape.")

        N = kp_current.shape[0]  # Number of keypoints

        # Compute the error vector (e)
        error = (kp_current - kp_target).flatten()  # Shape: (2N,)
        
        # Compute squared errors
        squared_errors = error ** 2

        # Compute mean of squared errors
        mean_squared_error = np.mean(squared_errors)

        # Compute RMSE
        rmse = np.sqrt(mean_squared_error)
        return rmse

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

    def compute_camera_velocity_pvs_internal(self, error_image, grad_x, grad_y, Z=None):
        """
        Computes the camera velocity to minimize the photometric error for PVS.

        Parameters:
            error_image (numpy array): Difference between current and target images.
            grad_x (numpy array): Gradient of current image in x-direction.
            grad_y (numpy array): Gradient of current image in y-direction.
            Z (numpy array or None): Depth map of the current image. If None, assumes Z=1.

        Returns:
            v (numpy array): Desired camera velocity [vx, vy, vz, wx, wy, wz].
        """
        # Flatten the arrays
        error_vector = error_image.flatten()
        grad_x_vector = grad_x.flatten()
        grad_y_vector = grad_y.flatten()
        
        # Coordinates of each pixel
        u_coords, v_coords = np.meshgrid(np.arange(self.W), np.arange(self.H))
        u_coords = u_coords.flatten()
        v_coords = v_coords.flatten()
        
        # Convert pixel coordinates to normalized image coordinates
        xn = (u_coords - self.cx) / self.fx
        yn = (v_coords - self.cy) / self.fy

        # Remove invalid entries where any of the arrays is not finite
        valid_indices = np.isfinite(error_vector) & np.isfinite(grad_x_vector) & np.isfinite(grad_y_vector)
        
        if Z is not None:
            Z_vector = Z.flatten()
            valid_indices &= (Z_vector > 0) & np.isfinite(Z_vector)
            Z_vector = Z_vector[valid_indices]
        else:
            Z_vector = np.ones_like(error_vector[valid_indices])

        # Filter out invalid entries
        error_vector = error_vector[valid_indices]
        grad_x_vector = grad_x_vector[valid_indices]
        grad_y_vector = grad_y_vector[valid_indices]
        xn = xn[valid_indices]
        yn = yn[valid_indices]

        N = error_vector.shape[0]

        if N < 6:
            raise ValueError("Not enough valid pixels to compute the camera velocity.")

        # Build the interaction matrix L using actual depth Z
        L = np.zeros((N, 6))

        # Compute the interaction matrix elements
        L[:, 0] = grad_x_vector / Z_vector
        L[:, 1] = grad_y_vector / Z_vector
        L[:, 2] = -(grad_x_vector * xn + grad_y_vector * yn) / Z_vector
        L[:, 3] = -(grad_x_vector * xn * yn + grad_y_vector * (1 + yn**2))
        L[:, 4] = grad_x_vector * (1 + xn**2) + grad_y_vector * xn * yn
        L[:, 5] = -(grad_x_vector * yn - grad_y_vector * xn)

        # Regularization parameter (gain)
        lambda_gain = 0.1

        # Compute camera velocity using least squares
        mu = 1e-3  # Damping factor for numerical stability
        H = L.T @ L + mu * np.eye(6)
        v = -lambda_gain * np.linalg.solve(H, L.T @ error_vector)

        return v  # Desired camera velocity [vx, vy, vz, wx, wy, wz]


    def compute_navigation_action_from_velocity(self, v):
        """
        Converts camera velocity v to one of the navigation actions.

        Parameters:
            v (numpy array): Desired camera velocity [vx, vy, vz, wx, wy, wz].

        Returns:
            action (str): One of 'forward', 'left', 'right', 'update'.
        """
        # Invert the sign of vx if necessary
        vx = -v[0]    # Linear velocity in x (forward)
        wz = v[5]     # Angular velocity around z (yaw)

        # Compute desired movements over a time step (assuming dt = 1 second)
        desired_linear_movement = vx * 1.0          # in meters
        desired_angular_movement = wz * 1.0         # in radians

        # Debugging prints
        print(f"Computed velocities: vx = {vx:.4f} m/s, wz = {wz:.4f} rad/s")
        print(f"Desired movements: linear = {desired_linear_movement:.4f} m, angular = {desired_angular_movement:.4f} rad")
        print(f"Current thresholds: linear_threshold = {self.linear_threshold:.4f} m, angular_threshold = {self.angular_threshold:.4f} rad")

        # Threshold checks
        if abs(desired_linear_movement) < self.linear_threshold and abs(desired_angular_movement) < self.angular_threshold:
            action = 'update'  # Robot has reached the target image
        elif desired_linear_movement >= self.linear_threshold:
            action = 'forward'
        elif abs(desired_angular_movement) > self.angular_threshold:
            # Decide to turn if angular deviation is significant
            if desired_angular_movement > 0:
                action = 'left'
            else:
                action = 'right'
        else:
            action = 'update'

        # Update action history
        self.action_history.append(action)

        # Oscillation detection logic
        oscillation_detected = self.detect_oscillation()

        if oscillation_detected:
            print("Oscillation detected. Adjusting action.")
            # Decide the most likely action between 'forward' and 'update'
            if desired_linear_movement >= self.linear_threshold:
                action = 'forward'
            else:
                action = 'update'

        # Print the chosen action
        print(f"Chosen action: {action}\n")

        return action

          
    def compute_camera_velocity_dvs_internal(self, error_image, grad_x, grad_y):
        """
        Computes the camera velocity to minimize the depth error for Direct Visual Servoing.

        Parameters:
            error_image (numpy array): Difference between current and target depth maps.
            grad_x (numpy array): Gradient of current depth map in x-direction.
            grad_y (numpy array): Gradient of current depth map in y-direction.

        Returns:
            v (numpy array): Desired camera velocity [vx, vy, vz, wx, wy, wz].
        """
        # Flatten the arrays
        error_vector = error_image.flatten()
        grad_x_vector = grad_x.flatten()
        grad_y_vector = grad_y.flatten()

        # Coordinates of each pixel
        u_coords, v_coords = np.meshgrid(np.arange(self.W), np.arange(self.H))
        u_coords = u_coords.flatten()
        v_coords = v_coords.flatten()

        # Convert pixel coordinates to normalized image coordinates
        xn = (u_coords - self.cx) / self.fx
        yn = (v_coords - self.cy) / self.fy

        # Assume unit depth Z = 1 for all pixels (or use actual depth if needed)
        Z = 1.0

        # Build the interaction matrix L
        N = error_vector.shape[0]
        L = np.zeros((N, 6))

        # Compute the interaction matrix elements
        L[:, 0] = grad_x_vector / Z  # Interaction with vx
        L[:, 1] = grad_y_vector / Z  # Interaction with vy
        L[:, 2] = -(grad_x_vector * xn + grad_y_vector * yn) / Z  # Interaction with vz
        L[:, 3] = -(grad_x_vector * xn * yn + grad_y_vector * yn**2)  # Interaction with wx
        L[:, 4] = grad_x_vector * (xn**2 + 1) + grad_y_vector * xn * yn  # Interaction with wy
        L[:, 5] = -(grad_x_vector * yn - grad_y_vector * xn)  # Interaction with wz

        # Regularization parameter (gain)
        lambda_gain = 0.1

        # Remove invalid entries where gradients might be NaN or Inf
        valid_indices = np.isfinite(L).all(axis=1) & np.isfinite(error_vector)
        L = L[valid_indices, :]
        error_vector = error_vector[valid_indices]

        if L.shape[0] < 6:
            raise ValueError("Not enough valid pixels to compute the camera velocity.")

        # Compute camera velocity using least squares with damping
        mu = 1e-3  # Damping factor for numerical stability
        H = L.T @ L + mu * np.eye(6)
        v = -lambda_gain * np.linalg.solve(H, L.T @ error_vector)

        return v  # Desired camera velocity [vx, vy, vz, wx, wy, wz]


    def compute_camera_velocity_dvs(self, dmap_current, dmap_target):
        """
        Compute the desired camera velocity using Direct Visual Servoing from depth maps.

        Parameters:
            dmap_current (numpy array): Depth map of the current image.
            dmap_target (numpy array): Depth map of the target image.

        Returns:
            v (numpy array): Desired camera velocity [vx, vy, vz, wx, wy, wz].
        """
        # Compute the error between current and desired depth maps
        error_image = dmap_current.astype(np.float32) - dmap_target.astype(np.float32)

        # Compute gradients of the current depth map
        grad_x = np.gradient(dmap_current.astype(np.float32), axis=1)
        grad_y = np.gradient(dmap_current.astype(np.float32), axis=0)

        # Compute camera velocity
        v = self.compute_camera_velocity_dvs_internal(error_image, grad_x, grad_y)
        return v

    def compute_camera_velocity_pvs_with_depth(self, img_current, img_target, dmap_current):
        """
        Compute the desired camera velocity using Photometric Visual Servoing with depth information.

        Parameters:
            img_current (numpy array): Current image (grayscale or color).
            img_target (numpy array): Target image (grayscale or color).
            dmap_current (numpy array): Depth map of the current image.

        Returns:
            v (numpy array): Desired camera velocity [vx, vy, vz, wx, wy, wz].
        """
        # Convert images to grayscale if they are color images
        if len(img_current.shape) == 3 and img_current.shape[2] == 3:
            img_current_gray = cv2.cvtColor(img_current, cv2.COLOR_BGR2GRAY)
            img_target_gray = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
        else:
            img_current_gray = img_current
            img_target_gray = img_target

        # Compute photometric error
        error_image = img_current_gray.astype(np.float32) - img_target_gray.astype(np.float32)
        photometric_error = np.mean(np.abs(error_image))
        self.photometric_error = photometric_error # not working

        # Compute image gradients
        grad_x = np.gradient(img_current_gray.astype(np.float32), axis=1)
        grad_y = np.gradient(img_current_gray.astype(np.float32), axis=0)

        # Call the internal method with depth information
        v = self.compute_camera_velocity_pvs_internal(error_image, grad_x, grad_y, dmap_current)
        return v


    def compute_photometric_error(self, img1, img2):
        """
        Computes the photometric error between two images.

        Parameters:
            img1 (numpy array): First image (grayscale or color).
            img2 (numpy array): Second image (grayscale or color).

        Returns:
            photometric_error (float): Mean absolute photometric error between the images.
        """
        # Convert images to grayscale if they are color images
        if len(img1.shape) == 3 and img1.shape[2] == 3:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img1_gray = img1
            img2_gray = img2

        # Ensure the images have the same size
        if img1_gray.shape != img2_gray.shape:
            img2_gray = cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0]))

        # Compute photometric error
        error_image = img1_gray.astype(np.float32) - img2_gray.astype(np.float32)
        photometric_error = np.mean(np.abs(error_image))
        return photometric_error

    def select_key_images_pvs(self, images):
        """
        Selects key images from a sequence based on photometric error.

        Parameters:
            images (list or array): List or array of images.

        Returns:
            key_indices (List[int]): Indices of selected key images.
        """
        num_images = len(images)
        key_indices = [0]  # Always select the first image

        current_image = images[0]

        for i in range(1, num_images):
            next_image = images[i]
            # Compute photometric error between current_image and next_image
            error = self.compute_photometric_error(current_image, next_image)
            if error > self.visual_threshold:
                key_indices.append(i)
                current_image = next_image

        # Ensure the last image is included
        if key_indices[-1] != num_images - 1:
            key_indices.append(num_images - 1)

        return key_indices

    def compute_feature_based_error(self, img1, img2):
        """
        Computes the RMSE between matched keypoints of two images.

        Parameters:
            img1 (numpy array): First image.
            img2 (numpy array): Second image.

        Returns:
            feature_error (float): RMSE of matched keypoints.
        """
        if self.feature_matcher is None:
            raise ValueError("Feature matcher is not set. Please provide a feature_matcher instance.")

        # Compute keypoints and matches using the feature matcher
        kp1, kp2 = self.feature_matcher.compute_matches(img1, img2)

        if len(kp1) == 0 or len(kp2) == 0:
            # No matches found, return a high error
            return float('inf')

        # Convert keypoints to NumPy arrays
        kp1 = np.array(kp1, dtype=np.float32)
        kp2 = np.array(kp2, dtype=np.float32)

        # Compute Euclidean distances between matched keypoints
        distances = np.linalg.norm(kp1 - kp2, axis=1)

        # Compute RMSE
        rmse = np.sqrt(np.mean(distances ** 2))

        return rmse

    def select_key_images_fbvs(self, images):
        """
        Selects key images from a sequence based on feature-based error (e.g., RMSE between matched keypoints).

        Parameters:
            images (list or array): List or array of images.

        Returns:
            key_indices (List[int]): Indices of selected key images.
        """
        num_images = len(images)
        key_indices = [0]  # Always select the first image

        current_image = images[0]

        for i in range(1, num_images):
            next_image = images[i]
            # Compute feature-based error between current_image and next_image
            error = self.compute_feature_based_error(current_image, next_image)
            if error > self.feature_error_threshold:
                key_indices.append(i)
                current_image = next_image

        # Ensure the last image is included
        if key_indices[-1] != num_images - 1:
            key_indices.append(num_images - 1)

        return key_indices
    
    def get_GZN_image(self, I, sigma):
        """
        Computes the Gaussian Zero-mean Normalized (GZN) image.

        Parameters:
            I (numpy array): Grayscale image.
            sigma (float): Standard deviation for Gaussian filter.

        Returns:
            I_GZN (numpy array): GZN image.
        """
        # Convert to float32
        I = I.astype(np.float32)
        # Apply Gaussian filter
        I_blur = cv2.GaussianBlur(I, (0, 0), sigmaX=sigma, sigmaY=sigma)
        # Zero-mean normalization
        mean_I_blur = np.mean(I_blur)
        I_GZN = I_blur - mean_I_blur
        return I_GZN
    
    def compute_camera_velocity_enhanced_pvs(self, img_current, img_target):
        """
        Compute the desired camera velocity using Enhanced Photometric Visual Servoing.
        
        Parameters:
            img_current (numpy array): Current image (RGB).
            img_target (numpy array): Target image (RGB).
        
        Returns:
            v (numpy array): Desired camera velocity [vx, vy, vz, wx, wy, wz].
            photometric_error (float): RMSE of the photometric error between current and target GZN images.
        """
        # Step 1: Convert images to grayscale if they are RGB
        if len(img_current.shape) == 3 and img_current.shape[2] == 3:
            img_current_gray = cv2.cvtColor(img_current, cv2.COLOR_RGB2GRAY)
            img_target_gray = cv2.cvtColor(img_target, cv2.COLOR_RGB2GRAY)
        else:
            img_current_gray = img_current
            img_target_gray = img_target

        # Get image dimensions
        H, W = img_current_gray.shape
        # Update self.H and self.W if necessary
        if self.H != H or self.W != W:
            self.H = H
            self.W = W
            # Recompute fx, fy, cx, cy
            self.fx = (self.W / 2) / np.tan(self.hfov / 2)
            self.fy = (self.H / 2) / np.tan(self.hfov / 2)
            self.cx = self.W / 2
            self.cy = self.H / 2

        # Step 2: Compute GZN images
        sigma = 1.0  # Standard deviation for Gaussian filter
        I_hat_current = self.get_GZN_image(img_current_gray, sigma)
        I_hat_target = self.get_GZN_image(img_target_gray, sigma)

        # Step 3: Compute photometric error (RMSE)
        error_image = I_hat_current - I_hat_target
        e_hat = error_image.flatten()

        # Compute RMSE
        photometric_error = np.sqrt(np.mean(e_hat ** 2))

        # Step 4: Compute gradient of I_hat_target
        grad_y, grad_x = np.gradient(I_hat_target.astype(np.float32))  # Note: np.gradient returns gradients in y and x order

        # Flatten gradients
        grad_x = grad_x.flatten()
        grad_y = grad_y.flatten()

        # Step 5: Compute x and y normalized coordinates
        u_coords, v_coords = np.meshgrid(np.arange(self.W), np.arange(self.H))
        u_coords = u_coords.flatten()
        v_coords = v_coords.flatten()
        xn = (u_coords - self.cx) / self.fx
        yn = (v_coords - self.cy) / self.fy

        # Assume Z = 1 for all pixels
        Z = 1.0

        # Compute elements of Lx_i
        Z_inv = 1.0 / Z
        x_Z_inv = xn * Z_inv
        y_Z_inv = yn * Z_inv

        # Interaction matrix components
        Lx_1 = np.column_stack((-Z_inv * np.ones_like(xn), np.zeros_like(xn), x_Z_inv, xn * yn, -(1 + xn ** 2), yn))
        Lx_2 = np.column_stack((np.zeros_like(yn), -Z_inv * np.ones_like(yn), y_Z_inv, 1 + yn ** 2, -xn * yn, -xn))

        # Compute L_i for all pixels
        L = - (grad_x[:, np.newaxis] * Lx_1 + grad_y[:, np.newaxis] * Lx_2)

        # Step 6: Remove invalid pixels
        valid_indices = np.isfinite(L).all(axis=1) & np.isfinite(e_hat)

        # Optionally, set a threshold for gradient magnitude
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_threshold = 1e-3  # Threshold for gradient magnitude
        valid_indices &= (grad_magnitude > grad_threshold)

        # Filter L and e_hat
        L = L[valid_indices, :]
        e_hat = e_hat[valid_indices]

        if L.shape[0] < 6:
            raise ValueError("Not enough valid pixels to compute the camera velocity.")

        # Regularization parameter (gain)
        lambda_gain = 0.5  # Adjust this gain as needed

        # Compute camera velocity using least squares with damping
        mu = 1e-3  # Damping factor for numerical stability
        H = L.T @ L + mu * np.eye(6)
        v = -lambda_gain * np.linalg.solve(H, L.T @ e_hat)

        return v, photometric_error  # Desired camera velocity and photometric error

    def select_key_images_epvs(self, images):
        """
        Selects key images from a sequence based on photometric error.

        Parameters:
            images (list or array): List or array of images.

        Returns:
            key_indices (List[int]): Indices of selected key images.
        """
        num_images = len(images)
        key_indices = [0]  # Always select the first image

        current_image = images[0]

        for i in range(1, num_images):
            next_image = images[i]
            # Compute photometric error between current_image and next_image
            _, error = self.compute_camera_velocity_enhanced_pvs(current_image, next_image)
            if error > self.visual_threshold:
                key_indices.append(i)
                current_image = next_image

        # Ensure the last image is included
        if key_indices[-1] != num_images - 1:
            key_indices.append(num_images - 1)

        return key_indices
    
    def compute_dvs_from_depth_maps(self, c_dmap, t_dmap):
        """
        Compute the desired camera velocity using Depth-Based Visual Servoing (DVS)
        based on the current and target depth maps.

        Parameters:
            c_dmap (numpy.ndarray): Current depth map (H x W x 1).
            t_dmap (numpy.ndarray): Target depth map (H x W x 1).

        Returns:
            cam_vel (numpy.ndarray): Desired camera velocity [vx, vy, vz, wx, wy, wz].
            rmse (float): Root Mean Squared Error between current and target depth maps.
        """
        # Ensure depth maps have the same shape
        if c_dmap.shape != t_dmap.shape:
            raise ValueError("Current and target depth maps must have the same dimensions.")

        # Remove singleton dimensions if present
        if c_dmap.ndim == 3 and c_dmap.shape[2] == 1:
            Z_2D = c_dmap.squeeze(axis=2)
        else:
            Z_2D = c_dmap

        if t_dmap.ndim == 3 and t_dmap.shape[2] == 1:
            Z_star_2D = t_dmap.squeeze(axis=2)
        else:
            Z_star_2D = t_dmap

        # Flatten the depth maps to create vectors Z and Z*
        Z = Z_2D.flatten()  # Current depth values
        Z_star = Z_star_2D.flatten()  # Target depth values

        # Compute the error vector e = Z - Z*
        e = Z - Z_star

        # Compute Root Mean Squared Error (RMSE)
        rmse = np.sqrt(np.mean(e ** 2))

        # Compute partial derivatives A = dZ/dx and B = dZ/dy using finite differences
        dZ_dy, dZ_dx = np.gradient(Z_2D)  # numpy.gradient returns [dZ/dy, dZ/dx]

        # Flatten the derivatives to match the flattened depth map
        A = dZ_dx.flatten()
        B = dZ_dy.flatten()

        # Generate pixel coordinates
        H, W = Z_2D.shape
        x_indices = np.tile(np.arange(W), H)  # Column indices repeated for each row
        y_indices = np.repeat(np.arange(H), W)  # Row indices repeated for each column

        # Normalize pixel coordinates using camera intrinsics
        x = (x_indices - self.cx) / self.fx
        y = (y_indices - self.cy) / self.fy

        # Avoid division by zero by setting a minimum depth threshold
        epsilon = 1e-6
        Z_safe = np.where(Z > epsilon, Z, epsilon)

        # Compute components of the interaction matrix L_Z
        L_Z_part1 = A / Z_safe  # [A/Z]
        L_Z_part2 = B / Z_safe  # [B/Z]
        L_Z_part3 = -(Z_safe + x * A + y * B) / Z_safe  # [-(Z + xA + yB)/Z]

        # Compute additional terms for rotational velocity components
        Z_wx = -y * Z_safe - x * y * A - (1 + y ** 2) * B  # [Z_wx]
        Z_wy = x * Z_safe + (1 + x ** 2) * A + x * y * B  # [Z_wy]
        Z_wz = x * B - y * A  # [Z_wz]

        # Stack all components horizontally to form the interaction matrix L_Z (N x 6)
        L_Z = np.vstack((L_Z_part1, L_Z_part2, L_Z_part3, Z_wx, Z_wy, Z_wz)).T

        # Optionally, you can subsample the depth maps to reduce computational load
        # For example, take every 4th pixel in both dimensions
        # subsample_factor = 4
        # L_Z = L_Z[::subsample_factor, ::subsample_factor].reshape(-1, 6)
        # e = e[::subsample_factor, ::subsample_factor].flatten()

        # Compute the pseudo-inverse of L_Z
        # Using a small regularization term to improve numerical stability
        lambda_reg = 1e-6  # Regularization parameter
        try:
            L_Z_T_L_Z = L_Z.T @ L_Z + lambda_reg * np.eye(6)
            L_Z_pinv = np.linalg.inv(L_Z_T_L_Z) @ L_Z.T
        except np.linalg.LinAlgError as err:
            print("Linear algebra error during pseudo-inverse computation:", err)
            return np.zeros(6), rmse

        # Define the scalar gain parameter lambda
        lambda_gain = 1.0  # You may adjust this value as needed

        # Compute the desired camera velocity v = -lambda * L_Z^+ * e
        cam_vel = -lambda_gain * (L_Z_pinv @ e)

        return cam_vel, rmse


    def select_key_maps_dvs(self, maps):
        """
        Selects key images from a sequence based on photometric error.

        Parameters:
            images (list or array): List or array of images.

        Returns:
            key_indices (List[int]): Indices of selected key images.
        """
        num_maps = len(maps)
        key_indices = [0]  # Always select the first image

        current_map = maps[0]

        for i in range(1, num_maps):
            next_map = maps[i]
            # Compute photometric error between current_image and next_image
            _, error = self.compute_dvs_from_depth_maps(current_map, next_map)
            if error > self.visual_threshold:
                key_indices.append(i)
                current_map = next_map

        # Ensure the last image is included
        if key_indices[-1] != num_maps - 1:
            key_indices.append(num_maps - 1)

        return key_indices
    
    def compute_camera_velocity_epvs_GM(self, img_current, img_target, max_iterations=10, tolerance=1e-6):
        """
        Compute the desired camera velocity using Enhanced Photometric Visual Servoing
        and optimize using the Gauss-Newton method.

        Parameters:
            img_current (numpy array): Current image (RGB).
            img_target (numpy array): Target image (RGB).
            max_iterations (int): Maximum number of Gauss-Newton iterations.
            tolerance (float): Tolerance for convergence.

        Returns:
            v (numpy array): Desired camera velocity [vx, vy, vz, wx, wy, wz].
            photometric_error (float): RMSE of the photometric error between current and target GZN images.
        """
        import numpy as np
        import cv2

        # Step 1: Convert images to grayscale if they are RGB
        if len(img_current.shape) == 3 and img_current.shape[2] == 3:
            img_current_gray = cv2.cvtColor(img_current, cv2.COLOR_RGB2GRAY)
            img_target_gray = cv2.cvtColor(img_target, cv2.COLOR_RGB2GRAY)
        else:
            img_current_gray = img_current
            img_target_gray = img_target

        # Get image dimensions
        H, W = img_current_gray.shape
        # Update self.H and self.W if necessary
        if self.H != H or self.W != W:
            self.H = H
            self.W = W
            # Recompute fx, fy, cx, cy
            self.fx = (self.W / 2) / np.tan(self.hfov / 2)
            self.fy = (self.H / 2) / np.tan(self.hfov / 2)
            self.cx = self.W / 2
            self.cy = self.H / 2

        # Step 2: Compute GZN images
        sigma = 1.0  # Standard deviation for Gaussian filter
        I_hat_current = self.get_GZN_image(img_current_gray, sigma)
        I_hat_target = self.get_GZN_image(img_target_gray, sigma)

        # Flatten images
        I_hat_current_flat = I_hat_current.flatten()
        I_hat_target_flat = I_hat_target.flatten()

        # Step 3: Initialize variables for Gauss-Newton optimization
        v = np.zeros(6)  # Initial camera velocity guess
        photometric_error_prev = np.inf

        # Step 4: Precompute gradients and coordinates
        # Compute gradient of I_hat_target
        grad_y, grad_x = np.gradient(I_hat_target.astype(np.float32))

        # Flatten gradients
        grad_x = grad_x.flatten()
        grad_y = grad_y.flatten()

        # Compute x and y normalized coordinates
        u_coords, v_coords = np.meshgrid(np.arange(self.W), np.arange(self.H))
        u_coords = u_coords.flatten()
        v_coords = v_coords.flatten()
        xn = (u_coords - self.cx) / self.fx
        yn = (v_coords - self.cy) / self.fy

        # Assume Z = 1 for all pixels
        Z = 1.0

        # Compute elements of Lx_i
        Z_inv = 1.0 / Z
        x_Z_inv = xn * Z_inv
        y_Z_inv = yn * Z_inv

        # Interaction matrix components
        Lx_1 = np.column_stack((
            -Z_inv * np.ones_like(xn),
            np.zeros_like(xn),
            x_Z_inv,
            xn * yn,
            -(1 + xn ** 2),
            yn
        ))
        Lx_2 = np.column_stack((
            np.zeros_like(yn),
            -Z_inv * np.ones_like(yn),
            y_Z_inv,
            1 + yn ** 2,
            -xn * yn,
            -xn
        ))

        # Compute L_i for all pixels
        L = - (grad_x[:, np.newaxis] * Lx_1 + grad_y[:, np.newaxis] * Lx_2)

        # Step 5: Remove invalid pixels
        valid_indices = np.isfinite(L).all(axis=1)

        # Optionally, set a threshold for gradient magnitude
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_threshold = 1e-3  # Threshold for gradient magnitude
        valid_indices &= (grad_magnitude > grad_threshold)

        # Filter L and images
        L = L[valid_indices, :]
        I_hat_current_flat = I_hat_current_flat[valid_indices]
        I_hat_target_flat = I_hat_target_flat[valid_indices]

        if L.shape[0] < 6:
            raise ValueError("Not enough valid pixels to compute the camera velocity.")

        # Step 6: Gauss-Newton optimization loop
        for iteration in range(max_iterations):
            # Compute error image with current velocity estimate
            # Linearize the error function
            e_hat = I_hat_current_flat - I_hat_target_flat + L @ v

            # Compute photometric error (RMSE)
            photometric_error = np.sqrt(np.mean(e_hat ** 2))

            # Check for convergence
            if abs(photometric_error_prev - photometric_error) < tolerance:
                break

            photometric_error_prev = photometric_error

            # Compute the update using Gauss-Newton
            H = L.T @ L
            g = L.T @ e_hat

            # Solve for delta_v
            delta_v = -np.linalg.solve(H, g)

            # Update the velocity estimate
            v += delta_v

        return v, photometric_error  # Desired camera velocity and photometric error

    def select_key_images_epvs_GM(self, images):
        """
        Selects key images from a sequence based on photometric error.

        Parameters:
            images (list or array): List or array of images.

        Returns:
            key_indices (List[int]): Indices of selected key images.
        """
        num_images = len(images)
        key_indices = [0]  # Always select the first image

        current_image = images[0]

        for i in range(1, num_images):
            next_image = images[i]
            # Compute photometric error between current_image and next_image
            _, error = self.compute_camera_velocity_epvs_GM(current_image, next_image)
            if error > self.visual_threshold:
                key_indices.append(i)
                current_image = next_image

        # Ensure the last image is included
        if key_indices[-1] != num_images - 1:
            key_indices.append(num_images - 1)

        return key_indices
    
    def compute_camera_velocity_from_keypoints_GM(self, kp_current, kp_target, max_iterations=10, tolerance=1e-6):
        """
        Compute the desired camera velocity to minimize the error between current and target keypoints
        using the Gauss-Newton method.

        Parameters:
            kp_current (list of lists): Keypoints from the current image [[x1, y1], [x2, y2], ...].
            kp_target (list of lists): Keypoints from the target image [[x1, y1], [x2, y2], ...].
            max_iterations (int): Maximum number of Gauss-Newton iterations.
            tolerance (float): Tolerance for convergence.

        Returns:
            v (numpy array): Desired camera velocity [vx, vy, vz, wx, wy, wz].
            rmse (float): Root Mean Square Error between current and target keypoints.
        """
        import numpy as np

        kp_current = np.array(kp_current, dtype=np.float64)
        kp_target = np.array(kp_target, dtype=np.float64)

        if kp_current.shape != kp_target.shape:
            raise ValueError("Current and target keypoints must have the same shape.")

        N = kp_current.shape[0]  # Number of keypoints

        # Initialize camera velocity
        v = np.zeros(6)
        rmse_prev = np.inf

        for iteration in range(max_iterations):
            # Compute the error vector (e)
            error = (kp_current - kp_target).flatten()  # Shape: (2N,)

            # Compute RMSE
            rmse = np.sqrt(np.mean(error ** 2))

            # Check convergence
            if abs(rmse_prev - rmse) < tolerance:
                break

            rmse_prev = rmse

            # Initialize the interaction matrix (L)
            L = np.zeros((2 * N, 6))

            for i in range(N):
                # Pixel coordinates
                u = kp_current[i, 0]
                v_coord = kp_current[i, 1]

                # Convert pixel coordinates to normalized image coordinates
                xn = (u - self.cx) / self.fx
                yn = (v_coord - self.cy) / self.fy

                # Assume unit depth Z = 1 (or use actual depth if available)
                Z = 1.0

                # Interaction matrix for point (xn, yn)
                Li = np.array([
                    [-1/Z,      0,    xn/Z,   xn*yn, -(1 + xn**2),     yn],
                    [    0, -1/Z,    yn/Z,  1 + yn**2,     -xn*yn,    -xn]
                ])
                L[2*i:2*i+2, :] = Li

            # Compute the Hessian approximation and gradient
            H = L.T @ L
            g = L.T @ error

            # Solve for delta_v
            delta_v = -np.linalg.solve(H, g)

            # Update camera velocity
            v += delta_v

            # Update kp_current based on the estimated camera motion
            # Since we lack depth information, we approximate the update
            delta_s = L @ delta_v  # Change in keypoint positions
            kp_current_flat = kp_current.flatten() + delta_s
            kp_current = kp_current_flat.reshape(-1, 2)

        return v, rmse

    def compute_camera_velocity_fbvs_GM_without_depth(self, img_current, img_target):
        """
        Compute the desired camera velocity using Feature-Based Visual Servoing (GM)
        without using depth.
        """
        # Compute keypoints and matches using the feature matcher
        kp_current, kp_target = self.feature_matcher.compute_matches(img_current, img_target)
        v, rmse = self.compute_camera_velocity_from_keypoints_GM(kp_current, kp_target)
        return v, rmse

    def select_key_images_fbvs_GM(self, images):
        """
        Selects key images from a sequence based on photometric error.

        Parameters:
            images (list or array): List or array of images.

        Returns:
            key_indices (List[int]): Indices of selected key images.
        """
        num_images = len(images)
        key_indices = [0]  # Always select the first image

        current_image = images[0]

        for i in range(1, num_images):
            next_image = images[i]
            # Compute photometric error between current_image and next_image
            _, error = self.compute_camera_velocity_fbvs_GM_without_depth(current_image, next_image)
            if error > self.visual_threshold:
                key_indices.append(i)
                current_image = next_image

        # Ensure the last image is included
        if key_indices[-1] != num_images - 1:
            key_indices.append(num_images - 1)

        return key_indices
    
    def compute_velocity_and_error(self, img_current=None, img_target=None, map_current=None, map_target=None):
        if self.method == 'epvsgm':
            v, e = self.compute_camera_velocity_epvs_GM(img_current, img_target)
        elif self.method == 'fbvsgm':
            pass
        elif self.method == 'dvs':
            pass
        elif self.method == 'il':
            v, e = [0,0,0,0,0,0], 0
        else:
            raise ValueError("Unknown or not implemented method: {}".format(self.method))
        return v, e