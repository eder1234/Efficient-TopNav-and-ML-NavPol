import numpy as np

class PhotometricVisualServoing:
    def __init__(self, image_width=256, image_height=256, hfov=np.pi/2):
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
        # Parameters for navigation action decision
        self.linear_threshold = 0.05  # meters
        self.angular_threshold = np.deg2rad(1)  # radians

    def compute_navigation_action_from_images(self, img_current, img_target):
        """
        Computes the navigation action based on the photometric error between current and target images.

        Parameters:
            img_current (numpy array): Grayscale current image.
            img_target (numpy array): Grayscale target image.

        Returns:
            action (str): One of 'Move Forward', 'Turn Left', 'Turn Right', 'Update Memory'.
        """
        # Compute photometric error
        error_image = img_current.astype(np.float32) - img_target.astype(np.float32)
        photometric_error = np.mean(np.abs(error_image))

        # Compute image gradients
        grad_x = np.gradient(img_current.astype(np.float32), axis=1)
        grad_y = np.gradient(img_current.astype(np.float32), axis=0)

        # Compute interaction matrix and camera velocity
        v = self.compute_camera_velocity(error_image, grad_x, grad_y)

        # Map camera velocity to navigation action
        action = self.compute_navigation_action(v, photometric_error)

        return action

    def compute_camera_velocity(self, error_image, grad_x, grad_y):
        """
        Computes the camera velocity to minimize the photometric error.

        Parameters:
            error_image (numpy array): Difference between current and target images.
            grad_x (numpy array): Gradient of current image in x-direction.
            grad_y (numpy array): Gradient of current image in y-direction.

        Returns:
            v (numpy array): Desired camera velocity [vx, vy, vz, wx, wy, wz].
        """
        # Flatten the arrays
        error_vector = error_image.flatten()
        grad_x_vector = grad_x.flatten()
        grad_y_vector = grad_y.flatten()

        N = error_vector.shape[0]

        # Initialize interaction matrix L
        L = np.zeros((N, 6))

        # Coordinates of each pixel
        u_coords, v_coords = np.meshgrid(np.arange(self.W), np.arange(self.H))
        u_coords = u_coords.flatten()
        v_coords = v_coords.flatten()

        # Convert pixel coordinates to normalized image coordinates
        xn = (u_coords - self.cx) / self.fx
        yn = (v_coords - self.cy) / self.fy

        # Assume unit depth Z = 1 for all pixels
        Z = 1.0

        # Build the interaction matrix L
        L[:, 0] = grad_x_vector / Z
        L[:, 1] = grad_y_vector / Z
        L[:, 2] = -(grad_x_vector * xn + grad_y_vector * yn) / Z
        L[:, 3] = -(grad_x_vector * xn * yn + grad_y_vector * (1 + yn**2))
        L[:, 4] = grad_x_vector * (1 + xn**2) + grad_y_vector * xn * yn
        L[:, 5] = -(grad_x_vector * yn - grad_y_vector * xn)

        # Regularization parameter (gain)
        lambda_gain = 0.1

        # Compute camera velocity using least squares
        # Adding a small damping factor mu for numerical stability
        mu = 1e-3
        H = L.T @ L + mu * np.eye(6)
        v = -lambda_gain * np.linalg.solve(H, L.T @ error_vector)

        return v  # Desired camera velocity [vx, vy, vz, wx, wy, wz]

    def compute_navigation_action(self, v, photometric_error, error_threshold=5.0):
        """
        Converts camera velocity v to one of the navigation actions based on photometric error.

        Parameters:
            v (numpy array): Desired camera velocity [vx, vy, vz, wx, wy, wz].
            photometric_error (float): Mean absolute photometric error.
            error_threshold (float): Threshold for photometric error to consider the target reached.

        Returns:
            action (str): One of 'Move Forward', 'Turn Left', 'Turn Right', 'Update Memory'.
        """
        vx = v[0]    # Linear velocity in x (forward)
        wz = v[5]    # Angular velocity around z (yaw)

        # Threshold checks based on photometric error
        if photometric_error < error_threshold:
            return 'Update Memory'  # Robot has reached the target image

        # Determine dominant motion
        if abs(vx) >= abs(wz):
            if vx > 0:
                return 'Move Forward'
            else:
                return 'Update Memory'  # Cannot move backward; consider updating memory
        else:
            if wz > 0:
                return 'Turn Left'
            elif wz < 0:
                return 'Turn Right'
            else:
                return 'Update Memory'

