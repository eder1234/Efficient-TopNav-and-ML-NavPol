import numpy as np
import pandas as pd
from joblib import load

class IBVSController:
    def __init__(self, 
                 lambda_gain=0.5, 
                 assumed_depth=1.0, 
                 model_path="/home/rodriguez/Documents/GitHub/habitat/habitat-lab/nav_pol_models/model_20250213_145247.joblib", 
                 scaler_path="/home/rodriguez/Documents/GitHub/habitat/habitat-lab/nav_pol_models/scaler_20250213_145247.joblib"):

        """
        Initialize the IBVS controller.
        
        Parameters:
            lambda_gain (float): Control gain for the IBVS law (default: 0.5).
            assumed_depth (float): Assumed depth value for all features (default: 1.0).
                                    Since you are not using depth maps, this constant can be tuned.
            K (np.ndarray or None): 3x3 camera intrinsic matrix. If provided, it is used to
                                    convert pixel coordinates to normalized image coordinates.
                                    If None, it is assumed that the keypoints are already normalized.
        """
        self.lambda_gain = lambda_gain
        self.assumed_depth = assumed_depth

        # Load the classifier model and scaler.
        self.model = load(model_path)
        self.scaler = load(scaler_path)
        
        # If the model was trained with a label encoder, you can load it here.
        # Otherwise, we rely on the model's classes_ attribute.
        self.label_encoder = None  # Modify if you have a label encoder.
        self.classes_ = self.model.classes_ if hasattr(self.model, "classes_") else None


        hfov = np.pi / 2  # example horizontal field of view
        focal_length = 1 / np.tan(hfov / 2.)

        # Define a 3x3 intrinsic matrix (no fourth row/column)
        self.K = np.array([
            [focal_length,       0, 0],
            [      0,   focal_length, 0],
            [      0,           0, 1]
        ])

        self.invK = np.linalg.inv(self.K)
        self.action_buffer = []


    def pixel_to_normalized(self, keypoints):
        """
        Convert keypoints from pixel coordinates to normalized image coordinates.
        
        Parameters:
            keypoints (np.ndarray): Array of shape (N, 2) containing keypoints in pixel coordinates.
            
        Returns:
            np.ndarray: Normalized keypoints of shape (N, 2).
        """
        keypoints = np.array(keypoints)
        num_points = keypoints.shape[0]
        # Create homogeneous coordinates [u, v, 1]
        keypoints_h = np.concatenate([keypoints, np.ones((num_points, 1))], axis=1)  # shape (N, 3)
        # Multiply by the inverse intrinsic matrix to get normalized coordinates.
        norm_coords = (self.invK @ keypoints_h.T).T  # shape (N, 3)
        return norm_coords[:, :2]

    def compute_desired_velocity(self, current_keypoints, desired_keypoints):
        """
        Compute the desired 6-DOF velocity twist using an IBVS control law.

        The IBVS error is computed from the difference between the current and desired
        feature locations in normalized image coordinates. For a feature at (x,y) with
        assumed depth Z, a typical interaction matrix row pair is:

            L_i = [ -1/Z,    0,   x/Z,   x*y,     -(1+x^2),   y ]
                [    0, -1/Z,   y/Z,  1+y^2,    -x*y,      -x ]

        The control law is:

            v = -λ · L⁺ · e

        where L⁺ is the pseudoinverse of the stacked interaction matrix and e is the error vector.

        Parameters:
            current_keypoints (np.ndarray): Current feature positions (Nx2). Expected in pixel coordinates.
            desired_keypoints (np.ndarray): Desired feature positions (Nx2). Expected in pixel coordinates.

        Returns:
            np.ndarray or None: A 6D velocity vector [vx, vy, vz, wx, wy, wz], or None if computation fails.
        """
        try:
            # If intrinsics are provided, convert pixel coordinates to normalized image coordinates.
            if self.K is not None:
                current_norm = self.pixel_to_normalized(current_keypoints)
                desired_norm = self.pixel_to_normalized(desired_keypoints)
            else:
                # Assume the keypoints are already in normalized coordinates.
                current_norm = np.array(current_keypoints)
                desired_norm = np.array(desired_keypoints)

            # Compute the feature error (stacking differences for each point).
            error = (current_norm - desired_norm).reshape(-1, 1)  # shape (2N, 1)

            num_points = current_norm.shape[0]
            if num_points == 0:
                return None  # No keypoints, cannot compute velocity

            L_list = []
            for i in range(num_points):
                x, y = current_norm[i]  # normalized coordinates
                # Build the 2x6 interaction matrix for this feature point.
                L_i = np.array([
                    [-1.0 / self.assumed_depth, 0.0, x / self.assumed_depth, x * y, -(1 + x**2), y],
                    [0.0, -1.0 / self.assumed_depth, y / self.assumed_depth, 1 + y**2, -x * y, -x]
                ])
                L_list.append(L_i)

            L = np.vstack(L_list)  # Shape: (2N, 6)

            # Check if the interaction matrix has sufficient rank for inversion
            if np.linalg.matrix_rank(L) < 6:
                return None  # The system is underdetermined

            # Compute the pseudoinverse of the interaction matrix.
            L_pinv = np.linalg.pinv(L)

            # Compute the 6D velocity vector using the IBVS control law.
            velocity = -self.lambda_gain * np.dot(L_pinv, error)  # shape (6, 1)

            return velocity.flatten()  # Return as a 1D array of length 6

        except Exception as e:
            return None  # Catch any unexpected errors and return None
        
    def suggest_action_ibvs(self, desired_velocity):
        """
        Suggest a discrete navigation action based on the desired velocity.
        
        This method scales the desired velocity vector, uses the classifier model to predict
        the action, and optionally post-processes the result.
        
        Parameters:
            desired_velocity (list or np.ndarray): A 6D velocity vector.
            
        Returns:
            str: The predicted discrete navigation action.
        """
        # Prepare input data: the model expects a DataFrame (one row with 6 columns).
        data = pd.DataFrame([desired_velocity])
        
        # Scale the input using the pre-loaded scaler.
        data_scaled = self.scaler.transform(data.values)
        
        # Get class probabilities and predicted class index.
        probabilities = self.model.predict_proba(data_scaled)
        predicted_class_index = np.argmax(probabilities)
        
        # Retrieve the predicted label via label encoder or directly from classes_.
        if self.label_encoder:
            predicted_action = self.label_encoder.inverse_transform([predicted_class_index])[0]
        elif self.classes_ is not None:
            predicted_action = self.classes_[predicted_class_index]
        else:
            predicted_action = str(predicted_class_index)

        #print(f"Predicted Action before preprocessing: {predicted_action}")
        
        # Optionally post-process the action (e.g., smoothing or buffering).
        #final_action_int = self._process_action(predicted_action, probabilities)
        final_action_int = predicted_action
        # map 0 to update, 1 to forward, 2 to left, 3 to right

        if final_action_int == 0:
            final_action = "update"
        elif final_action_int == 1:
            final_action = "forward"
        elif final_action_int == 2:
            final_action = "left"
        elif final_action_int == 3:
            final_action = "right"
        else:
            final_action = "unknown"
        
        final_action_processsed = self._process_action(final_action, probabilities)

        return final_action_processsed
    
    def _process_action(self, predicted_action, probabilities):
        """
        Processes the predicted action by checking the action buffer for oscillations
        between 'left' and 'right' and, if detected, suggests the most probable action
        between 'forward' and 'update'.

        :param predicted_action: The action predicted by the model.
        :param probabilities: The probability estimates from the model.
        :return: The final action after processing.
        """
        # Append the predicted action to the buffer
        self.action_buffer.append(predicted_action)

        # Keep only the last two actions
        if len(self.action_buffer) > 2:
            self.action_buffer.pop(0)

        # Detect oscillation between 'left' and 'right'
        if (len(self.action_buffer) == 2 and
            self.action_buffer[0] in ['left', 'right'] and
            self.action_buffer[1] in ['left', 'right'] and
            self.action_buffer[0] != self.action_buffer[1]):

            predicted_action = 'update'

            # Update the last action in the buffer
            self.action_buffer[-1] = predicted_action

        final_action = predicted_action

        return final_action

    def velocity_to_action(velocity, min_vx=1.0, angular_threshold=0.005, ratio_threshold=0.01, forward_threshold=3.0):
        """
        Modified controller that combines an absolute angular threshold with a turning ratio.
        
        Parameters:
            velocity (list or np.ndarray): A 6D velocity vector [vx, vy, vz, wx, wy, wz].
            min_vx (float): Minimum forward speed below which the action is "update".
            angular_threshold (float): If |wz| exceeds this, a turn is commanded.
            ratio_threshold (float): If |wz|/vx exceeds this value, a turn is also commanded.
            forward_threshold (float): If vx exceeds this and no turning is commanded, action is "forward".
            
        Returns:
            str: A discrete navigation action: "forward", "left", "right", or "update".
        """
        velocity = np.array(velocity)
        vx, vy, vz, wx, wy, wz = velocity

        # If the forward speed is too low, return update.
        if vx < min_vx:
            return "update"
        
        # Check turning conditions: either an absolute angular threshold or a ratio threshold.
        if abs(wz) > angular_threshold or (vx > 0 and (abs(wz)/vx) > ratio_threshold):
            return "right" if wz > 0 else "left"
        
        # If moving fast enough and not turning, go forward.
        if vx >= forward_threshold:
            return "forward"
        
        # Otherwise, return update.
        return "update"

# --- Example usage ---
if __name__ == "__main__":
    import cv2

    # Example intrinsic: For a 256x256 image with a horizontal FOV of 90°,
    # the focal length f = (image_width/2) / tan(FOV/2) = 128.
    #K_example = np.array([[128,   0, 128],
    #                      [  0, 128, 128],
    #                      [  0,   0,   1]])
    
    # Initialize the IBVS controller with your chosen gain, assumed depth, and camera intrinsics.
    ibvs = IBVSController(lambda_gain=0.5, assumed_depth=1.0)

    # Suppose you have 4 keypoints (for example, the corners of a square) detected by LightGlue.
    # These keypoints are in pixel coordinates.
    desired_keypoints = np.array([
        [100, 100],
        [156, 100],
        [156, 156],
        [100, 156]
    ])

    # Current keypoints (as might be extracted from a live image by LightGlue).
    current_keypoints = np.array([
        [105,  95],
        [160, 105],
        [155, 160],
        [ 95, 155]
    ])

    # Compute the desired camera velocity to reduce the feature error.
    desired_velocity = ibvs.compute_desired_velocity(current_keypoints, desired_keypoints)
    print("Desired velocity vector:", desired_velocity)

    # Convert the continuous velocity to a discrete navigation action.
    action = ibvs.velocity_to_action(desired_velocity)
    print("Discrete navigation action:", action)
