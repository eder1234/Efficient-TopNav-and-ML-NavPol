import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

class Navigator:
    def __init__(self, model_path, scaler_path, label_encoder_path=None, rgbd = False):
        """
        Initializes the Navigator by loading the pre-trained model, scaler, and optionally the label encoder.

        :param model_path: Path to the saved MLP model (.joblib file).
        :param scaler_path: Path to the saved scaler (.joblib file).
        :param label_encoder_path: (Optional) Path to the saved label encoder (.joblib file).
        """
        # Load the trained MLP model
        self.model = load(model_path)

        # Load the scaler used during training
        self.scaler = load(scaler_path)

        # Load the label encoder if provided
        if label_encoder_path:
            self.label_encoder = load(label_encoder_path)
            self.classes_ = self.label_encoder.classes_
        else:
            self.label_encoder = None
            self.classes_ = self.model.classes_

        # Initialize the action buffer
        self.action_buffer = []

    def suggest_action(self, rmse, estimated_translation, estimated_quaternion, similarity_score=None):
        """
        Suggests an action based on the RMSE, estimated translation, and estimated quaternion.
        This method is for the old classifier version without 'v_sim' and without label encoding.

        :param rmse: Root Mean Square Error of the point cloud registration.
        :param estimated_translation: Estimated translation vector (numpy array or list of [x, y, z]).
        :param estimated_quaternion: Estimated quaternion with attributes .w, .x, .y, .z.
        :return: Suggested action as a string.

        Note: similarity_score is not yet implemented in this method.
        """
        # Extract translation components
        x, y, z = estimated_translation

        # Extract quaternion components
        qw = estimated_quaternion.w
        qx = estimated_quaternion.x
        qy = estimated_quaternion.y
        qz = estimated_quaternion.z

        # Prepare the data as a DataFrame with appropriate column names
        data = pd.DataFrame([{
            'rmse': rmse,
            'x': x,
            'y': y,
            'z': z,
            'qw': qw,
            'qx': qx,
            'qy': qy,
            'qz': qz
        }])

        # Scale the data using the loaded scaler
        data_scaled = self.scaler.transform(data.values)

        # Make predictions with probability outcomes
        probabilities = self.model.predict_proba(data_scaled)
        predicted_class_index = np.argmax(probabilities)
        predicted_action = self.classes_[predicted_class_index]

        # Update the action buffer and apply the oscillation check
        final_action = self._process_action(predicted_action, probabilities)

        return final_action

    def suggest_action_automatic(self, rmse, estimated_translation, estimated_quaternion, v_sim):
        """
        Suggests an action based on the RMSE, estimated translation, estimated quaternion, and visual similarity.
        This method is for the new classifier version with 'v_sim' and uses the label encoder if available.

        :param rmse: Root Mean Square Error of the point cloud registration.
        :param estimated_translation: Estimated translation vector (numpy array or list of [x, y, z]).
        :param estimated_quaternion: Estimated quaternion with attributes .w, .x, .y, .z.
        :param v_sim: Visual similarity score (float).
        :return: Suggested action as a string.
        """
        # Extract translation components
        x, y, z = estimated_translation

        # Extract quaternion components
        qw = estimated_quaternion.w
        qx = estimated_quaternion.x
        qy = estimated_quaternion.y
        qz = estimated_quaternion.z

        # Prepare the data as a DataFrame with appropriate column names
        data = pd.DataFrame([{
            'rmse': rmse,
            'x': x,
            'y': y,
            'z': z,
            'qw': qw,
            'qx': qx,
            'qy': qy,
            'qz': qz,
            'v_sim': v_sim
        }])

        # Scale the data using the loaded scaler
        data_scaled = self.scaler.transform(data.values)

        # Make predictions
        probabilities = self.model.predict_proba(data_scaled)
        predicted_class_index = np.argmax(probabilities)

        if self.label_encoder:
            # Use the label encoder to get the action label
            predicted_action = self.label_encoder.inverse_transform([predicted_class_index])[0]
        else:
            # Use the classes_ attribute if label encoder is not available
            predicted_action = self.classes_[predicted_class_index]

        # Update the action buffer and apply the oscillation check
        final_action = self._process_action(predicted_action, probabilities)

        return final_action

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
            # If oscillation detected, choose the best among 'forward' or 'update'
            probabilities = probabilities[0]  # Extract the probabilities array
            # Get indices for 'forward' and 'update'
            forward_index = np.where(self.classes_ == 'forward')[0][0]
            update_index = np.where(self.classes_ == 'update')[0][0]
            forward_prob = probabilities[forward_index]
            update_prob = probabilities[update_index]

            if forward_prob > update_prob:
                predicted_action = 'forward'
            else:
                predicted_action = 'update'

            # Update the last action in the buffer
            self.action_buffer[-1] = predicted_action

        final_action = predicted_action

        return final_action
    
    def suggest_action_rgbd(self, rmse, estimated_translation, estimated_quaternion, sim):
        """
        Suggests an action based on the RMSE, estimated translation, estimated quaternion, and visual similarity.
        This method is for the new classifier version with 'v_sim' and uses the label encoder if available.

        :param rmse: Root Mean Square Error of the point cloud registration.
        :param estimated_translation: Estimated translation vector (numpy array or list of [x, y, z]).
        :param estimated_quaternion: Estimated quaternion with attributes .w, .x, .y, .z.
        :param v_sim: Visual similarity score (float).
        :return: Suggested action as a string.
        """
        # Extract translation components
        x, y, z = estimated_translation

        # Extract quaternion components
        qw = estimated_quaternion.w
        qx = estimated_quaternion.x
        qy = estimated_quaternion.y
        qz = estimated_quaternion.z

        # Prepare the data as a DataFrame with appropriate column names
        data = pd.DataFrame([{
            'rmse': rmse,
            'x': x,
            'y': y,
            'z': z,
            'qw': qw,
            'qx': qx,
            'qy': qy,
            'qz': qz,
            'sim': sim
        }])

        # Scale the data using the loaded scaler
        data_scaled = self.scaler.transform(data.values)

        # Make predictions
        probabilities = self.model.predict_proba(data_scaled)
        predicted_class_index = np.argmax(probabilities)

        if self.label_encoder:
            # Use the label encoder to get the action label
            predicted_action = self.label_encoder.inverse_transform([predicted_class_index])[0]
        else:
            # Use the classes_ attribute if label encoder is not available
            predicted_action = self.classes_[predicted_class_index]

        # Update the action buffer and apply the oscillation check
        final_action = self._process_action(predicted_action, probabilities)

        return final_action