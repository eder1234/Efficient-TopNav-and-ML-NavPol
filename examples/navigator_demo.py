import numpy as np
import quaternion
from modules.navigator import Navigator


def main():
    # Paths to trained model and scaler (replace with actual files)
    model_path = "path/to/model.joblib"
    scaler_path = "path/to/scaler.joblib"

    navigator = Navigator(model_path=model_path, scaler_path=scaler_path, rgbd=True)

    # Example inputs
    rmse = 0.5
    estimated_translation = np.array([0.1, 0.0, 1.0])
    estimated_quaternion = quaternion.from_float_array([1, 0, 0, 0])
    similarity = 0.9

    action = navigator.suggest_action_rgbd(rmse, estimated_translation, estimated_quaternion, similarity)
    print("Suggested action:", action)


if __name__ == "__main__":
    main()
