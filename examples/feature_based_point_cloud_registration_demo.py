import numpy as np
from modules.feature_based_point_cloud_registration import FeatureBasedPointCloudRegistration


def main():
    # Dummy configuration and parameters
    config = {}
    device = "cpu"
    id_run = 0
    feature_nav_conf = "ORB"  # use ORB feature matcher for the demo
    feature_mode = "star"

    registration = FeatureBasedPointCloudRegistration(
        config=config,
        device=device,
        id_run=id_run,
        feature_nav_conf=feature_nav_conf,
        feature_mode=feature_mode,
        topological_map=None,
        manual_operation=True,
    )

    # Create synthetic RGB-D images (256 x 256)
    src_color = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    src_depth = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    tgt_color = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    tgt_depth = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

    # Compute relative pose between the two frames
    result = registration.compute_relative_pose(src_color, src_depth, tgt_color, tgt_depth)
    bot_lost, quaternion, rmse, translation, transformation = result

    print("Bot lost:", bot_lost)
    print("RMSE:", rmse)
    print("Translation:", translation)
    print("Quaternion:", quaternion)
    print("Transformation matrix:\n", transformation)


if __name__ == "__main__":
    main()
