import numpy as np
from modules.rgbd_similarity import RGBDSimilarity


def main():
    # Initialize similarity module (requires RD3D+ weights)
    similarity = RGBDSimilarity(device="cpu")

    # Create synthetic RGB-D observations
    rgb1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    depth1 = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    rgb2 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    depth2 = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

    # Compute similarity
    score = similarity.compute_image_similarity(rgb1, depth1, rgb2, depth2)
    print("Similarity score:", score)


if __name__ == "__main__":
    main()
