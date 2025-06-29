import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, config={}, c_color=None, t_color=None, c_depth=None, t_depth=None):
        self.config = config
        self.c_color = c_color
        self.t_color = t_color
        self.c_depth = c_depth
        self.t_depth = t_depth
    
    def transform_rgb_bgr(self, color_image):
        return color_image[:, :, [2, 1, 0]]

    def transform_depth(self, image):
        depth_image = (1.0 - (image / np.max(image))) * 255.0
        depth_image = depth_image.astype(np.uint8)
        return depth_image
    
    def display_current_images(self, color_image, depth_image):
        self.c_color = self.transform_rgb_bgr(color_image)
        self.c_depth = self.transform_depth(depth_image)
        cv2.namedWindow("Current Color", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Current Depth", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("Current Color", 100, 100)
        cv2.moveWindow("Current Depth", 100, 500)
        cv2.imshow("Current Color", self.c_color)
        cv2.imshow("Current Depth", self.c_depth)
        return self.c_color, self.c_depth
    
    def display_rgbd_data(self, source_color, target_color, source_depth, target_depth):
        source_color_bgr = cv2.cvtColor(source_color, cv2.COLOR_RGB2BGR)
        target_color_bgr = cv2.cvtColor(target_color, cv2.COLOR_RGB2BGR)

        # Normalize source depth image
        source_depth_normalized = cv2.normalize(source_depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        source_depth_8bit = np.uint8(source_depth_normalized)
        # Normalize target depth image
        target_depth_normalized = cv2.normalize(target_depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        target_depth_8bit = np.uint8(target_depth_normalized)

        cv2.namedWindow("Source color", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Target color", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Source depth", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Target depth", cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow("Source color", 100, 100)
        cv2.moveWindow("Target color", 100, 500)
        cv2.moveWindow("Source depth", 600, 100)
        cv2.moveWindow("Target depth", 600, 500)
        cv2.imshow("Source color", source_color_bgr)
        cv2.imshow("Target color", target_color_bgr)
        cv2.imshow("Source depth", np.uint8(source_depth_8bit))
        cv2.imshow("Target depth", np.uint8(target_depth_8bit))