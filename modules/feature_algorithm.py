import cv2
import numpy as np
import torch
import logging
from lightglue import LightGlue, SuperPoint
from models.matching import Matching
from models.utils import frame2tensor
from lightglue.utils import numpy_image_to_torch, rbd

# Initialize logger
logger = logging.getLogger(__name__)

class FeatureAlgorithm:
    def __init__(self, config, device='cpu'):
        """
        Initializes the FeatureAlgorithm with the given configuration and device.

        Args:
            config (dict): Configuration dictionary.
            device (str): Device to use ('cpu' or 'cuda').
        """
        self.config = config
        self.device = device
        self.feature = None
        self.threshold = 0.5
        self.extractor = None
        self.matcher = None
        self.mode = 'star'

    def set_mode(self, mode):
        self.mode = mode

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_feature(self, feature):
        """
        Sets the feature extractor and matcher based on the specified feature.

        Args:
            feature (str): Name of the feature matching algorithm.
        """
        self.feature = feature
        if feature == 'LightGlue':
            self.extractor = SuperPoint(max_num_keypoints=400).eval().to(self.device)
            self.matcher = LightGlue(features='superpoint').eval().to(self.device)
        elif feature == 'SuperGlue':
            self.matcher = Matching({'superpoint': {}, 'superglue': {'weights': 'indoor'}}).to(self.device).eval()
        elif feature in ['AKAZE', 'ORB', 'BRISK', 'SIFT']:
            pass  # OpenCV features do not require special initialization
        else:
            raise ValueError(f"Feature '{feature}' is not supported.")

    def compute_matches(self, s_img, t_img):
        """
        Computes matched keypoints between two images using the specified feature matching algorithm.

        Args:
            s_img (np.ndarray): Source image.
            t_img (np.ndarray): Target image.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Matched keypoints from source and target images.
        """
        if self.feature == 'XFeat':
            return self.matched_points_with_xfeat(s_img, t_img)
        elif self.feature == 'LightGlue':
            return self.matched_points_with_lightglue(s_img, t_img)
        elif self.feature == 'SuperGlue':
            return self.matched_points_with_superglue(s_img, t_img)
        elif self.feature == 'AKAZE':
            return self.matched_points_with_akaze(s_img, t_img)
        elif self.feature == 'ORB':
            return self.matched_points_with_orb(s_img, t_img)
        elif self.feature == 'BRISK':
            return self.matched_points_with_brisk(s_img, t_img)
        elif self.feature == 'SIFT':
            return self.matched_points_with_sift(s_img, t_img)
        else:
            raise ValueError(f"Feature '{self.feature}' is not supported.")

    def matched_points_with_sift(self, s_img, t_img):
        # Convert images to grayscale if necessary
        s_img_gray = cv2.cvtColor(s_img, cv2.COLOR_BGR2GRAY) if s_img.ndim == 3 else s_img
        t_img_gray = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY) if t_img.ndim == 3 else t_img

        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(s_img_gray, None)
        kp2, des2 = sift.detectAndCompute(t_img_gray, None)

        if des1 is None or des2 is None:
            return np.array([]), np.array([])

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.threshold * n.distance:
                good_matches.append(m)

        if len(good_matches) == 0:
            return np.array([]), np.array([])

        points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        return points1, points2

    def matched_points_with_akaze(self, s_img, t_img):
        # Convert images to grayscale if necessary
        s_img_gray = cv2.cvtColor(s_img, cv2.COLOR_BGR2GRAY) if s_img.ndim == 3 else s_img
        t_img_gray = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY) if t_img.ndim == 3 else t_img

        # Initialize AKAZE detector
        akaze = cv2.AKAZE_create()
        kp1, des1 = akaze.detectAndCompute(s_img_gray, None)
        kp2, des2 = akaze.detectAndCompute(t_img_gray, None)

        if des1 is None or des2 is None:
            return np.array([]), np.array([])

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.threshold * n.distance:
                good_matches.append(m)

        if len(good_matches) == 0:
            return np.array([]), np.array([])

        points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        return points1, points2

    def matched_points_with_orb(self, s_img, t_img):
        # Convert images to grayscale if necessary
        s_img_gray = cv2.cvtColor(s_img, cv2.COLOR_BGR2GRAY) if s_img.ndim == 3 else s_img
        t_img_gray = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY) if t_img.ndim == 3 else t_img

        # Initialize ORB detector
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(s_img_gray, None)
        kp2, des2 = orb.detectAndCompute(t_img_gray, None)

        if des1 is None or des2 is None:
            return np.array([]), np.array([])

        # BFMatcher with Hamming distance and crossCheck
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if len(matches) == 0:
            return np.array([]), np.array([])

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        return points1, points2


    def matched_points_with_brisk(self, s_img, t_img):
        """
        Match keypoints between two images using BRISK.

        Args:
            s_img (np.ndarray): Source image.
            t_img (np.ndarray): Target image.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Matched keypoints from source and target images.
        """
        # Convert images to grayscale if necessary
        s_img_gray = cv2.cvtColor(s_img, cv2.COLOR_BGR2GRAY) if s_img.ndim == 3 else s_img
        t_img_gray = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY) if t_img.ndim == 3 else t_img

        # Initialize BRISK detector
        brisk = cv2.BRISK_create()
        # Find the keypoints and descriptors with BRISK
        kp1, des1 = brisk.detectAndCompute(s_img_gray, None)
        kp2, des2 = brisk.detectAndCompute(t_img_gray, None)

        # Check if descriptors are None
        if des1 is None or des2 is None:
            return np.array([]), np.array([])

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.threshold * n.distance:
                good_matches.append(m)

        # Check if there are any good matches
        if len(good_matches) == 0:
            return np.array([]), np.array([])

        # Extract location of good matches
        points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        return points1, points2


    def matched_points_with_superglue(self, s_img, t_img):
        # Convert images to grayscale and to tensors
        s_img_gray = cv2.cvtColor(s_img, cv2.COLOR_BGR2GRAY) if s_img.ndim == 3 else s_img
        t_img_gray = cv2.cvtColor(t_img, cv2.COLOR_BGR2GRAY) if t_img.ndim == 3 else t_img
        frame_tensor1 = frame2tensor(s_img_gray, self.device)
        frame_tensor2 = frame2tensor(t_img_gray, self.device)

        with torch.no_grad():
            pred = self.matcher({'image0': frame_tensor1, 'image1': frame_tensor2})

        kpts0 = pred['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        conf = confidence[valid]

        # Apply confidence threshold
        mask = conf > self.threshold
        kp1 = mkpts0[mask]
        kp2 = mkpts1[mask]

        return kp1, kp2

    def matched_points_with_lightglue(self, s_img, t_img):
        # Convert images to tensors
        img0_tensor = numpy_image_to_torch(s_img).to(self.device)
        img1_tensor = numpy_image_to_torch(t_img).to(self.device)

        with torch.no_grad():
            feats0 = self.extractor.extract(img0_tensor)
            feats1 = self.extractor.extract(img1_tensor)
            matches01 = self.matcher({"image0": feats0, "image1": feats1})

        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

        matches = matches01["matches"]
        if matches.shape[0] == 0:
            return np.array([]), np.array([])

        kpts0 = feats0["keypoints"][matches[:, 0]].cpu().numpy()
        kpts1 = feats1["keypoints"][matches[:, 1]].cpu().numpy()

        return kpts0, kpts1

    def matched_points_with_xfeat(self, s_img, t_img):
        # Implement your XFeat matching logic here
        # Since XFeat is custom, ensure it returns NumPy arrays
        raise NotImplementedError("XFeat matching is not implemented.")

    # Additional methods like visualization can be added if needed
