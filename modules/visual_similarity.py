import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

class VisualSimilarity:
    def __init__(self, device='cuda'):
        # Load pre-trained ResNet50 model and remove the classification layer
        self.model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.threshold = 0.95
        
        # Define preprocessing transformations
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def set_threshold(self, threshold):
        self.threshold = threshold

    def preprocess_image(self, image_np):
        """
        Preprocesses a NumPy image array for feature extraction.
        
        Args:
            image_np (numpy.ndarray): Input image as a NumPy array.
        
        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        # Convert NumPy array to PIL Image
        image = Image.fromarray(image_np.astype('uint8'), 'RGB')
        return self.preprocess(image).unsqueeze(0).to(self.device) # watch this

    def extract_features(self, image_tensor):
        """
        Extracts features from a preprocessed image tensor using the ResNet50 model.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor.
        
        Returns:
            torch.Tensor: Extracted feature tensor.
        """
        with torch.no_grad():
            features = self.model(image_tensor)
        return features.squeeze()

    def compute_similarity(self, features1, features2):
        """
        Computes cosine similarity between two feature vectors.
        
        Args:
            features1 (torch.Tensor): Feature tensor of the first image.
            features2 (torch.Tensor): Feature tensor of the second image.
        
        Returns:
            float: Cosine similarity value.
        """
        return F.cosine_similarity(features1.unsqueeze(0), features2.unsqueeze(0)).item()

    def compute_image_similarity(self, image_np1, image_np2, depth_np1=None, depth_np2=None):
        """
        Computes visual similarity between two images represented as NumPy arrays.
        
        Args:
            image_np1 (numpy.ndarray): First input image as a NumPy array.
            image_np2 (numpy.ndarray): Second input image as a NumPy array.
        
        Returns:
            float: Cosine similarity value between the two images.
        """
        preprocessed_image1 = self.preprocess_image(image_np1)
        preprocessed_image2 = self.preprocess_image(image_np2)
        features1 = self.extract_features(preprocessed_image1)
        features2 = self.extract_features(preprocessed_image2)
        return self.compute_similarity(features1, features2)


    def select_key_images(self, images_np, maps_np=None):
        """
        Select key images from a sequence of images based on visual similarity.

        Args:
            images_np (np.ndarray): Array of images with shape (N, H, W, C).
            threshold (float): Similarity threshold to select key images.

        Returns:
            List[int]: Indices of selected key images.
        """
        num_images = images_np.shape[0]
        key_indices = [0]  # Always select the first image
        current_key_index = 0

        # Preprocess and extract features for the first image
        current_key_image = images_np[current_key_index]
        current_key_features = self.extract_features(self.preprocess_image(current_key_image))

        for i in range(1, num_images):
            image = images_np[i]
            image_features = self.extract_features(self.preprocess_image(image))
            similarity = self.compute_similarity(current_key_features, image_features)
            if similarity < self.threshold and (i - 1) != key_indices[-1]:
                key_indices.append(i - 1)
                current_key_index = i - 1
                current_key_image = images_np[current_key_index]
                current_key_features = self.extract_features(self.preprocess_image(current_key_image))

        # Add the last image to the list of keys if not already added
        if key_indices[-1] != num_images - 1:
            key_indices.append(num_images - 1)

        return key_indices


# Example usage
# similarity_computer = VisualSimilarity()
# similarity = similarity_computer.compute_image_similarity(source_color, target_color)
