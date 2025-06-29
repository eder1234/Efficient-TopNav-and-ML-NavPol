import os
import numpy as np
import cv2
import glob
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from modules.scenes import Scene  # Assuming this is a user-defined module
import gc
from torch.cuda.amp import autocast
from modules.rgbd_similarity import RGBDSimilarity
from modules.visual_similarity import VisualSimilarity

# Set environment variable to handle memory fragmentation (optional)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def process_triplet(vp_id, vm_id, vm_id_output, root_path, visual_similarity, feature_alg='LightGlue', scene_pseudo='room', visual_threshold=0.95):
    """
    Process a triplet of vp_id, vm_id, and vm_id_output to select key images based on feature similarity.

    Args:
        vp_id (int): Viewpoint ID.
        vm_id (int): Virtual Map ID.
        vm_id_output (int): Output Virtual Map ID.
        root_path (str): Root directory path.
        visual_similarity (VisualSimilarity): Instance of VisualSimilarity class.
        feature_alg (str): Feature algorithm used.
        scene_pseudo (str): Scene pseudonym.
        visual_threshold (float): Similarity threshold for key image selection.
    """
    print(f"Processing vp_id: {vp_id}, vm_id: {vm_id}, vm_id_output: {vm_id_output}")

    # Construct paths
    images_path = os.path.join(root_path, 'manual_operation', 'all_frames', f'source_color_{vp_id}_{vm_id}_{feature_alg}.npy')
    maps_path = os.path.join(root_path, 'manual_operation', 'all_frames', f'source_depth_{vp_id}_{vm_id}_{feature_alg}.npy')
    poses_path = os.path.join(root_path, 'manual_operation', 'all_poses', f'agent_state_{vp_id}_{vm_id}_{feature_alg}.pkl')
    output_dir = os.path.join(root_path, 'selected_frames', f'path_{vm_id_output}')

    # Check if all input files exist
    if not os.path.exists(images_path):
        print(f"❌ Image file not found: {images_path}. Skipping this triplet.\n")
        return
    if not os.path.exists(maps_path):
        print(f"❌ Depth maps file not found: {maps_path}. Skipping this triplet.\n")
        return
    if not os.path.exists(poses_path):
        print(f"❌ Poses file not found: {poses_path}. Skipping this triplet.\n")
        return

    # Load images
    images = np.load(images_path)  # Expected shape: (N, H, W, C)
    maps = np.load(maps_path)
    print(f"📷 Loaded images with shape: {images.shape}")
    print(f"🗺️ Loaded depth maps with shape: {maps.shape}")

    # Select key images using VisualSimilarity class
    visual_similarity.set_threshold(visual_threshold)
    selected_indices = visual_similarity.select_key_images(images)

    print(f"✅ Selected {len(selected_indices)} key images out of {images.shape[0]} total images.")
    print(f"📌 Indices of selected key images: {selected_indices}")

    # Select images based on indices
    selected_images = images[selected_indices]

    # Load depth maps
    selected_maps = maps[selected_indices]
    print("🗺️ Selected corresponding depth maps.")

    # Load poses
    with open(poses_path, 'rb') as f:
        poses = pkl.load(f)
    selected_poses = [poses[i] for i in selected_indices]
    print("📍 Selected corresponding poses.")

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Ensured that output directory exists: {output_dir}")

    # Save selected indices as a CSV file
    selected_indices_df = pd.DataFrame({'selected_index': selected_indices})
    selected_indices_csv_path = os.path.join(output_dir, f'selected_indices_{vm_id_output}.csv')
    selected_indices_df.to_csv(selected_indices_csv_path, index=False)
    print(f"📄 Saved selected indices to: {selected_indices_csv_path}")

    # Optional: Save a mapping of selected indices to their order
    selected_order_df = pd.DataFrame({
        'order': list(range(1, len(selected_indices) + 1)),
        'selected_index': selected_indices
    })
    selected_order_csv_path = os.path.join(output_dir, f'selected_order_indices_{vm_id_output}.csv')
    selected_order_df.to_csv(selected_order_csv_path, index=False)
    print(f"📄 Saved selected indices with order to: {selected_order_csv_path}")

    # Prepare the metadata
    metadata = {
        'vp_id': vp_id,
        'vm_id': vm_id,
        'vm_id_output': vm_id_output,
        'feature_alg': feature_alg,
        'scene_pseudo': scene_pseudo,
        'visual_threshold': visual_threshold,
        'num_selected_images': len(selected_indices),
        'total_images': images.shape[0]
    }

    # Define the path for the metadata file
    metadata_file_path = os.path.join(output_dir, f'metadata_{vm_id_output}.txt')

    # Write the metadata to a text file
    with open(metadata_file_path, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    print(f"📝 Metadata has been written to: {metadata_file_path}")

    # Store the selected data
    np.save(os.path.join(output_dir, 'selected_rgbs.npy'), selected_images)
    np.save(os.path.join(output_dir, 'selected_depths.npy'), selected_maps)
    with open(os.path.join(output_dir, f'selected_poses_{vm_id_output}.pkl'), 'wb') as f:
        pkl.dump(selected_poses, f)
    print(f"💾 Selected images, depth maps, and poses have been saved successfully in {output_dir}.\n")

    # Clear GPU cache and collect garbage
    torch.cuda.empty_cache()
    gc.collect()

def main():
    OLD_VERSION = True
    # Define the root path
    root_path = '/home/rodriguez/Documents/GitHub/habitat/habitat-lab'

    # Define your list of [vp_id, vm_id, vm_id_output] triplets
    triplets = [
        [206, 104, 104],
    ]

    # Validate triplets
    for idx, triplet in enumerate(triplets):
        if not isinstance(triplet, (list, tuple)) or len(triplet) != 3:
            raise ValueError(f"Triplet at index {idx} is invalid: {triplet}. Each triplet must contain exactly three elements [vp_id, vm_id, vm_id_output].")

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create an instance of VisualSimilarity class
    if OLD_VERSION:
        rgbd_similarity = VisualSimilarity(device=device)
    else:
        rgbd_similarity = RGBDSimilarity(device=device, threshold=0.9)
    
    print("🔍 Initialized RGBDSimilarity instance for feature extraction.")

    # Process each triplet
    for triplet in triplets:
        vp_id, vm_id, vm_id_output = triplet
        process_triplet(vp_id, vm_id, vm_id_output, root_path, rgbd_similarity, visual_threshold=0.94)

if __name__ == "__main__":
    main()
