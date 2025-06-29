import open3d as o3d
import numpy as np
import os
import copy
import json

class AdvancedRegistration:
    def __init__(self, config, id=0):
        self.config = config
        self.pc_initial_source = None
        self.pc_global_source = None
        self.pc_fine_source = None
        self.pc_reference = None
        self.initial_rmse = -1
        self.global_rmse = -1
        self.fine_rmse = -1
        self.global_pose = np.eye(4)
        self.fine_pose = np.eye(4)
        self.run_id = id
        self.voxel_size = 0

    def load_point_clouds(self, source_path, target_path):
        """Load point clouds from the specified files."""
        self.pc_initial_source = o3d.io.read_point_cloud(source_path)
        self.pc_reference = o3d.io.read_point_cloud(target_path)

    def main_registration_process(self, source_files, target_files):
        """Main process for registering point clouds."""
        if len(source_files) != len(target_files):
            print("Error: The number of source and target files must be the same.")
            return

        results = []
        for i, (source_file, target_file) in enumerate(zip(source_files, target_files)):
            print(f"Processing pair {i+1}/{len(source_files)}: {source_file} & {target_file}")
            self.load_point_clouds(source_file, target_file)
            result = self.execute_registration(self.pc_initial_source, self.pc_reference, i)
            results.append(result)
        return results

    def set_pc_source(self, pc_initial_source):
        """Set the initial source point cloud."""
        self.pc_initial_source = pc_initial_source
    
    def set_pc_reference(self, pc_reference):
        """Set the initial reference point cloud."""
        self.pc_reference = pc_reference

    def execute_registration(self, pc_initial_source, pc_reference, it):
        self.set_pc_source(pc_initial_source)
        self.set_pc_reference(pc_reference)
        """Executes the registration process and logs the results."""
        if self.pc_initial_source is None or self.pc_reference is None:
            print("Initial source or reference point cloud not set.")
            return

        self.initial_rmse = self.calculate_rmse(self.pc_initial_source, self.pc_reference)
        # Reset or update pc_global_source before global registration
        self.pc_global_source = None  # or self.pc_global_source = copy.deepcopy(self.pc_initial_source)

        # Perform global registration
        #print("Starting global registration.")
        #print(f"Global registration method from config: {self.config['registration'].get('global')}")
        #print(f"Initial source point cloud is None: {self.pc_initial_source is None}")
        #print(f"Reference point cloud is None: {self.pc_reference is None}")
        #print(f"Before global registration: Source {len(self.pc_initial_source.points)} points, Target {len(self.pc_reference.points)} points")
        self.global_registration()
        #print(f"After global registration: Global Source {len(self.pc_global_source.points if self.pc_global_source else [])} points")
        #print("Global registration completed.")
        #print(f"Global source point cloud is None: {self.pc_global_source is None}")
        #print(f"Global RMSE: {self.global_rmse}")
        #print(f"Global pose: \n{self.global_pose}")

        if self.pc_global_source is None:
            self.pc_global_source = self.pc_initial_source

        # Perform fine registration
        #print("Starting fine registration.")
        #print(f"Fine registration method from config: {self.config['registration'].get('fine')}")
        #print(f"Global source point cloud for fine registration is None: {self.pc_global_source is None}")
        #print(f"Reference point cloud for fine registration is None: {self.pc_reference is None}")
        #print(f"Before fine registration: Global Source {len(self.pc_global_source.points)} points, Target {len(self.pc_reference.points)} points")
        self.fine_registration()
        #print(f"After fine registration: Fine Source {len(self.pc_fine_source.points)} points")
        #print(f"Fine registration completed with RMSE: {self.fine_rmse}")

        # Log the results
        self.log_results(it)

        # Save the merged point cloud with colored segments
        self.save_merged_point_cloud(it)
        print("Merged and colored point cloud saved.")
        
        if self.fine_pose is not None:
            transformation = self.fine_pose
        else:
            transformation = self.global_pose
        if self.fine_rmse is not None:
            rmse = self.fine_rmse
        else:
            rmse = self.global_rmse
        return transformation, rmse


    def global_registration(self):
        """Dynamically selects and performs the global registration method based on the configuration."""
        # Map configuration strings to registration method calls
        registration_methods = {
            'FPFH': self.global_registration_fpfh,
            'RSCS': self.global_registration_rscs,
            'SVD': self.global_registration_svd
        }

        alignment_option = self.config['registration'].get('global')

        if alignment_option in registration_methods:
            print(f"Performing {alignment_option} registration.")
            global_result = registration_methods[alignment_option]()
            
            # Some methods might not directly update self.initial_pose, so we check and update if needed
            if hasattr(global_result, 'transformation'):
                self.initial_pose = global_result.transformation
        else:
            print(f"Registration method '{alignment_option}' not recognized. No global registration performed.")

    def global_registration_fpfh(self):
        self.pc_initial_source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30))
        self.pc_reference.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30))
        
        optimal_voxel_size1 = self.estimate_optimal_voxel_size(self.pc_initial_source)
        optimal_voxel_size2 = self.estimate_optimal_voxel_size(self.pc_reference)

        self.voxel_size = np.mean([optimal_voxel_size1, optimal_voxel_size2])
        # Downsample point clouds to speed up computation
        pc1_down = self.pc_initial_source.voxel_down_sample(voxel_size=self.voxel_size)
        pc2_down = self.pc_reference.voxel_down_sample(voxel_size=self.voxel_size)

        # Compute FPFH features for both downsampled point clouds
        radius_feature = self.voxel_size * 3
        fpfh_pc1 = o3d.pipelines.registration.compute_fpfh_feature(pc1_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=30))
        fpfh_pc2 = o3d.pipelines.registration.compute_fpfh_feature(pc2_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=30))

        # Set RANSAC registration parameters
        distance_threshold = self.voxel_size * 1.5
        ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pc1_down, pc2_down, fpfh_pc1, fpfh_pc2, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 500)
        )

        # Update the global_pose with the obtained transformation
        self.global_pose = ransac_result.transformation

        # Transform the original source point cloud using the updated global_pose
        self.pc_global_source = copy.deepcopy(self.pc_initial_source).transform(self.global_pose)

        # Calculate the RMSE based on the transformed source and original target point clouds
        self.global_rmse = self.calculate_rmse(self.pc_global_source, self.pc_reference)

        print("Updated global pose from FPFH registration: \n", self.global_pose)
        print("Global RMSE: ", self.global_rmse)

        return ransac_result

    
    def estimate_optimal_voxel_size(self, point_cloud, k=1):

        # Convert Open3D PointCloud to numpy array
        points = np.asarray(point_cloud.points)
        
        # Build a KDTree for efficient nearest neighbor search
        kdtree = o3d.geometry.KDTreeFlann(point_cloud)
        
        distances = []
        for i in range(len(points)):
            # For each point, find its nearest neighbor
            _, idx, dist = kdtree.search_knn_vector_3d(point_cloud.points[i], k + 1)
            
            # Exclude the point itself and take the distance to the nearest neighbor
            distances.append(np.sqrt(dist[1]))  # dist[0] is the distance to itself, which is 0

        # Use the median of these distances as the optimal voxel size
        optimal_voxel_size = np.median(distances)

        return optimal_voxel_size

    def global_registration_rscs(self):
        pass

    def global_registration_svd(self):
        # Ensure the point clouds are converted to numpy arrays
        A = np.asarray(self.pc_initial_source.points)
        B = np.asarray(self.pc_reference.points)

        # Calculate centroids and centered arrays
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # Compute the matrix H
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)

        # Compute rotation
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)

        # Compute translation
        t = centroid_B - np.dot(R, centroid_A)

        # Update the global_pose of the class
        self.global_pose = np.eye(4)
        self.global_pose[:3, :3] = R
        self.global_pose[:3, 3] = t

        # Transform the initial source point cloud using the computed global pose
        self.pc_global_source = copy.deepcopy(self.pc_initial_source).transform(self.global_pose)

        # Calculate the RMSE based on the transformed source and reference point clouds
        self.global_rmse = self.calculate_rmse(self.pc_global_source, self.pc_reference)

    def fine_registration(self):
        """Dynamically selects and performs the fine registration method based on the configuration."""
        # Map configuration strings to ICP registration method calls
        icp_methods = {
            'generalized': self.fine_registration_generalized,
            'point_to_point': self.fine_registration_point_to_point,
            'point_to_plane': self.fine_registration_point_to_plane,
            'colored': self.fine_registration_colored  # Assuming you have a method for colored ICP
        }

        icp_option = self.config['registration'].get('fine')

        if icp_option in icp_methods:
            print(f"Performing {icp_option} ICP fine registration.")
            self.icp_result = icp_methods[icp_option]()
            # Apply the fine registration transformation to the global source point cloud
            self.fine_pose = self.icp_result.transformation
            self.pc_fine_source = copy.deepcopy(self.pc_global_source).transform(self.fine_pose)
            # Calculate and update the fine RMSE using the transformed fine source and reference point clouds
            self.fine_rmse = self.calculate_rmse(self.pc_fine_source, self.pc_reference)
        else:
            print(f"ICP method '{icp_option}' not recognized. No fine registration performed.")

    def fine_registration_colored(self):
        pass

    def fine_registration_generalized(self):
        icp_result = o3d.pipelines.registration.registration_generalized_icp(
            self.pc_global_source, self.pc_reference, self.config['registration']['distance_threshold'], self.global_pose,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.config['registration']['max_iteration']
                # You can add relative_fitness and relative_rmse if needed
            )
        )
        return icp_result

    def fine_registration_point_to_point(self):
        print("Performing point_to_point ICP fine registration.")
        print(f"Source point cloud (global source) for ICP has {len(self.pc_global_source.points)} points")
        print(f"Target point cloud (reference) for ICP has {len(self.pc_reference.points)} points")
        print(f"Initial transformation for ICP: \n{self.global_pose}")

        icp_result = o3d.pipelines.registration.registration_icp(
            self.pc_global_source, self.pc_reference, self.config['registration']['distance_threshold'], self.global_pose,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.config['registration']['max_iteration'])
        )

        self.pc_fine_source = copy.deepcopy(self.pc_global_source).transform(icp_result.transformation)
        print(f"Fine source point cloud after ICP has {len(self.pc_fine_source.points)} points")

        return icp_result


    def fine_registration_point_to_plane(self):
        # Compute normals for the reference point cloud
        target_with_normals = self.compute_normals(self.pc_reference)
        icp_result = o3d.pipelines.registration.registration_icp(
            self.pc_global_source, target_with_normals, self.config['registration']['distance_threshold'], self.global_pose,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.config['registration']['max_iteration'])
        )
        return icp_result

    def compute_normals(self, point_cloud, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)): # search_param could be modified and its params could be tuned

        point_cloud.estimate_normals(search_param=search_param)
        #point_cloud.orient_normals_consistent_tangent_plane(50)  # Optional, for better normal orientation
        return point_cloud
    
    def calculate_rmse(self, source, target):
        """Calculate the Root Mean Square Error (RMSE) between the source and target point clouds."""
        # This implementation assumes that source and target are aligned and have the same number of points
        differences = np.asarray(source.points) - np.asarray(target.points)
        squared_differences = np.sum(differences ** 2, axis=1)
        mean_squared_difference = np.mean(squared_differences)
        rmse = np.sqrt(mean_squared_difference)
        return rmse

    def log_results(self, it):
        """Log the registration results including point clouds and RMSE values."""
        log_data = {
            'initial_rmse': self.initial_rmse,
            'global_rmse': self.global_rmse,
            'fine_rmse': self.fine_rmse,
        }
        log_path = os.path.join(self.config['log_error_dir'], f'{self.run_id:04d}_registration_rmse{it:04}.json')
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=4)
        print(f'Results logged in {log_path}')

    def save_point_cloud(self, point_cloud, filename):
        """Save a point cloud to a file."""
        if not os.path.exists(self.config['output_pc_dir']):
            os.makedirs(self.config['output_pc_dir'])
        file_path = os.path.join(self.config['output_pc_dir'], filename)
        o3d.io.write_point_cloud(file_path, point_cloud)
        print(f'Point cloud saved to {file_path}')

    def merge_and_color_point_clouds(self):
        # Define colors for each point cloud
        colors = {
            'initial_source': [1, 0, 0],  # Red
            'global_source': [1, 1, 0],   # Yellow
            'fine_source': [0, 0, 1],     # Blue
            'reference': [0, 1, 0]        # Green
        }

        merged_pc = o3d.geometry.PointCloud()

        for pc, color in zip([self.pc_initial_source, self.pc_global_source, self.pc_fine_source, self.pc_reference], colors.values()):
            if pc is not None:
                colored_pc = copy.deepcopy(pc)
                colored_pc.paint_uniform_color(color)
                merged_pc += colored_pc

        return merged_pc
    
    def save_merged_point_cloud(self, it):
        # Use the merging and coloring utility to prepare the point cloud
        merged_pc = self.merge_and_color_point_clouds()

        # Determine the output directory and ensure it exists
        output_pc_dir = self.config.get('output_pc_dir', '.')
        if not os.path.exists(output_pc_dir):
            os.makedirs(output_pc_dir)

        # Construct the full path for the output file
        filename= f'{self.run_id}_merged_registration_result_{it:04d}.ply'
        file_path = os.path.join(output_pc_dir, filename)

        # Save the point cloud to the specified file
        o3d.io.write_point_cloud(file_path, merged_pc)
        print(f"Merged point cloud saved to: {file_path}")
