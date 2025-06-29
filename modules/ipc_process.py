import numpy as np
import open3d as o3d
from modules.fa3r import Vector3d, FA3R_int, FA3R_double, eig3D_eig
from dataclasses import dataclass
from modules.feature_matcher import FeatureMatcher

@dataclass
class Vector3d:
    x: float
    y: float
    z: float

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

class IPC_process:
    def __init__(self, o_cimg = None, o_dmap = None, k_cimg = None, k_dmap = None, kp_feature_algorithm='LightGlue', th_kp=0.5, registration_algorithm='SVD'):
        self.kp_feature_algorithm = kp_feature_algorithm
        self.o_cimg = o_cimg
        self.o_dmap = o_dmap
        self.k_cimg = k_cimg
        self.k_dmap = k_dmap
        self.registration_algorithm = registration_algorithm
        self.o_kp = None
        self.k_kp = None
        self.o_pc = None
        self.k_pc = None
        self.o_ipc = None
        self.k_ipc = None
        self.regis_error = None
        self.est_pose = None
        self.t_ipc = None
        self.th_regis_error = None
        self.kp_feature_matcher = FeatureMatcher(config={}, device='cuda')
        self.kp_feature_matcher.set_feature(kp_feature_algorithm)
        self.kp_feature_matcher.set_threshold(th_kp)

    def set_RGBD_data(self, o_cimg, o_dmap, k_cimg, k_dmap):
        self.o_cimg = o_cimg
        self.o_dmap = o_dmap
        self.k_cimg = k_cimg
        self.k_dmap = k_dmap

    def set_observed_data(self, o_cimg, o_dmap):
        self.o_cimg = o_cimg
        self.o_dmap = o_dmap

    def set_key_data(self, k_cimg, k_dmap):
        self.k_cimg = k_cimg
        self.k_dmap = k_dmap

    def compute_keypoints(self):
        o_kp, k_kp = self.kp_feature_matcher.compute_matches(self.o_cimg, self.k_cimg)
        self.o_kp, self.k_kp = o_kp, k_kp

    def set_kp_feature_algorithm(self, kp_feature_algorithm):
        self.kp_feature_matcher.set_feature(kp_feature_algorithm)

    def set_registration_algorithm(self, registration_algorithm):
        self.registration_algorithm = registration_algorithm

    def set_registration_error_threshold(self, th_regis_error):
        self.th_regis_error = th_regis_error

    def get_kp_feature_algorithm(self):
        return self.kp_feature_algorithm
    
    def get_observed_data(self):
        return self.o_cimg, self.o_dmap
    
    def get_key_data(self):
        return self.k_cimg, self.k_dmap
    
    def get_RGBD_data(self):
        return self.o_cimg, self.o_dmap, self.k_cimg, self.k_dmap

    def generate_point_clouds(self):
        o_pc = self.generate_pc_in_cam_ref_frame(self.o_dmap)
        k_pc = self.generate_pc_in_cam_ref_frame(self.k_dmap)
        self.o_pc, self.k_pc = o_pc, k_pc
        return o_pc, k_pc
    
    def generate_ipc(self):
        o_ipc = self.get_ipc_from_pc(self.o_pc, self.o_kp)
        k_ipc = self.get_ipc_from_pc(self.k_pc, self.k_kp)
        self.o_ipc, self.k_ipc = o_ipc, k_ipc
        return o_ipc, k_ipc
    
    def generate_pc_in_cam_ref_frame(self, depth_img, T_cam_world=None):
        #W = H = 256 # fixed for simplifity, but depth.shape could be used
        W, H, _ = depth_img.shape
        hfov = np.pi / 2
        K = np.array([
            [1 / np.tan(hfov / 2.), 0., 0., 0.],
            [0., 1 / np.tan(hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        
        # Now get an approximation for the true world coordinates -- see if they make sense
        # [-1, 1] for x and [1, -1] for y as array indexing is y-down while world is y-up
        xs, ys = np.meshgrid(np.linspace(-1,1,W), np.linspace(1,-1,W))
        depth = depth_img.reshape(1,W,W)
        xs = xs.reshape(1,W,W)
        ys = ys.reshape(1,W,W)

        # Unproject
        # negate depth as the camera looks along -Z
        xys = np.vstack((xs * depth , ys * depth, -depth, np.ones(depth.shape)))
        xys = xys.reshape(4, -1)
        pc_cam_h = np.matmul(np.linalg.inv(K), xys)
        return pc_cam_h
    
    def get_ipc_from_pc(self, pc_cam, kp_cam):
        W = 256
        cam_key_id = [int(kp[1]*W+ kp[0]) for kp in kp_cam]
        ipc_cam_h = pc_cam[:, cam_key_id]
        ipc_cam = ipc_cam_h[:3].T
        return ipc_cam
    
    def transform_point_cloud(self, point_cloud, transformation_matrix):
        # Add a ones column for homogeneous coordinates
        homogeneous_points = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
        # Apply the transformation
        return np.dot(homogeneous_points, transformation_matrix.T)[:, :3]    

    def IPC_registration(self):
        regis_error = None
        t_ipc = None
        est_pose = None
        if self.registration_algorithm == "SVD":
            regis_error, t_ipc, est_pose = self.execute_SVD_registration()
        if self.registration_algorithm == "ICP":
            regis_error, t_ipc, est_pose = self.execute_ICP_registration()
        if self.registration_algorithm == "FA3R_int" or self.registration_algorithm == "FA3R_double":
            regis_error, t_ipc, est_pose = self.execute_FA3R_registration()
        if self.registration_algorithm == "eig3D_eig":
            regis_error, t_ipc, est_pose = self.execute_eig3D_eig_registration()
        self.regis_error, self.t_ipc, self.est_pose = regis_error, t_ipc, est_pose
        return regis_error, t_ipc, est_pose
    
    def execute_ICP_registration(self):
        # Convert NumPy arrays to Open3D point clouds
        source_cloud = o3d.geometry.PointCloud()
        source_cloud.points = o3d.utility.Vector3dVector(self.o_pc)
        
        target_cloud = o3d.geometry.PointCloud()
        target_cloud.points = o3d.utility.Vector3dVector(self.k_pc)
        
        # Estimate normals
        source_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Perform ICP registration
        threshold = 0.02
        trans_init = np.identity(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_cloud, target_cloud, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        )
        
        # Get the transformation matrix (relative pose)
        est_pose = reg_p2p.transformation
        
        # Transform the source point cloud
        transformed_cloud = source_cloud.transform(est_pose)
        
        # Compute registration error
        regis_error = reg_p2p.inlier_rmse
        
        # Convert transformed point cloud back to NumPy array
        t_ipc = np.asarray(transformed_cloud.points)
        
        return regis_error, t_ipc, est_pose
    
    def compute_ICP_registration(self, o_pc, k_pc):
        # Convert NumPy arrays to Open3D point clouds
        source_cloud = o3d.geometry.PointCloud()
        source_cloud.points = o3d.utility.Vector3dVector(o_pc)
        
        target_cloud = o3d.geometry.PointCloud()
        target_cloud.points = o3d.utility.Vector3dVector(k_pc)
        
        # Preprocess point clouds
        voxel_size = 0.05  # Adjust based on your point cloud density
        source_cloud = source_cloud.voxel_down_sample(voxel_size)
        target_cloud = target_cloud.voxel_down_sample(voxel_size)
        
        source_cloud, _ = source_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        target_cloud, _ = target_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Estimate normals
        source_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Perform global registration
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source_cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target_cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
        
        # Updated RANSAC registration
        distance_threshold = 0.05
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_cloud, target_cloud, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,  # RANSAC n
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )
        
        # Perform ICP registration
        threshold = 0.05  # Adjust based on your point cloud scale
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_cloud, target_cloud, threshold, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100, relative_fitness=1e-6, relative_rmse=1e-6)
        )
        
        # Get the transformation matrix (relative pose)
        est_pose = reg_p2p.transformation
        
        # Transform the source point cloud
        transformed_cloud = source_cloud.transform(est_pose)
        
        # Compute registration error
        regis_error = reg_p2p.inlier_rmse
        
        # Convert transformed point cloud back to NumPy array
        t_ipc = np.asarray(transformed_cloud.points)
        
        return regis_error, t_ipc, est_pose

    def compute_simple_ICP_registration(self, o_pc, k_pc):
        # Convert NumPy arrays to Open3D point clouds
        source_cloud = o3d.geometry.PointCloud()
        source_cloud.points = o3d.utility.Vector3dVector(o_pc)
        
        target_cloud = o3d.geometry.PointCloud()
        target_cloud.points = o3d.utility.Vector3dVector(k_pc)
        
        # Preprocess point clouds
        voxel_size = 0.05  # Adjust based on your point cloud density
        source_cloud = source_cloud.voxel_down_sample(voxel_size)
        target_cloud = target_cloud.voxel_down_sample(voxel_size)
        
        source_cloud, _ = source_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        target_cloud, _ = target_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Estimate normals
        source_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Perform global registration
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source_cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target_cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
        
        # Updated RANSAC registration
        distance_threshold = 0.05
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_cloud, target_cloud, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,  # RANSAC n
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
        )
        
        # Perform ICP registration
        threshold = 0.05  # Adjust based on your point cloud scale
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_cloud, target_cloud, threshold, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100, relative_fitness=1e-6, relative_rmse=1e-6)
        )
        
        # Get the transformation matrix (relative pose)
        est_pose = reg_p2p.transformation
        
        R = est_pose[:3, :3]
        t = est_pose[:3, 3]

        return R, t

    def execute_SVD_registration(self):
        # Compute centroids
        centroid_o = np.mean(self.o_pc, axis=0)
        centroid_k = np.mean(self.k_pc, axis=0)
        
        # Center the point clouds
        o_centered = self.o_pc - centroid_o
        k_centered = self.k_pc - centroid_k
        
        # Cross-covariance
        H = np.dot(o_centered.T, k_centered)
        
        # SVD
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        # Translation
        t = centroid_k - np.dot(R, centroid_o)
        
        # Homogeneous transformation matrix
        T = np.identity(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        # Transform the o point cloud
        transformed_o_pc = self.transform_point_cloud(self.o_pc, T)
        
        # RMSE calculation
        distances = np.linalg.norm(transformed_o_pc - self.k_pc, axis=1)
        rmse = np.sqrt(np.mean(distances**2))
        
        return rmse, transformed_o_pc, T

    def transform_point_cloud(self, point_cloud, T):
        # Homogeneous coordinates
        homogeneous_pc = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
        
        # Apply transformation
        transformed_pc = np.dot(homogeneous_pc, T.T)
        
        # Return to 3D coordinates
        return transformed_pc[:, :3]

    def numpy_to_vector3d(self, point_cloud: np.ndarray) -> list[Vector3d]:
        if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
            raise ValueError("Input point cloud must be a 2D array with shape (n, 3)")
        
        return [Vector3d(x=point[0], y=point[1], z=point[2]) for point in point_cloud]

    def apply_transformation(self, points, R, t):
        return [Vector3d(*(R @ np.array([p.x, p.y, p.z]) + t)) for p in points]

    def vector3d_to_numpy(self, points):
        return np.array([[p.x, p.y, p.z] for p in points])

    def execute_FA3R_registration(self):
        s_ipc_as_vector3d = self.numpy_to_vector3d(self.o_ipc)
        k_ipc_as_vector3d = self.numpy_to_vector3d(self.k_ipc)
        if self.registration_algorithm == "FA3R_int":
            R, t = FA3R_int(s_ipc_as_vector3d, k_ipc_as_vector3d, None, 20)
        if self.registration_algorithm == "FA3R_double":
            R, t = FA3R_double(s_ipc_as_vector3d, k_ipc_as_vector3d, None, 20)
        Q_est = self.apply_transformation(s_ipc_as_vector3d, R, t)
        t_ipc = self.vector3d_to_numpy(Q_est)
        est_pose = np.zeros((4, 4))
        est_pose[:3, :3] = R
        est_pose[:3, 3] = t
        regis_error = self.compute_registration_error(self.o_ipc, t_ipc)
        self.regis_error, self.t_ipc, self.est_pose = regis_error, t_ipc, est_pose
        return regis_error, t_ipc, est_pose

    def compute_registration_error(self, s_ipc, t_ipc, error_type="rmse"):
        if error_type == "rmse":
            return np.sqrt(np.mean((s_ipc - t_ipc) ** 2))
        elif error_type == "mae":
            return np.mean(np.abs(s_ipc - t_ipc))
        else:
            raise ValueError("Invalid error type. Choose 'rmse' or 'mae'.")
    
    def execute_eig3D_eig_registration(self):
        s_ipc_as_vector3d = self.numpy_to_vector3d(self.o_ipc)
        k_ipc_as_vector3d = self.numpy_to_vector3d(self.k_ipc)
        R, t = eig3D_eig(s_ipc_as_vector3d, k_ipc_as_vector3d, None)
        Q_est = self.apply_transformation(s_ipc_as_vector3d, R, t)
        t_ipc = self.vector3d_to_numpy(Q_est)
        est_pose = np.zeros((4, 4))
        est_pose[:3, :3] = R
        est_pose[:3, 3] = t
        regis_error = self.compute_registration_error(self.o_ipc, t_ipc)
        self.regis_error, self.t_ipc, self.est_pose = regis_error, t_ipc, est_pose
        return regis_error, t_ipc, est_pose

    def compute_SVD_registration(self, source_pc, target_pc):
        # Compute centroids
        centroid_source = np.mean(source_pc, axis=0)
        centroid_target = np.mean(target_pc, axis=0)
        
        # Center the point clouds
        source_centered = source_pc - centroid_source
        target_centered = target_pc - centroid_target
        
        # Cross-covariance
        H = np.dot(source_centered.T, target_centered)
        
        # SVD
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        # Translation
        t = centroid_target - np.dot(R, centroid_source)
        
        # Homogeneous transformation matrix
        T = np.identity(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        # Transform the source point cloud
        transformed_source_pc = self.transform_point_cloud(source_pc, T)
        
        # RMSE calculation
        distances = np.linalg.norm(transformed_source_pc - target_pc, axis=1)
        rmse = np.sqrt(np.mean(distances**2))
        
        return rmse, transformed_source_pc, T
    
    def compute_simple_SVD_registration(self, source_pc, target_pc):
        # Compute centroids
        centroid_source = np.mean(source_pc, axis=0)
        centroid_target = np.mean(target_pc, axis=0)
        
        # Center the point clouds
        source_centered = source_pc - centroid_source
        target_centered = target_pc - centroid_target
        
        # Cross-covariance
        H = np.dot(source_centered.T, target_centered)
        
        # SVD
        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
        
        # Translation
        t = centroid_target - np.dot(R, centroid_source)
        
        # Homogeneous transformation matrix
        T = np.identity(4)
        T[:3, :3] = R
        T[:3, 3] = t

        return R, t

    def complete_ipc_process(self, kp_feature_algorithm, registration_algorithm, o_cimg, o_dmap, k_cimg, k_dmap):

        self.set_kp_feature_algorithm(kp_feature_algorithm)
        self.set_registration_algorithm(registration_algorithm)
        o_kp, k_kp = self.kp_feature_matcher.compute_matches(o_cimg, k_cimg)

        o_pc = self.generate_pc_in_cam_ref_frame(o_dmap)
        k_pc = self.generate_pc_in_cam_ref_frame(k_dmap)

        o_ipc = self.get_ipc_from_pc(o_pc, o_kp)
        k_ipc = self.get_ipc_from_pc(k_pc, k_kp)
        # others registration algorithms here
        print(f"Preparing registration using {self.registration_algorithm} algorithm")
        if registration_algorithm == 'SVD':
            print('SVD registration')
            rmse, t_ipc, est_pose = self.compute_SVD_registration(o_ipc, k_ipc)
            return  rmse, t_ipc, est_pose
        if registration_algorithm == 'ICP':
            print('ICP registration')
            rmse, t_ipc, est_pose = self.compute_ICP_registration(o_ipc, k_ipc)
            return  rmse, t_ipc, est_pose
        else:
            s_ipc_as_vector3d = self.numpy_to_vector3d(o_ipc)
            k_ipc_as_vector3d = self.numpy_to_vector3d(k_ipc)
            if registration_algorithm == "FA3R_int":
                print('FA3R_int registration')
                R, t = FA3R_int(s_ipc_as_vector3d, k_ipc_as_vector3d, None, 20)
            if registration_algorithm == "FA3R_double":
                print('FA3R_double registration')
                R, t = FA3R_double(s_ipc_as_vector3d, k_ipc_as_vector3d, None, 20)
            if registration_algorithm == "eig3D_eig":
                print('eig3D_eig registration')
                R, t = eig3D_eig(s_ipc_as_vector3d, k_ipc_as_vector3d, None)
            Q_est = self.apply_transformation(s_ipc_as_vector3d, R, t)
            t_ipc = self.vector3d_to_numpy(Q_est)
            est_pose = np.zeros((4, 4))
            est_pose[:3, :3] = R
            est_pose[:3, 3] = t
            rmse = self.compute_registration_error(o_ipc, t_ipc)

            return rmse, t_ipc, est_pose