import numpy as np
import pandas as pd
import heapq

class TopologicalMap:
    def __init__(self, id_vm, tm_path, vm_path, feature_registration, localization, th_rmse_nav, th_vis=0.8):
        self.id_vm = id_vm
        self.tm_path = tm_path
        self.vm_path = vm_path
        self.feature_registration = feature_registration  # Instance of FeatureBasedPointCloudRegistration
        self.localization = localization  # Passed-in instance of Localization class
        self.th_rmse_nav = th_rmse_nav
        self.th_vis = th_vis

        # Load the adjacency matrix
        self.adj_matrix = self.load_adj_matrix()

        # Load the topological map data frame
        self.data_df = self.load_topological_map()

    def load_adj_matrix(self):
        adj_matrix_path = self.tm_path + f'nav_matrix_{self.id_vm}.npy'
        try:
            adj_matrix = np.load(adj_matrix_path)
            return adj_matrix
        except FileNotFoundError:
            print(f"Adjacency matrix file not found at {adj_matrix_path}")
            return None

    def load_topological_map(self):
        data_df_path = self.tm_path + f'topological_map_{self.id_vm}.csv'
        try:
            data_df = pd.read_csv(data_df_path)
            return data_df
        except FileNotFoundError:
            print(f"Topological map file not found at {data_df_path}")
            return None

    def plan_path(self, source_color, source_depth, final_top_map_id):
        # Use the passed-in Localization instance to find the initial node
        init_top_map_id, _ = self.localization.localization(source_color, source_depth)

        # Compute the shortest path using Dijkstra's algorithm
        distance, path = self.dijkstra_algorithm(init_top_map_id, final_top_map_id)
        if distance == float('inf'):
            print(f"No path found from node {init_top_map_id} to node {final_top_map_id}")
            return []
        print(f"Planned path: {path}")
        return path

    def update_adjacency_matrix(self, o_node, c_node, k_node):
        # Adjust the adjacency matrix based on the robot's state
        self.adj_matrix[c_node][k_node] = 0
        self.adj_matrix[o_node][c_node] = 0

    def generate_visual_memory_from_visual_path(self, visual_path):
        target_colors = []
        target_depths = []
        for vp_id in visual_path:
            _, target_color, target_depth = self.localization.load_vm_arrays(vp_id)
            target_colors.append(target_color)
            target_depths.append(target_depth)
        return target_colors, target_depths

    def dijkstra_algorithm(self, start, end):
        num_nodes = len(self.adj_matrix)
        visited = [False] * num_nodes
        # Distance from start to each node, initialized to infinity
        dist = [float('inf')] * num_nodes
        dist[start] = 0
        # Priority queue to select the node with the smallest distance
        priority_queue = []
        heapq.heappush(priority_queue, (0, start))
        # To reconstruct the path
        predecessors = [-1] * num_nodes

        while priority_queue:
            current_dist, current_node = heapq.heappop(priority_queue)
            if visited[current_node]:
                continue

            if current_node == end:
                # Reconstruct the path using predecessors
                path = []
                step = end
                while step != -1:
                    path.append(step)
                    step = predecessors[step]
                return dist[end], path[::-1]  # Return distance and path

            visited[current_node] = True
            for neighbor in range(num_nodes):
                if self.adj_matrix[current_node][neighbor] > 0:  # there is a connection
                    distance = current_dist + self.adj_matrix[current_node][neighbor]
                    if distance < dist[neighbor]:
                        dist[neighbor] = distance
                        predecessors[neighbor] = current_node
                        heapq.heappush(priority_queue, (distance, neighbor))

        return float('inf'), []  # Return infinity and empty path if no path is found

    def select_targets(self, target_colors, target_depths, vm_index):
        # Select the target pair based on the index
        target_color = target_colors[vm_index]
        target_depth = target_depths[vm_index]
        return target_color, target_depth
    
    def load_vm_arrays(self, vm_image_index, vm_path):
        target_depths = np.load(vm_path + "selected_depths.npy")
        target_depth = target_depths[vm_image_index]
        target_colors = np.load(vm_path + "selected_rgbs.npy")
        target_color = target_colors[vm_image_index]
        vm_len = len(target_depths)
        return vm_len, target_color, target_depth
    
    def len_vm(self, vm_path):
        target_colors = np.load(vm_path + "selected_rgbs.npy")
        return len(target_colors)