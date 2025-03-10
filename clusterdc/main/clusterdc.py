import numpy as np
import pandas as pd
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import networkx as nx
import warnings
import copy
from typing import List, Tuple, Dict, Optional, Union, Any
from tqdm import tqdm
from .kde import KDE
from .data import Data
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

class ClusterDC:
    """
    Density-Contour Clustering with Automatic Peak Detection.
    
    This class implements a density-based clustering algorithm that uses kernel density
    estimation (KDE) and contour analysis to identify clusters in 2D data. The algorithm:
    1. Estimates density using KDE
    2. Creates density contours
    3. Analyzes contour hierarchies
    4. Identifies density peaks
    5. Determines cluster assignments based on density connectivity
    
    Key Features:
    - Multiple initialization methods (raw data or KDE object)
    - Automatic selection of number of clusters using gap analysis
    - Manual specification of desired number of clusters
    - Hierarchical density-based clustering
    - Visualization tools for clustering results
    - Support for different density estimation methods
    
    The algorithm is particularly effective for:
    - Identifying clusters of varying shapes and sizes
    - Handling noise in the data
    - Detecting natural hierarchies in cluster structure
    - Working with non-linear cluster boundaries
    """
    def __init__(
        self,
        data: Optional[Union[np.ndarray, pd.DataFrame, Tuple[np.ndarray, np.ndarray]]] = None,
        columns: Optional[List[str]] = None,
        kde_obj: Optional['KDE'] = None,
        levels: int = 50,
        min_point: int = 1,
        gap_order: Optional[Union[int, str, None]] = 1,
        n_clusters: Optional[int] = None,
        save_path: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        kde_method: str = 'scott',
        kde_kernel_types: Optional[List[str]] = None,
        kde_n_iter: int = 50,
        kde_k_limits: tuple = (1, 40)
    ):
        """
        Initialize the ClusterDC algorithm.

        Parameters:
            data: Input data in one of two formats:
                - np.ndarray: shape (n_samples, 2) for 2D data
                - Tuple[np.ndarray, np.ndarray]: x and y coordinates separately
            kde_obj: Pre-fitted KDE object (alternative to raw data)
            levels: Number of density contour levels
            min_point: Minimum points required for a valid cluster
            gap_order: Method to select number of clusters:
                - int: Gap order (1 for first gap, 2 for second, etc.)
                - 'max_clusters': Use all positive separability points
                - 'all': Return all possible clusterings
                - None: Use user-specified n_clusters
            n_clusters: Exact number of clusters (used when gap_order is None)
            save_path: Path to save visualizations
            xlabel: Label for x-axis in plots
            ylabel: Label for y-axis in plots
            kde_method: Method for KDE ('scott', 'silverman', 'local_bandwidth', 'global_bandwidth')
            kde_kernel_types: List of kernel functions for KDE
            kde_n_iter: Number of iterations for KDE optimization
            kde_k_limits: Range of k values for local bandwidth as percentage

        Returns:
            None

        Raises:
            ValueError: If neither data nor kde_obj is provided
            ValueError: If input validation fails
        """
        # Store original data
        self.original_data = data
        self.columns = columns
        self.levels = levels
        self.min_point = min_point
        self.gap_order = gap_order  # Changed from selection_method
        self.n_clusters = n_clusters
        self.save_path = save_path
        self.xlabel = xlabel if xlabel is not None else 'X'
        self.ylabel = ylabel if ylabel is not None else 'Y'
        self.cluster_assignments = None
        self._temp_max_flow = None
        self._temp_dens_list = None

        # Validate inputs
        if gap_order is None and n_clusters is None:
            raise ValueError("If gap_order is None, n_clusters must be specified")
        if gap_order is not None and n_clusters is not None:
            raise ValueError("Cannot specify both gap_order and n_clusters")
        if n_clusters is not None and n_clusters < 1:
            raise ValueError("n_clusters must be positive")
        if isinstance(gap_order, int) and gap_order < 1:
            raise ValueError("When using gap order (int), value must be positive")
        elif isinstance(gap_order, str) and gap_order not in ['max_clusters', 'all']:
            raise ValueError("When using string gap_order, must be 'max_clusters' or 'all'")

        if kde_obj is not None:
            self._initialize_from_kde(kde_obj)
        elif data is not None:
            self._initialize_from_data(data, kde_method, kde_kernel_types, kde_n_iter, kde_k_limits)
        else:
            raise ValueError("Either data or kde_obj must be provided")

    def _initialize_from_kde(self, kde_obj: 'KDE'):
        """
        Initialize using a fitted KDE object.

        Parameters:
            kde_obj: Pre-fitted KDE object containing density estimates

        Returns:
            None

        Raises:
            ValueError: If KDE object is not fitted or data is not 2D
        """
        if not hasattr(kde_obj, 'point_densities') or kde_obj.point_densities is None:
            raise ValueError("KDE object must be fitted before use")
    
        # Store the original data if available in the KDE object
        if hasattr(kde_obj, 'original_data'):
            self.original_data = kde_obj.original_data
                
        # Get the data points from KDE object
        if kde_obj.data.shape[1] != 2:
            raise ValueError("KDE data must be 2-dimensional")
                
        self.x_points = kde_obj.data[:, 0]
        self.y_points = kde_obj.data[:, 1]
        self.xy = kde_obj.data
        
        # Get the densities and grid from KDE object
        self.point_densities = kde_obj.get_point_densities()
        self.grid_densities = kde_obj.get_grid_densities()
        self.xx, self.yy, _ = kde_obj.calculate_grid_with_margin()

    def _initialize_from_data(
        self,
        data: Union[np.ndarray, pd.DataFrame, Tuple[np.ndarray, np.ndarray]],
        kde_method: str,
        kde_kernel_types: Optional[List[str]],
        kde_n_iter: int,
        kde_k_limits: tuple,
        columns: Optional[List[str]] = None
    ):
        """
        Initialize using raw data and KDE parameters.

        Parameters:
            data: Input data as array or tuple of arrays
            kde_method: Method for density estimation
            kde_kernel_types: List of kernel functions
            kde_n_iter: Number of optimization iterations
            kde_k_limits: Range for k nearest neighbors

        Returns:
            None

        Raises:
            ValueError: If data is not 2D
        """
        # Handle different input formats
        if isinstance(data, tuple):
            self.x_points, self.y_points = data
            self.xy = np.column_stack((self.x_points, self.y_points))
        elif isinstance(data, pd.DataFrame):
            # If no columns specified, use first two numeric columns
            if columns is None:
                numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) < 2:
                    raise ValueError("DataFrame must contain at least two numeric columns")
                columns = numeric_cols[:2].tolist()
                
            # Store column names
            self.columns = columns
            self.xy = data[columns].to_numpy()
            self.x_points = self.xy[:, 0]
            self.y_points = self.xy[:, 1]
        else:
            if data.shape[1] < 2:
                raise ValueError("Data must have at least 2 columns")
                
            # If no columns specified, use first two columns
            if columns is None:
                columns = [0, 1]
                
            # Extract data for selected columns
            self.xy = data[:, columns] if len(columns) == 2 else data[:, :2]
            self.x_points = self.xy[:, 0]
            self.y_points = self.xy[:, 1]

        # Calculate KDE
        kde = KDE(
            data=self.xy,
            k_limits=kde_k_limits,
            kernel_types=kde_kernel_types,
            n_iter=kde_n_iter
        )
        kde.fit(method=kde_method)
        
        # Get densities and grid
        self.grid_densities = kde.get_grid_densities()
        self.point_densities = kde.get_point_densities()
        self.xx, self.yy, _ = kde.calculate_grid_with_margin()

    def create_contours(self) -> Tuple[List, List, List]:
        """
        Create density contours from the grid density estimation.
        
        Uses matplotlib's contour function to create density level sets,
        which are then converted to polygons for further analysis.

        Returns:
            Tuple containing:
            - List of polygon coordinates (x, y pairs)
            - List of density values for each contour
            - List of contour level indices
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        cset = ax.contour(self.xx, self.yy, self.grid_densities, levels=self.levels)
        plt.close()
        
        polygons, density_values, density_contours = [], [], []
        for j, level in enumerate(cset.levels):
            for seg in cset.allsegs[j]:
                polygons.append([seg[:, 0], seg[:, 1]])
                density_values.append(level)
                density_contours.append(j)
        
        return polygons, density_values, density_contours

    def make_node_dic(
        self,
        polygons: List,
        density_values: List,
        density_contours: List
    ) -> Dict:
        """
        Create a hierarchical dictionary of contour nodes.
        
        Analyzes containment relationships between contours to build
        a tree structure representing the density hierarchy.
    
        Parameters:
            polygons: List of polygon coordinates
            density_values: List of density values for each contour
            density_contours: List of contour level indices
    
        Returns:
            Dictionary with hierarchical node structure where each node contains:
            - level: Contour level
            - density: Density value
            - childrens: List of child node indices
            - parent: List of parent node indices
            - token: List of assigned cluster tokens
        """
        dic_node = {
            ind: {
                'level': density_contours[ind],
                'density': density_values[ind],
                'childrens': [],
                'parent': [],
                'token': []
            } for ind in range(len(polygons))
        }
        
        for ind in range(len(polygons)):
            # Check if the polygon has valid coordinates
            if (len(polygons[ind][0]) > 0 and len(polygons[ind][1]) > 0):
                # Check if polygon is closed (first and last points match)
                if (polygons[ind][0][0] == polygons[ind][0][-1] and 
                    polygons[ind][1][0] == polygons[ind][1][-1]):
                    try:
                        poly = Polygon([
                            (polygons[ind][0][index], polygons[ind][1][index])
                            for index in range(len(polygons[ind][0]))
                        ])
    
                        for ind_sub in range(len(polygons)):
                            # Skip empty polygons or self-comparison
                            if ind_sub == ind or len(polygons[ind_sub][0]) == 0 or len(polygons[ind_sub][1]) == 0:
                                continue
                                
                            # Check if the sub-polygon is closed
                            if (polygons[ind_sub][0][0] == polygons[ind_sub][0][-1] and 
                                polygons[ind_sub][1][0] == polygons[ind_sub][1][-1]):
                                # Check level and density conditions
                                if (dic_node[ind_sub]['level'] == dic_node[ind]['level'] + 1 and 
                                    dic_node[ind_sub]['density'] > dic_node[ind]['density']):
                                    try:
                                        poly_sub = Polygon([
                                            (polygons[ind_sub][0][index], polygons[ind_sub][1][index])
                                            for index in range(len(polygons[ind_sub][0]))
                                        ])
                                        if poly.contains(poly_sub):
                                            dic_node[ind]['childrens'].append(ind_sub)
                                            dic_node[ind_sub]['parent'].append(ind)
                                    except Exception as e:
                                        print(f"Warning: Error creating sub-polygon {ind_sub}: {e}")
                                        continue
                    except Exception as e:
                        print(f"Warning: Error creating polygon {ind}: {e}")
                        continue
    
        # Check for maximum level
        max_level = max(node['level'] for node in dic_node.values())
        
        # Handle nodes at each level, from highest to lowest
        for level in range(max_level, 0, -1):
            list_node_level = [
                node for node in dic_node 
                if dic_node[node]['level'] == level
            ]
            for ind_node in list_node_level:
                # Check if the polygon exists and has valid coordinates
                if (ind_node in dic_node and 
                    ind_node < len(polygons) and 
                    len(polygons[ind_node][0]) > 0 and 
                    len(polygons[ind_node][1]) > 0):
                    try:
                        poly = Polygon([
                            (polygons[ind_node][0][index], polygons[ind_node][1][index])
                            for index in range(len(polygons[ind_node][0]))
                        ])
                        for ind_node_sub in list_node_level:
                            if ind_node != ind_node_sub and ind_node_sub in dic_node:
                                # Check if the sub-polygon has valid coordinates
                                if (ind_node_sub < len(polygons) and 
                                    len(polygons[ind_node_sub][0]) > 0 and 
                                    len(polygons[ind_node_sub][1]) > 0):
                                    try:
                                        poly_sub = Polygon([
                                            (polygons[ind_node_sub][0][index], polygons[ind_node_sub][1][index])
                                            for index in range(len(polygons[ind_node_sub][0]))
                                        ])
                                        if poly.contains(poly_sub) and ind_node_sub in dic_node:
                                            del dic_node[ind_node_sub]
                                    except Exception as e:
                                        print(f"Warning: Error checking polygon containment: {e}")
                                        continue
                    except Exception as e:
                        print(f"Warning: Error creating polygon {ind_node}: {e}")
                        continue
    
        # Set up root level connections
        children = [ind for ind in dic_node if dic_node[ind]['level'] == 1]
        for child in children:
            dic_node[child]['parent'] = ['*']
        dic_node['*'] = {
            'level': 0,
            'density': 0,
            'childrens': children,
            'parent': ['**'],
            'token': []
        }
        
        return dic_node

    def find_leaves(self, dic_node: Dict) -> List:
        """
        Find leaf nodes in the contour hierarchy.
        
        Leaf nodes represent local density maxima and are potential cluster centers.

        Parameters:
            dic_node: Dictionary of nodes in the contour hierarchy

        Returns:
            List of indices for leaf nodes (nodes with no children)
        """
        return [
            ind for ind in dic_node.keys()
            if ind != '*' and not dic_node[ind]['childrens']
        ]

    def check_node_unvalid(
        self,
        ind_node: str,
        polygons: List,
        min_point: int
    ) -> bool:
        """
        Check if a node is invalid based on minimum point criterion.
        
        A node is considered invalid if it contains fewer points than
        the specified minimum.

        Parameters:
            ind_node: Index of node to check
            polygons: List of polygon coordinates
            min_point: Minimum number of points required

        Returns:
            bool: True if node is invalid, False otherwise
        """
        if ind_node == '*':
            return False
            
        poly_current = Polygon([
            (polygons[ind_node][0][index], polygons[ind_node][1][index])
            for index in range(len(polygons[ind_node][0]))
        ])
        
        nb_point_inside = sum(
            1 for point in self.xy
            if Point(point[0], point[1]).within(poly_current)
        )
        
        return nb_point_inside < min_point

    def trim_unvalid_leaves(self, dic_node: Dict, polygons: List, min_point: int) -> Dict:
        """
        Remove invalid nodes from the hierarchy.

        Invalid nodes are those containing fewer points than the minimum threshold.

        Parameters:
            dic_node: Dictionary of nodes in contour hierarchy
            polygons: List of polygon coordinates
            min_point: Minimum number of points required for validity

        Returns:
            Dict: Updated node dictionary with invalid nodes removed

        Raises:
            ValueError: If dic_node is empty or if polygons list is empty
        """
        if not dic_node:
            raise ValueError("Empty node dictionary provided")
        if not polygons:
            raise ValueError("Empty polygons list provided")

        dic_node_copy = copy.deepcopy(dic_node)

        # Identify levels present in the node dictionary
        try:
            levels = sorted(set(
                node.get('level', -1) 
                for node in dic_node_copy.values() 
                if isinstance(node, dict)
            ), reverse=True)
        except Exception as e:
            raise ValueError(f"Error identifying hierarchy levels: {str(e)}")

        for level in levels:
            # Find nodes at this specific level
            nodes_at_level = [
                node for node, node_data in dic_node_copy.items() 
                if isinstance(node_data, dict) and node_data.get('level') == level
            ]

            for node in nodes_at_level:
                try:
                    # Skip special nodes
                    if node in ['*', '**']:
                        continue

                    # Safely get the parent
                    parent_list = dic_node_copy[node].get('parent', ['*'])
                    parent = parent_list[0] if parent_list else '*'

                    # Check if the node is invalid
                    if self.check_node_unvalid(node, polygons, min_point):
                        # Update parent's children list
                        if parent in dic_node_copy and isinstance(dic_node_copy[parent], dict):
                            children = dic_node_copy[parent].get('childrens', [])
                            dic_node_copy[parent]['childrens'] = [
                                child for child in children if child != node
                            ]

                        # Remove the invalid node
                        dic_node_copy.pop(node, None)

                except Exception as e:
                    print(f"Warning: Error processing node {node}: {str(e)}")
                    continue

        return dic_node_copy

    def trim_chain_nodes(self, dic_node: Dict) -> Dict:
        """
        Remove chain nodes from the hierarchy.

        Chain nodes are those with exactly one child. These are removed to
        simplify the hierarchy while preserving the cluster structure.

        Parameters:
            dic_node: Dictionary of nodes in contour hierarchy

        Returns:
            Dict: Updated node dictionary with chain nodes removed

        Raises:
            ValueError: If dic_node is empty
        """
        if not dic_node:
            raise ValueError("Empty node dictionary provided")

        dic_node_copy = copy.deepcopy(dic_node)

        try:
            # Identify levels present in the node dictionary
            levels = sorted(set(
                node.get('level', -1) 
                for node in dic_node_copy.values() 
                if isinstance(node, dict)
            ), reverse=True)
        except Exception as e:
            raise ValueError(f"Error identifying hierarchy levels: {str(e)}")

        for level in levels:
            # Find nodes at this specific level
            nodes_at_level = [
                node for node, node_data in dic_node_copy.items() 
                if isinstance(node_data, dict) and node_data.get('level') == level
            ]

            for node in nodes_at_level:
                try:
                    # Skip special nodes
                    if node in ['*', '**']:
                        continue

                    # Safely get the parent
                    parent_list = dic_node_copy[node].get('parent', ['**'])
                    parent = parent_list[0] if parent_list else '**'

                    # Skip root-level nodes
                    if parent == '**':
                        continue

                    # Check if the node has exactly one child
                    children = dic_node_copy[node].get('childrens', [])
                    if len(children) == 1:
                        child = children[0]

                        # Update parent's children
                        if parent in dic_node_copy and isinstance(dic_node_copy[parent], dict):
                            parent_children = dic_node_copy[parent].get('childrens', [])
                            dic_node_copy[parent]['childrens'] = [
                                child if n == node else n for n in parent_children
                            ]

                        # Update child's parent
                        if child in dic_node_copy and isinstance(dic_node_copy[child], dict):
                            dic_node_copy[child]['parent'] = [parent]

                        # Remove the chain node
                        dic_node_copy.pop(node, None)

                except Exception as e:
                    print(f"Warning: Error processing node {node}: {str(e)}")
                    continue

        return dic_node_copy

    def find_closest_point(
        self,
        point: np.ndarray,
        points: np.ndarray
    ) -> Tuple[np.ndarray, int, float]:
        """
        Find the closest point in a set of points to a given point.
        
        Uses KD-tree for efficient nearest neighbor search.

        Parameters:
            point: Target point coordinates
            points: Array of points to search in

        Returns:
            Tuple containing:
            - np.ndarray: Coordinates of closest point
            - int: Index of closest point
            - float: Distance to closest point
        """
        tree = KDTree(points)
        dist, idx = tree.query(point)
        return points[idx], idx, dist

    def compute_density_peak(
        self,
        polygons: List,
        ind_peak: List
    ) -> Tuple[List, List]:
        """
        Compute density peaks for each leaf node.
        
        For each leaf node (potential cluster center), finds the point
        with maximum density within its contour.

        Parameters:
            polygons: List of polygon coordinates
            ind_peak: List of indices for leaf nodes

        Returns:
            Tuple containing:
            - List: Coordinates of density peaks
            - List: Density values at peaks
        """
        peak_center, peak_center_density = [], []
        
        for ind in ind_peak:
            poly_current = Polygon([
                (polygons[ind][0][index], polygons[ind][1][index])
                for index in range(len(polygons[ind][0]))
            ])
            
            inside_points = [
                self.point_densities[i]
                for i, point in enumerate(self.xy)
                if Point(point[0], point[1]).within(poly_current)
            ]
            
            if inside_points:
                max_density_index = np.argmax(inside_points)
                peak_center_density.append(inside_points[max_density_index])
                peak_center.append(self.xy[max_density_index])
                
        return peak_center, peak_center_density

    def create_graph(
        self,
        polygons: List,
        density_values: List,
        dic_node: Dict
    ) -> nx.Graph:
        """
        Create a graph representation of the contour hierarchy.
        
        Constructs a graph where nodes represent contours and edges
        represent containment relationships. Edge weights are based
        on density values.

        Parameters:
            polygons: List of polygon coordinates
            density_values: List of density values for each contour
            dic_node: Dictionary of nodes in contour hierarchy

        Returns:
            nx.Graph: NetworkX graph object representing the hierarchy
        """
        G = nx.Graph()

        for node in dic_node.keys():
            try:
                # Safely get the parent, with fallback to '*' or '**'
                parent_list = dic_node[node].get('parent', ['*'])
                parent = parent_list[0] if parent_list else '*'

                # Add nodes to the graph
                G.add_node(node)
                G.add_node(parent)

                # Determine edge weight
                if parent in ['**', '*']:
                    # Use random weight for root-level nodes
                    edge_weight = (0, np.random.uniform(0, 1))
                else:
                    # Use density value for non-root nodes
                    edge_weight = (density_values[parent], np.random.uniform(0, 1))

                # Add edge with weight
                G.add_edge(node, parent, weight=edge_weight)

            except (KeyError, IndexError) as e:
                # Log any problematic nodes
                print(f"Error processing node {node} in create_graph: {e}")
                # Skip this node to continue processing
                continue

        return G

    def path_max(
        self,
        G: nx.Graph,
        s: Any,
        t: Any,
        current_depth: int
    ) -> float:
        """
        Find the maximum density path between two nodes.
        
        Uses a recursive algorithm to find the path with maximum density
        between two nodes in the graph. This is used to determine cluster
        separability.

        Parameters:
            G: NetworkX graph
            s: Source node
            t: Target node
            current_depth: Current recursion depth

        Returns:
            float: Maximum density along the path
        """
        if len(list(G.edges.data())) == 1:
            return list(G.edges.data())[0][2]['weight'][0]
        if s == t:
            return 0

        weight = np.array([w[0] for u, v, w in G.edges(data='weight')])
        rand_weight = np.array([w[1] for u, v, w in G.edges(data='weight')])
        median = np.median(weight)
        
        G_k = nx.Graph()
        G_k.add_nodes_from(G)
        
        if len(np.unique(weight)) == 1:
            return median
            
        for edge in list(G.edges.data()):
            if edge[2]['weight'][0] > median:
                G_k.add_edge(edge[0], edge[1], weight=edge[2]['weight'])
            if edge[2]['weight'][0] == median:
                if edge[2]['weight'][1] >= np.random.uniform(0, 1):
                    G_k.add_edge(edge[0], edge[1], weight=edge[2]['weight'])

        if t in list(nx.node_connected_component(G_k, s)):
            return self.path_max(G_k.subgraph(nx.node_connected_component(G_k, s)), s, t, current_depth + 1)
        else:
            S = [G_k.subgraph(c) for c in nx.connected_components(G_k)]
            G_bar_k = nx.Graph()
            x, y = None, None
            
            for u, G_u in enumerate(S):
                G_bar_k.add_node(u)
                if s in list(G_u.nodes()):
                    x = u
                if t in list(G_u.nodes()):
                    y = u
                    
                for v, G_v in enumerate(S):
                    if u != v:
                        maximum_weight = None
                        maximum_link = [None, None]
                        for node_u in list(G_u.nodes()):
                            for node_v in list(G_v.nodes()):
                                if node_v in list(G.neighbors(node_u)):
                                    w = G[node_u][node_v]['weight'][0]
                                    rand_value = G[node_u][node_v]['weight'][1]
                                    if maximum_weight is None:
                                        maximum_weight = w
                                        maximum_link[0] = node_u
                                        maximum_link[1] = node_v
                                    else:
                                        if w > maximum_weight:
                                            maximum_weight = w
                                            maximum_link[0] = node_u
                                            maximum_link[1] = node_v
                        if maximum_weight is not None:
                            G_bar_k.add_edge(u, v, weight=(maximum_weight, np.random.uniform(0, 1)))
                            
            return self.path_max(G_bar_k, x, y, current_depth + 1)

    def calculate_maximum_flow(self, G, peak_center_density, ind_peak):
        """
        Calculate maximum flow between density peaks.

        This method computes the maximum density path between all pairs of peaks
        to determine cluster separability. The flows are used to calculate
        gap values that determine distinct clusters.

        Parameters:
            G (nx.Graph): NetworkX graph representing the density hierarchy.
                Nodes represent contours and edges represent containment relationships.
                Edge weights are tuples of (density_value, random_value).

            peak_center_density (List[float]): List of density values at peak centers.
                These values represent the estimated density at each potential
                cluster center.

            ind_peak (List[int]): List of indices for density peaks.
                These indices correspond to the leaf nodes in the contour hierarchy
                that represent potential cluster centers.

        Returns:
            Tuple containing:
            - List[float]: Maximum flow values between peaks.
                Each value represents the highest density path between a peak
                and all higher-density peaks.

            - np.ndarray: Array of sorted density values.
                Contains the density values at peak centers, sorted in
                descending order.

            - pd.DataFrame: DataFrame containing peak information.
                Includes columns for peak indices and their density values,
                sorted by density in descending order.

        Raises:
            ValueError: If G is empty or not a valid NetworkX graph
            ValueError: If peak_center_density is empty or contains invalid values
            ValueError: If ind_peak is empty or contains invalid indices

        Note:
            The maximum flow values are used to calculate separability scores
            that determine how distinct each cluster is. A higher flow value
            indicates a stronger connection (less separation) between peaks,
            while a lower flow value indicates better cluster separation.

        Example:
            >>> max_flow, dens_list, dens_df = self.calculate_maximum_flow(G, peak_densities, peak_indices)
            >>> print(f"Number of potential clusters: {len(max_flow)}")
            >>> print(f"Maximum density value: {dens_list[0]}")
        """
        # Create and sort density DataFrame
        dens_df = pd.DataFrame({'density': peak_center_density}, index=ind_peak)
        dens_df = dens_df.sort_values(by=['density'], ascending=False)
        dens_list = dens_df.to_numpy()
        ind_cand = dens_df.index

        # Calculate maximum flows
        max_flow = []
        for ind, i in enumerate(ind_cand):
            maximum = -np.inf
            for j in ind_cand[:ind]:
                current_depth = 0
                flow = self.path_max(G, i, j, current_depth)
                if flow > maximum:
                    maximum = flow

            # Handle case of first point or no flow found
            if maximum == -np.inf:
                if ind == 0:
                    max_flow.append(0)
                else:
                    max_flow.append(np.inf)
            else:
                max_flow.append(maximum)

        # Store results for later use
        self._max_flow = max_flow
        self._dens_list = dens_list
        self._dens_df = dens_df

        return max_flow, dens_list, dens_df

    def get_well_separated_points(self, max_flow, dens_list, dens_df, print_table=True):
        """
        Returns well-separated points from the input data based on the specified selection method.
        This function is central to the clustering process, analyzing the separability between 
        density peaks and determining cluster cutoffs.

        Parameters:
        -----------
        max_flow : list
            Maximum flow values between peaks, representing the highest density path
            between each peak and all higher-density peaks.

        dens_list : numpy.ndarray
            Array of density values at peak centers, sorted in descending order.

        dens_df : pandas.DataFrame
            DataFrame containing peak information with indices mapping to peak locations
            and density values.

        print_table : bool, default=True
            Whether to print the separability table. Set to False to delay printing.

        Returns:
        --------
        tuple
            (numpy.ndarray, pandas.DataFrame): 
            - Indices of selected peaks that will serve as cluster centers
            - Display DataFrame for later printing if print_table=False
        """
        # Calculate separability for each point
        nb_values = len(max_flow)
        separability = np.array([1 - max_flow[i]/dens_list[i][0] for i in range(nb_values)])

        # Add separability to DataFrame and sort
        dens_df['separability'] = separability
        dens_df = dens_df.sort_values(by=['separability'], ascending=False)

        if self.n_clusters is not None:
            well_separated = dens_df[dens_df['separability'] > 0]
            max_possible_clusters = len(well_separated)

            if self.n_clusters > max_possible_clusters:
                if print_table:
                    print(f"\nWarning: Requested number of clusters ({self.n_clusters}) exceeds maximum possible clusters ({max_possible_clusters})")
                    print(f"Using maximum available clusters: {max_possible_clusters}")

            return well_separated.iloc[:self.n_clusters].index.to_numpy(), None

        # Get positive separability points and add reference point
        well_separated = dens_df[dens_df['separability'] > 0].copy()
        well_separated.loc['*',:] = [0, 0]

        # Calculate gaps
        well_separated['gap'] = -well_separated.separability.diff()

        # Create a display DataFrame
        display_df = well_separated.copy()
        display_df = display_df.reset_index()

        # Find gap order (excluding first row which has NaN gap)
        gaps = display_df['gap'].iloc[1:].to_numpy()
        gap_order_indices = np.argsort(-gaps)  # Sort in descending order

        # Add gap_order column for display
        display_df['gap_order'] = None
        for order, idx in enumerate(gap_order_indices, 1):
            display_df.loc[idx + 1, 'gap_order'] = order

        # Display the formatted output if requested
        if print_table:
            print("\nSeparability and gaps:")
            display_output = display_df[['separability', 'gap', 'gap_order']].fillna('--').round(6)
            print(display_output.to_string())

        if isinstance(self.gap_order, int):
            if self.gap_order <= len(gap_order_indices):
                # Get the position of the nth largest gap
                cut_position = gap_order_indices[self.gap_order - 1] + 1
                # Return points up to and including the cut position
                return well_separated.iloc[:cut_position + 1].index.to_numpy()[:-1], display_df

            else:
                return well_separated.index.to_numpy()[:-1], display_df  # Exclude reference point

        elif self.gap_order == 'max_clusters':
            return well_separated.index.to_numpy()[:-1], display_df  # Exclude reference point

        elif self.gap_order == 'all':
            return well_separated.index.to_numpy()[:-1], display_df  # Exclude reference point

        else:
            raise ValueError("Invalid gap_order parameter. Must be int, 'max_clusters', or 'all'.")
   
    
    def clean_dic_after_choice(self, ind_selection, dic_node):
        """
        Clean and update node dictionary after cluster selection.

        This method updates the hierarchical structure to reflect the selected clusters.
        It maintains the token hierarchy and ensures proper parent-child relationships.

        Parameters:
            ind_selection: List of indices for selected cluster centers
            dic_node: Dictionary representing the node hierarchy

        Returns:
            dict: Updated node dictionary reflecting cluster selection
        """
        dic_node_copy = copy.deepcopy(dic_node)

        # First, add tokens to leaves
        list_leaves = self.find_leaves(dic_node_copy)
        for leaf in list_leaves:
            if leaf in ind_selection:
                dic_node_copy[leaf]['token'] = [leaf]
            else:
                dic_node_copy[leaf]['token'] = []

        # Process nodes from highest level to root
        max_level = max(
            node.get('level', -1) 
            for node in dic_node_copy.values() 
            if isinstance(node, dict)
        )

        for level in range(max_level, -1, -1):
            nodes_at_level = [
                node for node, node_data in dic_node_copy.items() 
                if isinstance(node_data, dict) and node_data.get('level') == level
            ]

            for node in nodes_at_level:
                if node == '*':
                    continue

                parent_list = dic_node_copy[node].get('parent', ['*'])
                parent = parent_list[0] if parent_list else '*'

                tokens = dic_node_copy[node].get('token', [])
                valid_tokens = [t for t in tokens if t in ind_selection]

                dic_node_copy[node]['token'] = valid_tokens

                if valid_tokens and parent in dic_node_copy:
                    dic_node_copy[parent]['token'].extend(valid_tokens)

        # Add all selected indices to root node's token
        if '*' in dic_node_copy:
            dic_node_copy['*']['token'] = list(ind_selection)

        return dic_node_copy


    def get_point_polygon(
        self,
        polygons: List,
        dic_node: Dict
    ) -> Dict:
        """
        Create mapping of points to polygon nodes.
        
        For each node, identifies which points lie within its polygon.

        Parameters:
            polygons: List of polygon coordinates
            dic_node: Dictionary of nodes in contour hierarchy

        Returns:
            Dict: Mapping of node indices to lists of contained points
        """
        dic_point_node = {ind: [] for ind in dic_node.keys()}

        # Create KDTree once for all points
        tree = KDTree(self.xy)

        for ind_node in dic_node.keys():
            if ind_node != '*':
                try:
                    # Convert polygon to shapely Polygon once
                    polygon_coords = [(polygons[ind_node][0][i], polygons[ind_node][1][i]) 
                                    for i in range(len(polygons[ind_node][0]))]
                    poly_current = Polygon(polygon_coords)

                    # Get polygon bounds for quick filtering
                    minx, miny, maxx, maxy = poly_current.bounds

                    # Query KDTree for points in bounding box
                    potential_points = tree.query_ball_point(
                        [(minx + maxx)/2, (miny + maxy)/2],
                        r=max(maxx - minx, maxy - miny)
                    )

                    # Filter points using shapely
                    for k in potential_points:
                        point = self.xy[k]
                        if Point(point[0], point[1]).within(poly_current):
                            dic_point_node[ind_node].append([k, point, self.point_densities[k]])

                except Exception:
                    continue

        # Add all points to root node more efficiently
        dic_point_node['*'] = [[k, point, self.point_densities[k]] 
                              for k, point in enumerate(self.xy)]

        return dic_point_node
    
    def find_closest_point_vectorized(point, points):
        """Vectorized version of closest point finding"""
        distances = np.sum((points - point) ** 2, axis=1)
        idx = np.argmin(distances)
        return points[idx], idx, np.sqrt(distances[idx])

    def assign_points(self, xy, dic_point_node, dic_node):
        """
        Assigns cluster labels to points using a hierarchical density-based approach.
        This function implements a top-down point assignment strategy, processing nodes
        from highest to lowest density levels and ensuring connectivity within clusters.

        Parameters:
        -----------
        xy : numpy.ndarray
            Array of point coordinates with shape (n_points, 2)

        dic_point_node : dict
            Dictionary mapping node indices to lists of contained points, where each point
            is represented as [index, coordinates, density]

        dic_node : dict
            Hierarchical structure of density contours where each node contains:
            - level: Contour level in hierarchy
            - childrens: List of child node indices
            - token: List of cluster assignments
            - parent: Parent node index

        Returns:
        --------
        numpy.ndarray
            Array of cluster assignments for each point, numbered from 1 to n_clusters

        Algorithm Steps:
        ---------------
        1. Point Assignment Strategy:
            - Processes nodes level by level, from highest to lowest density
            - Assigns points in leaf nodes directly using node tokens
            - For non-leaf nodes:
                * Separates points into assigned and unassigned groups
                * Processes unassigned points based on density and proximity

        2. Assignment Rules:
            For leaf nodes:
                - Points get the node's token value directly
            For non-leaf nodes:
                - Unassigned points are handled in two ways:
                    a) If there are already assigned points:
                       * Assigns based on nearest assigned neighbor
                    b) If no assigned points but node has token:
                       * All points get the node's token value

        3. Ordering and Processing:
            - Orders unassigned points by density (highest first)
            - Ensures high-density points are assigned first
            - Maintains cluster connectivity through nearest neighbor assignments

        Notes:
        ------
        - Final cluster labels are remapped to start from 1
        - Points in the same contour get the same cluster label
        - Assignment preserves the density-based hierarchy
        - Method ensures all points get assigned to a cluster
        """
        # Initialize assignment array
        assignment = np.full(len(xy), None)

        # Find maximum level in the hierarchy
        max_level = -1
        for node in dic_node.keys():
            if dic_node[node]['level'] > max_level:
                max_level = dic_node[node]['level']

        # Process levels from highest to lowest
        for level in range(max_level, -1, -1):
            # Get nodes at current level
            list_node_level = []
            for node in dic_node.keys():
                if dic_node[node]['level'] == level:
                    list_node_level.append(node)

            # Process each node at current level
            for ind_node in list_node_level:
                # Handle leaf nodes
                if len(dic_node[ind_node]['childrens']) == 0:
                    if dic_node[ind_node]['token']:  # If node has tokens
                        for info_point in dic_point_node[ind_node]:
                            k = info_point[0]
                            assignment[k] = int(dic_node[ind_node]['token'][0])
                # Handle non-leaf nodes
                else:
                    # Initialize lists for assigned and unassigned points
                    already_assigned = []
                    not_assigned = []
                    already_assigned_value = []
                    already_assigned_point = []
                    not_assigned_point = []
                    not_assigned_den = []

                    # Split points into assigned and unassigned
                    for info_point in dic_point_node[ind_node]:
                        k = info_point[0]
                        if assignment[k] is not None:
                            already_assigned.append(k)
                            already_assigned_value.append(int(assignment[k]))
                            already_assigned_point.append(info_point[1])
                        else:
                            not_assigned.append(k)
                            not_assigned_point.append(info_point[1])
                            not_assigned_den.append(info_point[2])

                    # Process unassigned points if any exist
                    if len(not_assigned) != 0:
                        # Sort points by density in descending order
                        sorted_zip = sorted(zip(not_assigned_den, not_assigned, not_assigned_point), 
                                         key=lambda x: x[0], 
                                         reverse=True)
                        not_assigned_den, not_assigned, not_assigned_point = zip(*sorted_zip)

                        # Assign based on nearest assigned point or node token
                        if already_assigned:
                            already_assigned_point = np.array(already_assigned_point)
                            for ind in range(len(not_assigned)):
                                _, idx, _ = self.find_closest_point(not_assigned_point[ind], already_assigned_point)
                                assignment[not_assigned[ind]] = already_assigned_value[idx]
                        elif dic_node[ind_node]['token']:
                            for ind in range(len(not_assigned)):
                                assignment[not_assigned[ind]] = int(dic_node[ind_node]['token'][0])

        # Make cluster numbers start from 1
        if assignment is not None:
            unique_clusters = np.unique(assignment)
            mapping = {old: idx + 1 for idx, old in enumerate(unique_clusters)}
            assignment = np.array([mapping[x] for x in assignment])

        return assignment

    def run_clustering(self):
        """
        Executes the complete ClusterDC algorithm, implementing the full density-based 
        clustering workflow from contour creation to point assignment.

        Returns:
        --------
        tuple:
           - assignments: List of cluster assignments
               * For gap_order='all': List of multiple clustering solutions
               * For other modes: List containing single clustering solution
           - density_info: List containing [point_densities, grid_densities, xx, yy]
               Used for visualization and analysis

        Algorithm Workflow:
        ------------------
        1. Contour Analysis:
           - Creates density contours from KDE
           - Builds hierarchical structure of contours
           - Removes invalid leaves and chain nodes

        2. Peak Detection:
           - Identifies leaf nodes as potential peaks
           - Computes density values at peak centers
           - Creates graph representation of hierarchy

        3. Separability Analysis:
           - Calculates maximum flows between peaks
           - Determines well-separated points
           - Applies selected gap order criteria

        4. Cluster Assignment:
           For gap_order='all':
               - Generates multiple clustering solutions
               - Creates solutions from 1 to max clusters
               - Stores results in DataFrame with columns per solution

           For other modes:
               - Generates single clustering solution
               - Assigns points to clusters
               - Creates DataFrame with final assignments
        """

        # Create single progress bar for entire process
        with tqdm(total=100, desc="ClusterDC Clustering Progress", unit="%") as pbar:
            # Stage 1: Contour Analysis (25%)
            polygons, density_values, density_contours = self.create_contours()
            dic_node = self.make_node_dic(polygons, density_values, density_contours)
            dic_node = self.trim_unvalid_leaves(dic_node, polygons, self.min_point)
            dic_node = self.trim_chain_nodes(dic_node)
            pbar.update(25)

            # Stage 2: Peak Detection (25%)
            ind_peak = self.find_leaves(dic_node)
            peak_center, peak_center_density = self.compute_density_peak(polygons, ind_peak)
            G = self.create_graph(polygons, density_values, dic_node)
            pbar.update(25)

            # Stage 3: Flow Analysis (25%)
            max_flow, dens_list, dens_df = self.calculate_maximum_flow(G, peak_center_density, ind_peak)
            # Important change: pass print_table=False to delay printing
            ind_selection, separability_df = self.get_well_separated_points(max_flow, dens_list, dens_df, print_table=False)
            pbar.update(25)

            # Stage 4: Point Assignment (25%)
            assignments = []
            if self.gap_order == 'all':
                # If we have the original dataframe
                if hasattr(self, 'original_data') and isinstance(self.original_data, pd.DataFrame):
                    df = self.original_data.copy()
                else:
                    df = pd.DataFrame({'X': self.x_points, 'Y': self.y_points})
                    
                for j in range(len(ind_selection)):
                    ind_selection_sub = ind_selection[:j + 1]
                    dic_node_sub = self.clean_dic_after_choice(ind_selection_sub, dic_node)
                    dic_point_node = self.get_point_polygon(polygons, dic_node_sub)
                    assignment_sub = self.assign_points(self.xy, dic_point_node, dic_node_sub)
                    assignments.append(assignment_sub)
                    df[f'{j+1} clusters'] = assignment_sub
            else:
                dic_node = self.clean_dic_after_choice(ind_selection, dic_node)
                dic_point_node = self.get_point_polygon(polygons, dic_node)
                assignment = self.assign_points(self.xy, dic_point_node, dic_node)
                assignments.append(assignment)
                
                # If we have the original dataframe
                if hasattr(self, 'original_data') and isinstance(self.original_data, pd.DataFrame):
                    df = self.original_data.copy()
                    df['Cluster'] = assignment
                else:
                    df = pd.DataFrame({
                        'X': self.x_points,
                        'Y': self.y_points,
                        'Cluster': assignment
                    })
            pbar.update(25)

        # Now that progress is 100%, print the separability table if available
        if separability_df is not None:
            print("\nSeparability and gaps:")
            display_output = separability_df[['separability', 'gap', 'gap_order']].fillna('--').round(6)
            print(display_output.to_string())

        print('\n============================')
        if self.gap_order == 'all':
            print('Max. number of clusters: ', len(np.unique(assignments[-1])))
        else:
            print('Number of clusters: ', len(np.unique(assignments[0])))
        print('============================\n')

        self.cluster_assignments = df
        density_info = [self.point_densities, self.grid_densities, self.xx, self.yy]
        # Add cluster assignments to original dataframe if it's a pandas DataFrame
        if isinstance(self.original_data, pd.DataFrame):
            if self.gap_order == 'all':
                for j in range(len(ind_selection)):
                    self.original_data[f'cluster_{j+1}'] = df[f'{j+1} clusters']
            else:
                self.original_data['cluster'] = df['Cluster']

        return assignments, density_info
    
    def get_gap_position(self, max_flow, dens_list, dens_df):
        """
        Helper method to get the position based on gap order.

        Parameters:
            max_flow: List of maximum flow values
            dens_list: Array of density values
            dens_df: DataFrame containing peak information

        Returns:
            int: Position based on gap order
        """
        # Calculate separability for each point
        nb_values = len(max_flow)
        separability = np.array([1 - max_flow[i]/dens_list[i][0] for i in range(nb_values)])

        # Add separability to DataFrame and sort
        dens_df['separability'] = separability
        dens_df = dens_df.sort_values(by=['separability'], ascending=False)

        # Get positive separability points and add reference point
        well_separated = dens_df[dens_df['separability'] > 0].copy()
        well_separated.loc['*',:] = [0, 0]

        # Calculate gaps
        well_separated['gap'] = -well_separated.separability.diff()
        well_separated = well_separated.reset_index(drop=False)

        # Find gaps and their positions
        gaps = []
        for i in range(1, len(well_separated)):  # Start from 1 to skip reference point
            gap = well_separated.iloc[i]['gap']
            if not np.isnan(gap):
                gaps.append((abs(gap), i))

        # Sort gaps by size in descending order
        gaps.sort(reverse=True)

        if not gaps:
            print("\nNo gaps found in the data.")
            return 1

        if isinstance(self.gap_order, int):
            if self.gap_order > len(gaps):
                print(f"\nWarning: Requested gap order ({self.gap_order}) exceeds number of available gaps ({len(gaps)}).")
                print(f"Using the smallest available gap (gap order {len(gaps)}).")
                return gaps[-1][1]
            return gaps[self.gap_order - 1][1]
        elif self.gap_order == 'max_clusters':
            return len(well_separated) - 1  # Exclude reference point
        else:
            raise ValueError("Invalid gap_order parameter for get_gap_position method.")


    def get_cluster_assignments(self) -> pd.DataFrame:
        """
        Get cluster assignments for all points, added to the original dataframe.
    
        Returns:
            pd.DataFrame: Original dataframe with added cluster assignments
        
        Raises:
            RuntimeError: If clustering hasn't been performed yet
        """
        if self.cluster_assignments is None:
            raise RuntimeError("Clustering has not been performed yet. Run clustering first.")
        
        # If we have the original dataframe, return it with the cluster assignments
        if hasattr(self, 'original_data') and isinstance(self.original_data, pd.DataFrame):
            return self.original_data
        
        # If not, return the stored cluster assignments
        return self.cluster_assignments

    def plot_separability(self, save_path=None):
        """
        Creates a visualization of cluster separability analysis, showing the relationships
        between density peaks and the gaps that separate them.

        Parameters:
        -----------
        save_path : str, optional
           Path to save the plot as an image file. If None, plot is only displayed.

        Plot Components:
        ---------------
        1. Separability Curve:
           - Blue line with markers showing separability values
           - X-axis: Number of clusters
           - Y-axis: Separability values (0 to 1)
           - Higher values indicate better cluster separation

        2. Gap Visualization:
           - Red dashed vertical lines showing gaps
           - Gaps ordered by magnitude (Gap 1 = largest)
           - Labels showing gap number and magnitude
           - Positioned at midpoints between separability values

        3. Plot Elements:
           - Grid lines for better readability
           - Legend identifying curve and gaps
           - Axis labels and title
           - Automatic axis scaling with padding

        Visual Interpretation:
        ---------------------
        1. Separability Values:
           - Range from 0 to 1
           - 1 = Perfectly separated cluster
           - 0 = Complete connection to other clusters

        2. Gaps:
           - Represent natural breaks between clusters
           - Larger gaps suggest stronger cluster separation
           - Gap order indicates relative importance

        3. Pattern Analysis:
           - Sharp drops indicate clear cluster boundaries
           - Gradual changes suggest hierarchical structure
           - Flat regions indicate similar cluster strengths

        Notes:
        ------
        - Requires clustering to be run first (checks for _max_flow and _dens_list)
        - Automatically adjusts figure size and layout
        - Uses high-resolution output (300 dpi) when saving
        - Shows reference point (*) at zero for computational completeness

        Example Interpretation:
        ----------------------
        - Large first gap: Clear primary cluster separation
        - Multiple similar gaps: Several equally valid clustering options
        - Small gaps: Potential sub-cluster structure
        """
        if self._max_flow is None or self._dens_list is None:
            raise RuntimeError("No clustering results available. Run clustering first.")

        # Calculate separability values
        max_flow = self._max_flow
        dens_list = self._dens_list
        nb_values = len(max_flow)
        separability = np.array([1 - max_flow[i]/dens_list[i][0] for i in range(nb_values)])

        # Create DataFrame with separability values
        dens_df = self._dens_df.copy()
        dens_df['separability'] = separability
        dens_df = dens_df.sort_values(by=['separability'], ascending=False)

        # Get positive separability points and add reference point
        well_separated = dens_df[dens_df['separability'] > 0].copy()
        well_separated.loc['*',:] = [0, 0]

        # Calculate gaps
        well_separated['gap'] = -well_separated.separability.diff()
        well_separated = well_separated.reset_index(drop=False)

        # Sort gaps by magnitude (excluding first row which has NaN gap)
        gaps = well_separated['gap'].iloc[1:].to_numpy()
        gap_order = np.argsort(-gaps)  # Sort in descending order

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Plot separability values
        x_values = np.arange(1, len(well_separated) + 1)
        plt.plot(x_values, well_separated['separability'], 'bo-', 
                 label='Separability', linewidth=2, markersize=8)

        # Add dummy plot for gap legend
        plt.plot([], [], 'r--', alpha=0.5, label='Gap Magnitude')

        # Plot gaps in order of magnitude
        for gap_num, gap_idx in enumerate(gap_order, 1):
            gap_idx = gap_idx + 1  # Adjust index since we excluded first row
            gap = well_separated.iloc[gap_idx]['gap']
            if not np.isnan(gap):
                # Calculate midpoint between points for vertical line
                mid_x = (x_values[gap_idx-1] + x_values[gap_idx]) / 2
                y_top = well_separated.iloc[gap_idx-1]['separability']
                y_bottom = well_separated.iloc[gap_idx]['separability']

                # Plot vertical line for gap
                plt.plot([mid_x, mid_x], [y_bottom, y_top], 'r--', alpha=0.5)

                # Add gap label
                plt.text(mid_x, y_bottom + (y_top - y_bottom)/2, 
                    f'Gap {gap_num}',
                    transform=plt.gca().transData,
                    rotation=45,
                    rotation_mode='anchor',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=10,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

        # Customize the plot
        plt.xlabel('Number of Clusters', fontsize=12)
        plt.ylabel('Separability', fontsize=12)
        plt.title('Cluster Separability Analysis', fontsize=14, pad=20)

        # Set axis limits with padding
        plt.xlim(0.5, len(well_separated) + 0.5)
        y_min = min(0, well_separated['separability'].min()) - 0.1
        y_max = well_separated['separability'].max() + 0.1
        plt.ylim(y_min, y_max)

        # Add grid and legend
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(x_values)
        plt.legend(loc='upper right', fontsize=10)

        # Adjust layout
        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
   
        
    def plot_results(self, assignments, density_info, figsize=(20, 6), vmin=None):
        """
        Creates comprehensive visualization of clustering results, showing original data,
        density estimation, and cluster assignments.

        Parameters:
        -----------
        assignments : list
           List of cluster assignments:
           - For gap_order='all': Multiple clustering solutions
           - For other modes: Single clustering solution

        density_info : list
           List containing [point_densities, grid_densities, xx, yy]:
           - point_densities: Density values at each data point
           - grid_densities: Density values on regular grid
           - xx, yy: Grid coordinates for contour plotting

        figsize : tuple, optional
           Figure size in inches (width, height). Default is (20, 6)

        vmin : float, optional
           Minimum density value for visualization. Default is 0.01

        Plot Types:
        -----------
        1. For gap_order='all':
           Creates N+3 subplots, where N is number of clustering solutions:
           - Original data scatter plot
           - KDE density contour plot
           - Point density scatter plot
           - N clustering result plots (1 to N clusters)

        2. For other modes:
           Creates three subplots:
           - Original data (black points)
           - KDE density (contour plot with colorbar)
           - Clustering results (colored by cluster with legend)

        Visual Elements:
        ---------------
        1. Original Data:
           - Black scatter points
           - Consistent axis limits
           - Custom axis labels

        2. Density Visualization:
           - Contour plot with viridis colormap
           - Overlaid white points
           - Density colorbar
           - 20 contour levels

        3. Clustering Results:
           - Points colored by cluster
           - Automatic legend generation
           - Cluster count in title
           - Customized titles based on method

        Notes:
        ------
        - Maintains consistent axis limits across all plots
        - Automatically adjusts figure size for multiple solutions
        - Uses tab10 colormap for cluster visualization
        - Saves high-resolution output if save_path provided
        - Handles both hierarchical and single clustering modes
        - Creates legends for cluster identification
        """
        point_densities, grid_densities, xx, yy = density_info

        # Dynamically calculate vmin if not provided
        if vmin is None:
            # Find a small non-zero minimum, or use a very small value
            non_zero_densities = grid_densities[grid_densities > 0]
            vmin = non_zero_densities.min() if len(non_zero_densities) > 0 else 1e-10

        # Calculate axis limits from the grid
        x_min, x_max = xx[0, 0], xx[0, -1]
        y_min, y_max = yy[0, 0], yy[-1, 0]

        # Prepare contour levels - use vmin properly
        max_density = np.max(grid_densities)
        # Ensure vmin doesn't exceed max_density
        vmin = min(vmin, max_density)

        # Create levels that are strictly increasing
        levels = np.linspace(vmin, max_density, 20)
        levels = np.unique(levels)  # Ensure unique, increasing values

        # Ensure at least two levels
        if len(levels) < 2:
            levels = [vmin, max_density]

        if self.gap_order == 'all':
            # Calculate required number of subplots
            n_plots = len(assignments) + 3  # +3 for original data, KDE density, and point densities
            n_cols = 3  # Fixed at 3 columns
            n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division

            # Adjust figure size based on number of rows
            fig_height = 6 * n_rows
            plt.figure(figsize=(20, fig_height))

            # Plot original data
            plt.subplot(n_rows, n_cols, 1)
            plt.scatter(self.x_points, self.y_points, c='black', s=10)
            plt.title('Original Data')
            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

            # Plot KDE density - use vmin in contour
            plt.subplot(n_rows, n_cols, 2)
            contour = plt.contourf(xx, yy, grid_densities, 
                                 levels=levels,
                                 cmap='viridis', 
                                 alpha=0.6)
            plt.colorbar(contour, label='KDE')
            plt.title('KDE Density')
            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

            # Plot point densities - use provided vmin
            plt.subplot(n_rows, n_cols, 3)
            point_max = np.max(point_densities)
            sc = plt.scatter(self.x_points, self.y_points, 
                           c=point_densities, 
                           cmap='viridis', 
                           s=10,
                           vmin=vmin,
                           vmax=point_max)
            plt.colorbar(sc, label='Density')
            plt.title('Point Densities')
            plt.xlabel(self.xlabel)
            plt.ylabel(self.ylabel)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

            # Plot all clustering results
            for i, assignment in enumerate(assignments):
                plt.subplot(n_rows, n_cols, i + 4)
                plt.scatter(self.x_points, self.y_points, c=assignment, cmap='tab10', s=10)
                plt.title(f'ClusterDC Clustering ({len(np.unique(assignment))} {"cluster" if len(np.unique(assignment)) == 1 else "clusters"})')
                plt.xlabel(self.xlabel)
                plt.ylabel(self.ylabel)
                plt.xlim(x_min, x_max)
                plt.ylim(y_min, y_max)

        else:  # For gap order, max_clusters, or user-specified number of clusters
            fig, ax = plt.subplots(1, 3, figsize=figsize)

            # Original Data plot
            ax[0].scatter(self.x_points, self.y_points, c='black', s=10)
            ax[0].set_title('Original Data')
            ax[0].set_xlabel(self.xlabel)
            ax[0].set_ylabel(self.ylabel)
            ax[0].set_xlim(x_min, x_max)
            ax[0].set_ylim(y_min, y_max)

            # KDE Density plot - use vmin in contour
            contour = ax[1].contourf(xx, yy, grid_densities, 
                                    levels=levels,
                                    cmap='viridis', 
                                    alpha=0.6)
            plt.colorbar(contour, ax=ax[1], label='KDE')
            ax[1].set_title('KDE Density')
            ax[1].set_xlabel(self.xlabel)
            ax[1].set_ylabel(self.ylabel)
            ax[1].set_xlim(x_min, x_max)
            ax[1].set_ylim(y_min, y_max)

            # Clustering results plot
            if assignments and len(assignments) > 0:  # Check if we have assignments
                assignment = assignments[-1]  # Use the last assignment

                # Create scatter plot for clusters
                scatter = ax[2].scatter(self.x_points, self.y_points, 
                                      c=assignment, cmap='tab10', s=10)

                # Get unique clusters and their colors
                unique_clusters = np.unique(assignment)
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

                # Get unique clusters and their colors
                unique_clusters = np.unique(assignment)
                n_clusters = len(unique_clusters)
                colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

                # Calculate number of legend columns needed (15 items per column)
                n_cols = (n_clusters - 1) // 15 + 1  # Integer division rounded up

                # Create legend handles
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=colors[i], 
                                            label=f'Cluster {cluster}',
                                            markersize=10)
                                  for i, cluster in enumerate(unique_clusters)]

                # Add legend with dynamic columns
                if n_cols > 1:
                    # For multi-column legend, adjust bbox_to_anchor to prevent overlap
                    ax[2].legend(handles=legend_elements, 
                                loc='center left', 
                                bbox_to_anchor=(1, 0.5),
                                ncol=n_cols,
                                columnspacing=1.5,  # Add more space between columns
                                handletextpad=0.5)  # Reduce space between marker and text
                else:
                    # For single column legend, use original formatting
                    ax[2].legend(handles=legend_elements, 
                                loc='center left', 
                                bbox_to_anchor=(1, 0.5))

                # Set title based on clustering method
                if self.n_clusters is not None:
                    title = f'ClusterDC Clustering\n(User-specified {self.n_clusters} clusters)'
                elif self.gap_order == 'max_clusters':
                    title = f'ClusterDC Clustering\n(Maximum {len(np.unique(assignment))} clusters)'
                else:
                    title = f'ClusterDC Clustering\n({len(np.unique(assignment))} clusters)'

                ax[2].set_title(title)
                ax[2].set_xlabel(self.xlabel)
                ax[2].set_ylabel(self.ylabel)
                ax[2].set_xlim(x_min, x_max)
                ax[2].set_ylim(y_min, y_max)

        plt.tight_layout()

        # Save the figure if save_path is provided
        if self.save_path:
            plt.savefig(self.save_path, dpi=300, bbox_inches='tight')

        plt.show()


    def find_optimal_clusters(self, save_path=None, max_elbows=4, method='direct_gap'):
        """
        Identifies potential elbow points in the cluster separability curve using either 
        direct gap analysis or Gaussian Process Regression (GPR).
    
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot. If None, plot is only displayed.
        max_elbows : int, optional
            Maximum number of elbow points to return (default: 4)
        method : str, optional
            Method to use for finding elbow points:
            - 'direct_gap': Analyzes gaps directly from separability values (default)
            - 'gpr': Uses Gaussian Process Regression (original method)
    
        Returns:
        --------
        list
            List of potential optimal cluster counts corresponding to main elbow points,
            sorted by importance (most pronounced elbows first)
    
        Raises:
        -------
        RuntimeError
            If clustering has not been performed yet
        ImportError
            If required packages are not installed
        """
        # Check if clustering has been run
        if self._max_flow is None or self._dens_list is None:
            raise RuntimeError("No clustering results available. Run clustering first.")
    
        # Calculate separability values
        max_flow = self._max_flow
        dens_list = self._dens_list
        nb_values = len(max_flow)
        separability = np.array([1 - max_flow[i]/dens_list[i][0] for i in range(nb_values)])
    
        # Create DataFrame with separability values
        dens_df = self._dens_df.copy()
        dens_df['separability'] = separability
        dens_df = dens_df.sort_values(by=['separability'], ascending=False)
    
        # Get positive separability points and add reference point
        well_separated = dens_df[dens_df['separability'] > 0].copy()
        well_separated.loc['*',:] = [0, 0]
    
        # Calculate gaps
        well_separated['gap'] = -well_separated.separability.diff()
        well_separated = well_separated.reset_index(drop=False)
        
        if method == 'direct_gap':
            return self._find_optimal_clusters_direct_gap(well_separated, max_elbows, save_path)
        else:  # 'gpr' method
            return self._find_optimal_clusters_gpr(well_separated, max_elbows, save_path)
    
    def _find_optimal_clusters_direct_gap(self, well_separated, max_elbows=4, save_path=None):
        """
        Find optimal clusters by directly analyzing the gaps in separability values.
        This approach is more straightforward and can better detect clear separations
        in the data, especially sharp drops in separability.
        
        Parameters:
        -----------
        well_separated : pd.DataFrame
            DataFrame containing separability and gap values
        max_elbows : int
            Maximum number of elbow points to return
        save_path : str, optional
            Path to save visualization
            
        Returns:
        --------
        list
            List of optimal cluster counts
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Skip the first row which has NaN gap (reference point)
        gaps = well_separated['gap'].iloc[1:].fillna(0).values
        indices = np.arange(1, len(gaps) + 1)  # Cluster counts (index positions)
        separability_values = well_separated['separability'].iloc[1:].values
        
        # Special case: Check for sharp drop from high to low separability
        # This indicates a clear cluster boundary
        for i in range(len(separability_values)-1):
            if (separability_values[i] > 0.9 and  # High separability
                separability_values[i+1] < 0.1 and  # Low separability
                i+1 < len(indices)):  # Valid index
                # Return the index before the drop
                optimal_clusters = [i+2]  # +2 because indices start at 1 and we already skipped first row
                
                # Create visualization
                plt.figure(figsize=(12, 8))
                
                # Plot separability values
                plt.plot(indices, separability_values, 'bo-', 
                         label='Separability', linewidth=2, markersize=6)
                
                # Highlight the optimal cluster point
                i_opt = optimal_clusters[0] - 1  # Adjust for 0-indexing
                plt.axvline(x=optimal_clusters[0], color='red', linestyle='--', alpha=0.7, linewidth=2)
                plt.scatter([optimal_clusters[0]], [separability_values[i_opt-1]], 
                            c='r', s=100, marker='o', zorder=5)
                plt.annotate(f'{optimal_clusters[0]} clusters - Sharp separability drop', 
                            xy=(optimal_clusters[0], separability_values[i_opt-1]),
                            xytext=(10, 0), textcoords='offset points',
                            fontsize=10, fontweight='bold', color='darkred')
                
                # Customize the plot
                plt.xlabel('Number of Clusters', fontsize=12)
                plt.ylabel('Separability', fontsize=12)
                plt.title('Sharp Drop Analysis for Optimal Cluster Selection', fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(loc='upper right')
                
                # Set axis limits
                plt.xlim(0.5, len(indices) + 0.5)
                plt.xticks(np.arange(1, len(indices) + 1, max(1, len(indices) // 20)))
                
                # Tight layout and save if path provided
                plt.tight_layout()
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
                plt.show()
                
                print(f"Optimal cluster count identified: {optimal_clusters[0]} (based on sharp separability drop)")
                return optimal_clusters
        
        # If no sharp drop, continue with the regular gap analysis
        # Find all gaps and sort by magnitude
        gap_data = [(i+1, gaps[i]) for i in range(len(gaps)) if gaps[i] > 0]
        gap_data.sort(key=lambda x: x[1], reverse=True)
        
        # Take only significant gaps
        # Define significance: at least 25% of the maximum gap or above the 90th percentile
        if gap_data:
            max_gap = gap_data[0][1]
            threshold = max(max_gap * 0.25, np.percentile(gaps[gaps > 0], 90))
            significant_gaps = [g for g in gap_data if g[1] >= threshold]
            
            # Limit to max_elbows
            significant_gaps = significant_gaps[:max_elbows]
            
            # Sort by index (cluster count) for easier interpretation
            optimal_clusters = sorted([g[0] for g in significant_gaps])
        else:
            optimal_clusters = []
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot separability values
        plt.plot(indices, separability_values, 'bo-', 
                 label='Separability', linewidth=2, markersize=6)
        
        # Plot gaps as vertical lines
        for i, gap in gap_data[:max_elbows]:
            color = 'r' if (i in optimal_clusters) else 'gray'
            alpha = 0.7 if (i in optimal_clusters) else 0.3
            lw = 2 if (i in optimal_clusters) else 1
            
            plt.axvline(x=i, color=color, linestyle='--', alpha=alpha, linewidth=lw)
            if i in optimal_clusters:
                plt.annotate(f'Gap: {gap:.4f}', 
                            xy=(i, well_separated['separability'].iloc[i]),
                            xytext=(5, 0), textcoords='offset points',
                            fontsize=9, fontweight='bold', color='darkred')
        
        # Highlight optimal cluster counts
        for i in optimal_clusters:
            plt.scatter([i], [well_separated['separability'].iloc[i]], 
                        c='r', s=100, marker='o', zorder=5)
            plt.annotate(f'{i} clusters', 
                        xy=(i, well_separated['separability'].iloc[i]),
                        xytext=(0, -15), textcoords='offset points',
                        fontsize=10, fontweight='bold', color='darkred',
                        ha='center')
        
        # Customize the plot
        plt.xlabel('Number of Clusters', fontsize=12)
        plt.ylabel('Separability', fontsize=12)
        plt.title('Gap Analysis for Optimal Cluster Selection', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        
        # Set axis limits
        plt.xlim(0.5, len(indices) + 0.5)
        plt.xticks(np.arange(1, len(indices) + 1, max(1, len(indices) // 20)))
        
        # Tight layout and save if path provided
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print results
        if optimal_clusters:
            print(f"Optimal cluster counts identified: {optimal_clusters}")
        else:
            print("No clear optimal cluster counts identified.")
        
        return optimal_clusters
    
    def _find_optimal_clusters_gpr(self, well_separated, max_elbows=4, save_path=None):
        """
        Original GPR-based method for finding optimal clusters.
        This is the original method implemented with clearer code organization.
        """
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
            from scipy.signal import find_peaks
            from scipy.ndimage import gaussian_filter1d
            import numpy as np
        except ImportError:
            raise ImportError("Required packages not installed. Please install scikit-learn and scipy.")
    
        # Extract x and y values for GPR
        x_values = np.arange(1, len(well_separated)).reshape(-1, 1)  # Skip the reference point
        y_values = well_separated['separability'].iloc[1:].values  # Skip the reference point
    
        # If we have fewer than 5 clusters, just use the maximum number of clusters
        if len(x_values) < 5:
            optimal_clusters = len(x_values)
            print(f"Fewer than 5 potential clusters detected. Using maximum number of clusters: {optimal_clusters}")
            return [optimal_clusters]
    
        # Minimum data points required for GPR analysis
        if len(x_values) < 3:
            print("Not enough data points for GPR analysis. Need at least 3 points.")
            if len(x_values) > 0:
                optimal_clusters = len(x_values)
                print(f"Using the maximum available clusters: {optimal_clusters}")
                return [optimal_clusters]
            else:
                print("No valid clusters detected.")
                return []
    
        # Set up the Gaussian Process Regression
        length_scale = max(1.0, len(x_values) / 10)  # Scale based on data size
        kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=length_scale, 
                                           length_scale_bounds=(length_scale/2, length_scale*2)) + \
                 WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1.0))
    
        gpr = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=5,
            alpha=0.1,
            normalize_y=True,
            random_state=42
        )
    
        # Fit the model to the data
        gpr.fit(x_values, y_values)
    
        # Create a fine-grained x-axis for prediction
        x_pred = np.linspace(x_values.min(), x_values.max(), 100).reshape(-1, 1)
    
        # Predict with GPR (mean and standard deviation)
        y_pred, y_std = gpr.predict(x_pred, return_std=True)
    
        # Calculate derivatives
        dx = x_pred[1, 0] - x_pred[0, 0]
        first_derivative = np.gradient(y_pred, dx)
        second_derivative = np.gradient(first_derivative, dx)
    
        # Apply Gaussian smoothing to reduce noise
        sigma = max(2.0, len(x_values) / 15)
        smooth_second_derivative = gaussian_filter1d(second_derivative, sigma=sigma)
    
        # Find candidate points (positive curvature, negative slope)
        candidate_indices = np.where(
            (smooth_second_derivative > 0) & 
            (first_derivative < 0)
        )[0]
    
        # Initialize elbow_points
        elbow_points = []
    
        # Find significant positive curvature points
        if len(candidate_indices) > 0:
            # Set threshold for significant curvature
            max_curvature = np.max(smooth_second_derivative[candidate_indices])
            curvature_threshold = max_curvature * 0.3  # Consider only top 30% of curvatures
    
            # Filter candidates by curvature threshold
            significant_candidates = [idx for idx in candidate_indices 
                                     if smooth_second_derivative[idx] >= curvature_threshold]
    
            # If too few points pass the threshold, take the top 1
            if len(significant_candidates) < 1 and len(candidate_indices) >= 1:
                # Sort candidates by curvature and take top 1
                significant_candidates = sorted(candidate_indices, 
                                              key=lambda idx: smooth_second_derivative[idx],
                                              reverse=True)[:1]
    
            # Sort candidate indices by curvature value
            sorted_candidates = sorted(significant_candidates, 
                                      key=lambda idx: smooth_second_derivative[idx],
                                      reverse=True)
    
            # Limit to max_elbows
            sorted_candidates = sorted_candidates[:max_elbows]
    
            # Get cluster counts for all candidates
            for idx in sorted_candidates:
                x_val = x_pred[idx, 0]
                cluster_count = int(np.round(x_val))
                # Ensure cluster count is within valid range
                cluster_count = max(1, min(cluster_count, len(x_values)))
                elbow_points.append((idx, cluster_count, smooth_second_derivative[idx]))
    
            # Filter out duplicates (when multiple points map to the same cluster count)
            unique_clusters = set()
            filtered_elbow_points = []
            for idx, count, curvature in elbow_points:
                if count not in unique_clusters:
                    unique_clusters.add(count)
                    filtered_elbow_points.append((idx, count, curvature))
    
            elbow_points = filtered_elbow_points
    
            # Further filter to remove adjacent cluster counts
            if len(elbow_points) > 1:
                # Sort by cluster count
                elbow_points.sort(key=lambda x: x[1])
    
                # Identify groups of adjacent clusters
                adjacent_groups = []
                current_group = [elbow_points[0]]
    
                for i in range(1, len(elbow_points)):
                    prev_count = elbow_points[i-1][1]
                    curr_count = elbow_points[i][1]
    
                    if curr_count - prev_count <= 1:  # Adjacent clusters
                        current_group.append(elbow_points[i])
                    else:
                        adjacent_groups.append(current_group)
                        current_group = [elbow_points[i]]
    
                # Add the last group
                if current_group:
                    adjacent_groups.append(current_group)
    
                # From each group, select the point with highest curvature
                filtered_again = []
                for group in adjacent_groups:
                    best_in_group = max(group, key=lambda x: x[2])
                    filtered_again.append(best_in_group)
    
                elbow_points = filtered_again
        else:
            print("No clear elbow points with positive curvature found.")
    
        # Create the visualization
        plt.figure(figsize=(12, 8))
    
        # Plot original data points
        plt.scatter(x_values, y_values, c='blue', s=50, label='Separability values')
    
        # Plot GPR prediction with uncertainty
        plt.plot(x_pred, y_pred, 'k-', label='GPR mean')
        plt.fill_between(x_pred.flatten(), 
                         y_pred - 1.96 * y_std, 
                         y_pred + 1.96 * y_std, 
                         alpha=0.2, 
                         color='k', 
                         label='GPR 95% confidence interval')
    
        # Create a secondary axis for showing the second derivative
        ax1 = plt.gca()
        ax2 = ax1.twinx()
    
        # Plot the second derivative (curvature) curve
        ax2.plot(x_pred, smooth_second_derivative, 'g-', alpha=0.7, label='Curvature (2nd derivative)')
        ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.7)  # zero line for reference
        ax2.set_ylabel('Curvature', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
    
        # Mark the positive curvature regions that meet the threshold
        if len(candidate_indices) > 0:
            # Only show candidate points if we have more than one elbow point
            if len(elbow_points) > 1:
                ax2.scatter(x_pred[candidate_indices], smooth_second_derivative[candidate_indices], 
                           c='lightgreen', alpha=0.3, s=20, marker='o')
    
        # Define colors and markers for identified elbow points
        markers = ['o', 's', 'd', '^']  # Different marker shapes
        colors = ['darkred', 'darkorange', 'darkgreen', 'darkblue']
    
        # Plot vertical lines for all elbow points
        for i, (idx, cluster_count, curvature) in enumerate(elbow_points):
            color_idx = i % len(colors)
            marker_idx = i % len(markers)
            x_val = x_pred[idx, 0]
    
            # Plot vertical line at elbow point
            plt.axvline(x=x_val, color=colors[color_idx], linestyle='--', alpha=0.4)
    
            # Mark the point on the curve
            plt.scatter([x_val], [y_pred[idx]], 
                        c=colors[color_idx], 
                        s=100, 
                        marker=markers[marker_idx],
                        label=f'Elbow point: {cluster_count} clusters', 
                        zorder=10-i)
    
            # Add small annotation with cluster count
            plt.annotate(f'{cluster_count}', 
                        (x_val, y_pred[idx]), 
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9,
                        fontweight='bold')
    
        # Customize the plot
        ax1.set_xlabel('Number of Clusters', fontsize=12)
        ax1.set_ylabel('Separability', fontsize=12)
        plt.title('Elbow Point Analysis for Cluster Selection', fontsize=14, pad=20)
    
        # Set axis limits with padding
        plt.xlim(x_values.min() - 0.5, x_values.max() + 0.5)
        y_min = max(0, np.min(y_values) - 0.1)
        y_max = np.max(y_values) + 0.1
        plt.ylim(y_min, y_max)
    
        # Add grid and legend
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(np.arange(1, len(x_values) + 1, max(1, len(x_values) // 20)))
        plt.legend(loc='upper right', fontsize=10)
    
        # Adjust layout
        plt.tight_layout()
    
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
        plt.show()
    
        # Print results
        if elbow_points:
            print(f"Main elbow points identified (clusters): {[count for _, count, _ in elbow_points]}")
        else:
            print("No clear elbow points identified.")
    
        # Return all identified elbow points as cluster counts
        return [count for _, count, _ in elbow_points]

    def save_clustering(self, file_path: str) -> None:
        """
        Saves the clustering results to a file.
        
        This method saves the ClusterDC object including all clustering
        results, parameters, and cached computations.
        
        Parameters:
            file_path (str): Path where the clustering should be saved
                             .cdc extension recommended
        
        Returns:
            None
            
        Raises:
            RuntimeError: If clustering has not been performed
            Exception: If there are errors during saving
        """
        import pickle
        
        if self.cluster_assignments is None:
            raise RuntimeError("Clustering has not been performed yet. Run clustering first.")
            
        try:
            # Create directory if it doesn't exist
            save_dir = Path(file_path).parent
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the clustering results
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
                
            print(f"Clustering results saved successfully to {file_path}")
            
        except Exception as e:
            print(f"Error saving clustering results: {str(e)}")
            raise
    
    @classmethod
    def load_clustering(cls, file_path: str):
        """
        Loads previously saved clustering results from a file.
        
        This static method loads a ClusterDC object with all its state
        from a previously saved results file.
        
        Parameters:
            file_path (str): Path to the saved results file
        
        Returns:
            ClusterDC: Loaded ClusterDC object with restored state
            
        Raises:
            FileNotFoundError: If the file does not exist
            Exception: If there are errors during loading
        """
        import pickle
        
        try:
            with open(file_path, 'rb') as f:
                loaded_clustering = pickle.load(f)
                
            print(f"Clustering results loaded successfully from {file_path}")
            return loaded_clustering
            
        except FileNotFoundError:
            print(f"Results file not found: {file_path}")
            raise
        except Exception as e:
            print(f"Error loading clustering results: {str(e)}")
            raise