import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from skopt import gp_minimize
from skopt.space import Integer, Categorical, Real
from skopt.utils import use_named_args
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.linalg import inv, det
from collections import namedtuple
import pandas as pd
from pathlib import Path

OptimizationResult = namedtuple(
    "OptimizationResult",
    ["method_used", "kernel_type", "k", "scaling_factor"]
)

class KDE:
    """
    Kernel Density Estimation (KDE) with Adaptive and Global Bandwidth Selection.
    
    This class implements Kernel Density Estimation with support for both adaptive 
    (local) and global bandwidth selection methods. It includes:
    - Multiple kernel functions (gaussian, epanechnikov, laplacian)
    - Automatic bandwidth selection using Bayesian optimization
    - Rule-of-thumb bandwidth methods (Scott's rule, Silverman's rule)
    - Visualization tools for density estimation and optimization progress
    
    The class supports both univariate and multivariate data, with specialized 
    optimizations for 2D visualizations. It uses Bayesian optimization to automatically 
    select optimal parameters for bandwidth and kernel type.
    
    Key Features:
    - Adaptive local bandwidth based on k-nearest neighbors
    - Global bandwidth with anisotropic covariance estimation
    - Multiple kernel functions for different types of data
    - Built-in visualization and optimization monitoring
    - Parallel processing for performance optimization
    """

    def __init__(self, data, columns=None, k_limits=(1, 40), kernel_types=None, n_iter=50, save_path=None):
        """
        Initialize the KDE class.

        Parameters:
            data: Input dataset in one of these formats:
                - pd.DataFrame: DataFrame with numeric columns
                - np.ndarray: Array with shape (n_samples, n_features)
            columns (list): the two columns that will be used to estimate KDE
            k_limits (tuple): Range of neighbors (min%, max%) for local bandwidth as percentage of data points
            kernel_types (list): List of kernel functions. Options: ['gaussian', 'epanechnikov', 'laplacian']
            n_iter (int): Number of iterations for Bayesian optimization
            save_path (str, optional): Path to save visualizations

        Returns:
            None
        """
        # Store the original data
        self.original_data = data
        
        # Handle DataFrame input
        if isinstance(data, pd.DataFrame):
            # If no columns specified, use first two numeric columns
            if columns is None:
                numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) < 2:
                    raise ValueError("DataFrame must contain at least two numeric columns")
                columns = numeric_cols[:2].tolist()
            
            # Check if columns exist and are numeric
            for col in columns:
                if col not in data.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame")
                if not pd.api.types.is_numeric_dtype(data[col]):
                    raise ValueError(f"Column '{col}' is not numeric")
            
            # Store the selected columns
            self.columns = columns
            
            # Extract data for KDE analysis
            self.data = data[columns].to_numpy()
            self.column_names = columns
        else:
            # Handle numpy array input
            if not isinstance(data, np.ndarray):
                raise ValueError("Input data must be either pandas DataFrame or numpy array")
            
            # If no columns specified, use first two columns
            if columns is None:
                if data.shape[1] < 2:
                    raise ValueError("Array must have at least two columns")
                columns = [0, 1]
            
            # Extract data for KDE analysis
            self.data = data[:, columns]
            self.column_names = [f"Feature_{i}" for i in columns]
            self.columns = columns
            
        self.k_limits = (int(len(data) * (k_limits[0] / 100)), int(len(data) * (k_limits[1] / 100)))
        self.kernel_types = kernel_types or ['gaussian', 'epanechnikov', 'laplacian']
        self.n_iter = n_iter
        self.best_kernel = None
        self.best_bandwidth = None
        self.best_scaling_factor = 1.0
        self.best_k = None
        self.method_used = None
        self.point_densities = None
        self.grid_densities = None
        self.iteration_results = None
        self.save_path = save_path
        self.optimization_progress = []
        
        # Add storage for grid calculation results
        self.stored_grid = None
        self.stored_grid_resolution = None
        self.stored_grid_buffer = None
        self.stored_grid_boundaries = None

    @staticmethod
    def _kernel_function(diff, inv_bandwidth, kernel_type):
        """
        Compute kernel density values using specified kernel type.

        Parameters:
            diff (np.ndarray): Difference matrix between points, shape (n_samples, n_features)
            inv_bandwidth (np.ndarray): Inverse of bandwidth matrix
            kernel_type (str): Type of kernel ('gaussian', 'epanechnikov', 'laplacian')

        Returns:
            np.ndarray: Kernel density values for each point
        """
        if diff.ndim == 1:
            diff = diff.reshape(1, -1)

        d = diff.shape[1]
        quadratic_form = np.einsum('ij,ji->i', np.dot(diff, inv_bandwidth), diff.T)

        if kernel_type == 'gaussian':
            coefficient = 1 / (np.sqrt((2 * np.pi) ** d * det(inv_bandwidth)))
            return coefficient * np.exp(-0.5 * quadratic_form)

        elif kernel_type == 'epanechnikov':
            valid_mask = quadratic_form <= 1
            coefficient = (d + 2) / (2 * np.pi * det(inv_bandwidth) ** 0.5)
            return np.where(valid_mask, coefficient * (1 - quadratic_form), 0)

        elif kernel_type == 'laplacian':
            coefficient = 1 / (2 * np.sqrt((2 * np.pi) ** d * det(inv_bandwidth)))
            return coefficient * np.exp(-np.sqrt(quadratic_form))

        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

    def _calculate_local_bandwidths(self, k):
        """
        Calculate local bandwidth matrices for each point using k-nearest neighbors.

        Parameters:
            k (int): Number of nearest neighbors to use

        Returns:
            list: List of local bandwidth matrices for each point
        """
        nbrs = NearestNeighbors(n_neighbors=k).fit(self.data)
        _, indices = nbrs.kneighbors(self.data)

        bandwidths = []
        for i in range(len(self.data)):
            local_points = self.data[indices[i]]
            local_cov = np.cov(local_points, rowvar=False) + np.eye(local_points.shape[1]) * 1e-8
            local_cov /= np.trace(local_cov)
            bandwidths.append(local_cov * self.best_scaling_factor)

        return bandwidths

    def _calculate_global_bandwidth(self, method=None):
        """
        Calculate global bandwidth matrix using specified method.

        Parameters:
            method (str, optional): Method for bandwidth calculation ('scott', 'silverman', or None)

        Returns:
            np.ndarray: Global bandwidth matrix
        """
        # Calculate covariance with stability check
        covariance_matrix = np.cov(self.data, rowvar=False)

        # Add small constant to diagonal for numerical stability
        covariance_matrix += np.eye(covariance_matrix.shape[0]) * 1e-10

        if method in ['scott', 'silverman']:
            n, d = self.data.shape

            # Standard deviation scaling
            std_dev = np.sqrt(np.diag(covariance_matrix))
            iqr = np.array([np.percentile(self.data[:, i], 75) - 
                            np.percentile(self.data[:, i], 25) 
                            for i in range(d)])

            # Use minimum of std and IQR/1.34 (robust estimate)
            sigma = np.minimum(std_dev, iqr/1.34)

            if method == 'scott':
                # Scott's rule with robust scaling
                scaling_factor = n ** (-1 / (d + 4))
            elif method == 'silverman':
                # Silverman's rule with robust scaling
                scaling_factor = (4 / (d + 2)) ** (1 / (d + 4)) * n ** (-1 / (d + 4))

            self.best_scaling_factor = scaling_factor

            # Scale covariance by robust estimates
            bandwidth_matrix = np.diag(sigma) @ \
                             (covariance_matrix / np.outer(std_dev, std_dev)) @ \
                             np.diag(sigma) * scaling_factor

        else:
            bandwidth_matrix = covariance_matrix * self.best_scaling_factor

        return bandwidth_matrix

    def _determine_optimal_batch_size(self, data_shape, safety_factor=0.05):
        """
        Determine optimal batch size based on available system memory.

        Parameters:
            data_shape: Shape of the full data array
            safety_factor: Fraction of available memory to leave as buffer (0.0-1.0)

        Returns:
            int: Optimal batch size
        """
        try:
            import psutil

            # Get available memory in bytes
            available_memory = psutil.virtual_memory().available

            # Apply safety factor to leave some memory free
            usable_memory = available_memory * (1 - safety_factor)

            # Calculate memory per row in the diff calculation
            # For each row, we need to store differences with all data points
            # Each difference is a 2D point (2 float64 values)
            bytes_per_float = 8  # float64 = 8 bytes
            n_points = data_shape[0]
            n_dims = data_shape[1]
            memory_per_row = n_points * n_dims * bytes_per_float

            # Calculate max batch size that fits in usable memory
            max_batch_size = int(usable_memory / memory_per_row)

            # Ensure batch size is at least 1 and no more than the dataset size
            batch_size = max(1, min(max_batch_size, n_points))

            # Round to a nice number for better cache performance
            if batch_size > 1000:
                batch_size = round(batch_size / 1000) * 1000
            elif batch_size > 100:
                batch_size = round(batch_size / 100) * 100
            elif batch_size > 10:
                batch_size = round(batch_size / 10) * 10

            return batch_size

        except (ImportError, AttributeError):
            # Fallback if psutil is not available or fails
            # A conservative default batch size
            return min(1000, data_shape[0])

    def _fit_kde(self, bandwidths, kernel_type, adaptive):
        """
        Fit KDE model using specified bandwidths and kernel. Optimized for memory
        efficiency by processing data in batches to avoid large intermediate arrays.

        Parameters:
            bandwidths (Union[np.ndarray, List[np.ndarray]]): Bandwidth matrix(ces) to use
            kernel_type (str): Type of kernel function
            adaptive (bool): Whether to use adaptive (local) bandwidths

        Returns:
            np.ndarray: Density estimates for each point
        """
        # Determine optimal batch size based on available memory
        batch_size = self._determine_optimal_batch_size(self.data.shape)

        # Initialize density array
        densities = np.zeros(len(self.data))

        # Process data in batches
        for i in range(0, len(self.data), batch_size):
            end_idx = min(i + batch_size, len(self.data))
            current_batch = slice(i, end_idx)

            # Get current batch of points
            batch_points = self.data[current_batch]

            if adaptive:
                # For adaptive bandwidth, process each point individually 
                # but vectorize the calculation against all data points
                for j, point_idx in enumerate(range(i, end_idx)):
                    point = batch_points[j]
                    diff = point.reshape(1, -1) - self.data  # shape (n_data, n_dims)

                    # Get the bandwidth for this point
                    inv_bandwidth = inv(bandwidths[point_idx])
                    d = diff.shape[1]  # dimensionality

                    # Calculate quadratic form for all data points
                    quad_form = np.sum(diff @ inv_bandwidth * diff, axis=1)

                    if kernel_type == 'gaussian':
                        coefficient = 1 / (np.sqrt((2 * np.pi) ** d * det(inv_bandwidth)))
                        kernel_values = coefficient * np.exp(-0.5 * quad_form)

                    elif kernel_type == 'epanechnikov':
                        coefficient = (d + 2) / (2 * np.pi * det(inv_bandwidth) ** 0.5)
                        kernel_values = coefficient * np.where(quad_form <= 1, (1 - quad_form), 0)

                    elif kernel_type == 'laplacian':
                        coefficient = 1 / (2 * np.sqrt((2 * np.pi) ** d * det(inv_bandwidth)))
                        kernel_values = coefficient * np.exp(-np.sqrt(quad_form))

                    else:
                        # Fallback for any other kernel types
                        kernel_values = np.array([
                            self._kernel_function(point - self.data[k], inv_bandwidth, kernel_type) 
                            for k in range(len(self.data))
                        ])

                    densities[point_idx] = np.mean(kernel_values)

            else:
                # For global bandwidth
                inv_bandwidth = inv(bandwidths)
                d = self.data.shape[1]  # dimensionality

                # Pre-compute coefficients based on kernel type
                if kernel_type == 'gaussian':
                    coefficient = 1 / (np.sqrt((2 * np.pi) ** d * det(inv_bandwidth)))
                elif kernel_type == 'epanechnikov':
                    coefficient = (d + 2) / (2 * np.pi * det(inv_bandwidth) ** 0.5)
                elif kernel_type == 'laplacian':
                    coefficient = 1 / (2 * np.sqrt((2 * np.pi) ** d * det(inv_bandwidth)))

                # Process each point in the batch
                for j, point_idx in enumerate(range(i, end_idx)):
                    point = batch_points[j]
                    diff = point.reshape(1, -1) - self.data  # shape (n_data, n_dims)

                    # Compute quadratic form for all differences at once
                    quad_form = np.sum(diff @ inv_bandwidth * diff, axis=1)

                    if kernel_type == 'gaussian':
                        kernel_values = coefficient * np.exp(-0.5 * quad_form)

                    elif kernel_type == 'epanechnikov':
                        kernel_values = coefficient * np.where(quad_form <= 1, (1 - quad_form), 0)

                    elif kernel_type == 'laplacian':
                        kernel_values = coefficient * np.exp(-np.sqrt(quad_form))

                    else:
                        # Use standard kernel function for other types
                        kernel_values = np.array([
                            self._kernel_function(point - self.data[k], inv_bandwidth, kernel_type)
                            for k in range(len(self.data))
                        ])

                    densities[point_idx] = np.mean(kernel_values)

        return densities

    def _bayesian_optimization(self, use_global):
        """
        Optimize KDE parameters using Bayesian optimization.

        Parameters:
            use_global (bool): Whether to use global bandwidth optimization

        Returns:
            tuple: (best_parameters, best_objective_value)
                - best_parameters: List of optimized parameter values
                - best_objective_value: Final objective function value
        """
        # Ensure kernel_types is not empty
        if not self.kernel_types:
            self.kernel_types = ['gaussian']  # Default to gaussian if empty

        if use_global:
            search_space = [
                Categorical(list(range(len(self.kernel_types))), name='kernel'),
                Real(0.1, 2.0, name='scaling_factor')
            ]
            n_iter = self.n_iter  # Use full number of iterations for global bandwidth
        else:
            search_space = [
                Integer(self.k_limits[0], self.k_limits[1], name='k'),
                Categorical(list(range(len(self.kernel_types))), name='kernel'),
                Real(0.1, 2.0, name='scaling_factor')
            ]
            # For local bandwidth, limit iterations based on k range
            n_iter = min(self.n_iter, self.k_limits[1] - self.k_limits[0] + 1)

        @use_named_args(search_space)
        def objective(**params):
            kernel_index = params['kernel']
            kernel_type = self.kernel_types[kernel_index]
            self.best_scaling_factor = params.get('scaling_factor', 1.0)

            try:
                if use_global:
                    global_bandwidth = self._calculate_global_bandwidth()
                    densities = self._fit_kde(global_bandwidth, kernel_type, adaptive=False)
                else:
                    local_bandwidths = self._calculate_local_bandwidths(k=int(params['k']))
                    densities = self._fit_kde(local_bandwidths, kernel_type, adaptive=True)

                return -np.mean(np.log(np.maximum(densities, 1e-10)))
            except ValueError:
                return np.inf

        with tqdm(total=n_iter, desc="Bayesian Optimization") as pbar:
            def tqdm_callback(res):
                pbar.update(1)
                current_iter = len(self.optimization_progress) + 1
                self.optimization_progress.append({
                    'iteration': current_iter,
                    'best_objective': -np.min(res.func_vals),
                    'current_objective': -res.func_vals[-1]
                })

            result = gp_minimize(
                func=objective,
                dimensions=search_space,
                n_calls=n_iter,
                random_state=42,
                callback=[tqdm_callback],
                n_random_starts=n_iter
            )

        if use_global:
            self.best_k = None
            self.best_kernel = self.kernel_types[result.x[0]]
            self.best_scaling_factor = result.x[1]
        else:
            self.best_k = result.x[0]
            self.best_kernel = self.kernel_types[result.x[1]]
            self.best_scaling_factor = result.x[2]

        self.optimization_result = OptimizationResult(
            method_used="global_bandwidth" if use_global else "local_bandwidth",
            kernel_type=self.best_kernel,
            k=self.best_k,
            scaling_factor=self.best_scaling_factor
        )

        self._log_iterations(result)

        return result.x, -result.fun

    def _log_iterations(self, result):
        """
        Log parameters and objective values for each optimization iteration.

        Parameters:
            result: Optimization result object from gp_minimize

        Returns:
            None
        """
        iteration_data = []

        if self.method_used == 'global_bandwidth':
            for i, (params, value) in enumerate(zip(result.x_iters, result.func_vals)):
                kernel_index, scaling_factor = params
                kernel_type = self.kernel_types[kernel_index]
                iteration_data.append({
                    'iteration': i + 1,
                    'kernel': kernel_type,
                    'scaling_factor': scaling_factor,
                    'objective_value': value
                })
        else:
            for i, (params, value) in enumerate(zip(result.x_iters, result.func_vals)):
                k, kernel_index, scaling_factor = params
                kernel_type = self.kernel_types[kernel_index]
                iteration_data.append({
                    'iteration': i + 1,
                    'k': k,
                    'kernel': kernel_type,
                    'scaling_factor': scaling_factor,
                    'objective_value': value
                })

        self.iteration_results = pd.DataFrame(iteration_data)
        self.iteration_results = self.iteration_results.sort_values(by='objective_value', ascending=True).reset_index(drop=True)

    def fit(self, method='local_bandwidth'):
        """
        Fit the KDE model using specified method.

        Parameters:
            method (str): Method to use ('local_bandwidth', 'global_bandwidth', 'scott', 'silverman')

        Returns:
            None
        """
        self.method_used = method

        if method == 'local_bandwidth':
            best_params, _ = self._bayesian_optimization(use_global=False)
            local_bandwidths = self._calculate_local_bandwidths(k=int(best_params[0]))
            kernel_type = self.kernel_types[best_params[1]]
            self.point_densities = self._fit_kde(local_bandwidths, kernel_type, adaptive=True)
            self.best_kernel = kernel_type
            self.best_bandwidth = local_bandwidths

        elif method == 'global_bandwidth':
            best_params, _ = self._bayesian_optimization(use_global=True)
            global_bandwidth = self._calculate_global_bandwidth()
            kernel_type = self.kernel_types[best_params[0]]
            self.point_densities = self._fit_kde(global_bandwidth, kernel_type, adaptive=False)
            self.best_kernel = kernel_type
            self.best_bandwidth = global_bandwidth

        elif method in ['scott', 'silverman']:
            global_bandwidth = self._calculate_global_bandwidth(method=method)
            self.point_densities = self._fit_kde(global_bandwidth, kernel_type='gaussian', adaptive=False)
            self.best_kernel = 'gaussian'
            self.best_bandwidth = global_bandwidth

            self.optimization_result = OptimizationResult(
                method_used=method,
                kernel_type=self.best_kernel,
                k=None,
                scaling_factor=self.best_scaling_factor
            )
        else:
            raise ValueError("Invalid method. Choose from 'local_bandwidth', 'global_bandwidth', 'scott', or 'silverman'.")
       
        # Add density values to original dataframe if it's a pandas DataFrame
        if isinstance(self.original_data, pd.DataFrame):
            self.original_data['kde_density'] = self.point_densities

    def calculate_grid_with_margin(self, resolution=50, buffer_fraction=0.05):
        """
        Calculate grid points with margin for density estimation.

        Parameters:
            resolution (int): Number of points along each axis for grid
            buffer_fraction (float): Fraction of data range to add as buffer

        Returns:
            tuple: (xx, yy, boundaries)
                - xx (np.ndarray): X coordinates of grid points
                - yy (np.ndarray): Y coordinates of grid points
                - boundaries (tuple): (x_min, x_max, y_min, y_max)
        """
        if (self.stored_grid is not None and 
            self.stored_grid_resolution == resolution and 
            self.stored_grid_buffer == buffer_fraction):
            return self.stored_grid[0], self.stored_grid[1], self.stored_grid_boundaries

        x_min, x_max = self.data[:, 0].min(), self.data[:, 0].max()
        y_min, y_max = self.data[:, 1].min(), self.data[:, 1].max()

        x_range = x_max - x_min
        y_range = y_max - y_min

        if self.best_bandwidth is not None:
            if isinstance(self.best_bandwidth, (int, float)):
                h_x = h_y = self.best_bandwidth
            elif isinstance(self.best_bandwidth, np.ndarray):
                h_x, h_y = np.sqrt(np.diag(self.best_bandwidth))
            elif isinstance(self.best_bandwidth, list):
                h_x = max(np.sqrt(np.diag(bw)[0]) for bw in self.best_bandwidth)
                h_y = max(np.sqrt(np.diag(bw)[1]) for bw in self.best_bandwidth)
            else:
                raise ValueError("Unsupported bandwidth type.")
        else:
            h_x = h_y = 0

        margin_x = 3 * h_x + buffer_fraction * x_range
        margin_y = 3 * h_y + buffer_fraction * y_range

        x_min -= margin_x
        x_max += margin_x
        y_min -= margin_y
        y_max += margin_y

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )

        self.stored_grid = (xx, yy)
        self.stored_grid_resolution = resolution
        self.stored_grid_buffer = buffer_fraction
        self.stored_grid_boundaries = (x_min, x_max, y_min, y_max)

        return xx, yy, (x_min, x_max, y_min, y_max)

    def calculate_grid_densities(self, resolution=50, buffer_fraction=0.05):
        """
        Compute density estimates on a grid with vectorized operations for speed.
        Optimized for all kernel types with progress bar support.

        Parameters:
            resolution (int): Number of points along each axis for grid
            buffer_fraction (float): Fraction of data range to add as buffer

        Returns:
            np.ndarray: Grid of density estimates, shape (resolution, resolution)
        """
        if self.best_kernel is None or self.best_bandwidth is None:
            raise RuntimeError("Model must be fit before calculating grid densities.")

        # Get grid coordinates
        xx, yy, _ = self.calculate_grid_with_margin(resolution=resolution, buffer_fraction=buffer_fraction)
        grid_points = np.column_stack((xx.ravel(), yy.ravel()))

        # Set up progress bar
        pbar = tqdm(total=len(grid_points), desc="Estimating grid density values", unit="point")

        if isinstance(self.best_bandwidth, list):
            # For local bandwidth - process in batches for memory efficiency
            batch_size = 100  # Adjust based on available memory
            densities = []

            for i in range(0, len(grid_points), batch_size):
                batch = grid_points[i:i + batch_size]
                batch_densities = []

                # Reshape for broadcasting
                batch_points = batch.reshape(-1, 1, 2)  # (batch, 1, 2)
                data_points = self.data.reshape(1, -1, 2)  # (1, n_data, 2)

                # Calculate differences for all points at once
                diffs = batch_points - data_points  # (batch, n_data, 2)

                for j in range(len(self.data)):
                    diff = diffs[:, j, :]  # (batch, 2)
                    inv_bandwidth = inv(self.best_bandwidth[j])

                    # Compute quadratic form for all differences at once
                    quad_form = np.einsum('bi,ij,bj->b', diff, inv_bandwidth, diff)

                    if self.best_kernel == 'gaussian':
                        coefficient = 1 / (np.sqrt((2 * np.pi) ** 2 * det(inv_bandwidth)))
                        kernel_values = coefficient * np.exp(-0.5 * quad_form)

                    elif self.best_kernel == 'epanechnikov':
                        coefficient = (2 + 2) / (2 * np.pi * det(inv_bandwidth) ** 0.5)  # d=2 for 2D data
                        kernel_values = coefficient * np.where(quad_form <= 1, (1 - quad_form), 0)

                    elif self.best_kernel == 'laplacian':
                        coefficient = 1 / (2 * np.sqrt((2 * np.pi) ** 2 * det(inv_bandwidth)))
                        kernel_values = coefficient * np.exp(-np.sqrt(quad_form))

                    else:
                        # Use standard kernel function for other types
                        kernel_values = np.array([
                            self._kernel_function(d, inv_bandwidth, self.best_kernel)
                            for d in diff
                        ])

                    batch_densities.append(kernel_values)

                # Average across all data points
                batch_result = np.mean(np.array(batch_densities), axis=0)
                densities.extend(batch_result)
                pbar.update(len(batch))

        else:
            # For global bandwidth - fully vectorized
            inv_bandwidth = inv(self.best_bandwidth)
            batch_size = 100
            densities = []
            d = 2  # dimensionality (2D data)

            # Pre-compute coefficients based on kernel type
            if self.best_kernel == 'gaussian':
                coefficient = 1 / (np.sqrt((2 * np.pi) ** d * det(inv_bandwidth)))
            elif self.best_kernel == 'epanechnikov':
                coefficient = (d + 2) / (2 * np.pi * det(inv_bandwidth) ** 0.5)
            elif self.best_kernel == 'laplacian':
                coefficient = 1 / (2 * np.sqrt((2 * np.pi) ** d * det(inv_bandwidth)))

            for i in range(0, len(grid_points), batch_size):
                batch = grid_points[i:i + batch_size]

                # Reshape for broadcasting
                batch_points = batch.reshape(-1, 1, 2)  # (batch, 1, 2)
                data_points = self.data.reshape(1, -1, 2)  # (1, n_data, 2)

                # Calculate all differences at once
                diffs = batch_points - data_points  # (batch, n_data, 2)

                # Initialize array for kernel values
                kernel_values = np.zeros((len(batch), len(self.data)))

                for b in range(len(batch)):
                    # Compute quadratic form for each grid point against all data points
                    quad_form = np.einsum('ni,ij,nj->n', 
                                        diffs[b], inv_bandwidth, diffs[b])

                    if self.best_kernel == 'gaussian':
                        kernel_values[b] = coefficient * np.exp(-0.5 * quad_form)

                    elif self.best_kernel == 'epanechnikov':
                        kernel_values[b] = coefficient * np.where(quad_form <= 1, (1 - quad_form), 0)

                    elif self.best_kernel == 'laplacian':
                        kernel_values[b] = coefficient * np.exp(-np.sqrt(quad_form))

                    else:
                        # Use standard kernel function for other types
                        kernel_values[b] = np.array([
                            self._kernel_function(d, inv_bandwidth, self.best_kernel)
                            for d in diffs[b]
                        ])

                batch_result = np.mean(kernel_values, axis=1)
                densities.extend(batch_result)
                pbar.update(len(batch))

        pbar.close()

        # Reshape to grid - same format as original
        grid_densities = np.array(densities).reshape(xx.shape)
        self.grid_densities = grid_densities
        self.stored_grid_resolution = resolution
        self.stored_grid_buffer = buffer_fraction

        return grid_densities

    def _process_point_local(self, point, bandwidths, kernel_type, data):
        """Helper function for parallel processing with local bandwidth."""
        return np.mean([
            self._kernel_function(point - data[i], inv(bandwidths[i]), kernel_type)
            for i in range(len(data))
        ])

    def _process_point_global(self, point, inv_bandwidth, kernel_type, data):
        """Helper function for parallel processing with global bandwidth."""
        return np.mean(self._kernel_function(data - point, inv_bandwidth, kernel_type))

    def plot_results(self, vmin=0, resolution=50, buffer_fraction=0.05):
        """
        Visualize KDE results with point and density plots.
        Parameters:
            vmin (float): Minimum density value for visualization
            resolution (int): Number of points along each axis for grid
            buffer_fraction (float): Fraction of data range to add as buffer
        Returns:
            None
        """
        if self.point_densities is None:
            raise RuntimeError("Model must be fit before plotting results.")
        # Compute grid densities
        grid_densities = self.get_grid_densities(resolution=resolution, buffer_fraction=buffer_fraction)
        xx, yy, (x_min, x_max, y_min, y_max) = self.calculate_grid_with_margin(
            resolution=resolution, buffer_fraction=buffer_fraction
        )
        point_densities = self.get_point_densities()
        # Create figure with two subplots
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        # Method display mapping
        method_display = {
            'global_bandwidth': 'Global Bandwidth',
            'local_bandwidth': 'Local Bandwidth',
            'scott': "Scott's Rule Bandwidth",
            'silverman': "Silverman's Rule Bandwidth"
        }.get(self.method_used, self.method_used)
        # Construct detailed method information
        method_info = method_display
        if self.best_kernel:
            method_info += f", {self.best_kernel} kernel"
        if hasattr(self, 'best_k') and self.best_k:
            method_info += f", k={self.best_k}"
        if hasattr(self, 'best_scaling_factor') and self.best_scaling_factor is not None:
            method_info += f", scaling factor={self.best_scaling_factor:.2f}"
        # Point density plot - use vmin parameter
        point_max = np.max(point_densities)
        sc = ax[0].scatter(self.data[:, 0], self.data[:, 1], 
                          c=point_densities, 
                          cmap='viridis', 
                          s=10,
                          vmin=vmin,
                          vmax=point_max)
        plt.colorbar(sc, ax=ax[0], label='KDE')
        ax[0].set_title(f'Point Densities\n({method_info})')
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Y')
        ax[0].set_xlim(x_min, x_max)
        ax[0].set_ylim(y_min, y_max)
        # Robust contour level generation - use vmin parameter
        grid_min = max(vmin, np.min(grid_densities))
        grid_max = np.max(grid_densities)
        # Ensure unique, increasing levels
        if grid_min == grid_max:
            # If all densities are the same, create a small range
            levels = [grid_min, grid_min * 1.1]
        else:
            # Generate levels ensuring they are strictly increasing
            levels = np.unique(np.linspace(grid_min, grid_max, num=20))
            # Add a small epsilon to ensure levels are strictly increasing
            if len(levels) < 2:
                levels = [grid_min, grid_min * 1.1]
            elif len(levels) == 2 and levels[0] == levels[1]:
                levels[1] = levels[0] * 1.1
        # Contour plot with robust level generation
        contour = ax[1].contourf(xx, yy, grid_densities, 
                                levels=levels,
                                cmap='viridis', 
                                alpha=0.6)
        plt.colorbar(contour, ax=ax[1], label='KDE')
        ax[1].set_title('Density Contours')
        ax[1].set_xlabel('X')
        ax[1].set_ylabel('Y')
        ax[1].set_xlim(x_min, x_max)
        ax[1].set_ylim(y_min, y_max)
        plt.tight_layout()
        # Save figure if save path is specified
        if self.save_path:
            plt.savefig(self.save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_optimization_progress(self, save_path=None):
        """
        Plot the optimization progress over iterations.

        Parameters:
            save_path (str, optional): Path to save the plot

        Returns:
            None
        """
        if not self.optimization_progress:
            raise RuntimeError("No optimization progress available. Run fit() first.")

        plt.figure(figsize=(12, 8))

        iterations = [prog['iteration'] for prog in self.optimization_progress]
        best_objectives = [prog['best_objective'] for prog in self.optimization_progress]
        current_objectives = [prog['current_objective'] for prog in self.optimization_progress]

        plt.plot(iterations, best_objectives, 'b-', label='Best value')
        plt.scatter(iterations, current_objectives, c='red', alpha=0.3, s=30, label='Evaluated points')

        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')

        method = "Global" if self.method_used == "global_bandwidth" else "Local"
        final_score = best_objectives[-1]
        plt.title(f'{method} Bandwidth Optimization Progress\n'
                 f'Final parameters: {self.best_kernel} kernel'
                 f'{", k=" + str(self.best_k) if self.best_k else ""}'
                 f', scaling={self.best_scaling_factor:.3f}\n'
                 f'Final objective value: {final_score:.4f}')

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='lower right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def get_point_densities(self):
        """
        Get density estimates for input points.

        Parameters:
            None

        Returns:
            np.ndarray: Density estimates for each input point
        """
        return self.point_densities

    def get_grid_densities(self, resolution=50, buffer_fraction=0.05):
        """
        Get or compute density estimates on a grid.

        Parameters:
            resolution (int): Number of points along each axis for grid
            buffer_fraction (float): Fraction of data range to add as buffer

        Returns:
            np.ndarray: Grid of density estimates
        """
        if self.grid_densities is None:
            self.calculate_grid_densities(resolution=resolution, buffer_fraction=buffer_fraction)
        return self.grid_densities

    def get_optimization_report(self):
        """
        Get report of optimization parameters.

        Parameters:
            None

        Returns:
            OptimizationResult: Named tuple containing optimization results
        """
        if self.optimization_result is None:
            raise RuntimeError("Model must be fit before generating a report.")
        return self.optimization_result
    
    def print_optimization_report(self):
        """
        Print formatted report of optimization parameters.

        Parameters:
            None

        Returns:
            None
        """
        if self.optimization_result is None:
            raise RuntimeError("Model must be fit before printing a report.")

        print("Optimization Report:")
        print(f"- Method Used: {self.optimization_result.method_used}")
        print(f"- Kernel Type: {self.optimization_result.kernel_type}")
        print(f"- k: {self.optimization_result.k}")
        print(f"- Scaling Factor: {self.optimization_result.scaling_factor:.4f}")
    
    def get_iteration_results(self):
        """
        Get DataFrame of iteration results.

        Parameters:
            None

        Returns:
            pd.DataFrame: Results from each optimization iteration
        """
        if self.iteration_results is None:
            raise RuntimeError("Iteration results are not available.")
        return self.iteration_results
    
    def save_iteration_results(self, file_path):
        """
        Save iteration results to CSV file.

        Parameters:
            file_path (str): Path to save CSV file

        Returns:
            None
        """
        if not hasattr(self, 'iteration_results'):
            raise RuntimeError("Iteration results are not available.")
        self.iteration_results.to_csv(file_path, index=False)

    def save_model(self, file_path: str) -> None:
        """
        Saves the fitted KDE model to a file.
        
        This method saves the KDE model including bandwidth parameters,
        kernel types, and cached results to avoid recomputing.
        
        Parameters:
            file_path (str): Path where the model should be saved
                             .kde extension recommended
        
        Returns:
            None
            
        Raises:
            RuntimeError: If the model has not been fitted
            Exception: If there are errors during saving
        """
        import pickle
        
        if self.point_densities is None:
            raise RuntimeError("Model must be fit before saving.")
            
        try:
            # Create directory if it doesn't exist
            save_dir = Path(file_path).parent
            if not save_dir.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the KDE model
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
                
            print(f"KDE model saved successfully to {file_path}")
            
        except Exception as e:
            print(f"Error saving KDE model: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, file_path: str):
        """
        Loads a previously saved KDE model from a file.
        
        This static method loads a KDE object with all its state
        from a previously saved model file.
        
        Parameters:
            file_path (str): Path to the saved model file
        
        Returns:
            KDE: Loaded KDE object with restored state
            
        Raises:
            FileNotFoundError: If the file does not exist
            Exception: If there are errors during loading
        """
        import pickle
        
        try:
            with open(file_path, 'rb') as f:
                loaded_model = pickle.load(f)
                
            print(f"KDE model loaded successfully from {file_path}")
            return loaded_model
            
        except FileNotFoundError:
            print(f"Model file not found: {file_path}")
            raise
        except Exception as e:
            print(f"Error loading KDE model: {str(e)}")
            raise

    @staticmethod
    def benchmark_and_predict(data_input, sample_sizes=None, target_size=None, method='scott', n_runs=3, columns=None, 
                             sample_by_percentage=False, min_sample_sizes=5, min_samples=100, allow_extrapolation=True,
                             save_path=None):
        """
        Comprehensive function that benchmarks KDE performance and predicts processing time for larger datasets.
        
        Parameters:
            data_input: Input data in one of these formats:
                - str: Path to a data file (CSV)
                - pd.DataFrame: DataFrame with numeric columns
                - Data: Data object with loaded dataframe
            sample_sizes (list): List of sample sizes to benchmark. If None:
                - If sample_by_percentage=True: Uses percentages from 10% to 90% of data
                - If sample_by_percentage=False: Uses appropriate range based on data size
            target_size (int): Size to predict processing time for. If None, uses 2x the largest sample size
            method (str): KDE bandwidth method ('scott', 'silverman', 'local_bandwidth', 'global_bandwidth')
            n_runs (int): Number of runs per sample size for averaging
            columns (list): The columns to use for KDE (only for DataFrame or Data input)
            sample_by_percentage (bool): If True, interpret sample_sizes as percentages of the data
            min_sample_sizes (int): Minimum number of different sample sizes to use (default: 5)
            min_samples (int): Minimum number of samples to use in smallest benchmark (default: 100)
            allow_extrapolation (bool): If True, allows extrapolation beyond the dataset size (default: True)
            save_path (str): Directory path to save figures. If None, figures will not be saved.
            
        Returns:
            tuple: (results_df, polynomial_function, prediction_df)
            
        Raises:
            ValueError: If not enough data for reliable benchmarking
            TypeError: If data_input is not a supported type
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import time
        import os
        from pathlib import Path
        
        # Function to round to significant digits
        def round_to_nice_number(num, max_value=None, min_value=None):
            """
            Round to a 'nice' number with few significant digits.
            Ensures result is between min_value and max_value if provided.
            """
            if num <= 0:
                return min_value if min_value is not None else 100
                
            magnitude = 10 ** (np.floor(np.log10(num)))
            
            # Round based on relative value within order of magnitude
            if num / magnitude < 1.5:
                rounded = int(np.round(num / magnitude) * magnitude)
            elif num / magnitude < 3.5:
                rounded = int(np.round(num / (magnitude/2)) * (magnitude/2))
            elif num / magnitude < 7.5:
                rounded = int(np.round(num / (magnitude/5)) * (magnitude/5))
            else:
                rounded = int(np.round(num / magnitude) * magnitude)
            
            # Apply constraints
            if max_value is not None and rounded > max_value:
                # Try several round-down options, from least to most aggressive
                options = [
                    int(np.floor(num / magnitude) * magnitude),
                    int(np.floor(num / (magnitude/2)) * (magnitude/2)),
                    int(np.floor(num / (magnitude/5)) * (magnitude/5)),
                    max_value
                ]
                # Take the largest option that doesn't exceed max_value
                rounded = max([opt for opt in options if opt <= max_value])
            
            if min_value is not None and rounded < min_value:
                rounded = min_value
                
            return rounded
        
        # Function to create output file path
        def create_figure_path(base_path, figure_type, data_size, target, method_name):
            """Create a file path for saving figures with descriptive name."""
            if base_path is None:
                return None
                
            # Create directory if it doesn't exist
            if not os.path.exists(base_path):
                os.makedirs(base_path)
                
            # Create a descriptive filename
            filename = f"kde_{figure_type}_d{data_size}_t{target}_{method_name}.png"
            return os.path.join(base_path, filename)
        
        # Part 0: Data loading and preparation
        # ====================================
        
        # Initialize dataframe based on input type
        if isinstance(data_input, str):
            # Load from file path
            file_path = Path(data_input)
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            print(f"Successfully loaded data from {file_path}")
            
        elif isinstance(data_input, pd.DataFrame):
            # Use the provided DataFrame
            df = data_input
            print("Using provided DataFrame")
            
        elif hasattr(data_input, 'data') and isinstance(data_input.data, pd.DataFrame):
            # Extract from Data object
            df = data_input.data
            print("Using DataFrame from Data object")
            
        elif hasattr(data_input, 'original_data') and isinstance(data_input.original_data, pd.DataFrame):
            # Extract from a KDE object
            df = data_input.original_data
            print("Using DataFrame from KDE object")
            
        else:
            raise TypeError("data_input must be a file path, DataFrame, Data object, or KDE object")
        
        print(f"Data shape: {df.shape}")
        
        # Select columns if specified
        if columns is not None:
            # Validate columns
            for col in columns:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame")
            # Select only the specified columns
            df = df[columns]
            print(f"Using columns: {columns}")
        
        # Determine sample sizes if not provided
        total_samples = len(df)
        
        # Check if we have enough samples for meaningful benchmarking
        if total_samples < min_samples * 2:
            raise ValueError(f"Dataset has only {total_samples} samples, which is too small for meaningful benchmarking.")
        
        # Round the total samples to nearest 100 for maximum sample size
        max_sample_rounded = int(np.floor(total_samples / 100) * 100)
        if max_sample_rounded < total_samples * 0.9:  # Make sure we don't lose too much
            max_sample_rounded = int(np.floor(total_samples / 100) * 100)
        else:
            max_sample_rounded = int(np.floor((total_samples - 100) / 100) * 100)  # Go one step lower
        
        if sample_sizes is None:
            if sample_by_percentage:
                # For small datasets, use percentages
                # Create a range from 10% to 90% of the data
                step = (90 - 10) / (min_sample_sizes - 1)
                percentages = [10 + step * i for i in range(min_sample_sizes)]
                raw_sizes = [max(min_samples, int(total_samples * p/100)) for p in percentages]
                # Round to nice numbers with constraint
                sample_sizes = [round_to_nice_number(size, max_value=max_sample_rounded, min_value=min_samples) for size in raw_sizes]
                print(f"Using {len(sample_sizes)} sample sizes based on data percentages: {[f'{p:.1f}%' for p in percentages]}")
            else:
                # Generate evenly spaced sample sizes
                # Use max_sample_rounded as the upper limit
                max_raw_size = max_sample_rounded
                step = (max_raw_size - min_samples) / (min_sample_sizes - 1)
                raw_sizes = [min_samples + step * i for i in range(min_sample_sizes)]
                # Round to nice numbers with constraint
                sample_sizes = [round_to_nice_number(size, max_value=max_sample_rounded, min_value=min_samples) for size in raw_sizes]
                print(f"Using {len(sample_sizes)} evenly distributed sample sizes: {sample_sizes}")
        elif sample_by_percentage:
            # Convert percentages to actual sample sizes and round
            raw_sizes = [max(min_samples, int(total_samples * p/100)) for p in sample_sizes]
            sample_sizes = [round_to_nice_number(size, max_value=max_sample_rounded, min_value=min_samples) for size in raw_sizes]
            print(f"Using {len(sample_sizes)} sample sizes based on specified percentages: {sample_sizes}")
        else:
            # Ensure user-provided sample sizes don't exceed total
            sample_sizes = [min(s, max_sample_rounded) for s in sample_sizes]
        
        # Ensure minimum number of sample sizes
        if len(sample_sizes) < min_sample_sizes:
            print(f"Warning: Only {len(sample_sizes)} sample sizes provided. For optimal polynomial fitting, {min_sample_sizes} are recommended.")
            print("Proceeding with the available sample sizes, but predictions may be less accurate.")
        
        # Sort sample sizes and remove duplicates
        sample_sizes = sorted(list(set(sample_sizes)))
        
        # For small datasets, ensure all sample sizes are at least min_samples and don't exceed max
        sample_sizes = [max(min_samples, min(s, max_sample_rounded)) for s in sample_sizes]
        
        # Ensure target size is a nice round number if auto-generated
        if target_size is None:
            # Use 2x the largest sample size for default target (no limit for target)
            target_size = round_to_nice_number(max(sample_sizes) * 2)
            print(f"Target size not specified. Using {target_size:,} samples (2x largest sample size).")
        elif isinstance(target_size, float):
            # Round to nice number if it's a float (no limit for target)
            target_size = round_to_nice_number(target_size)
            print(f"Rounded target size to {target_size:,} samples")
        
        # Display warning about extrapolation if target exceeds data size
        if target_size > total_samples:
            if allow_extrapolation:
                extrapolation_factor = target_size / total_samples
                print(f"Note: Target size {target_size:,} exceeds available data ({total_samples:,}).")
                print(f"Extrapolating {extrapolation_factor:.1f}x beyond dataset size.")
            else:
                print(f"Warning: Target size {target_size:,} exceeds available data ({total_samples:,}). Using maximum available: {total_samples:,}.")
                target_size = total_samples
        
        # Part 1: Benchmarking
        # ====================
        
        # Function to measure KDE processing time
        def measure_kde_time(df, n_samples, method='scott', n_runs=3, columns=None):
            """
            Measure the time needed to fit KDE with a specific sample size.
            """
            times = []
            
            for run in range(n_runs):
                # Sample the data using pandas DataFrame sample method
                df_sample = df.sample(n=n_samples, random_state=run)
                
                # Initialize KDE with appropriate columns
                kde = KDE(data=df_sample, columns=columns)
                
                # Measure time to fit
                start_time = time.time()
                kde.fit(method=method)
                end_time = time.time()
                
                times.append(end_time - start_time)
                print(f"  Run {run+1}/{n_runs}: {times[-1]:.2f} seconds")
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            print(f"  Average time for {n_samples:,} samples: {avg_time:.2f} seconds (±{std_time:.2f})")
            return avg_time, std_time
        
        # Initialize results dataframe with explicit numeric dtypes
        results = pd.DataFrame({
            'sample_size': [],
            'time_seconds': [],
            'std_deviation': []
        }, dtype=float)
        
        # Run benchmark for each sample size
        print(f"\nRunning KDE benchmark with '{method}' method...")
        for size in sample_sizes:
            print(f"\nBenchmarking with {size:,} samples:")
            time_taken, time_std = measure_kde_time(df, size, method, n_runs, columns)
            
            # Create a new row with explicit numeric types
            new_row = pd.DataFrame({
                'sample_size': [float(size)],
                'time_seconds': [float(time_taken)],
                'std_deviation': [float(time_std)]
            })
            
            results = pd.concat([results, new_row], ignore_index=True)
        
        # Part 2: Fitting & Prediction
        # ============================
        
        # Convert data to numpy arrays
        x_data = results['sample_size'].to_numpy(dtype=float)
        y_data = results['time_seconds'].to_numpy(dtype=float)
        
        # Determine polynomial degree based on number of sample points
        if len(sample_sizes) >= 5:
            degree = 2  # Quadratic for 5+ points
        else:
            degree = 1  # Linear for fewer points
            print("Warning: Using linear fit due to limited data points. Consider using more sample sizes for better accuracy.")
        
        # Fit polynomial regression
        coefficients = np.polyfit(x_data, y_data, degree)
        polynomial = np.poly1d(coefficients)
        
        # Print the fitted polynomial
        print("\n==== KDE Timing Analysis ====")
        print(f"Fitted polynomial: {polynomial}")
        
        if degree == 2:
            print(f"Formula: y = {coefficients[0]:.3e}x² + {coefficients[1]:.3e}x + {coefficients[2]:.3f}")
        else:
            print(f"Formula: y = {coefficients[0]:.3e}x + {coefficients[1]:.3f}")
        
        # Calculate R-squared
        y_mean = np.mean(y_data)
        ss_total = sum((y_data - y_mean) ** 2)
        ss_residual = sum((y_data - polynomial(x_data)) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        print(f"R-squared: {r_squared:.4f}")
        
        # Predict processing time for target size
        predicted_time = polynomial(target_size)
        print(f"\nPredicted processing time for {target_size:,} samples: {predicted_time:.2f} seconds")
        print(f"That's approximately {predicted_time/60:.1f} minutes or {predicted_time/3600:.2f} hours")
        
        # Part 3: Scaling Analysis
        # ========================
        
        print("\nScaling Factors Between Sample Sizes:")
        for i in range(1, len(sample_sizes)):
            size_ratio = float(sample_sizes[i]) / float(sample_sizes[i-1])
            time_ratio = float(results.iloc[i]['time_seconds']) / float(results.iloc[i-1]['time_seconds'])
            print(f"Size {sample_sizes[i-1]:,} → {sample_sizes[i]:,} ({size_ratio:.1f}x): Time increase {time_ratio:.2f}x")
        
        # Determine appropriate prediction sizes based on data scale
        max_benchmark = max(sample_sizes)
        
        # Generate prediction sizes - include beyond dataset size if allow_extrapolation
        prediction_sizes = []
        
        if allow_extrapolation:
            # Create rounded prediction points between max_benchmark and target_size
            # Use more points for larger extrapolations
            if target_size > max_benchmark:
                extrapolation_factor = target_size / max_benchmark
                
                if extrapolation_factor <= 2:
                    # For smaller extrapolations, use 25% steps
                    steps = [1.25, 1.5, 1.75, 2.0]
                elif extrapolation_factor <= 5:
                    # For medium extrapolations, use 1x steps
                    steps = list(range(2, int(extrapolation_factor) + 1))
                else:
                    # For large extrapolations, use logarithmic steps
                    steps = [2]
                    current = 2
                    while current < extrapolation_factor:
                        current *= 2
                        if current < extrapolation_factor:
                            steps.append(current)
                
                # Convert steps to actual sizes and round to nice numbers
                prediction_sizes = [round_to_nice_number(int(max_benchmark * step)) for step in steps]
                
                # Add target_size (already nicely rounded)
                if target_size not in prediction_sizes:
                    prediction_sizes.append(target_size)
        else:
            # Only include sizes up to total_samples
            steps = [1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]
            raw_sizes = [int(max_benchmark * step) for step in steps]
            prediction_sizes = [round_to_nice_number(size, max_value=total_samples, min_value=min_samples) 
                               for size in raw_sizes if size > max_benchmark and size <= total_samples]
        
        # Sort prediction sizes and remove duplicates
        prediction_sizes = sorted(list(set(prediction_sizes)))
        
        # Create predictions table
        prediction_data = []
        
        print("\nPredictions for various sample sizes:")
        print("Sample Size    | Predicted Time (seconds) | Minutes   | Hours")
        print("-" * 65)
        
        # Add max_benchmark as a reference point
        print(f"{max_benchmark:12,} | {polynomial(max_benchmark):24.2f} | {polynomial(max_benchmark)/60:9.2f} | {polynomial(max_benchmark)/3600:6.3f} (benchmark)")
        
        for size in prediction_sizes:
            pred_time = polynomial(size)
            minutes = pred_time / 60
            hours = minutes / 60
            
            # Add an indicator for extrapolated sizes
            indicator = " (extrapolated)" if size > total_samples else ""
            print(f"{size:12,} | {pred_time:24.2f} | {minutes:9.2f} | {hours:6.3f}{indicator}")
            
            prediction_data.append({
                'sample_size': size,
                'seconds': pred_time,
                'minutes': minutes,
                'hours': hours,
                'extrapolated': size > total_samples
            })
        
        predictions_df = pd.DataFrame(prediction_data)
        
        # Part 4: Visualization
        # =====================
        try:
            import seaborn as sns
            sns_available = True
        except ImportError:
            sns_available = False
            print("Seaborn not available. Using matplotlib for plotting.")
        
        # 1. Benchmark Results Plot
        plt.figure(figsize=(10, 6))
        if sns_available:
            sns.scatterplot(x=x_data, y=y_data, s=80, color='blue', alpha=0.7)
        else:
            plt.scatter(x_data, y_data, s=80, color='blue', alpha=0.7)
        
        # Add trend line
        x_trend = np.linspace(0, max(x_data), 100)
        plt.plot(x_trend, polynomial(x_trend), 'r--', linewidth=2)
        
        # Add annotations for the polynomial equation
        if degree == 2:
            equation = f"y = {coefficients[0]:.2e}x² + {coefficients[1]:.2e}x + {coefficients[2]:.2f}"
        else:
            equation = f"y = {coefficients[0]:.2e}x + {coefficients[1]:.2f}"
            
        plt.annotate(f"{equation}\nR² = {r_squared:.3f}", 
                    xy=(0.05, 0.9), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Add labels and title
        plt.title(f'KDE Processing Time vs. Sample Size (Method: {method})', fontsize=14)
        plt.xlabel('Sample Size', fontsize=12)
        plt.ylabel('Processing Time (seconds)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ticklabel_format(style='plain', axis='x')
        plt.ylim(bottom=0)
        plt.tight_layout()
        
        # Create filename with dataset and target size info
        benchmark_fig_path = create_figure_path(
            save_path, 
            "benchmark", 
            total_samples, 
            target_size, 
            method
        )
        if benchmark_fig_path:
            plt.savefig(benchmark_fig_path, dpi=300)
            print(f"Saved benchmark plot to: {benchmark_fig_path}")
            
        plt.show()
        
        # Only create extrapolation plot if we have predictions
        if len(prediction_sizes) > 0:
            # 2. Prediction Extrapolation Plot
            plt.figure(figsize=(12, 8))
            
            # Plot the original data points
            plt.scatter(x_data, y_data, s=100, color='blue', label='Benchmark data')
            
            # Plot the fitted curve for the range of measured data
            x_fit = np.linspace(0, max(x_data), 100)
            plt.plot(x_fit, polynomial(x_fit), 'r-', linewidth=2, label='Fitted curve')
            
            # Plot the extrapolation
            x_extrapolation = np.linspace(max(x_data), max(prediction_sizes), 100)
            plt.plot(x_extrapolation, polynomial(x_extrapolation), 'r--', linewidth=2, label='Extrapolation')
            
            # Add a vertical line at the total dataset size
            if allow_extrapolation and max(prediction_sizes) > total_samples:
                plt.axvline(x=total_samples, color='gray', linestyle='--', alpha=0.7)
                plt.annotate('Available data limit', 
                             xy=(total_samples, 0), 
                             xytext=(total_samples, -0.05 * polynomial(max(prediction_sizes))),
                             ha='center', 
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
            
            # Mark the target prediction point
            plt.scatter([target_size], [polynomial(target_size)], s=150, color='green', marker='*', 
                        label=f'Prediction: {polynomial(target_size):.1f} seconds')
            
            # Add labels and title
            plt.title('KDE Processing Time Extrapolation', fontsize=16)
            plt.xlabel('Sample Size', fontsize=14)
            plt.ylabel('Processing Time (seconds)', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            
            # Format axes
            plt.ticklabel_format(style='plain', axis='x')
            plt.ticklabel_format(style='plain', axis='y')
            
            # Add extrapolation warning if applicable
            if allow_extrapolation and target_size > total_samples:
                extrapolation_factor = target_size / total_samples
                warning_text = f"Note: Extrapolating {extrapolation_factor:.1f}x beyond available data"
                plt.annotate(warning_text, 
                            xy=(0.5, 0.01), 
                            xycoords='figure fraction',
                            ha='center',
                            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
            
            # Add text annotation for the prediction
            plt.annotate(f"Predicted time for {target_size:,} samples:\n{polynomial(target_size):.1f} seconds\n({polynomial(target_size)/60:.1f} minutes)",
                        xy=(target_size, polynomial(target_size)), 
                        xytext=(target_size*0.7, polynomial(target_size)*0.6),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                        bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.8),
                        fontsize=12)
            
            plt.tight_layout()
            
            # Save extrapolation plot
            extrapolation_fig_path = create_figure_path(
                save_path, 
                "extrapolation", 
                total_samples, 
                target_size, 
                method
            )
            if extrapolation_fig_path:
                plt.savefig(extrapolation_fig_path, dpi=300)
                print(f"Saved extrapolation plot to: {extrapolation_fig_path}")
                
            plt.show()
        
        # 3. Simple Plot of Benchmark Results
        plt.figure(figsize=(10, 6))
        plt.plot(results['sample_size'], results['time_seconds'], 'o-', linewidth=2, markersize=10)
        plt.title('KDE Processing Time vs. Sample Size', fontsize=14)
        plt.xlabel('Sample Size', fontsize=12)
        plt.ylabel('Processing Time (seconds)', fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        
        # Save simple plot
        simple_fig_path = create_figure_path(
            save_path, 
            "simple", 
            total_samples, 
            target_size, 
            method
        )
        if simple_fig_path:
            plt.savefig(simple_fig_path, dpi=300)
            print(f"Saved simple plot to: {simple_fig_path}")
            
        plt.show()
        
        # 4. Prediction Bar Chart (only if we have predictions)
        if len(prediction_sizes) > 0:
            plt.figure(figsize=(12, 8))
            
            # Create bar chart including the last benchmark and predictions
            display_sizes = [max(sample_sizes)] + prediction_sizes
            display_times = [polynomial(size) for size in display_sizes]
            
            # Create a list of colors to distinguish benchmark, in-range predictions and extrapolated predictions
            colors = ['blue']  # Benchmark is blue
            for size in prediction_sizes:
                if size > total_samples:
                    colors.append('red')  # Extrapolated predictions in red
                else:
                    colors.append('skyblue')  # In-range predictions in skyblue
            
            bars = plt.bar([f"{size:,}" for size in display_sizes], 
                           display_times, 
                           color=colors)
            
            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                minutes = height / 60
                hours = minutes / 60
                
                if hours < 1:
                    time_label = f"{minutes:.1f} min"
                else:
                    time_label = f"{hours:.1f} hrs"
                    
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.02*max(display_times),
                        time_label, ha='center', va='bottom', fontsize=12)
            
            plt.title('Predicted KDE Processing Times for Various Sample Sizes', fontsize=16)
            plt.xlabel('Sample Size', fontsize=14)
            plt.ylabel('Processing Time (seconds)', fontsize=14)
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            
            # Add a legend for the color coding
            if any(size > total_samples for size in prediction_sizes):
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='blue', label='Benchmark (measured)'),
                    Patch(facecolor='skyblue', label='Prediction (within data range)'),
                    Patch(facecolor='red', label='Extrapolation (beyond data range)')
                ]
                plt.legend(handles=legend_elements, loc='upper left')
            
            plt.tight_layout()
            
            # Save bar chart
            barchart_fig_path = create_figure_path(
                save_path, 
                "barchart", 
                total_samples, 
                target_size, 
                method
            )
            if barchart_fig_path:
                plt.savefig(barchart_fig_path, dpi=300)
                print(f"Saved bar chart to: {barchart_fig_path}")
                
            plt.show()
        
        return results, polynomial, predictions_df    