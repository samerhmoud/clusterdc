import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List, Dict, Tuple
from pathlib import Path
import urllib.parse
import requests
from io import StringIO, BytesIO
import os
from .kde import KDE
from sklearn.neighbors import KDTree


class Data:
    """
    A comprehensive data handling and visualization utility class that provides functionality 
    for loading, analyzing, and visualizing data from various file formats, with additional 
    capabilities for density estimation and cluster analysis.
    """
    def __init__(self):
        """
        Initializes a new Data object with empty placeholders for data, file path, file type,
        KDE model, and sample weights. Establishes a fixed random state (42) for reproducibility.
        """
        self.data = None
        self.file_path = None
        self.file_type = None
        self.kde_model = None
        self.sample_weights = None
        
        # Set fixed random state for reproducibility
        np.random.seed(42)
        self.random_state = 42
        
        # Built-in dataset URLs
        self.datasets = {
            "training_data": "https://github.com/samerhmoud/clusterdc/blob/main/clusterdc/datasets/training_data.csv",
            "spiral_data": "https://github.com/samerhmoud/clusterdc/blob/main/clusterdc/datasets/spiral_data.csv"
        }
        
    def list_available_datasets(self) -> List[str]:
        """
        Returns a list of all available built-in datasets.
        
        Returns:
            List[str]: Names of available datasets
        """
        return list(self.datasets.keys())
        
    def read_file(self, 
                  file_path: str, 
                  sheet_name: Optional[str] = None,
                  **kwargs) -> pd.DataFrame:
        """
        Reads data from local files, URLs, or built-in datasets into a pandas DataFrame.

        Parameters:
            file_path (str): Path to the file, URL, or name of built-in dataset to be read
            sheet_name (Optional[str]): Name of the sheet for Excel files
            **kwargs: Additional parameters to pass to pandas read functions

        Returns:
            pd.DataFrame: The loaded DataFrame

        Raises:
            ValueError: If file format is not supported or dataset name not found
            Exception: If there are errors during file reading
        """
        # Check if the file_path is a built-in dataset name
        if file_path in self.datasets:
            print(f"Loading built-in dataset: {file_path}")
            return self._read_from_url(self.datasets[file_path], sheet_name, **kwargs)
        
        # Check if the file_path is a URL
        is_url = self._is_url(file_path)
        
        if is_url:
            # Handle URL
            return self._read_from_url(file_path, sheet_name, **kwargs)
        else:
            # Handle local file
            return self._read_from_local(file_path, sheet_name, **kwargs)
    
    def _is_url(self, path: str) -> bool:
        """
        Determines if a path is a URL.
        
        Parameters:
            path (str): The path to check
            
        Returns:
            bool: True if the path is a URL, False otherwise
        """
        try:
            result = urllib.parse.urlparse(path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False
    
    def _read_from_url(self, 
                      url: str, 
                      sheet_name: Optional[str] = None,
                      **kwargs) -> pd.DataFrame:
        """
        Reads data from a URL into a pandas DataFrame.
        
        Parameters:
            url (str): URL of the file to be read
            sheet_name (Optional[str]): Name of the sheet for Excel files
            **kwargs: Additional parameters to pass to pandas read functions
            
        Returns:
            pd.DataFrame: The loaded DataFrame
            
        Raises:
            ValueError: If file format is not supported
            Exception: If there are errors during file reading
        """
        # Store original URL
        original_url = url
        
        # Handle GitHub URLs - convert to raw content URL
        if 'github.com' in url and '/blob/' in url:
            # Convert GitHub URL to raw content URL
            url = url.replace('github.com', 'raw.githubusercontent.com')
            url = url.replace('/blob/', '/')
            print(f"GitHub URL detected. Using raw content URL: {url}")
        
        self.file_path = url
        
        # Determine file type from URL
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path
        self.file_type = os.path.splitext(path)[1].lower()
        
        try:
            # Download the file content
            response = requests.get(url)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            if self.file_type == '.csv':
                # Use StringIO for CSV content
                content = StringIO(response.text)
                self.data = pd.read_csv(content, **kwargs)
            elif self.file_type in ['.xlsx', '.xls']:
                # Use BytesIO for binary content (Excel)
                content = BytesIO(response.content)
                self.data = pd.read_excel(content, sheet_name=sheet_name, **kwargs)
            elif self.file_type == '.json':
                # Handle JSON data
                self.data = pd.read_json(StringIO(response.text), **kwargs)
            elif self.file_type == '.html' or self.file_type == '.htm':
                # Handle HTML tables
                self.data = pd.read_html(url, **kwargs)[0]  # Get first table by default
            else:
                raise ValueError(f"Unsupported file format: {self.file_type}")
            
            print(f"Successfully loaded data from URL: {original_url}")
            print(f"Shape: {self.data.shape}")
            return self.data
            
        except Exception as e:
            print(f"Error reading file from URL: {str(e)}")
            
            # If GitHub URL failed, try to read tables directly from the HTML
            if 'github.com' in original_url and '/blob/' in original_url and '.csv' in original_url:
                try:
                    print("Attempting to read CSV data directly from GitHub HTML page...")
                    tables = pd.read_html(original_url)
                    if tables:
                        self.data = tables[0]  # Usually the first table contains the CSV data
                        print(f"Successfully extracted table from GitHub HTML page")
                        print(f"Shape: {self.data.shape}")
                        return self.data
                except Exception as table_error:
                    print(f"Failed to extract table from GitHub page: {str(table_error)}")
            
            raise e
    
    def _read_from_local(self, 
                       file_path: str, 
                       sheet_name: Optional[str] = None,
                       **kwargs) -> pd.DataFrame:
        """
        Reads data from a local file into a pandas DataFrame.
        
        Parameters:
            file_path (str): Path to the local file to be read
            sheet_name (Optional[str]): Name of the sheet for Excel files
            **kwargs: Additional parameters to pass to pandas read functions
            
        Returns:
            pd.DataFrame: The loaded DataFrame
            
        Raises:
            ValueError: If file format is not supported
            Exception: If there are errors during file reading
        """
        self.file_path = Path(file_path)
        self.file_type = self.file_path.suffix.lower()
        
        try:
            if self.file_type == '.csv':
                self.data = pd.read_csv(file_path, **kwargs)
            elif self.file_type in ['.xlsx', '.xls']:
                self.data = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            elif self.file_type == '.json':
                self.data = pd.read_json(file_path, **kwargs)
            elif self.file_type == '.html' or self.file_type == '.htm':
                self.data = pd.read_html(file_path, **kwargs)[0]  # Get first table by default
            else:
                raise ValueError(f"Unsupported file format: {self.file_type}")
                
            print(f"Successfully loaded data from {self.file_path}")
            print(f"Shape: {self.data.shape}")
            return self.data
            
        except Exception as e:
            print(f"Error reading local file: {str(e)}")
            raise

    # The rest of the class methods remain the same...

    def get_summary(self) -> pd.DataFrame:
        """
        Generates a comprehensive summary of the loaded data.

        Returns:
            pd.DataFrame: A DataFrame containing:
                - Data types for each column
                - Count of values
                - Missing values (count and percentage)
                - Unique value counts
                - For numeric columns: mean, std, min, quartiles, max (rounded to 3 decimals)

        Raises:
            ValueError: If no data has been loaded
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        # Initialize summary dictionary
        summary_dict = {
            'dtype': self.data.dtypes,
            'count': self.data.count(),
            'missing': self.data.isna().sum(),
            'missing_%': (self.data.isna().sum() / len(self.data) * 100).round(2),
            'unique': self.data.nunique()
        }
        
        # Create initial DataFrame
        summary_df = pd.DataFrame(summary_dict)
        
        # Get numeric columns
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        
        # Add numeric statistics where applicable
        if len(numeric_cols) > 0:
            desc = self.data[numeric_cols].describe()
            summary_df.loc[numeric_cols, 'mean'] = desc.loc['mean']
            summary_df.loc[numeric_cols, 'std'] = desc.loc['std']
            summary_df.loc[numeric_cols, 'min'] = desc.loc['min']
            summary_df.loc[numeric_cols, '25%'] = desc.loc['25%']
            summary_df.loc[numeric_cols, '50%'] = desc.loc['50%']
            summary_df.loc[numeric_cols, '75%'] = desc.loc['75%']
            summary_df.loc[numeric_cols, 'max'] = desc.loc['max']
            
        # Round numeric columns
        numeric_summary_cols = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
        summary_df[numeric_summary_cols] = summary_df[numeric_summary_cols].round(3)
        
        return summary_df

    def save_data(self, 
                  file_path: str,
                  sheet_name: Optional[str] = None,
                  index: bool = False,
                  **kwargs) -> None:
        """
        Saves the current data to a file in CSV or Excel format.

        Parameters:
            file_path (str): Path where the file should be saved
            sheet_name (Optional[str]): Name of the sheet for Excel files
            index (bool): Whether to include index in the output file
            **kwargs: Additional parameters to pass to pandas write functions

        Raises:
            ValueError: If no data has been loaded or file format is not supported
            Exception: If there are errors during file saving
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        save_path = Path(file_path)
        file_type = save_path.suffix.lower()
        
        try:
            if file_type == '.csv':
                self.data.to_csv(file_path, index=index, **kwargs)
            elif file_type in ['.xlsx', '.xls']:
                self.data.to_excel(file_path, 
                                 sheet_name=sheet_name or 'Sheet1',
                                 index=index, 
                                 **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_type}")
                
            print(f"Successfully saved data to {save_path}")
            
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            raise

    def make_folder(self, directory_path: str) -> Path:
        """
        Creates a folder at the specified path if it doesn't already exist.
        If the folder already exists, the function will return its path without any error.
    
        Parameters:
            directory_path (str): Path where the folder should be created
    
        Returns:
            Path: Path object pointing to the created or existing folder
    
        Raises:
            OSError: If there are permission errors or other issues creating the folder
        """
        try:
            # Convert to Path object
            dir_path = Path(directory_path)
            
            # Create directory and any necessary parent directories
            dir_path.mkdir(parents=True, exist_ok=True)
            
            return dir_path
            
        except OSError as e:
            print(f"Error creating folder: {str(e)}")
            raise
    
    def plot_scatter(self,
                    x: str,
                    y: str,
                    labels: Optional[Union[str, List[int], List[str]]] = None,
                    title: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 8),
                    save_path: Optional[str] = None,
                    buffer_fraction: float = 0.05,
                    point_size: int = 10,
                    alpha: float = 0.6,
                    dpi: int = 300,
                    show_kde: bool = False,
                    kernel_type: str = 'gaussian'):
        """
        Creates a customizable scatter plot of two variables with optional features.

        Parameters:
            x (str): Column name for x-axis
            y (str): Column name for y-axis
            labels (Optional[Union[str, List[int], List[str]]]): Labels for color-coding points
            title (Optional[str]): Plot title
            figsize (Tuple[int, int]): Figure size in inches
            save_path (Optional[str]): Path to save the plot
            buffer_fraction (float): Fraction of range to add as buffer around plot
            point_size (int): Size of scatter points
            alpha (float): Transparency of points
            dpi (int): DPI for saved figure
            show_kde (bool): Whether to show kernel density estimation
            kernel_type (str): Type of kernel for density estimation

        Raises:
            ValueError: If no data has been loaded or specified columns not found
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        if x not in self.data.columns or y not in self.data.columns:
            raise ValueError(f"Columns {x} or {y} not found in data")

        # Create figure and axis objects explicitly
        fig, ax = plt.subplots(figsize=figsize)

        # Calculate axis limits with buffer
        x_min, x_max = self.data[x].min(), self.data[x].max()
        y_min, y_max = self.data[y].min(), self.data[y].max()

        x_range = x_max - x_min
        y_range = y_max - y_min

        x_buffer = buffer_fraction * x_range
        y_buffer = buffer_fraction * y_range

        plt.xlim(x_min - x_buffer, x_max + x_buffer)
        plt.ylim(y_min - y_buffer, y_max + y_buffer)

        if show_kde:
            # Initialize and fit KDE with Scott's bandwidth
            kde = KDE(data=self.data[[x, y]].values, kernel_types=[kernel_type])
            kde.fit(method='scott')

            # Get point densities
            densities = kde.get_point_densities()

            # Create scatter plot colored by density
            scatter = plt.scatter(self.data[x], 
                                self.data[y], 
                                c=densities,
                                cmap='viridis',
                                alpha=alpha,
                                s=point_size)
            plt.colorbar(scatter, label='Density (Scott\'s Rule)')

        elif labels is not None:
            # Handle different types of labels
            if isinstance(labels, str):
                if labels not in self.data.columns:
                    raise ValueError(f"Labels column {labels} not found in data")
                label_values = self.data[labels]
            else:
                if len(labels) != len(self.data):
                    raise ValueError("Length of labels must match length of data")
                label_values = labels

            # Get unique labels and sort them for consistent colors
            unique_labels = sorted(np.unique(label_values))
            n_clusters = len(unique_labels)
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

            # Create legend elements
            legend_elements = [
                plt.Line2D([0], [0], 
                          marker='o', 
                          color='w',
                          markerfacecolor=colors[i], 
                          label=f'Cluster {cluster}',
                          markersize=10)
                for i, cluster in enumerate(unique_labels)
            ]

            # Plot points for each cluster
            for label, color in zip(unique_labels, colors):
                mask = label_values == label
                plt.scatter(self.data[x][mask], 
                           self.data[y][mask],
                           c=[color], 
                           alpha=alpha,
                           s=point_size)

            # Calculate number of legend columns needed (15 items per column)
            n_cols = (n_clusters - 1) // 15 + 1

            # Add legend with dynamic columns
            if n_cols > 1:
                plt.legend(handles=legend_elements,
                          loc='center left',
                          bbox_to_anchor=(1, 0.5),
                          ncol=n_cols,
                          columnspacing=1.5,
                          handletextpad=0.5)
            else:
                plt.legend(handles=legend_elements,
                          loc='center left',
                          bbox_to_anchor=(1, 0.5))

        else:
            plt.scatter(self.data[x], 
                       self.data[y], 
                       alpha=alpha,
                       s=point_size,
                       c='black')

        # Set labels and title
        plt.xlabel(x)
        plt.ylabel(y)
        plot_title = title or f"Scatter Plot of {y} vs {x}"
        if show_kde:
            plot_title += "\nwith Density Estimate (Scott's Rule)"
        plt.title(plot_title)

        # Add grid
        plt.grid(True, linestyle='--', alpha=0.3)

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            

    def estimate_density(self, 
                        columns: Optional[List[str]] = None,
                        kernel_type: str = 'gaussian') -> np.ndarray:
        """
        Estimates the density of points in the dataset using Kernel Density Estimation.

        Parameters:
            columns (Optional[List[str]]): Columns to use for density estimation
            kernel_type (str): Type of kernel to use for density estimation

        Returns:
            np.ndarray: Array of density estimates for each point

        Raises:
            ValueError: If no data has been loaded or columns are invalid
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        # Select numeric columns if none specified
        if columns is None:
            numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
            columns = numeric_cols.tolist()

        # Validate columns
        for col in columns:
            if col not in self.data.columns:
                raise ValueError(f"Column {col} not found in data")
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                raise ValueError(f"Column {col} is not numeric")

        # Get data for selected columns
        X = self.data[columns].values

        # Set random state before initializing KDE
        np.random.seed(self.random_state)

        # Initialize and fit KDE with Scott's bandwidth
        self.kde_model = KDE(data=X, kernel_types=[kernel_type])
        self.kde_model.fit(method='scott')

        # Get point densities
        densities = self.kde_model.get_point_densities()

        # Normalize to get proper weights
        self.sample_weights = densities / densities.sum()

        self.data['kde_density'] = densities

        return densities

    def sample(self, 
              n_samples: int,
              method: str = 'random',
              columns: Optional[List[str]] = None,
              kernel_type: str = 'gaussian',
              replace: bool = True,
              random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Sample data points using either random or density-based sampling.

        Parameters:
            n_samples (int): Number of samples to draw
            method (str): Sampling method ('random' or 'density')
            columns (Optional[List[str]]): Columns to use for density estimation (only for density sampling)
            kernel_type (str): Type of kernel to use for density estimation (only for density sampling)
            replace (bool): Whether to sample with replacement (default True)
            random_state (Optional[int]): Random state for reproducibility. If None, uses class random_state

        Returns:
            pd.DataFrame: Sampled data

        Raises:
            ValueError: If no data has been loaded or invalid method specified

        Notes:
            - If random_state is not provided, uses the class's random_state (default=42)
            - Setting a specific random_state allows for reproducible sampling results
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        if method not in ['random', 'density']:
            raise ValueError("Method must be either 'random' or 'density'")

        # Set random state for reproducibility
        # If random_state is provided, use it; otherwise use class random_state
        current_random_state = random_state if random_state is not None else self.random_state
        np.random.seed(current_random_state)

        if method == 'density':
            if self.sample_weights is None:
                self.estimate_density(columns=columns, kernel_type=kernel_type)

            # Sample indices based on density weights
            sampled_indices = np.random.choice(
                len(self.data),
                size=n_samples,
                p=self.sample_weights,
                replace=replace
            )
        else:  # random sampling
            sampled_indices = np.random.choice(
                len(self.data),
                size=n_samples,
                replace=replace
            )

        # Return sampled data
        return self.data.iloc[sampled_indices].reset_index(drop=True)

    def plot_density_samples(self,
                            x: str,
                            y: str,
                            n_samples: int,
                            kernel_type: str = 'gaussian',
                            figsize: Tuple[int, int] = (15, 5),
                            random_state: Optional[int] = None,
                            return_samples: bool = False) -> Optional[pd.DataFrame]:
        """
        Creates a three-panel visualization comparing original and sampled data.

        Parameters:
            x (str): Column name for x-axis
            y (str): Column name for y-axis
            n_samples (int): Number of samples to draw
            kernel_type (str): Type of kernel for density estimation
            figsize (Tuple[int, int]): Figure size in inches
            random_state (Optional[int]): Random state for reproducibility. If None, uses class random_state
            return_samples (bool): If True, returns the sampled DataFrame

        Returns:
            Optional[pd.DataFrame]: If return_samples is True, returns the sampled DataFrame

        Raises:
            ValueError: If no data has been loaded
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        # Set random state for reproducibility
        current_random_state = random_state if random_state is not None else self.random_state
        np.random.seed(current_random_state)

        # Estimate density if not already done
        if self.sample_weights is None:
            self.estimate_density(columns=[x, y], kernel_type=kernel_type)

        # Get sampled data using density method
        sampled_data = self.sample(
            n_samples=n_samples,
            method='density',
            columns=[x, y],
            kernel_type=kernel_type,
            random_state=current_random_state
        )

        # Set random state before second KDE
        np.random.seed(current_random_state)

        # Estimate density for sampled data
        sampled_kde = KDE(data=sampled_data[[x, y]].values, kernel_types=[kernel_type])
        sampled_kde.fit(method='scott')
        sampled_weights = sampled_kde.get_point_densities()
        sampled_weights = sampled_weights / sampled_weights.sum()

        # Create plots
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Original data (black points)
        axes[0].scatter(self.data[x], self.data[y], alpha=0.5, s=20, c='black')
        axes[0].set_title('Original Data')
        axes[0].set_xlabel(x)
        axes[0].set_ylabel(y)

        # Original data density plot
        scatter = axes[1].scatter(
            self.data[x], 
            self.data[y],
            c=self.sample_weights,
            cmap='viridis',
            alpha=0.5,
            s=20
        )
        plt.colorbar(scatter, ax=axes[1], label='Density')
        axes[1].set_title('Original Data \nKernel Density Estimate (Scott\'s Rule)')
        axes[1].set_xlabel(x)
        axes[1].set_ylabel(y)

        # Sampled data with density estimation
        scatter = axes[2].scatter(
            sampled_data[x], 
            sampled_data[y], 
            c=sampled_weights,
            cmap='viridis',
            alpha=0.5, 
            s=20
        )
        plt.colorbar(scatter, ax=axes[2], label='Density')
        axes[2].set_title(f'Sampled Data (n={n_samples}) \nKernel Density Estimate (Scott\'s Rule)')
        axes[2].set_xlabel(x)
        axes[2].set_ylabel(y)

        plt.tight_layout()

        if return_samples:
            return sampled_data
    
    def assign_nearest_cluster(self,
                             sampled_data: pd.DataFrame,
                             cluster_labels: Union[List[int], np.ndarray],
                             columns: List[str],
                             k_neighbors: int = 1) -> np.ndarray:
        """
        Assigns cluster labels to the original dataset by:
        1. First matching exact coordinates using pandas merge
        2. Then using k-nearest neighbors from original sampled points for remaining points,
           assigning labels based on the mode of the k neighbors' labels.

        Parameters:
            sampled_data: DataFrame containing the sampled points
            cluster_labels: Labels for the sampled points
            columns: List of column names used for matching
            k_neighbors: Number of nearest neighbors to consider (default=1)
                        Mode of their labels will be used for assignment

        Returns:
            np.ndarray: Array of cluster assignments
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        if len(sampled_data) != len(cluster_labels):
            raise ValueError("Length of sampled_data and cluster_labels must match")

        if k_neighbors < 1:
            raise ValueError("k_neighbors must be at least 1")

        # Set random state for reproducibility
        np.random.seed(self.random_state)

        # Add cluster labels to sampled data and drop duplicates
        sampled_with_labels = sampled_data.copy()
        sampled_with_labels['cluster'] = cluster_labels.astype(int)
        sampled_with_labels = sampled_with_labels.drop_duplicates(subset=columns)

        # First attempt: exact coordinate matching
        result_df = self.data.copy()
        result_df['cluster'] = pd.NA

        # Merge to get exact matches
        for idx, row in self.data.iterrows():
            matches = sampled_with_labels[
                (sampled_with_labels[columns[0]] == row[columns[0]]) & 
                (sampled_with_labels[columns[1]] == row[columns[1]])
            ]
            if len(matches) > 0:
                result_df.loc[idx, 'cluster'] = matches.iloc[0]['cluster']

        # Find rows with no labels
        mask_no_label = result_df['cluster'].isna()
        rows_no_label = self.data[mask_no_label]

        if len(rows_no_label) > 0:
            print(f"Found {len(rows_no_label)} points without exact matches. Using {k_neighbors}-nearest neighbors for these points.")

            # Ensure k_neighbors doesn't exceed number of sampled points
            k = min(k_neighbors, len(sampled_with_labels))
            if k < k_neighbors:
                print(f"Warning: Reduced k_neighbors to {k} due to sample size limitation")

            # Build KD-tree from original sampled points
            tree = KDTree(sampled_with_labels[columns].values)

            # Find k nearest sampled points for each unlabeled point
            _, indices = tree.query(rows_no_label[columns].values, k=k)

            # For each unlabeled point, get the mode of its neighbors' labels
            for i, idx in enumerate(mask_no_label[mask_no_label].index):
                neighbor_labels = sampled_with_labels['cluster'].values[indices[i]]

                # Get unique labels and their counts
                unique_labels, counts = np.unique(neighbor_labels, return_counts=True)

                # Find labels with maximum count
                max_count = np.max(counts)
                max_labels = unique_labels[counts == max_count]

                # If multiple modes exist, randomly choose one (with fixed random state)
                if len(max_labels) > 1:
                    result_df.loc[idx, 'cluster'] = np.random.choice(max_labels)
                else:
                    result_df.loc[idx, 'cluster'] = max_labels[0]
        
        self.data['cluster'] = result_df['cluster'].values.astype(int)

        return result_df['cluster'].values.astype(int)

    def save_session(self, file_path: str) -> None:
        """
        Saves the current session state to a file for later resumption.
        
        This method saves the data object, including loaded data, file paths,
        and any computed results to avoid recomputing expensive operations.
        
        Parameters:
            file_path (str): Path where the session file should be saved
                             .pkl extension recommended
        
        Returns:
            None
            
        Raises:
            Exception: If there are errors during saving
        """
        import pickle
        
        try:
            # Create directory if it doesn't exist
            save_dir = Path(file_path).parent
            self.make_folder(str(save_dir))
            
            # Save the session state
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
                
            print(f"Session saved successfully to {file_path}")
            
        except Exception as e:
            print(f"Error saving session: {str(e)}")
            raise
    
    @classmethod
    def load_session(cls, file_path: str):
        """
        Loads a previously saved session state from a file.
        
        This static method loads a Data object with all its state
        from a previously saved session file.
        
        Parameters:
            file_path (str): Path to the saved session file
        
        Returns:
            Data: Loaded Data object with restored state
            
        Raises:
            FileNotFoundError: If the file does not exist
            Exception: If there are errors during loading
        """
        import pickle
        
        try:
            with open(file_path, 'rb') as f:
                loaded_object = pickle.load(f)
                
            print(f"Session loaded successfully from {file_path}")
            return loaded_object
            
        except FileNotFoundError:
            print(f"Session file not found: {file_path}")
            raise
        except Exception as e:
            print(f"Error loading session: {str(e)}")
            raise