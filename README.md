# ClusterDC

## Overview

ClusterDC is a powerful density-based clustering library tailored for identifying clusters in two-dimensional embedding spaces. It is fast, robust, flexible, and data-driven. Initially created to address the clustering challenges faced by geochemists, it has evolved into a comprehensive toolkit for data analysis, visualization, and clustering that can be applied across multiple domains.

ClusterDC helps in analyzing two-dimensional embeddings of multivariate data, such as multielement assay datasets, to identify meaningful patterns and groups. At its core, ClusterDC leverages advanced Kernel Density Estimation (KDE) techniques to accurately model the underlying density distribution of your data. This density-based approach excels at identifying natural clusters of arbitrary shapes and varying densities, making it particularly effective for real-world data with complex structures.

ClusterDC operates only on two-dimensional data, so high-dimensional datasets must first be reduced to 2D using dimension reduction techniques. Based on extensive testing across numerous projects, we strongly recommend PaCMAP for its robust non-linear dimensionality reduction capabilities that effectively preserve cluster structures. LocalMAP, an improved version of PaCMAP, is another excellent option we're currently evaluating. While UMAP and t-SNE are also compatible with ClusterDC, our experience shows that PaCMAP and LocalMAP typically produce superior clustering results by maintaining better separation between natural data groups.

The library's advanced KDE implementation offers multiple kernel functions and bandwidth selection methods, including adaptive local bandwidths that automatically adjust to variations in your data density. This provides superior performance compared to traditional clustering methods, especially for datasets with irregular distributions, outliers, or varying cluster densities. We recommend using local bandwidth KDE for understanding clusters within your data first before start lumping these clusters into bigger ones. Try always to use max_clusters option at the first attempt to understand the clusters within the data to see the maximum number of clusters that can be generated using local bandwidth KDE.

While originally focused on geological applications, ClusterDC can be used in many fields beyond geosciences, such as environmental engineering, biological sciences, financial analysis, and other natural sciences. These areas often struggle with clustering due to natural variations and complex real-world phenomena that ClusterDC's density-based approach handles effectively.

For more information about how the core algorithm works, please refer to the publication:
<br> [Meyrieux, M., Hmoud, S., van Geffen, P., Kaeter, D. CLUSTERDC: A New Density-Based Clustering Algorithm and its Application in a Geological Material Characterization Workflow. Nat Resour Res (2024). https://doi.org/10.1007/s11053-024-10379-5](https://link.springer.com/article/10.1007/s11053-024-10379-5)

The publication presents case studies demonstrating the application of ClusterDC in geological contexts, showing how the algorithm supports the characterization of geological material types based on multi-element geochemistry.

![3D plot of the Kernel Density Estimation](https://github.com/Maximilien42/ClusterDC/blob/main/Images/3D%20plot%20of%20the%20Kernel%20Density%20Estimation%20-%204.svg)

![Contour plot of the Kernel Density Estimation](https://github.com/Maximilien42/ClusterDC/blob/main/Images/Contour%20plot%20of%20the%20Kernel%20Density%20Estimation%20-%204.svg)

## Installation

ClusterDC can be installed using pip:

```bash
pip install clusterdc
```

If you're using a conda environment, you may want to install certain dependencies with conda first:

```bash
conda install -y numpy=1.24.3 matplotlib=3.7.1
pip install clusterdc
```

## Key Components

ClusterDC now consists of three main classes that work together to provide a complete data analysis pipeline:

### 1. Data Class

A comprehensive data handling utility that simplifies loading, processing, analyzing, and visualizing data from various sources:

- **Versatile data loading** from local files, URLs, or built-in datasets
- **Support for multiple formats** including CSV, Excel, JSON, and HTML tables
- **Data summarization** with comprehensive statistics
- **Rich visualization capabilities** including:
  - Scatter plots with customizable markers and coloring
  - Density-colored scatter plots for pattern identification
  - Combined visualizations showing original and sampled data
  - Customizable plotting parameters (colors, sizes, transparency)
  - Plot saving in various formats with adjustable resolution
- **Advanced sampling techniques** including density-based sampling for large datsets
- **Seamless integration** with KDE and clustering components

### 2. KDE (Kernel Density Estimation) Class

An advanced implementation of kernel density estimation with multiple bandwidth selection methods:

- **Multiple kernel functions** (Gaussian, Epanechnikov, Laplacian)
- **Automatic bandwidth selection** using Bayesian optimization
- **Adaptive local bandwidth** estimation based on k-nearest neighbors
- **Global bandwidth** with anisotropic covariance estimation
- **Rule-of-thumb methods** (Scott's rule, Silverman's rule, both are same when having 2D datasets)
- **High-performance implementation** with memory optimization
- **Visualization tools** for analyzing density distributions
- **Performance benchmarking** capabilities
- **saving and loading KDE models** to save time

### 3. ClusterDC Class

The core clustering algorithm that identifies natural clusters based on density patterns:

- **Automatic selection** of the optimal number of clusters using gap analysis
- **Manual specification** of desired number of clusters
- **Rich visualization tools** for cluster interpretation
- **Comprehensive separability analysis**

## Getting Started

Here's a simple example of using the ClusterDC library:

### Data Loading and Processing

The `Data` class provides versatile data handling capabilities:

```python
from clusterdc import Data

# Initialize data handler
data = Data()

# Load from various sources
df1 = data.read_file('training_data') # url link to clusterdc github
# df2 = data.read_file('https://example.com/data.csv')
# df3 = data.read_file('excel_file.xlsx', sheet_name='Sheet1')

# Get comprehensive summary statistics
summary = data.get_summary()
print(summary)

# Perform dimension reduction if needed
# (e.g., using PaCMAP, LocalMAP, UMAP, t-SNE, etc.)
# This step depends on your specific needs
# for this training data, you don't need dimension reduction. 
# Data is already in 2D embedding space

# Create visualization
data.plot_scatter('PaCMAP_X', 'PaCMAP_Y', labels='category', 
                  title='Data Visualization', save_path='scatter.png')

# Perform density-based sampling
estimated_densities = data.estimate_density(['PaCMAP_X', 'PaCMAP_Y'])
sampled_data = data.sample(n_samples=1000, method='density')

# Create comparison visualization
result_df = data.plot_density_samples('PaCMAP_X', 'PaCMAP_Y', n_samples=500, 
                                      return_samples=True)
```
once data is imported, summarized, and visualized. Next step is to model kernel density estimate.

### Kernel Density Estimation

The `KDE` class offers powerful density estimation:

```python
from clusterdc import KDE

# Initialize KDE with options
kde = KDE(
    data=df[['PaCMAP_X', 'PaCMAP_Y']],
    kernel_types=['gaussian', 'epanechnikov'],
    n_iter=50,
    k_limits=(1, 40)  # as percentage of data points
)

# Fit KDE with different methods
kde.fit(method='scott')  # or 'silverman', 'local_bandwidth', 'global_bandwidth'

# Get density values
point_densities = kde.get_point_densities()
grid_densities = kde.get_grid_densities()

# Visualize results
kde.plot_results()

# Analyze optimization results
kde.plot_optimization_progress()
kde.print_optimization_report()

# Benchmark performance for larger datasets
results, model, predictions = KDE.benchmark_and_predict(
    data_input=df,
    target_size=100000,
    method='scott'
)
```

The `KDE` class offers also localized and globally optimized bandwidths using Bayesian Optimization:

```python
from clusterdc import KDE

# Initialize KDE with options
kde = KDE(
    data=df[['PaCMAP_X', 'PaCMAP_Y']],
    kernel_types=['gaussian', 'epanechnikov'],
    n_iter=50,
    k_limits=(1, 40)  # as percentage of data points
)

# Fit KDE with different methods
kde.fit(method='local_bandwidth') # finding best localized bandwidths using Bayesian Optimization.

# Get density values
point_densities = kde.get_point_densities()
grid_densities = kde.get_grid_densities()

# Visualize results
kde.plot_results()

# Analyze optimization results
kde.plot_optimization_progress()
kde.print_optimization_report()

```
### ClusterDC

Aftet estimating data density using `KDE`, `ClusterDC` will identify clusters in data based on `KDE`.

```python
from clusterdc import ClusterDC

# Create ClusterDC object with 2D data
cluster_dc = ClusterDC(
    data=df,
    columns=['PaCMAP_X', 'PaCMAP_Y'],  # Specify which columns to use
    kde_method='scott',  # Use Scott's rule for bandwidth
    gap_order=1  # Use first major gap for cluster selection
)

# Run clustering
assignments, density_info = cluster_dc.run_clustering()

# Visualize results
cluster_dc.plot_results(assignments, density_info)

# Find optimal number of clusters
optimal_clusters = cluster_dc.find_optimal_clusters()
print(f"Optimal number of clusters: {optimal_clusters}")

# Get cluster assignments
cluster_df = cluster_dc.get_cluster_assignments()
```

### Advanced Clustering

The `ClusterDC` class provides sophisticated clustering capabilities:

```python
from clusterdc import ClusterDC

# Create ClusterDC with options
cluster_dc = ClusterDC(
    data=df,
    columns=['x', 'y'],
    levels=50,  # Number of contour levels
    min_point=5,  # Minimum points per cluster
    gap_order='max_clusters',  # Get maximum possible clusters
    kde_method='local_bandwidth'  # Use adaptive bandwidth
)

# Run clustering
assignments, density_info = cluster_dc.run_clustering()

# Analyze separability between clusters
cluster_dc.plot_separability(save_path='separability.png')

# Find optimal number of clusters automatically
optimal_clusters = cluster_dc.find_optimal_clusters(
    method='direct_gap',
    save_path='optimal_clusters.png'
)

# Save clustering model for later use
cluster_dc.save_clustering('my_clustering_model.cdc')

# Load previously saved model
loaded_model = ClusterDC.load_clustering('my_clustering_model.cdc')
```

## Example Notebook

To see detailed examples of how to use the library, please refer to the provided Jupyter Notebook files in the [examples directory](https://github.com/samerhmoud/clusterdc/tree/main/clusterdc/examples). These notebooks demonstrate the usage of the functions with sample data and provide visualizations of the analysis and clustering results. They serve as practical guides to help you get started with ClusterDC and understand its various capabilities.

## Upcoming Improvements

We're actively working on enhancing ClusterDC with:

- More advanced visualization capabilities
- Extended documentation and tutorials

## Contact

If you have any questions or feedback, please feel free to contact:
- Samer Hmoud: geo.samer.hmoud@gmail.com
- Maximilien Meyrieux: maximilien.meyrieux@gmail.com 

## Attribution

If you use ClusterDC in your work, please include the following attribution:

[Meyrieux, M., Hmoud, S., van Geffen, P., Kaeter, D. CLUSTERDC: A New Density-Based Clustering Algorithm and its Application in a Geological Material Characterization Workflow. Nat Resour Res (2024). https://doi.org/10.1007/s11053-024-10379-5](https://link.springer.com/article/10.1007/s11053-024-10379-5)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/samerhmoud/clusterdc/blob/main/LICENSE.txt) file for details.

## Acknowledgements

We would like to acknowledge:

- The PaCMAP team for providing the PaCMAP dimension reduction algorithm, which is useful for reducing the dimensionality of the data before applying ClusterDC. For more information, refer to the [PaCMAP & LocalMAP GitHub repository](https://github.com/YingfanWang/PaCMAP).

- The ClusterDV team for the development of the ClusterDV MATLAB code and the synthetic datasets provided with it. ClusterDC was developed as an extension of ClusterDV to overcome its limitations in processing large datasets. For more details on ClusterDV, see the [ClusterDV GitHub repository](https://github.com/jcbmarques/clusterdv).

Please refer to the following references for more information:

- [Wang, Y., Huang, H., Rudin, C., & Shaposhnik, Y. (2021). Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization. Journal of Machine Learning Research, 22(201), 1-73.](http://jmlr.org/papers/v22/20-1061.html)

- [Wang, Y., Sun, Y., Huang, H., & Rudin, C. (2024). Dimension Reduction with Locally Adjusted Graphs. arXiv preprint arXiv:2412.15426.](https://arxiv.org/pdf/2412.15426)

- [Marques, J. C., & Orger, M. B. (2019). Clusterdv: a simple density-based clustering method that is robust, general and automatic. Bioinformatics, 35(12), 2125-2132.](https://doi.org/10.1093/bioinformatics/bty907)