# test_basic.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Import your library
from clusterdc import ClusterDC, KDE, Data

# Generate synthetic data
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
df = pd.DataFrame(X, columns=['x', 'y'])

# Test your clustering
clusterer = ClusterDC(
    data=df,
    columns=['x', 'y'],
    gap_order=None,
    n_clusters=4,
    kde_method="local_bandwidth"
)

# Run clustering
print("Running clustering...")
assignments, density_info = clusterer.run_clustering()

# Get cluster assignments
clusters = clusterer.get_cluster_assignments()
print(f"Found {len(np.unique(assignments[0]))} clusters")

# Visualize results
clusterer.plot_results(assignments, density_info)

print("Test completed successfully!")