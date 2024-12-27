# Import the libraries
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np

def plot_clusters(x, y, clusters, contour = False, levels = None, xx = None, yy = None, f = None):
    """
    This function is used to plot a scatter plot of the points assignments.
    Parameters:
        x_points (list): The list of x-coordinates.
        y_points (list): The list of y-coordinates.
        clusters (array): An array of the assigned values for each point in xy.
    """
    # Create a color palette of length of unique assignments.
    palette = sns.color_palette("rainbow", n_colors = len(np.unique(clusters)))
    # Get colormap from seaborn
    cmap = ListedColormap(palette.as_hex())
    # Create a figure of size (10, 10)
    plt.figure(figsize = (10, 10))
    # Plot the scatter plot using seaborn's scatterplot function
    plt.scatter(x = x, y = y, c = clusters, cmap = cmap)
    # Display the contour
    plt.contour(xx, yy, f, levels = levels, colors = "k")
    # Plot the axis labels and title
    plt.xlabel('X', fontsize = 15)
    plt.ylabel('Y', fontsize = 15)
    if contour == True:
        plt.title('Contour plot of the Kernel Density Estimation:', fontsize = 15)    
    # Show the plot
    plt.show()

def plot_clusters_3D(x, y, xx, yy, f, clusters, density):
    """
    This function is used to plot a 3D scatter plot of the points assignments.
    Parameters:
        x_points (list): The list of x-coordinates.
        y_points (list): The list of y-coordinates.
        xx (array): The x-coordinates of the grid points.
        yy (array): The y-coordinates of the grid points.
        f (array): An array of density values for each point on the grid.
        clusters (array): An array of the assigned values for each point in xy.
        density (list): List of density values of the points.
    """
    # Create a color palette of length of unique assignments.
    palette = sns.color_palette("rainbow", n_colors = len(np.unique(clusters)))
    # Create figure and 3D axis
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Get colormap from seaborn
    cmap = ListedColormap(palette.as_hex())
    # Plot the wireframe of function f
    ax.plot_wireframe(xx, yy, f, color='gray', rstride=5, cstride = 5, alpha = 0.3)
    # Plot the scatter points colored by assignment
    ax.scatter(xs = x, ys = y, zs = density, c = clusters, cmap = cmap)
    # Plot the axis labels and title
    plt.xlabel('X', fontsize = 15)
    plt.ylabel('Y', fontsize = 15)
    ax.set_zlabel('Kernel Density Estimation', fontsize = 15)
    plt.title('3D plot of the Kernel Density Estimation:', fontsize = 15)
    # Show the plot
    plt.show()
