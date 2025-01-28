# Import the librairies
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.spatial import KDTree
import seaborn as sns
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import networkx as nx
import random
import copy
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)


def create_grid(x_points, y_points, border_fraction = 0, num_points = 100):
    """
    Creates a grid with a specified border size and number of points.
    Parameters:
        x_points (list): The list of x-coordinates.
        y_points (list): The list of y-coordinates.
        border_fraction (float): The fraction to use as border size.
        num_points (int): The number of points used for each side of the grid.
    Returns:
        xx (array): The x-coordinates of the grid points.
        yy (array): The y-coordinates of the grid points.
    """
    # Calculate the border size by multiplying the range of x and y by the border_fraction
    deltaX = (np.max(x_points) - np.min(x_points)) * border_fraction
    deltaY = (np.max(y_points) - np.min(y_points)) * border_fraction
    # Define the minimum and maximum x and y values with the added border size
    xmin = np.min(x_points) - deltaX
    xmax = np.max(x_points) + deltaX
    ymin = np.min(y_points) - deltaY
    ymax = np.max(y_points) + deltaY
    # Create the grid with num_points number of points between the min and max values
    xx, yy = np.mgrid[xmin:xmax:num_points * 1j, ymin:ymax:num_points * 1j]
    # Return the meshgrid components
    return xx, yy


def get_kde_point(point, bw_method = None, kernel = None, bw = None, x_points = None, y_points = None):
    """
    This function takes a point in the form of a tuple (x,y) and returns the density at that point according to the 
    kernel density estimate.
    Parameters:
        point (array): A array representing the point for which the density is to be computed.
        bw_method (str/float): The method used for computing the bandwidth.
        kernel (gaussian kernel): The kernel density estimate.
        bw (array): The bandwidth used at each point if the bw_method is 'local'.
        x_points (list): The list of x-coordinates.
        y_points (list): The list of y-coordinates.
    Returns:
        density (float): the density at the point to return.
    """
    if bw_method == 'local':
        # If bw_method is 'local', iterate over all x_points, y_points, and bw to calculate the density
        density = 0
        for i, bwi in enumerate(bw):
            # Add the density contribution of each data point to the total density
            density = density + (1 / (bwi ** 2)) * np.exp(-((point[0] - x_points[i]) ** 2 + (point[1] - y_points[i]) ** 2)/(2 * (bwi ** 2)))
        return (density / len(bw))
    else:
        # If bw_method is not 'local', use the kernel to calculate the density at the point
        return kernel.evaluate(np.array([point[0], point[1]]))


def evaluate_kde(x_points, y_points, xx, yy, bw_method, kernel, bw):
    """
    This function evaluates the kernel density estimate (KDE) on a grid of points.
    Parameters:
        x_points (list): The list of x-coordinates.
        y_points (list): The list of y-coordinates.
        xx (array): The x-coordinates of the grid points.
        yy (array): The y-coordinates of the grid points.
        bw_method (str/float): The method used for computing the bandwidth.
        kernel (gaussian kernel): The kernel density estimate.
        bw (array): The bandwidth used at each point if the bw_method is 'local'.
    Returns:
        f (array): An array of density values for each point on the grid.
    """
    positions = np.vstack([xx.ravel(), yy.ravel()])
    density_map = []
    for i in range(np.vstack([xx.ravel(), yy.ravel()]).shape[1]):
        xi = np.vstack([xx.ravel(), yy.ravel()])[0][i] 
        yi = np.vstack([xx.ravel(), yy.ravel()])[1][i]
        # Evaluate the density at each point on the grid
        density_map.append(get_kde_point(point = [xi, yi], bw_method = bw_method, kernel = kernel, bw = bw, x_points = x_points, y_points = y_points))
    # Reshape the density_map into the same shape as xx, which is the shape of the grid
    f = np.reshape(density_map, xx.shape)
    # Return the array of density values for each point on the grid
    return f

def create_contours(xx, yy, f, levels):
    """
    This function creates contours of a given density map.
    Parameters:
        xx (array): The x-coordinates of the grid points.
        yy (array): The y-coordinates of the grid points.
        f (array): An array of density values for each point on the grid.
        levels (int): The number of contour levels to create.
    Returns:
        polygon (list): A list of polygons representing the contours. Each polygon is a list of x and y coordinates of the contour.
        density_values (list): A list of density values for each contour.
        density_contours (list): A list of contour indices for each polygon.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    # Create contours with matplotlib's contour function
    cset = ax.contour(xx, yy, f, levels = levels)
    # Close the plot to avoid displaying it
    plt.close()
    polygon = []
    density_values = []
    density_contours = []
    density_levels = cset.levels
    for j in range(len(cset.allsegs)):
        level = density_levels[j]
        for ii, seg in enumerate(cset.allsegs[j]):
            # Append the x,y coordinates of each contour to the polygon list
            polygon.append([seg[:,0], seg[:,1]])
            # Append the density value for this contour to the density_values list
            density_values.append(level)
            # Append the contour index for this polygon to the density_contours list
            density_contours.append(j)
    # Return the informations about the contour polygons
    return polygon, density_values, density_contours

# def make_node_dic(polygon, density_values, density_contours):
#     """
#     This function creates a dictionary of nodes, representing the contours, with each node containing information about its level, density, children, parent, and token.
#     Parameters:
#         polygon (list): A list of polygons representing the contours. Each polygon is a list of x and y coordinates of the contour.
#         density_values (list): A list of density values for each contour.
#         density_contours (list): A list of contour indices for each polygon.
#     Returns:
#         dic_node (dict): A dictionary of nodes, each representing a contour.
#     """
#     # Create dictionary of nodes with level, density, children, parent, and token information
#     dic_node = {ind: {'level': density_contours[ind], 'density': density_values[ind], 'childrens': [], 'parent': [], 'token': []} for ind in range(len(polygon))}
#     for ind in range(len(polygon)):
#         if polygon[ind][0][0] == polygon[ind][0][-1] and polygon[ind][1][0] == polygon[ind][1][-1]:
#             for ind_sub in range(len(polygon)):
#                 if polygon[ind_sub][0][0] == polygon[ind_sub][0][-1] and polygon[ind_sub][1][0] == polygon[ind_sub][1][-1]:
#                     # Check if the subpolygon is one level higher than polygon and has higher density
#                     if dic_node[ind_sub]['level'] == dic_node[ind]['level'] + 1 and dic_node[ind_sub]['density'] > dic_node[ind]['density']:
#                         # Check if polygon contains subpolygon
#                         poly = Polygon([(polygon[ind][0][index], polygon[ind][1][index]) for index in range(len(polygon[ind][0]))])
#                         poly_sub = Polygon([(polygon[ind_sub][0][index], polygon[ind_sub][1][index]) for index in range(len(polygon[ind_sub][0]))])
#                         if poly.contains(poly_sub):
#                             # Add subpolygon to the children list of the polygon
#                             dic_node[ind]['childrens'].append(ind_sub)
#                             # Add polygon to the parent list of the subpolygon
#                             dic_node[ind_sub]['parent'].append(ind)  
#                 else:
#                     if ind_sub in dic_node.keys():
#                         del dic_node[ind_sub]
#         else:
#             if ind in dic_node.keys():
#                 del dic_node[ind]
#     # Find the maximum level in the dictionary
#     max_level = -1
#     for node in dic_node.keys():
#         if dic_node[node]['level'] > max_level:
#             max_level = dic_node[node]['level']
#     # Loop through levels from max level to 1
#     for level in range(max_level, 0, -1):
#         list_node_level = []
#         # Find all nodes of the current level
#         for node in dic_node.keys():
#             if dic_node[node]['level'] == level:
#                 list_node_level.append(node)
#         # Loop through nodes of the current level
#         for ind_node in list_node_level:    
#             for ind_node_sub in list_node_level:
#                 poly = Polygon([(polygon[ind_node][0][index], polygon[ind_node][1][index]) for index in range(len(polygon[ind_node][0]))])
#                 poly_sub = Polygon([(polygon[ind_node_sub][0][index], polygon[ind_node_sub][1][index]) for index in range(len(polygon[ind_node_sub][0]))])
#                 # Check if polygon contains subpolygon
#                 if poly.contains(poly_sub) and ind_node != ind_node_sub:
#                     # Delete subpolygon node
#                     if ind_node_sub in dic_node.keys():
#                         del dic_node[ind_node_sub]
#     # Find all children of the root node
#     children = []
#     for ind in dic_node.keys():
#         if dic_node[ind]['level'] == 1:
#             dic_node[ind]['parent'] = '*'
#             children.append(ind)
#     # Create root node with all children
#     dic_node['*'] = {'level': 0, 'density': 0, 'childrens': children, 'parent': ['**'], 'token': []}      
#     # Return the built dictionary.
#     return dic_node

def make_node_dic(polygon, density_values, density_contours):
    """
    This function creates a dictionary of nodes, representing the contours, with each node containing information about its level, density, children, parent, and token.
    Parameters:
        polygon (list): A list of polygons representing the contours. Each polygon is a list of x and y coordinates of the contour.
        density_values (list): A list of density values for each contour.
        density_contours (list): A list of contour indices for each polygon.
    Returns:
        dic_node (dict): A dictionary of nodes, each representing a contour.
    """
    # Create dictionary of nodes with level, density, children, parent, and token information
    dic_node = {ind: {'level': density_contours[ind], 'density': density_values[ind], 'childrens': [], 'parent': [], 'token': []} for ind in range(len(polygon))}
    
    for ind in range(len(polygon)):
        if len(polygon[ind][0]) > 0 and len(polygon[ind][1]) > 0:
            if polygon[ind][0][0] == polygon[ind][0][-1] and polygon[ind][1][0] == polygon[ind][1][-1]:
                for ind_sub in range(len(polygon)):
                    if len(polygon[ind_sub][0]) > 0 and len(polygon[ind_sub][1]) > 0:
                        if polygon[ind_sub][0][0] == polygon[ind_sub][0][-1] and polygon[ind_sub][1][0] == polygon[ind_sub][1][-1]:
                            # Check if the subpolygon is one level higher than polygon and has higher density
                            if dic_node[ind_sub]['level'] == dic_node[ind]['level'] + 1 and dic_node[ind_sub]['density'] > dic_node[ind]['density']:
                                # Check if polygon contains subpolygon
                                poly = Polygon([(polygon[ind][0][index], polygon[ind][1][index]) for index in range(len(polygon[ind][0]))])
                                poly_sub = Polygon([(polygon[ind_sub][0][index], polygon[ind_sub][1][index]) for index in range(len(polygon[ind_sub][0]))])
                                if poly.contains(poly_sub):
                                    # Add subpolygon to the children list of the polygon
                                    dic_node[ind]['childrens'].append(ind_sub)
                                    # Add polygon to the parent list of the subpolygon
                                    dic_node[ind_sub]['parent'].append(ind)
                        else:
                            if ind_sub in dic_node.keys():
                                del dic_node[ind_sub]
            else:
                if ind in dic_node.keys():
                    del dic_node[ind]
        else:
            if ind in dic_node.keys():
                del dic_node[ind]

    # Find the maximum level in the dictionary
    max_level = -1
    for node in dic_node.keys():
        if dic_node[node]['level'] > max_level:
            max_level = dic_node[node]['level']
    
    # Loop through levels from max level to 1
    for level in range(max_level, 0, -1):
        list_node_level = []
        # Find all nodes of the current level
        for node in dic_node.keys():
            if dic_node[node]['level'] == level:
                list_node_level.append(node)
        # Loop through nodes of the current level
        for ind_node in list_node_level:
            for ind_node_sub in list_node_level:
                poly = Polygon([(polygon[ind_node][0][index], polygon[ind_node][1][index]) for index in range(len(polygon[ind_node][0]))])
                poly_sub = Polygon([(polygon[ind_node_sub][0][index], polygon[ind_node_sub][1][index]) for index in range(len(polygon[ind_node_sub][0]))])
                # Check if polygon contains subpolygon
                if poly.contains(poly_sub) and ind_node != ind_node_sub:
                    # Delete subpolygon node
                    if ind_node_sub in dic_node.keys():
                        del dic_node[ind_node_sub]

    # Find all children of the root node
    children = []
    for ind in dic_node.keys():
        if dic_node[ind]['level'] == 1:
            dic_node[ind]['parent'] = '*'
            children.append(ind)
    
    # Create root node with all children
    dic_node['*'] = {'level': 0, 'density': 0, 'childrens': children, 'parent': ['**'], 'token': []}
    
    # Return the built dictionary.
    return dic_node
    
def find_leaves(dic_node):
    """
    This function finds all leaf nodes in the dictionary of nodes.
    Parameters:
        dic_node (dict): A dictionary of nodes, each representing a contour.
    Returns:
        list_leaves (list): A list of leaf nodes in the dictionary.
    """
    list_leaves = []
    # Loop through nodes in the dictionary
    for ind in dic_node.keys():
        # Check if the node is not the root node and has no children
        if ind != '*':
            if len(dic_node[ind]['childrens']) == 0:
                # Add the node to the list of leaves
                list_leaves.append(ind)
    # Return the list of leaf nodes in the dictionary
    return list_leaves

def check_node_unvalid(xy, polygon, ind_node, min_point):
    """
    This function checks if a node is valid, meaning it contains not enough data points.
    Parameters:
        xy (list): A list of the x,y coordinates of data points.
        polygon (list): A list of polygons representing the contours. Each polygon is a list of x and y coordinates of the contour.
        ind_node (int/str): the index of the node to check for validness.
        min_point (int): the minimum number of points to include inside a valid polygon.
    Returns:
        unvalid (bool): A boolean value indicating if the node is unvalid.
    """
    unvalid = True
    # Check if the node is the root node
    if ind_node == '*':
        return False
    # Create a shapely Polygon object for the node
    poly_current = Polygon([(polygon[ind_node][0][index], polygon[ind_node][1][index]) for index in range(len(polygon[ind_node][0]))])
    nb_point_inside = 0
    # Loop through data points
    for point in xy:
        # Create a shapely Point object for the data point
        point_current = Point(point[0], point[1])
        # Check if the point is within the node's polygon
        if point_current.within(poly_current):
            nb_point_inside += 1
        if nb_point_inside >= min_point:
            unvalid = False
            break
            break 
    # Return the unvalid boolean
    return unvalid

def trim_unvalid_leaves(xy, dic_node, polygon, min_point):
    """
    This function removes unvalid leaves from the dictionary of nodes.
    Parameters:
        xy (list): A list of the x,y coordinates of data points.
        polygon (list): A list of polygons representing the contours. Each polygon is a list of x and y coordinates of the contour.
        ind_node (int/str): the index of the node to check for unvalidness.
        min_point (int): the minimum number of points to include inside a valid polygon.
    Returns:
        dic_node (dict): A modified dictionary of nodes with unvalid leaves removed.
    """
    list_leaves = find_leaves(dic_node)
    # Find the maximum level in the dictionary
    max_level = -1
    for node in dic_node.keys():
        if dic_node[node]['level'] > max_level:
            max_level = dic_node[node]['level']
    # Loop through levels from max level to 1
    for level in range(max_level, 0, -1):
        list_node_level = []
        # Find all nodes of the current level
        for node in dic_node.keys():
            if dic_node[node]['level'] == level:
                list_node_level.append(node)
        # Loop through nodes of the current level
        for ind_node in list_node_level:
            # Get the index of the parent node
            ind_parent = dic_node[ind_node]['parent'][0]
            # Check if the node is unvalid
            if check_node_unvalid(xy, polygon, ind_node, min_point):
                dic_node[ind_parent]['childrens'] = [ind for ind in dic_node[ind_parent]['childrens'] if ind != ind_node]
                if ind_node in dic_node.keys():
                    del dic_node[ind_node]      
    # Return the modified dictionary of nodes with unvalid leaves removed
    return dic_node

def trim_chain_nodes(dic_node):
    """
    This function removes chain nodes from the dictionary of nodes.
    Parameters:
        dic_node (dict): A dictionary of nodes, each representing a contour.
    Returns:
        dic_node (dict): A modified dictionary of nodes with chain nodes removed.
    """
    list_leaves = find_leaves(dic_node)
    # Find the maximum level in the dictionary
    max_level = -1
    for node in dic_node.keys():
        if dic_node[node]['level'] > max_level:
            max_level = dic_node[node]['level']
    # Loop through levels from max level to 1
    for level in range(max_level, 0, -1):
        list_node_level = []
        # Find all nodes of the current level
        for node in dic_node.keys():
            if dic_node[node]['level'] == level:
                list_node_level.append(node)
        # Loop through nodes of the current level
        for ind_node in list_node_level:
            # Get the index of the parent node and the grandparent node
            ind_parent = dic_node[ind_node]['parent'][0]
            ind_parent_parent = dic_node[ind_parent]['parent'][0]
            # Check if the node has only one child
            if len(dic_node[ind_node]['childrens']) <= 1:
                # Check if the parent node has only one child
                if len(dic_node[ind_parent]['childrens']) == 1:
                    # Remove the node from the children list of the parent
                    dic_node[ind_parent]['childrens'] = [ind for ind in dic_node[ind_parent]['childrens'] if ind != ind_node]
                    # If the node has one child, make it a child of the grandparent node
                    if len(dic_node[ind_node]['childrens']) == 1:
                        ind_children = dic_node[ind_node]['childrens'][0]
                        dic_node[ind_parent]['childrens'] = [*dic_node[ind_parent]['childrens'], *[ind_children]]
                        dic_node[ind_children]['parent'][0] = ind_parent
                    # Delete the node from the dictionary
                    if ind_node in dic_node.keys():
                        del dic_node[ind_node]
    # Return the modified dictionary of nodes with chain nodes removed
    return dic_node

def find_plateau(dic_node):
    """
    This function finds the nodes that are plateaus in the tree of contours.
    Parameters:
        dic_node (dict): A dictionary of nodes, each representing a contour.
    Returns:
        ind_plateau (list): A list of indices of nodes that are plateaus.
    """
    ind_plateau = []
    # Loop through all nodes in the dictionary
    for ind in dic_node.keys():
        # Check if the node has more than one child
        if len(dic_node[ind]['childrens']) > 1:
            ind_plateau.append(ind)
    # Return the list of indices of nodes that are plateaus
    return ind_plateau

def create_graph(polygon, density_values, dic_node):
    """
    This function creates a graph from the tree of contours.
    Parameters:
        polygon (list): A list of polygons representing the contours. Each polygon is a list of x and y coordinates of the contour.
        density_values (list): A list of density values for each contour.
        dic_node (dict): A dictionary of nodes, each representing a contour.
    Returns:
        G (Graph): a graph of the tree of contours.
    """
    G = nx.Graph()
    # Loop through all nodes in the dictionary
    for node in dic_node.keys():
        # Add node to the graph
        G.add_node(node)
        G.add_node(dic_node[node]['parent'][0])
        # Add edge between node and its parent, with a weight of random value between 0 and eps
        if dic_node[node]['parent'][0] == '**' or dic_node[node]['parent'][0] == '*' :
            G.add_edge(node, dic_node[node]['parent'][0], weight = (0, np.random.uniform(0, 1)))
        else:
            # Add edge between node and its parent, with a weight of random value between 0 and eps + density value of the parent
            G.add_edge(node, dic_node[node]['parent'][0], weight = (density_values[dic_node[node]['parent'][0]], np.random.uniform(0, 1)))
    # Return the built graph G.
    return G

def path_max(G, s, t, current_depth):
    """
    Finds the maximum weight path between two nodes s and t in a graph G.
    Parameters:
        G (Graph): The input graph.
        s (int): The starting node. 
        t (int): The target node.
        current_depth (int): The current depth of recursion. This value is used to keep track of the recursion depth.
    Returns:
        (float): The maximum weight path between s and t in G.
    """
    # Check if the graph only has one edge and return its weight
    if len(list(G.edges.data())) == 1:
        return list(G.edges.data())[0][2]['weight'][0]
    # Check if the starting and target nodes are the same
    if s == t:
        return 0
    # Create an array of all edge weights in G
    weight = np.array([w[0] for u, v, w in G.edges(data = 'weight')])
    rand_weight = np.array([w[1] for u, v, w in G.edges(data = 'weight')])
    # Calculate the median weight of the edges
    median = np.median(weight)
    rand_median = np.median(rand_weight)
    # Create a new graph G_k and add all nodes from G to it
    G_k = nx.Graph()
    G_k.add_nodes_from(G)
    # Add edges to G_k if their weight is greater than or equal to the median weight
    if len(np.unique(weight)) == 1:
        return median
    for edge in list(G.edges.data()):
        if edge[2]['weight'][0] > median:
            G_k.add_edge(edge[0], edge[1], weight = edge[2]['weight'])
        if edge[2]['weight'][0] == median:
            if edge[2]['weight'][1] >= np.random.uniform(0, 1):
                G_k.add_edge(edge[0], edge[1], weight = edge[2]['weight'])
    # Check if the target node is reachable from the starting node in G_k
    if t in list(nx.node_connected_component(G_k, s)):
        # If it is, recursively call the function with the subgraph containing s and t and increment current depth
        return path_max(G_k.subgraph(nx.node_connected_component(G_k, s)), s, t, current_depth + 1)
    else:
        # Create a list of all connected components in G_k
        S = [G_k.subgraph(c) for c in nx.connected_components(G_k)]
        G_bar_k = nx.Graph()
        x = None
        y = None
        # For each connected component, check if it contains s or t
        for u, G_u in enumerate(S):
            G_bar_k.add_node(u)
            if s in list(G_u.nodes()):
                x = u
            if t in list(G_u.nodes()):
                y = u
            # Check the maximum weight between each component and if it exist add edge to G_bar_k
            for v, G_v in enumerate(S):
                if u != v:
                    maximum_weight = None
                    maximum_link = [None, None]
                    for node_u in list(G_u.nodes()):
                        for node_v in list(G_v.nodes()):
                            if node_v in list(G.neighbors(node_u)):
                                w = G[node_u][node_v]['weight'][0]
                                rand_value = G[node_u][node_v]['weight'][1]
                                if maximum_weight == None:
                                    maximum_weight = w
                                    maximum_link[0] = node_u
                                    maximum_link[1] = node_v 
                                else:
                                    if w > maximum_weight:
                                        maximum_weight = w
                                        maximum_link[0] = node_u
                                        maximum_link[1] = node_v
                    # If there is a maximum weight edge between the two connected components, add it to G_bar_k
                    if maximum_weight != None:
                        G_bar_k.add_edge(u, v, weight = (maximum_weight, np.random.uniform(0, 1)))
    # Call recursively the function with G_bar_k as the new graph, x and y as the new starting and target nodes, and increment current depth
    return path_max(G_bar_k, x, y, current_depth + 1)  

def calculate_maximum_flow(G, peak_center_density, ind_peak):
    """
    Calculates the maximum flow between any two peak centers in a graph G.
    Parameters:
        G (Graph): a graph of the tree of contours.
        peak_center_density (list): A list of densities of peak centers.
        ind_peak (list): A list of indices corresponding to the peak centers.
    Returns:
        (tuple): A tuple containing the maximum flow, the dens_list, and the dens_df.
    """
    # Create a DataFrame from the input peak center densities and indices
    dens_df = pd.DataFrame(peak_center_density, columns = ['density'])
    dens_df.index = ind_peak
    # Sort the DataFrame by density in descending order
    dens_df = dens_df.sort_values(by = ['density'], ascending = False)
    # Convert the DataFrame to a numpy array
    dens_list = dens_df.to_numpy()
    # Get the indices of the peak centers
    ind_cand = dens_df.index
    # Initialize an empty list to store the maximum flow values
    max_flow = []
    # Iterate over the peak centers
    for ind, i in enumerate(ind_cand):
        # Initialize the maximum flow as negative infinity
        maximum = - np.inf
        # Iterate over the peak centers with a lower density than the current center
        for j in ind_cand[:ind]:
            # Call the path_max function to find the maximum flow between the current peak center and the other peak center
            current_depth = 0
            flow = path_max(G, i, j, current_depth)
            # Update the maximum flow if the current flow is greater than the current maximum flow
            if flow > maximum:
                maximum = flow
        # Check if the maximum flow is negative infinity
        if maximum == - np.inf:
            # Check if it is the first peak center
            if ind == 0:
                max_flow.append(0)
            else:
                max_flow.append(np.inf)
        else:
            # Otherwise, append the maximum flow to the max_flow list
            max_flow.append(maximum)
    # Return the maximum flow list, the dens_list, and the dens_df
    return max_flow, dens_list, dens_df


def get_well_separated_points(max_flow, dens_list, dens_df, selection_method):
    """
    Returns the well-separated points from the input data.
    Parameters:
        max_flow (list): A list of maximum flow values.
        dens_list (array): An array of peak center densities.
        dens_df (pd.DataFrame): A DataFrame containing the density values.
        selection_method (str/int): Method to select well-separated peaks. Can be 'first_gap', 'second_gap', 'third_gap', an int or 'all'.
    Returns:
        (array): An array of indices of the well-separated points.
    """
    # Calculate the separability values
    nb_values = len(max_flow)
    separability = np.array([1 - max_flow[i]/dens_list[i] for i in range(nb_values)])
    # Add the separability values to the dens_df DataFrame
    dens_df['separability'] = separability
    # Sort the dens_df DataFrame by separability in descending order
    dens_df = dens_df.sort_values(by = ['separability'], ascending = False)
    # Check the selection method
    if selection_method == 'first_gap':
        # Select the well-separated points
        well_separated = dens_df[dens_df['separability'] > 0]
        # Append a row with negative infinity values
        well_separated.loc['*',:] = [0, 0]
        # Calculate the gaps between separability values
        well_separated['gap'] = - well_separated.separability.diff()
        # Reset the index of the DataFrame
        well_separated = well_separated.reset_index(drop = False)
        # Print the cluster center candidates separabilities
        print(well_separated.loc[:,['separability', 'gap']].round(2))
        # Get the index of the first gap
        first_gap_index = (-well_separated['gap']).argsort()[::-1][1]
        # Return the indices of the well-separated points up to the first gap
        return well_separated.iloc[:first_gap_index + 1]['index'].to_numpy()
    if selection_method == 'second_gap':
        # Select the well-separated points
        well_separated = dens_df[dens_df['separability'] > 0]
        # Append a row with negative infinity values
        well_separated.loc['*',:] = [0, 0]
        # Calculate the gaps between separability values
        well_separated['gap'] = -well_separated.separability.diff()
        # Reset the index of the DataFrame
        well_separated = well_separated.reset_index(drop = False)
        # Print the cluster center candidates separabilities
        print(well_separated.loc[:,['separability', 'gap']].round(2))
        # Get the index of the second gap
        second_gap_index = (-well_separated['gap']).argsort()[::-1][2]
        # Return the indices of the well-separated points up to the second gap
        return well_separated.iloc[:second_gap_index + 1, :]['index'].to_numpy()
    if selection_method == 'third_gap':
        # Select the well-separated points
        well_separated = dens_df[dens_df['separability'] > 0]
        # Append a row with negative infinity values
        well_separated.loc['*',:] = [0, 0]       
        # Calculate the gaps between separability values
        well_separated['gap'] = -well_separated.separability.diff()
        # Reset the index of the DataFrame
        well_separated = well_separated.reset_index(drop = False)
        # Print the cluster center candidates separabilities
        print(well_separated.loc[:,['separability', 'gap']].round(2))
        # Get the index of the third gap
        third_gap_index = (-well_separated['gap']).argsort()[::-1][3]
        # Return the indices of the well-separated points up to the second gap
        return well_separated.iloc[:third_gap_index + 1, :]['index'].to_numpy()
    if isinstance(selection_method, int):
        # If the selection method is an integer, return the first n indices where n is the selection_method
        return dens_df[dens_df['separability'] > 0].iloc[:selection_method].index.to_numpy()
    if selection_method == 'all':
        # Select the well-separated points
        well_separated = dens_df[dens_df['separability'] > 0]
        # Append a row with negative infinity values
        well_separated.loc['*',:] = [0, 0]
        # Calculate the gaps between separability values
        well_separated['gap'] = -well_separated.separability.diff()
        # Reset the index of the DataFrame
        well_separated = well_separated.reset_index(drop = False)
        # Print the cluster center candidates separabilities
        print(well_separated.loc[:,['separability', 'gap']].round(2))
        # If the selection method is 'all', return all the indices where separability is greater than 0
        return dens_df[dens_df['separability'] > 0].index.to_numpy()
    else:
        # If the inputed selection method is invalid, raise an error message
        raise ValueError("Invalid selection method. Must be 'first_gap', 'second_gap', 'third_gap', an integer or 'all'.")

def find_closest_point(point, points):
    """
    Finds the closest point to the input point from a set of input points.
    Parameters:
        point (array): An array representing the point whose closest point is to be found.
        points (array): An array of points.
    Returns:
        (tuple): A tuple with the closest point, the index of closest point, and the distance to closest point.
    """
    # Create a KDTree using the input points
    tree = KDTree(points)
    # Find the distance and index of the closest point to the input point using the KDTree
    dist, idx = tree.query(point)
    # Return the closest point, the index of the closest point and the distance to closest point
    return points[idx], idx, dist

def clean_dic_after_choice(ind_selection, dic_node):
    """
    Cleans the input dictionary by removing nodes and updating tokens after the selection of points.
    Parameters:
        ind_selection (list): A list of selected point indices.
        dic_node (dict): A dictionary of nodes, each representing a contour.
    Returns:
        dic_node (dict): the cleaned dictionary.
    """
    # Find the leaves in the dictionary
    list_leaves = find_leaves(dic_node)
    # Add the leaf indices to the token list of the leaf nodes
    for leaf in list_leaves:
        dic_node[leaf]['token'].append(leaf)
    # Find the maximum level of the tree
    max_level = -1
    for node in dic_node.keys():
        if dic_node[node]['level'] > max_level:
            max_level = dic_node[node]['level']
    # Iterate over the levels of the tree in reverse order
    for level in range(max_level, -1, -1):
        list_node_level = []
        # Get the list of nodes at the current level
        for node in dic_node.keys():
            if dic_node[node]['level'] == level:
                list_node_level.append(node)
        # Iterate over the nodes at the current level
        for node in list_node_level:
            token_to_keep = []
            # Get the tokens that need to be kept
            for token in dic_node[node]['token']:
                if token in ind_selection:
                    token_to_keep.append(token)
            # Get the parent node of the current node
            ind_parent = dic_node[node]['parent'][0]
            if ind_parent != '**':
                if len(dic_node[node]['childrens']) > 1:
                    dic_node[node]['token'] = token_to_keep
                    # If the current node has more than one child
                    if len(token_to_keep) == 0:
                        # If there are no tokens to keep, delete the current node
                        if node in dic_node.keys():
                            del dic_node[node]
                        dic_node[ind_parent]['childrens'] = [ind for ind in dic_node[ind_parent]['childrens'] if ind != node]
                    else:
                        # If there are tokens to keep, update the token list of the parent node
                        dic_node[ind_parent]['token'] = [*dic_node[ind_parent]['token'], *token_to_keep] 
                else:
                    # If the current node has only one child
                    if len(token_to_keep) == 0:
                        # If there are no tokens to keep, delete the current node
                        if node in dic_node.keys():
                            del dic_node[node]
                        dic_node[ind_parent]['childrens'] = [ind for ind in dic_node[ind_parent]['childrens'] if ind != node]              
                    else:
                        # If there are tokens to keep, update the token list of the parent node
                        dic_node[ind_parent]['token'] = [*dic_node[ind_parent]['token'], *dic_node[node]['token']]
    # Remove unnecessary chain nodes
    dic_node = trim_chain_nodes(dic_node)
    # Update the token list of the root node
    dic_node['*']['token'] = ind_selection
    # Return the cleaned dictionary
    return dic_node

def compute_density_peak(xy, polygon, ind_peak, density):
    """
    This function finds the point of maximum density in a contour.
    Parameters:
        xy (list): A list of the x,y coordinates of data points.
        polygon (list): A list of polygons representing the contours. Each polygon is a list of x and y coordinates of the contour.
        ind_peak (list): A list of indices corresponding to the peak centers.
        density (list): List of density values of the points.
    Returns: 
        (tuple): Tuple containing the density peak coordinates and the density values.
    """
    peak_center = []
    peak_center_density = []
    for ind in ind_peak:
        if ind != '*':
            # Create a polygon from the polygon data at the current index
            poly_current = Polygon([(polygon[ind][0][index], polygon[ind][1][index]) for index in range(len(polygon[ind][0]))])
            list_den = []
            list_cen = []
            for k, point in enumerate(xy):
                point_current = Point(point[0], point[1])
                if point_current.within(poly_current):
                    # Append the density value of the point if it is within the current polygon
                    list_den.append(density[k])
                    list_cen.append([point[0], point[1]])
            # Append the maximum density value found within the current polygon
            peak_center_density.append(np.max(list_den))
            # Append the coordinates of the point with the maximum density value found within the current polygon
            peak_center.append(list_cen[np.argmax(list_den)])
    # Return the density peak coordinates and the density values
    return peak_center, peak_center_density

def get_point_polygon(density, xy, polygon, dic_node):
    """
    Given a set of points and their corresponding density and polygon, the function returns a dictionary containing the points included inside each contour.
    Parameters:
        density (list): List of density values of the points.
        xy (list): A list of the x,y coordinates of data points.
        polygon (list): A list of polygons representing the contours. Each polygon is a list of x and y coordinates of the contour.
        dic_node (dict): A dictionary of nodes, each representing a contour.
    Returns:
        dic_point_node (dict): A dictionary containing the points and their corresponding density and coordinates for each contour.
    """
    dic_point_node = {ind: [] for ind in dic_node.keys()}
    # Iterate through each node in the density tree
    for ind_node in dic_node.keys():
        if ind_node != '*':
            poly_current = Polygon([(polygon[ind_node][0][index], polygon[ind_node][1][index]) for index in range(len(polygon[ind_node][0]))])
            #iterate through each point, check if it is within the current polygon
            for k, point in enumerate(xy):
                point_current = Point(point[0], point[1])   
                if point_current.within(poly_current):
                    dic_point_node[ind_node].append([k, point, density[k]])
    # Add all points to the root of the density tree
    dic_point_node['*'] = [[k, xy[k], density[k]] for k in range(len(xy))]
    # Return the dictionary containing the points and their corresponding density and coordinates for each contour
    return dic_point_node  

def assign_points(xy, dic_point_node, dic_node):
    """
    This function assigns a cluster to each point in xy.
    Parameters:
        xy (list): A list of the x,y coordinates of data points.
        dic_point_node (dict): A dictionary containing the points and their corresponding density and coordinates for each contour.
        dic_node (dict): A dictionary of nodes, each representing a contour.
    Returns:
        assignment (array): An array of the assigned values for each point in xy.
    """
    # Create an array to store the assigned values
    assignment = np.array([None for i in range(len(xy))])
    # Find the maximum level of the nodes in dic_node
    max_level = -1
    for node in dic_node.keys():
        if dic_node[node]['level'] > max_level:
            max_level = dic_node[node]['level']
    # Loop through the levels of the nodes in reverse order
    for level in range(max_level, -1, -1):
        # Create a list of nodes at the current level
        list_node_level = []
        for node in dic_node.keys():
            if dic_node[node]['level'] == level:
                list_node_level.append(node)
        # Loop through the nodes at the current level
        for ind_node in list_node_level:
            # If the current node is a leaf node
            if len(dic_node[ind_node]['childrens']) == 0:
                # Assign the value of the token to all points that belong to the current node
                for info_point in dic_point_node[ind_node]:
                    k = info_point[0]
                    assignment[k] = int(dic_node[ind_node]['token'][0])
            # If the current node is not a leaf node
            else:
                # Lists to store the information of points that have already been assigned a value, and points that have not been assigned a value
                already_assigned = []
                not_assigned = []
                already_assigned_value = []
                already_assigned_point = []
                not_assigned_point = []
                not_assigned_den = []
                # Split the points that belong to the current node into two groups: points that have already been assigned a value, and points that have not been assigned a value
                for info_point in dic_point_node[ind_node]:
                    k = info_point[0]
                    if assignment[k] != None:
                        already_assigned.append
                        already_assigned_value.append(int(assignment[k]))
                        already_assigned_point.append(info_point[1])
                    else:
                        not_assigned.append(k)
                        not_assigned_point.append(info_point[1])
                        not_assigned_den.append(info_point[2])
                # Sort the not assigned points by density in descending order
                if len(not_assigned)!=0:
                    not_assigned_den, not_assigned, not_assigned_point = list(zip(*sorted(zip(not_assigned_den, not_assigned, not_assigned_point), key=lambda x: x[0], reverse=True)))
                    # Assign the value of the closest point that has already been assigned a value to each point that has not been assigned a value
                    for ind in range(len(not_assigned)):
                        assignment[not_assigned[ind]] = int(already_assigned_value[find_closest_point(not_assigned_point[ind], already_assigned_point)[1]])
    # Return the assigned values
    return assignment

def clusterdc(x_points, y_points, levels = 50, num_points = 100, min_point = 1, border_fraction = 0.5, bw_method = 'scott', selection_method = 'first_gap', bw = None):
    """
    This function is used to perform density-based clustering of the given points using the clusterdc approach.
    Parameters:
        x_points (list): The list of x-coordinates.
        y_points (list): The list of y-coordinates.
        levels (int): The number of contour levels to create.
        num_points (int): The number of points used for each side of the grid.
        border_fraction (float): The fraction to use as border size.
        bw_method (str/float): The method used for computing the bandwidth.
        selection_method (str/int): Method to select well-separated peaks. Can be 'first_gap', 'second_gap', 'third_gap', an int or 'all'.
        bw (array): The bandwidth used at each point if the bw_method is 'local'.
        min_point (int): the minimum number of points to include inside a valid polygon.
    Returns:
        density_info (list): A list containing all the information to plot the KDE function.
        assignment (array): An array of the assigned values for each point in xy.
        list_assignment (list): A list of array of the assigned values, if the 'all' selection method is used.
    """
    # Create a list of [x, y] arrays representing the points in the dataset 
    xy = np.array([[x_points[i], y_points[i]] for i in range(len(x_points))])    
    # Set bandwidth to None if bw_method is not 'local'
    if bw_method != 'local':
        bw = None
        kernel = gaussian_kde(np.vstack([x_points, y_points]), bw_method)
    else:
        kernel = None    
    # Create a grid of points
    xx, yy = create_grid(x_points, y_points, border_fraction = border_fraction, num_points = num_points)
    # Evaluate the kde on the grid
    f = evaluate_kde(x_points = x_points, y_points = y_points, xx = xx, yy = yy, bw_method = bw_method, kernel = kernel, bw = bw)
    # Create contours of the density levels
    polygon, density_values, density_contours = create_contours(xx, yy, f, levels = levels)
    # Create dic_node of the polygons of each density level
    dic_node = make_node_dic(polygon, density_values, density_contours)
    # Trim unvalid leaves and chain nodes
    dic_node = trim_unvalid_leaves(xy, dic_node, polygon, min_point)
    dic_node = trim_chain_nodes(dic_node)
    # Find the plateau and peak nodes
    ind_plateau = find_plateau(dic_node)
    ind_peak = find_leaves(dic_node)
    # Evaluate the density of each point
    density = []        
    for point in xy:
        density.append(get_kde_point(point = point, bw_method = bw_method, kernel = kernel, bw = bw, x_points = x_points, y_points = y_points))
    # Compute density peak for each peak node
    peak_center, peak_center_density = compute_density_peak(xy, polygon, ind_peak, density)
    # Create a graph of the polygons
    G = create_graph(polygon, density_values, dic_node)
    # Calculate the maximum flow
    max_flow, dens_list, dens_df = calculate_maximum_flow(G, peak_center_density, ind_peak)
    # Find the well separated peaks
    ind_selection = list(get_well_separated_points(max_flow, dens_list, dens_df, selection_method))
    ind_peak = list([int(k) for k in ind_peak])
    if selection_method != 'all':        
        # Get indices of the selected peak nodes
        indices = [ind for ind, k in enumerate(ind_peak) if (k in ind_selection)]
        # Find the peak center and the peak center density for the selected peak nodes
        peak_center_selection = [peak_center[ind] for ind in indices]
        peak_center_selection_density = [peak_center_density[ind] for ind in indices]        
        # Clean dic_node after selection of peak nodes
        dic_node = clean_dic_after_choice(ind_selection, dic_node)        
        # Update ind_peak
        ind_peak = find_leaves(dic_node)       
        # Get the points inside each polygon in dic_point_node
        dic_point_node = get_point_polygon(density, xy, polygon, dic_node)        
        # Assign points to clusters
        assignment = assign_points(xy, dic_point_node, dic_node)
        # Reindex the clusters with indexes starting from 0
        dic_reassignment = {selection:index for index,selection in enumerate(ind_selection)}
        assignment = [dic_reassignment[selection] for selection in assignment]
        # Return the list of the cluster assignment per data point and information on the KDE function
        density_info = [density, f, xx, yy]
        print('============================')
        print('Number of clusters: ', len(np.unique(assignment)))
        print('============================')
        return assignment, density_info
    else:
        # Get the points inside each polygon in dic_point_node
        dic_point_node = get_point_polygon(density, xy, polygon, dic_node)
        list_assignment = []
        for j in range(len(ind_selection)):
            dic_node_sub = copy.deepcopy(dic_node)
            # Get the current subset of peak selections
            ind_selection_sub = ind_selection[:j+1]
            # Get the indices of the selected peaks
            indices_sub = [ind for ind, k in enumerate(ind_peak) if (k in ind_selection_sub)]
            # Get the coordinates of the selected peaks
            peak_center_selection_sub = [peak_center[ind] for ind in indices_sub]
            # Get the density values of the selected peaks
            peak_center_selection_density_sub = [peak_center_density[ind] for ind in indices_sub]
            # Clean the dictionary of nodes
            dic_node_sub = clean_dic_after_choice(ind_selection_sub, dic_node_sub)
            # Find the leaves of the tree
            ind_peak_sub = find_leaves(dic_node_sub)
            # Assign points to clusters
            assignment_sub = assign_points(xy, dic_point_node, dic_node_sub)
            # Reindex the clusters with indexes starting from 0
            dic_reassignment = {selection:index for index,selection in enumerate(ind_selection)}
            assignment_sub = [dic_reassignment[selection] for selection in assignment_sub]
            # Clear the dictionary of nodes
            dic_node_sub.clear()
            # Append the current cluster assignments to the list of all assignments
            list_assignment.append(assignment_sub)
        # Return the list of all cluster assignments and information on the KDE function
        density_info = [density, f, xx, yy]
        print('============================')
        print('Max. number of clusters: ', len(np.unique(assignment_sub)))
        print('============================')
        return list_assignment, density_info