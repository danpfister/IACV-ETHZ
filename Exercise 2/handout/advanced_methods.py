import numpy as np

# Import maxflow (PyMaxflow library) which can be used to solve min-cut problem
import maxflow


def perform_min_cut(unary_potential_foreground, unary_potential_background, pairwise_potential):
    """
    We provide a simple fuction to perform min cut using PyMaxFlow library. You
    may use this function to implement your algorithm if you wish. Feel free to
    modify this function as desired, or implement your own function to perform
    min cut.
    
    TO PASS THE EXERCISE, YOU DO NOT NEED TO USE THIS FRAMEWORK.

    args:
        unary_potential_foreground - A single channel NumPy array specifying the
            source (foreground) unary potentials for each pixel in the image
        unary_potential_background - A single channel NumPy array specifying the
            sink (background) unary potentials for each pixel in the image
        pairwise_potential - A single channel NumPy array specifying the pairwise
            potentials. We assume a graph where each pixel in the image is 
            connected to its four neighbors (left, right, top, and bottom). 
            Furthermore, we assume that the pairwise potential for all these 4
            edges are same, and set to the value of pairwise_potential at that 
            pixel location
    """    
    
    # create graph
    maxflow_graph = maxflow.Graph[float]()
    
    # add a node for each pixel in the image
    nodeids = maxflow_graph.add_grid_nodes(unary_potential_foreground.shape[:2])

    # Add edges for pairwise potentials. We use 4 connectivety, i.e. each pixel 
    # is connected to its 4 neighbors (up, down, left, right). Also we assume
    # that pairwise potential for all these 4 edges are same
    # Feel free to change this if you wish
    maxflow_graph.add_grid_edges(nodeids, pairwise_potential)

    # Add unary potentials
    maxflow_graph.add_grid_tedges(nodeids, unary_potential_foreground, unary_potential_background)

    maxflow_graph.maxflow()
    
    # Get the segments of the nodes in the grid.
    mask_bg = maxflow_graph.get_grid_segments(nodeids)
    mask_fg = (1 - mask_bg.astype(np.uint8))* 255

    return mask_fg