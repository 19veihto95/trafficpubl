import numpy as np
from numpy import dot
from numpy.linalg import norm
import scipy.spatial as spatial
from scipy.spatial.distance import hamming


def calculate_linestring_length(coords):
    length = 0
    for idx, coord in enumerate(coords):
        if idx > 0:
            length += norm(np.array(coord)-np.array(coords[idx-1]))
    return length

def h(x, y):
    denominator = len(x)
    counter = 0
    for node in x:
        node = np.array(node)
        min_dist = np.inf
        for inner_node in y:
            inner_node = np.array(inner_node)
            dist = norm(node-inner_node)
            if dist < min_dist:
                min_dist = dist
        counter += min_dist
    return counter / denominator
    

def hausdorff_mod(La, Lb):
    h_ab = h(La, Lb)
    h_ba = h(Lb, La)
    return max(h_ab, h_ba)


def get_angle_similarity(tomtom_vec, sumo_vec):
    cos_sim = dot(tomtom_vec, sumo_vec)/(norm(tomtom_vec)*norm(sumo_vec))
    return cos_sim

def vectorize(one, two):
    return tuple([one[i] - two[i] for i in range(len(one))])      

def cosine_sim(sumo, tomtom_xy):
    one = tomtom_xy[0]
    two = tomtom_xy[len(tomtom_xy)-1]
    
    tomtom_vec = vectorize(one, two)
    sumo_vec = vectorize(sumo[0], sumo[len(sumo)-1])
    
    cos_sim = get_angle_similarity(tomtom_vec, sumo_vec)
    return cos_sim


def hausdorff_dist(ls1, ls2):
    dist = spatial.distance.directed_hausdorff(ls1, ls2)
    return dist[0]

def sinuosity(ls):
    length = calculate_linestring_length(ls)
    straight_line_length = norm(np.array(ls[0])-np.array(ls[-1]))
    sinuosity = length / straight_line_length
    return sinuosity

def calculate_curvature(coordinates):
    curvatures = []  
    for i in range(len(coordinates)):
        x, y = coordinates[i]
        if i > 0:
            x_prev, y_prev = coordinates[i - 1]
        else:
            x_prev, y_prev = x, y
        if i < len(coordinates) - 1:
            x_next, y_next = coordinates[i + 1]
        else:
            x_next, y_next = x, y

        dx = x_next - x_prev
        dy = y_next - y_prev

        d2x = x_next - 2 * x + x_prev
        d2y = y_next - 2 * y + y_prev
        epsilon = 1e-8  # Small value to avoid division by nearly zero
        denominator = (dx ** 2 + dy ** 2) ** (3/2)
        if abs(denominator) < epsilon:
            curvature = 0  # or any other suitable value
        else:
            curvature = np.abs((dx * d2y - dy * d2x) / denominator)
        curvatures.append(curvature)  

    return curvatures 


def compare_curvatures(ls1, ls2):
    curvatures_1 = calculate_curvature(ls1)
    curvatures_2 = calculate_curvature(ls2)
    res = hamming(curvatures_1, curvatures_2)
    return res
