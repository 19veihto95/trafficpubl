import copy
import warnings

from scipy.interpolate import interp1d
from shapely.geometry import LineString, Point
import numpy as np
from numpy.linalg import norm

import similarity

def get_transformed_coordinates(edge, current_net, other_net):
    coords = edge.getShape()
    coords_lonlat = [current_net.convertXY2LonLat(coord[0], coord[1]) for coord in coords]
    coords_transformed = [other_net.convertLonLat2XY(coord[0], coord[1]) for coord in coords_lonlat]
    return coords_transformed


def intersection_on_straight(p1, p2, p3):
    #if p1 == p2:
    #    raise ValueError("p1 must differ from p2. They are the same, thus, this is not a linestring.")
    
    demoninator = (norm(np.array(p2)-np.array(p1))**2)
    u = ((p3[0] - p1[0])*(p2[0]-p1[0]) + (p3[1] - p1[1])*(p2[1]-p1[1])) / demoninator
    p = (p1[0] + u*(p2[0]-p1[0]), p1[1] + u*(p2[1]-p1[1]))
    return p, u

def intersection_on_vector(p1, p2, p3):
    p, u = intersection_on_straight(p1, p2, p3)
    if 0 <= u <= 1:
        return p
    return None

def prune_orig(pos, ls1, ls2, radius):
    p3 = ls2[pos]
    for idx, point in enumerate(ls1[:-1]):
        # no idx -1 because enumerate starts with 1. element, index with 0
        p1 = ls1[idx]
        p2 = ls1[idx +1]
        if p1 == p2:
            warnings.warn('p1 EQUALS p2. This is ls1: {}'.format(ls1))
        
        new_point = intersection_on_vector(p1, p2, p3)
        # maybe here should be a check whether distance between new_point and p3 is < radius.
        # otherwise it should set new_point to None
        if new_point is not None:
            distance = norm(np.array(p3)-np.array(new_point))
            # TODO: This may not be appropriate
            if distance <= 2*radius:
                return new_point, idx
    return None, None  

def prune(pos, ls1, ls2, radius):
    ls1str = copy.deepcopy(ls1)
    ls1str = LineString(ls1str)
    p3 = ls2[pos]
    p3 = Point(p3)
    
    distance_along_line = ls1str.project(p3)        
    length_ls1 = ls1str.length
    if 0 < distance_along_line < length_ls1:    
        new_point = ls1str.interpolate(distance_along_line)
        for idx, point in enumerate(ls1[:-1]):
            if LineString(ls1[:idx+2]).length > distance_along_line:
                distance = norm(np.array(p3)-np.array(new_point))
                # TODO: This may not be appropriate
                if distance <= 2*radius:
                    return (new_point.x, new_point.y), idx
    return None, None   

def shorten_linestring(ls, first_idx, second_idx, first_point, second_point, dist1 = None, dist2 = None):
    if dist1 is None:
        dist1 = first_idx
    if dist2 is None:
        dist2 = second_idx
        
    if dist1 < dist2:
        ls[first_idx] = first_point
        ls[second_idx + 1] = second_point
        ls = ls[first_idx : second_idx + 2]
        #print('First before second')
    else:
        ls[first_idx + 1] = first_point
        ls[second_idx] = second_point
        ls = ls[second_idx : first_idx + 2]
        #print('Second before first')
    return ls


def rework_linestring(ls, first_idx, second_idx, first_point, second_point):
    if first_idx == second_idx:
        start_idx = first_idx
        dist_first_point = norm(np.array(first_point)-np.array(ls[start_idx]))
        dist_second_point = norm(np.array(second_point)-np.array(ls[start_idx]))
        #print('First idx == second idx')
        return shorten_linestring(ls, first_idx, second_idx, first_point, second_point, dist_first_point, dist_second_point)
    else:
        return shorten_linestring(ls, first_idx, second_idx, first_point, second_point)
    

def pruning(ls1, ls2, radius, is_stroke = False):
    
    if len(ls1) < 2:
        warnings.warn("Ls1 must be linestring, not point")
        return None, None
    if len(ls2) < 2:
        warnings.warn("Ls2 must be linestring, not point")
        return None, None
    
    if (len(ls1) == 2) and (ls1[0] == ls1[1]):
        warnings.warn("Ls1 consists of two equal points. Return None.")
        
    if (len(ls2) == 2) and (ls2[0] == ls2[1]):
        warnings.warn("Ls2 consists of two equal points. Return None.")
    
    # Prune first ls
    pos = 0
        
    new_ls1_point_first, idx_ls1_first = prune(pos, ls1, ls2, radius)
    
    pos = -1
        
    new_ls1_point_second, idx_ls1_second = prune(pos, ls1, ls2, radius)
    

    # Prune second ls
    pos = 0
        
    new_ls2_point_first, idx_ls2_first = prune(pos, ls2, ls1, radius)
    
    pos = -1
        
    new_ls2_point_second, idx_ls2_second = prune(pos, ls2, ls1, radius)
    
    
    # Check how much has been pruned
    n_prunes = sum(x is not None for x in [new_ls1_point_first, new_ls1_point_second,
                                          new_ls2_point_first, new_ls2_point_second])
    if n_prunes != 2:    
        warnings.warn("VALUE ERROR: There must be exactly two prunes. There were {} instead".format(n_prunes))
        if n_prunes == 0:
            if is_stroke:
                return ls1, ls2
            else:
                return None, None
        
    if n_prunes == 1:
        if similarity.cosine_sim(ls1, ls2) > 0:
            if new_ls1_point_first is not None:
                ls1[idx_ls1_first] = new_ls1_point_first
                ls1 = ls1[idx_ls1_first :]
            elif new_ls1_point_second is not None:
                ls1[idx_ls1_second + 1] = new_ls1_point_second
                ls1 = ls1[: idx_ls1_second + 2]
            elif new_ls2_point_first is not None:
                ls2[idx_ls2_first] = new_ls2_point_first
                ls2 = ls2[idx_ls2_first:]
            elif new_ls2_point_second is not None:
                ls2[idx_ls2_second + 1] = new_ls2_point_second
                ls2 = ls2[: idx_ls2_second + 2]
        else:
            if new_ls1_point_first is not None:
                ls1[idx_ls1_first + 1] = new_ls1_point_first
                ls1 = ls1[: idx_ls1_first + 2]
            elif new_ls1_point_second is not None:
                ls1[idx_ls1_second] = new_ls1_point_second
                ls1 = ls1[idx_ls1_second :]
            elif new_ls2_point_first is not None:
                ls2[idx_ls2_first + 1] = new_ls2_point_first
                ls2 = ls2[: idx_ls2_first + 2]         
            elif new_ls2_point_second is not None:
                ls2[idx_ls2_second] = new_ls2_point_second
                ls2 = ls2[idx_ls2_second :]    
        
    # If both prunes in ls1
    if new_ls1_point_first is not None and new_ls1_point_second is not None:
        # if ls1 and ls2 have same orientation        
        ls1 = rework_linestring(ls1, idx_ls1_first, idx_ls1_second, new_ls1_point_first, new_ls1_point_second)
        
    # if both prunes in ls2    
    elif new_ls2_point_first is not None and new_ls2_point_second is not None:
        ls2 = rework_linestring(ls2, idx_ls2_first, idx_ls2_second, new_ls2_point_first, new_ls2_point_second)
        
    elif new_ls1_point_first is not None and new_ls2_point_first is not None:
        # they must be in opposite directions
        #print('Case ls1 first, ls2 first')
        ls1[idx_ls1_first + 1] = new_ls1_point_first
        ls1 = ls1[: idx_ls1_first + 2]
        ls2[idx_ls2_first + 1] = new_ls2_point_first
        ls2 = ls2[: idx_ls2_first + 2]          
        
    elif new_ls1_point_first is not None and new_ls2_point_second is not None:
        #print('Case ls1 first, ls2 second')
        # they must be in same directions
        ls1[idx_ls1_first] = new_ls1_point_first
        ls1 = ls1[idx_ls1_first :]
        ls2[idx_ls2_second + 1] = new_ls2_point_second
        ls2 = ls2[: idx_ls2_second + 2]
        
        
    elif new_ls1_point_second is not None and new_ls2_point_first is not None:
        # they must be in same direction
        #print('Case ls1 second, ls2 first')
        ls1[idx_ls1_second + 1] = new_ls1_point_second
        ls1 = ls1[: idx_ls1_second + 2]
        ls2[idx_ls2_first] = new_ls2_point_first
        ls2 = ls2[idx_ls2_first:]
        
        
    elif new_ls1_point_second is not None and new_ls2_point_second is not None:
        # they must be in opposite direction
        #print('Case ls1 second, ls2 second')
        ls1[idx_ls1_second] = new_ls1_point_second
        ls1 = ls1[idx_ls1_second :]
        ls2[idx_ls2_second] = new_ls2_point_second
        ls2 = ls2[idx_ls2_second :]        
      
    return ls1, ls2


def interpolate(x, y, num_points):
    coords_orig = [(x[i], y[i]) for i in range(len(x))]
    ls = LineString(coords_orig)
    
    coords = []
    for i in range(num_points):
        if i == 0:
            point = (x[0], y[0])
        elif i == num_points - 1:
            point = (x[len(x)-1], y[len(x)-1])
        else:
            shapely_point = ls.interpolate(i * (1/num_points), normalized = True)
            point = (shapely_point.x, shapely_point.y)
        coords.append(point)
    return coords

def interpolate_orig(x, y, num_points):
    virtual_x = list(range(len(x)))    
    
    # Create an interpolation function
    interpolation_function = interp1d(virtual_x, x, kind='linear')

    # Generate num_points evenly spaced X coordinates within the range of the given X coordinates
    new_virtual_x = np.linspace(min(virtual_x), max(virtual_x), num_points)

    # Use the interpolation function to calculate the corresponding Y coordinates
    new_x = interpolation_function(new_virtual_x)
    
    # Create an interpolation function
    interpolation_function = interp1d(virtual_x, y, kind='linear')
    
    # Use the interpolation function to calculate the corresponding Y coordinates
    new_y = interpolation_function(new_virtual_x)
    
    coords = [(new_x[i], new_y[i]) for i in range(len(new_x))]
    
    return coords

def control_lengths(sumo, tomtom):
    
    if len(sumo) < len(tomtom):
        x = [i[0] for i in sumo]
        y = [i[1] for i in sumo]
        sumo = interpolate(x, y, len(tomtom))        
        
    elif len(tomtom) < len(sumo):
        x = [i[0] for i in tomtom]
        y = [i[1] for i in tomtom]
        tomtom = interpolate(x, y, len(sumo))
        
    return sumo, tomtom