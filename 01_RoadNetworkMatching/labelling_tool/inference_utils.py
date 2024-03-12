import copy
import os
from pathlib import Path
import warnings

from frechetdist import frdist
from IPython.utils import io
import numpy as np
from numpy.linalg import norm
import pandas as pd
from rtree import index
from shapely.geometry import LineString

import preprocessing, similarity, strokeutils

def get_stroke_coords(net, _id, mid_coordinates, is_osm, net_osm, net_tomtom):
    coords_before, coords_after = strokeutils.get_whole_stroke(net.getEdge(_id), mid_coordinates, net_osm, net_tomtom, is_osm)
    stroke_coords = coords_before + mid_coordinates + coords_after
    return stroke_coords

def frechet_dist(ls1, ls2):
    dist = frdist(ls1, ls2)
    return dist

def calculate_linestring_length(coords):
    length = 0
    for idx, coord in enumerate(coords):
        if idx > 0:
            length += norm(np.array(coord)-np.array(coords[idx-1]))
    return length

def overlap_for_shorter(ls1_orig, ls2_orig, ls1, ls2):
    length_ls1orig = calculate_linestring_length(ls1_orig)
    length_ls2orig = calculate_linestring_length(ls2_orig)
    
    if length_ls1orig < length_ls2orig:
        length_ls1 = calculate_linestring_length(ls1)
        return length_ls1 / length_ls1orig
    else:
        length_ls2 = calculate_linestring_length(ls2)
        return length_ls2 / length_ls2orig
    
def calculate_features(ls1_orig, ls2_orig, ls1, ls2, prefix = ''):
    
    metrics = dict()
    
    # get overlap ratio for shorter linestring
    metrics['{}overlap_shorter'.format(prefix)] = overlap_for_shorter(ls1_orig, ls2_orig, ls1, ls2)

    # get curvature comparison
    metrics['{}curvature_difference'.format(prefix)]= similarity.compare_curvatures(ls1, ls2)
    
    # get hausdorff distance
    metrics['{}hausdorff'.format(prefix)] = similarity.hausdorff_dist(ls1, ls2)
    metrics['{}hausdorff_mod'.format(prefix)] = similarity.hausdorff_mod(ls1, ls2)
    metrics['{}frechet'.format(prefix)] = frechet_dist(ls1, ls2)

    length_diff = abs(calculate_linestring_length(ls1) - calculate_linestring_length(ls2))
    metrics['{}hausdorff_mod_to_distdiff'.format(prefix)] = metrics['{}hausdorff_mod'.format(prefix)] / length_diff
    
    # get sinuosity feature
    sinuosity_ls1 = similarity.sinuosity(ls1)
    sinuosity_ls2 = similarity.sinuosity(ls2)
    metrics['{}sinuosity_sim'.format(prefix)] = min(sinuosity_ls1, sinuosity_ls2) / max(sinuosity_ls1, sinuosity_ls2)
    
    return metrics

def get_candidates(reference_id, reference_coordinates, radius, tomtom_rtree_idx, net_tomtom, net_osm, stroke = True):
    reference_linestring = LineString(reference_coordinates)
    search_buffer = reference_linestring.buffer(radius)
    
    if stroke:
        reference_stroke_coords = get_stroke_coords(net_osm, reference_id, reference_coordinates, True, net_osm, net_tomtom)

    candidate_indices = list(tomtom_rtree_idx.intersection(search_buffer.bounds))
    
    data = dict()
        
    for idx in candidate_indices:
        idx = str(idx)
        candidate_edge = net_tomtom.getEdge(idx)
        candidate_id = candidate_edge.getID()
        if str(idx) != candidate_id:
            print(str(idx))
            print(candidate_id)
            raise ValueError('there is something wrong')
        candidate_coords = candidate_edge.getShape()
        ls1, ls2 = preprocessing.pruning(copy.deepcopy(reference_coordinates), copy.deepcopy(candidate_coords), radius, False)
        if stroke:
            candidate_stroke_coords = get_stroke_coords(net_tomtom, candidate_id, candidate_coords, False, net_osm, net_tomtom)
            stroke_ls1, stroke_ls2 = preprocessing.pruning(copy.deepcopy(reference_stroke_coords), copy.deepcopy(candidate_stroke_coords), radius, True)
        
        if ls1 is not None and ls2 is not None:
            angle_similarity  = similarity.cosine_sim(ls1, ls2)
            print('angle: {}'.format(angle_similarity))
            if angle_similarity > 0:
                ls1, ls2 = preprocessing.control_lengths(ls1, ls2)
                if stroke:
                    stroke_ls1, stroke_ls2 = preprocessing.control_lengths(stroke_ls1, stroke_ls2)
                features = calculate_features(copy.deepcopy(reference_coordinates), 
                                              copy.deepcopy(candidate_coords), ls1, ls2)
                if stroke:
                    stroke_features = calculate_features(copy.deepcopy(reference_stroke_coords), 
                                                  copy.deepcopy(candidate_stroke_coords), stroke_ls1, stroke_ls2, 'stroke_')
                    stroke_angle_similarity  = similarity.cosine_sim(stroke_ls1, stroke_ls2)
                features['cosine_sim'] = angle_similarity
                features['coords_ls1'] = ls1
                features['coords_ls2'] = ls2
                features['ls1_orig'] = copy.deepcopy(reference_coordinates)
                features['ls2_orig'] = copy.deepcopy(candidate_coords)
                if stroke:
                    for key in stroke_features.keys():
                        features[key] = stroke_features[key]
                    features['stroke_cosine_sim'] = stroke_angle_similarity
                data[candidate_id] = features
    print('data: {}'.format(data))
    return data

def save_to_json(df, filename, target_dir):
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
    df.to_json(os.path.join(target_dir, filename))

def create_tomtom_index(net_tomtom):
    intids = []
    tomtom_rtree_idx = index.Index()

    for edge in net_tomtom.getEdges(withInternal=False):
        tomtom_id = edge.getID()
        if int(tomtom_id) in intids:
            raise ValueError("Int value already in idlist. You need another mapping.")
        tomtom_id = int(tomtom_id)        
        shape = edge.getShape()
        shape = LineString(shape)
        bounding_box = shape.bounds
        tomtom_rtree_idx.insert(tomtom_id, bounding_box)
    return tomtom_rtree_idx

def create_candidate_df(net_osm, net_tomtom, radius, tomtom_rtree_idx):

    candidate_dfs = []

    with io.capture_output() as captured: # do not show printing

        for edge in net_osm.getEdges(withInternal=False):  
            reference_coordinates = preprocessing.get_transformed_coordinates(edge, net_osm, net_tomtom)
            reference_id = edge.getID()
            candidates = get_candidates(reference_id, reference_coordinates, radius, tomtom_rtree_idx, net_tomtom, net_osm)
            tomtom_ids = list(candidates.keys())  
            if len(tomtom_ids) > 0:
                osm_names = [reference_id for i in range(len(tomtom_ids))]
                candidate_df = pd.DataFrame({'OSM':osm_names, 'Tomtom':tomtom_ids})

                feature_names = candidates[tomtom_ids[0]].keys()
                for name in feature_names:
                    candidate_df[name] = [candidates[tomtom_ids[i]][name] for i in range(len(tomtom_ids))]            

                candidate_dfs.append(candidate_df)
                
    candidate_df = pd.concat(candidate_dfs).reset_index(drop = True)
    
    return candidate_df

def preprocess_data(net_osm, net_tomtom, tomtom_rtree_idx, radius, preprocessed_file):
    candidate_df = create_candidate_df(net_osm, net_tomtom, radius, tomtom_rtree_idx)
    save_to_json(candidate_df, preprocessed_file)
    return candidate_df

def filecheck(out_dir, preprocessed_file):
    preprocess = True
    if Path(os.path.join(os.path.dirname(os.path.realpath('__file__')), out_dir, preprocessed_file)).is_file():
        preprocess = False
    return preprocess

def get_candidate_df(out_dir, preprocessed_file, net_osm, net_tomtom, radius):
    preprocess = filecheck(out_dir, preprocessed_file)

    if preprocess:
        warnings.warn("No preprocessed file found. Executing entire preprocessing.")
        tomtom_rtree_idx = create_tomtom_index(net_tomtom)
        candidate_df = preprocess_data(net_osm, net_tomtom, tomtom_rtree_idx, radius, preprocessed_file)
    else:
        candidate_df = pd.read_json(os.path.join(out_dir, preprocessed_file))
        candidate_df['Tomtom'] = candidate_df['Tomtom'].astype('str')
        
    return candidate_df