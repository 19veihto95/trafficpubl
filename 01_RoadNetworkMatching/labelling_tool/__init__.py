import sys

from rtree import index
from shapely.geometry import LineString

from config import config

path_osm_network, path_tomtom_network, path_sumolib = config.read_config()

sys.path.append(r'{}'.format(path_sumolib))
import sumolib


def create_tomtom_rtree(net_tomtom):
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


def get_network(network_name):
    if network_name == 'osm':
        network = sumolib.net.readNet(path_osm_network)
    elif network_name == 'tomtom':
        network = sumolib.net.readNet(path_tomtom_network)
    else:
        raise ValueError('Such a network name does not exist: {}'.format(network_name))
    return network