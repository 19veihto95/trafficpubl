import preprocessing
import similarity

threshold = 0.8

def select(net, edge_list, idx, ref_coords, net_tomtom, osm):
    if osm:
        edge = net.getEdge(edge_list[idx].getID())
        co = preprocessing.get_transformed_coordinates(edge, net, net_tomtom)
    else:
        edge = net_tomtom.getEdge(edge_list[idx].getID())
        co = edge.getShape()
    sim = similarity.cosine_sim(ref_coords, co)
    if sim <= threshold:
        edge = None
        sim = -1
    return edge, sim

def handle_multiple(edge_list, net, ref_coords, net_tomtom, osm):
    max_sim = -1
    new_edge = None
    for i, e in enumerate(edge_list):
        ed, sim = select(net, edge_list, i, ref_coords, net_tomtom, osm)
        if sim > max_sim:
            max_sim = sim
            new_edge = ed
    if max_sim > threshold:
        return new_edge
    return None


def get_edge_after(edge, net, ref_coords, net_tomtom, osm):
    after = None
    edge_list = list(edge.getOutgoing().keys())
    edge_count = len(edge_list)
    if edge_count == 1:
        after, _ = select(net, edge_list, 0, ref_coords, net_tomtom, osm)
    elif edge_count > 1:
        return handle_multiple(edge_list, net, ref_coords, net_tomtom, osm)
    return after

def get_edge_before(edge, net, ref_coords, net_tomtom, osm):
    before = None
    edge_list = list(edge.getIncoming().keys())
    edge_count = len(edge_list)
    if edge_count == 1:
        before, _ = select(net, edge_list, 0, ref_coords, net_tomtom, osm)
    elif edge_count > 1:
        return handle_multiple(edge_list, net, ref_coords, net_tomtom, osm)
    return before


def get_edges_before(edge, reference_coordinates, net_osm, net_tomtom, osm):
    edges_before = []
    
    edge_before = get_edge_before(edge, net_osm, reference_coordinates, net_tomtom, osm)
    
    while edge_before is not None:
        if osm:
            coords = preprocessing.get_transformed_coordinates(edge_before, net_osm, net_tomtom)
        else:
            coords = edge_before.getShape()
        edges_before.append(edge_before)
        edge_before = get_edge_before(edge_before, net_osm, coords, net_tomtom, osm)
        
    return edges_before
    
def get_edges_after(edge, reference_coordinates, net_osm, net_tomtom, osm):
    edges_after = []
    
    edge_after = get_edge_after(edge, net_osm, reference_coordinates, net_tomtom, osm)
    
    while edge_after is not None:
        if osm:
            coords = preprocessing.get_transformed_coordinates(edge_after, net_osm, net_tomtom)
        else:
            coords = edge_after.getShape()
        edges_after.append(edge_after)
        edge_after = get_edge_after(edge_after, net_osm, coords, net_tomtom, osm)   

    return edges_after

def get_whole_stroke(edge, reference_coordinates, net_osm, net_tomtom, osm):
    edges_before = get_edges_before(edge, reference_coordinates, net_osm, net_tomtom, osm)
    edges_before.reverse()
    
    coords_before = []
    for edge in edges_before:
        if osm:
            coords_before += preprocessing.get_transformed_coordinates(edge, net_osm, net_tomtom)
        else:
            coords_before += edge.getShape()
        

    edges_after = get_edges_after(edge, reference_coordinates, net_osm, net_tomtom, osm)

    coords_after = []
    for edge in edges_after:
        if osm:
            coords_after += preprocessing.get_transformed_coordinates(edge, net_osm, net_tomtom)
        else:
            coords_after += edge.getShape()

    return coords_before, coords_after
