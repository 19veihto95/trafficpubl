import copy
import os
import pathlib
from threading import Timer
import webbrowser

import dash
from dash import dcc
from dash import html
from dash import Input, Output, State, ctx
import dash_bootstrap_components as dbc
from flask import Flask
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shapely.geometry import LineString
from waitress import serve

import __init__
import preprocessing
import similarity
import strokeutils



out_dir = pathlib.Path(pathlib.Path(__file__).parent.resolve(), 'data_fin')

if not out_dir.is_dir():
    os.mkdir(out_dir)

def read_new_file():
    print('TRIGGERED: read new file')
    global net_tomtom
    net_tomtom = __init__.get_network('tomtom')
    global net_osm
    net_osm = __init__.get_network('osm')
    print('First ID: {}'.format(net_osm.getEdges(withInternal=False)[0].getID()))
    global tomtom_rtree
    tomtom_rtree = __init__.create_tomtom_rtree(net_tomtom)

def get_candidates(reference_id, reference_coordinates, radius, tomtom_rtree_idx, net_tomtom):
    reference_linestring = LineString(reference_coordinates)
    #print('reference linestring: {}'.format(reference_linestring))
    #print('radius: {}'.format(radius))
    search_buffer = reference_linestring.buffer(radius)

    candidate_indices = list(tomtom_rtree_idx.intersection(search_buffer.bounds))

    #print('Number of candidates: {}'.format(len(candidate_indices)))
    
    data = dict()
        
    for idx in candidate_indices:
        #print('idx: {}'.format(idx))
        idx = str(idx)
        candidate_edge = net_tomtom.getEdge(idx)
        candidate_id = candidate_edge.getID()
        original_shape = candidate_edge.getShape()
        candidate_coords = copy.deepcopy(original_shape)
        #print('candidate_id: {}'.format(candidate_id))
        #print('candidate coords: {}'.format(candidate_coords))
        ##### Preprocessing
        ls1, ls2 = preprocessing.pruning(copy.deepcopy(reference_coordinates), candidate_coords, radius, False)
        if ls1 is not None and ls2 is not None:
            angle_similarity  = similarity.cosine_sim(ls1, ls2)
            #print('angle: {}'.format(angle_similarity))
            if angle_similarity > 0:
                ls1, ls2 = preprocessing.control_lengths(ls1, ls2)
                
                data[candidate_id] = copy.deepcopy(ls2)
                
                #print('Length: {}'.format(length_similarity))
                #print('Angle: {}'.format(angle_similarity))
                #print('Hausdorff: {}'.format(hausdorff_diff))
                #print('Tomtom ID: {}'.format(idx))
                #print('________________________________________________________________________________')
    #print('data: {}'.format(data))
    return data

def set_width(max_x, min_x, max_y, min_y):
    max_width = 600
    x_range = max_x - min_x
    y_range = max_y - min_y
    x_to_y = x_range/y_range

    bigger_x = False

    if x_to_y < 0.3:
        x_to_y = 0.3

    if x_to_y > 5:
        x_to_y = 5

    if x_to_y > 1:
        #width = max_width
        bigger_x = True
    #else:
        #width = max_width * x_to_y
        #return width, x_to_y, bigger_x
    return max_width, x_to_y, bigger_x

def add_plotly_trace(fig, x, y, i, key, color, mode, j):
    fig.append_trace(go.Scatter(
            x=x,
            y=y,
            hovertemplate="%{y}%{_xother}",
            marker = dict(color=color),
            mode = mode,
            name = key
        ), 
                     row=i, col=j)


def plotly_plot(reference_id, reference_coordinates, candidates):

    #fig = go.Figure()

    reference_x = [coord[0] for coord in reference_coordinates]
    reference_y = [coord[1] for coord in reference_coordinates]

    n_subplots = len(candidates)

    candidate_titles = tuple(candidates.keys())

    #print(candidates)
    if n_subplots == 0:
        n_subplots = 1
        candidate_titles = ['No candidate found']

    if n_subplots <= 1:
        n_cols = 1
    else:
        n_cols = 2
    n_rows = int(n_subplots / n_cols) + n_subplots % n_cols

    fig = make_subplots(rows= n_rows, cols=n_cols,
                       subplot_titles = candidate_titles
                       #, vertical_spacing = 0.05
                       )

    i = 1
    j = 1

    #print('N Candidates: {}'.format(len(candidates)))

    min_x = min(reference_x)
    max_x = max(reference_x)
    min_y = min(reference_y)
    max_y = max(reference_y)

    for key in candidates:
        x_tomtom = [coord[0] for coord in candidates[key]]
        y_tomtom = [coord[1] for coord in candidates[key]]

        add_plotly_trace(fig, reference_x, reference_y, i, reference_id, 'green', 'lines', j)
        add_plotly_trace(fig, [reference_x[0]], [reference_y[0]], i, reference_id, 'green', 'markers', j)
        add_plotly_trace(fig, [reference_x[-1]], [reference_y[-1]], i, reference_id, 'red', 'markers', j)
        add_plotly_trace(fig, x_tomtom, y_tomtom, i, key, 'blue', 'lines', j)    
        add_plotly_trace(fig, [x_tomtom[0]], [y_tomtom[0]], i, key, 'green', 'markers', j)
        add_plotly_trace(fig, [x_tomtom[-1]], [y_tomtom[-1]], i, key, 'red', 'markers', j)
        if min(x_tomtom) < min_x:
            min_x = min(x_tomtom)
        if min(y_tomtom) < min_y:
            min_y = min(y_tomtom)
        if max(x_tomtom) > max_x:
            max_x = max(x_tomtom)
        if max(y_tomtom) > max_y:
            max_y = max(y_tomtom)

        if j == 2:
            i += 1
            j -= 1
        else:
            j += 1

    width, x_to_y, bigger_x = set_width(max_x, min_x, max_y, min_y)
    h = n_rows * width / x_to_y
    if h < 250:
        h = 250

    fig.update_layout(height=h, width=n_cols * width, title_text=reference_id,
                     showlegend = False)

    return fig



def plotly_plot_extended(reference_id, reference_coordinates, candidates):

    #fig = go.Figure()

    reference_x = [coord[0] for coord in reference_coordinates]
    reference_y = [coord[1] for coord in reference_coordinates]

    n_subplots = len(candidates)

    candidate_titles = tuple(candidates.keys())

    #print(candidates)
    if n_subplots == 0:
        n_subplots = 1
        candidate_titles = ['No candidate found']

    if n_subplots <= 1:
        n_cols = 1
    else:
        n_cols = 2
    n_rows = int(n_subplots / n_cols) + n_subplots % n_cols

    fig = make_subplots(rows= n_rows, cols=n_cols,
                       subplot_titles = candidate_titles
                       #, vertical_spacing = 0.05
                       )

    i = 1
    j = 1

    coords_before, coords_after = strokeutils.get_whole_stroke(net_osm.getEdge(reference_id), reference_coordinates, net_osm, net_tomtom, True)
    reference_x_before = [coord[0] for coord in coords_before]
    reference_y_before = [coord[1] for coord in coords_before]
    reference_x_after = [coord[0] for coord in coords_after]
    reference_y_after = [coord[1] for coord in coords_after]

    #print('coords_before: {}'.format(coords_before))
    #print('coords_after: {}'.format(coords_after))

    min_x = min(reference_x + reference_x_before + reference_x_after)
    max_x = max(reference_x + reference_x_before + reference_x_after)
    min_y = min(reference_x + reference_x_before + reference_x_after)
    max_y = max(reference_x + reference_x_before + reference_x_after)

    for key in candidates:
        x_tomtom = [coord[0] for coord in candidates[key]]
        y_tomtom = [coord[1] for coord in candidates[key]]

        candidate_coords_before, candidate_coords_after = strokeutils.get_whole_stroke(net_tomtom.getEdge(key), candidates[key], net_osm, net_tomtom, False)
        candidate_x_before = [coord[0] for coord in candidate_coords_before]
        candidate_y_before = [coord[1] for coord in candidate_coords_before]
        candidate_x_after = [coord[0] for coord in candidate_coords_after] #'#17becf'
        candidate_y_after = [coord[1] for coord in candidate_coords_after] #'#bcbd22'

        add_plotly_trace(fig, reference_x_before, reference_y_before, i, reference_id, '#bcbd22', 'lines', j)
        add_plotly_trace(fig, reference_x_after, reference_y_after, i, reference_id, '#bcbd22', 'lines', j)
        add_plotly_trace(fig, candidate_x_before, candidate_y_before, i, key, '#17becf', 'lines', j) 
        add_plotly_trace(fig, candidate_x_after, candidate_y_after, i, key, '#17becf', 'lines', j)         

        add_plotly_trace(fig, reference_x, reference_y, i, reference_id, 'green', 'lines', j)        
        add_plotly_trace(fig, [reference_x[0]], [reference_y[0]], i, reference_id, 'green', 'markers', j)
        add_plotly_trace(fig, [reference_x[-1]], [reference_y[-1]], i, reference_id, 'red', 'markers', j)
        
        add_plotly_trace(fig, x_tomtom, y_tomtom, i, key, 'blue', 'lines', j)            
        add_plotly_trace(fig, [x_tomtom[0]], [y_tomtom[0]], i, key, 'green', 'markers', j)
        add_plotly_trace(fig, [x_tomtom[-1]], [y_tomtom[-1]], i, key, 'red', 'markers', j)

        if min(x_tomtom + candidate_x_before + candidate_x_after) < min_x:
            min_x = min(x_tomtom + candidate_x_before + candidate_x_after)
        if min(y_tomtom + candidate_y_before + candidate_y_after) < min_y:
            min_y = min(y_tomtom)
        if max(x_tomtom + candidate_x_before + candidate_x_after) > max_x:
            max_x = max(x_tomtom + candidate_x_before + candidate_x_after)
        if max(y_tomtom + candidate_y_before + candidate_y_after) > max_y:
            max_y = max(y_tomtom + candidate_y_before + candidate_y_after)

        if j == 2:
            i += 1
            j -= 1
        else:
            j += 1

    width, x_to_y, bigger_x = set_width(max_x, min_x, max_y, min_y)
    h = n_rows * width / x_to_y
    if h < 250:
        h = 250

    fig.update_layout(height=h, width=n_cols * width, title_text=reference_id,
                     showlegend = False)

    return fig


##### Layout #####

graph_alerts = html.Div(
    [
        dbc.Alert(
            [
                "You have reached the end of the dataset. Please change dataset."
            ],
            color="warning",
            id = "alert-end-of-dataframe",
            fade = False,
            is_open = False,
            dismissable = True
        ),
        dbc.Alert(
            [
                "You have reached the start of the dataset. To get to an earlier day, please change dataset"
            ],
            color="warning",
            id = "alert-start-of-dataframe",
            fade = False,
            is_open = False,
            dismissable = True
        )
    ]
)

io_alerts = html.Div(
    [
        dbc.Alert(
            [],
            color="success",
            id = "saving-successful",
            fade = False,
            is_open = False,
            dismissable = True
        ),
        dbc.Alert(
            ['Something went wrong. Try again'],
            color="danger",
            id = "saving-error",
            fade = False,
            is_open = False,
            dismissable = True
        )
    ]
)

text_input = html.Div(
    [
        dbc.Input(id="input", placeholder="Enter specific OSM ID...", type="text"),
        dbc.Button("Go to entered ID", id='goto', color="light", className="me-1"),
    ]
)

flask_server = Flask(__name__)
app = dash.Dash(__name__, server = flask_server, external_stylesheets=[dbc.themes.BOOTSTRAP])

read_new_file()

app.layout = html.Div([
        html.Div([
            html.Div([
                dcc.Graph(figure=go.Figure(), id = 'main_graph'),
                graph_alerts
            ], #style={'width': '40vw', #'height': '80vh'
               #                                  'float':'left'}
               ),

            html.Div([
                dbc.Label(children = ["Select matching linestrings"], id = "linestringlab"),
                dbc.Checklist(
                    options=[
                        {"label": "Dummy", "value": 1},
                    ],
                    value=[],
                    id="matching_checklist",
                ),
            ], #style = {'width': '38vw', #'height': '80vh'
               #                                  'float':'left'}
               )
        ]),    
    text_input,  
    html.Div([
        dbc.Button('Back', id = 'back', color = 'secondary',
                        className = 'me-1'),
        dbc.Button('Save matches for current linestring!', id = 'save', color = 'primary',
                        className = 'me-1'),
        dbc.Button('Next', id = 'next', color = 'secondary',
                        className = 'me-1'),
        #dbc.Button('Change data source', id = 'cdata', color = 'danger',
        #                className = 'me-1'),
        ], style = {'padding-left':'50px'}),
    io_alerts,
    dcc.Store(id = 'reference_edge_idx', data = 0, storage_type='local'),
    dcc.Store(id = 'reset', data = 0, storage_type = 'local'),
    dcc.Store(id = 'candidatelist', data = [], storage_type = 'local')],    
)

def get_reference_edge(reference_edge_idx):
    try:
        reference_edge = net_osm.getEdges(withInternal=False)[int(reference_edge_idx)]
        return reference_edge
    except IndexError:
        raise

##### Callbacks #####

@app.callback(
    Output('matching_checklist', 'options'),
    Output('matching_checklist', 'value'),
    Input('candidatelist', 'data')
)
def update_candidates(candidatelist):
    checklist = [{"label": str(id), "value": idx + 1} for idx, id in enumerate(candidatelist)]
    return checklist, []


@app.callback(
    Output('main_graph', 'figure'),
    Output('alert-end-of-dataframe', 'is_open'),
    Output('candidatelist', 'data'),
    Output('linestringlab', 'children'),
    Input('reference_edge_idx', 'data')
)
def plot(reference_edge_idx):
    try:
        
        reference_edge = get_reference_edge(int(reference_edge_idx))        

        reference_coordinates = preprocessing.get_transformed_coordinates(reference_edge, net_osm, net_tomtom)
        reference_id = reference_edge.getID()

        radius = 10 # define a radius

        candidates = get_candidates(reference_id, reference_coordinates, radius, tomtom_rtree, net_tomtom)
        candidateList = list(candidates.keys())
        #print('N Candidates: {}'.format(len(candidates)))

        fig = plotly_plot_extended(reference_id, reference_coordinates, candidates)

    except IndexError:
        return dash.no_update, True, dash.no_update, dash.no_update
    return fig, dash.no_update, candidateList, "Select matching linestrings for {}".format(reference_id)
    

@app.callback(
    Output('reference_edge_idx', 'data'),
    State('reference_edge_idx', 'data'),
    State("input", "value"),
    Input("goto", "n_clicks"),
    Input('next', 'n_clicks'),
    Input('back', 'n_clicks'),
    Input('reset', 'data')
)
def update_linestring(reference_edge, value, clicks_goto, clicks_next, clicks_back, reset):
    button_clicked = ctx.triggered_id
    reference_edge = int(reference_edge)
    if button_clicked == 'back':
        reference_edge -= 1
    elif button_clicked == 'reset':
        reference_edge = 0
    elif button_clicked == 'goto':
        idx = 0
        for edge in net_osm.getEdges(withInternal=False):
            if idx == 0:
                print(edge.getID())
            if edge.getID() == value:
                return idx
            idx += 1
        return dash.no_update
    else:
        if button_clicked is None:
            raise dash.exceptions.PreventUpdate()
        reference_edge += 1
    return reference_edge   

def save_file(reference_id, reference_idlist, matched_ids):
    print(reference_id)
    print(matched_ids)
    print(reference_idlist)
    filename = 'OSM_{}.csv'.format(reference_id)
    df = pd.DataFrame(data = {'OSM':reference_idlist, 'Tomtom':matched_ids})
    filepath = pathlib.Path(out_dir, filename)
    df.to_csv(filepath, index = False)
    
    if os.path.isfile(filepath.resolve()):
        print('successful')
        return 0, filename
    return -1, filename


@app.callback(
    Output("saving-successful", "children"),
    Output("saving-successful", "is_open"),
    Output("saving-error", "is_open"),
    State("matching_checklist", "value"),
    State('reference_edge_idx', 'data'),
    Input("save", "n_clicks"),
    State('matching_checklist', 'options') # changed from input
)
def callback(selection, reference_idx, clicks, options):
    trigger = ctx.triggered_id
    if trigger is None:
        return dash.no_update, dash.no_update, dash.no_update    
    
    matched_ids = []

    for item in options:
        if item['value'] in selection:
            matched_ids.append(item['label'])


    reference_id = get_reference_edge(reference_idx).getID()
    reference_idlist = [reference_id for i in range(len(matched_ids))]
    status, filename = save_file(reference_id, reference_idlist, matched_ids)
    
    #df_plot, date, location = get_df_plot(int(day))
    #df_plot['manually_cs_labelled'] = np.where(df_plot.index.isin(selectedpoints), True, False)

    #status, filename = save_file(location, date, df_plot)

    if status == 0:
        return 'Successfully saved as {} in {}'.format(filename, out_dir), True, False
    else:
        return dash.no_update, False, True





def open_browser():
    webbrowser.open_new("http://localhost:{}".format(8050))


if __name__ == '__main__':
    Timer(1, open_browser).start()
    serve(app.server, host = '127.0.0.1', port = 8050)
