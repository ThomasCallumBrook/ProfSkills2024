from dash import Dash, html, dcc, Input, Output
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import folium
import csv

sewage = pd.read_csv('SewageLeaksRefined.csv')
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    #Header
    html.H1("Example Libidos Data Dash", className='mb-2', style={'textAlign':'center'}),
    html.A("Sewage CSV Update", href='/csv_update'),
    html.A("Sewage Map", href='/sewage_map'),
    html.A("Sewage Grid", href='/sewage_grid'),
    # Dropdown menu
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='category',
                value='Duration',
                clearable=False,
                options=sewage.columns[1:])
        ], width=4)
    ]),
    # Matplot Output
    dbc.Row([
        dbc.Col([
            html.Img(id='bar-graph-matplotlib')
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
        ], width=12)
    ]),
    # Dataframe table
    dbc.Row([
        dbc.Col([
            dag.AgGrid(
                id='grid',
                rowData=sewage.to_dict("records"),
                columnDefs=[{"field": i} for i in sewage.columns],
                columnSize="sizeToFit",
            )
        ], width=12)
    ]),
])

# Create interactivity between dropdown component and graph
@app.callback(
    Output(component_id='bar-graph-matplotlib', component_property='src'),
    Output('grid', 'defaultColDef'),
    Input('category', 'value'),
)


def plot_data(selected_yaxis):

    # Build the matplotlib figure
    fig = plt.figure(figsize=(14, 5))
    limited = sewage.head(50)
    plt.bar(limited['Name'], limited[selected_yaxis])
    plt.ylabel(selected_yaxis)
    plt.xticks(rotation=45)

    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    fig_bar_matplotlib = f'data:image/png;base64,{fig_data}'

    return fig_bar_matplotlib, {} 



 # Lobitos Individual Leak Display

colours = ['gray','blue', 'green', 'orange', 'red', 'darkred']
def assign_colour(days):
    if days >= 200:
        return colours[5]
    elif days >= 100:
        return colours[4]
    elif days >= 75:
        return colours[3]
    elif days >= 50:
        return colours[2]
    elif days >= 25:
        return colours[1]
    else:
        return colours[0]

lobitos = folium.Map(location=[-4.457310481797269, -81.2811891931266], zoom_start=17)
# Add bubbles for each sewage pool
for _, row in sewage.iterrows():
    # Scale the radius based on duration (adjust multiplier as needed)
    radius = np.sqrt(row['Diameter']) * 2

    tooltip_text = f"""
    <b>{row['Name']}</b><br>
    Noted Date: {row['Noted']}<br>
    Resolved Date: {row['Resolved']}<br>
    Duration (Days): {row['Duration']}
    """

    # Add circle marker
    folium.CircleMarker(
        location=[row['Lat'], row['Lon']],
        radius=radius,
        color=assign_colour(row['Duration']),
        fill=True,
        fill_color=assign_colour(row['Duration']),
        fill_opacity=0.6,
        opacity=0.6,
        tooltip=tooltip_text
    ).add_to(lobitos)



# Lobitos Grid Display of Grouped Sector Leaks

lobitos_max_lat = -4.449793440238931
lobitos_min_lat = -4.460475724172084
lobitos_min_lon = -81.28962377680455
lobitos_max_lon = -81.2738937464837

delta_lat = lobitos_max_lat - lobitos_min_lat
delta_lon = lobitos_max_lon - lobitos_min_lon
grid_size = 10

# Generate center points for each Lobitos sector
lobitos_lat_center = np.linspace(lobitos_min_lat, lobitos_max_lat, grid_size)
lobitos_lon_center = np.linspace(lobitos_min_lon, lobitos_max_lon, grid_size)

# Make even lat-lon squares for each Lobitos sector center point 
lat_diffs = (lobitos_lat_center[1] - lobitos_lat_center[0])/2 # Lat difference between two center points
lobitos_lat_min = lobitos_lat_center - lat_diffs # Find Lat min by taking the Lat diff from the center point
lobitos_lat_max = lobitos_lat_center + lat_diffs # Find Lat max by adding the Lat diff to the center point

lon_diffs = (lobitos_lon_center[1] - lobitos_lon_center[0])/2
lobitos_lon_min = lobitos_lon_center - lon_diffs
lobitos_lon_max = lobitos_lon_center + lon_diffs


# Make an empty dataframe to contain all the lat/lon data required for each sector
lobitos_sectors = pd.DataFrame({
    'sector': [],
    'center_lat': [],
    'center_lon': [],
    'lat_min': [],
    'lat_max': [],
    'lon_min': [],
    'lon_max': []
})

def assign_sector(row):
    mask = (
        (row['Lat'] >= lobitos_sectors['lat_min']) &
        (row['Lat'] <= lobitos_sectors['lat_max']) &
        (row['Lon'] >= lobitos_sectors['lon_min']) &
        (row['Lon'] <= lobitos_sectors['lon_max'])
    )

    matching_sectors = lobitos_sectors[mask]
    if len(matching_sectors) > 0:
        return matching_sectors.iloc[0]['sector']
    return 'Other'

sewage['coord_sector'] = sewage.apply(assign_sector, axis=1)

# Group by assigned sector and calculate metrics
sector_counts = sewage.groupby('coord_sector').agg({
    'Name': 'count',
    'Diameter': ['sum', 'mean']  # Adding sum and mean of diameter
}).reset_index()

# Flatten column names
sector_counts.columns = ['sector', 'count', 'total diameter', 'average diameter']

# Merge with sector coordinates
lobitos_data = sector_counts.merge(
    lobitos_sectors[['sector', 'center_lat', 'center_lon']],
    how='left',
    on='sector'
)
lobitos_data = lobitos_data[(lobitos_data['sector'] != 'Other')]

lobitos_sectors = pd.DataFrame(columns=['sector','center_lat','center_lon','lat_min','lat_max','lon_min','lon_max'])

# Loop through a set grid size e.g. 10x10. Adding Sector names, and Lats: center, min, max, Lons: center, min, max
n = 0
for i in range(grid_size):
    for j in range(grid_size):
        lobitos_sectors.loc[n] = [f"Sector {n}"] + [lobitos_lat_center[i]] + [lobitos_lon_center[j]] + [lobitos_lat_min[i]] + [lobitos_lat_max[i]] + [lobitos_lon_min[j]] + [lobitos_lon_max[j]]
        n += 1


def assign_colour_on_leaks(leaks):
    if leaks >= 5:
        return colours[5]
    elif leaks >= 4:
        return colours[4]
    elif leaks >= 3:
        return colours[3]
    elif leaks >= 2:
        return colours[2]
    elif leaks >= 1:
        return colours[1]
    else:
        return colours[0]

lobitos_grid = folium.Map(location=[-4.457310481797269, -81.2811891931266], zoom_start=16)
for _, row in lobitos_sectors.iterrows():
    kw = {
        "color": "blue",
        "line_cap": "round",
        "fill": True,
        "fill_color": "white",
        "weight": 1,
        "tooltip": row['sector']
    }
    folium.Rectangle(
        bounds=[[row['lat_min'], row['lon_min']], [row['lat_max'], row['lon_max']]],
        line_join="round",
        dash_array="5, 5",
        **kw,
    ).add_to(lobitos_grid)


for _, row in lobitos_data.iterrows():
    # Scale the radius based on count (adjust multiplier as needed)
    radius = np.sqrt(row['count']) * 2
    print(f"Test {row['sector']}")
    tooltip_text = f"""
    <b>{row['sector']}</b><br>
    Number of Leaks: {row['count']}<br>
    Total Diameter: {row['total diameter']:,.2f}m^2<br>
    Average Diameter: {row['average diameter']:,.2f}m
    """
    
    # Add circle marker
    folium.CircleMarker(
        location=[row['center_lat'], row['center_lon']],
        radius=radius,
        color=assign_colour_on_leaks(row['count']),
        fill=True,
        fill_color=assign_colour_on_leaks(row['count']),
        fill_opacity=0.6,
        opacity=0.6,
        tooltip=tooltip_text
    ).add_to(lobitos_grid)

@app.server.route("/sewage_map")
def full():
    """Map of individual Lobitos Sewage Leaks"""
    return lobitos.get_root().render()


@app.server.route("/sewage_grid")
def grid():
    """Grid overlay of Lobitos sectors and merged sewage leaks counts"""
    return lobitos_grid.get_root().render()


@app.server.route("/csv_update", methods=["GET", "POST"])
def csv_update():
    if request.method == "POST":
        name = request.form['Name']
        lat = request.form['Lat']
        lon = request.form['Lon']
        noted = request.form['Noted']
        resolved = request.form['Resolved']
        diameter = request.form['Diameter']
        severity = request.form['Severity']
        issue_type = request.form['IssueType']
        with open('sewage_data.csv', mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([name, lat, lon, noted, resolved, diameter, severity, issue_type])
        return f"Your data has been saved to the CSV file."
    return render_template('csv_update.html')

if __name__ == '__main__':
    app.run_server(debug=False, port=8002)
