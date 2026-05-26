import pandas as pd
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output

input_file = "state_crime_census_2022.csv"
df = pd.read_csv(input_file)

us_state_to_abbrev = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "District of Columbia": "DC",
}
df["abbr"] = df["NAME"].map(us_state_to_abbrev)

# Copying flagged states logic over from EDA
mean_rate = df["crime_rate_per_100k"].mean()
std_rate = df["crime_rate_per_100k"].std()
cutoff = mean_rate - (1 * std_rate)

def flag_states(row):
    if row["NAME"] == "District of Columbia":
        return "City-state (DC)"
    if row["crime_rate_per_100k"] < cutoff:
        return "NIBRS partial"
    return "full_report"

df["flag_label"] = df.apply(flag_states, axis=1)

def get_status_msg(f):
    if f == "NIBRS partial":
        return "⚠ Partial Reports in 2022 — crime rate understated"
    if f == "City-state (DC)":
        return "⚠ DC — city-state outlier (excluded from modelling)"
    return "✓ Full Reports in 2022"

df["status_msg"] = df["flag_label"].apply(get_status_msg)

# Clustering groups from the H2o and k-means analysis
clusters = {
    "c_0": [
        "Alaska", "Arizona", "Colorado", "Connecticut", "Delaware", "Hawaii",
        "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Maine", "Maryland",
        "Massachusetts", "Minnesota", "Missouri", "Montana", "Nebraska",
        "New Hampshire", "New Jersey", "North Dakota", "Ohio", "Oregon",
        "Pennsylvania", "Rhode Island", "South Dakota", "Utah", "Vermont",
        "Virginia", "Washington", "Wisconsin", "Wyoming",
    ],
    "c_1": ["California", "Florida", "New York", "Texas"],
    "c_2": [
        "Alabama", "Arkansas", "Georgia", "Kentucky", "Louisiana", "Michigan",
        "Mississippi", "Nevada", "New Mexico", "North Carolina", "Oklahoma",
        "South Carolina", "Tennessee", "West Virginia",
    ],
}

state_to_cluster = {s: lbl for lbl, states in clusters.items() for s in states}
state_to_cluster["District of Columbia"] = "c_DC"
df["notebook_cluster"] = df["NAME"].map(state_to_cluster).fillna("Unassigned")

cluster_map_labels = {
    "c_0":  "C0: Mainstream",
    "c_1":  "C1: Large-state",
    "c_2":  "C2: High-risk Southern",
    "c_DC": "DC: Outlier",
}
df["cluster_short"] = df["notebook_cluster"].map(cluster_map_labels)

color_map_int = {
    "C0: Mainstream": 0,
    "C1: Large-state": 1,
    "C2: High-risk Southern": 2,
    "DC: Outlier": 3,
}
df["cluster_int"] = df["cluster_short"].map(color_map_int).astype(int)

# Finding the centers of flagged states to make them apparent
flagged_centers = {
    "Alaska": (64.20, -153.49),
    "California": (36.78, -119.42),
    "Florida": (27.66, -81.52),
    "New Jersey": (40.06, -74.41),
    "New York": (42.17, -74.95),
    "Pennsylvania": (41.20, -77.19),
    "District of Columbia": (38.91, -77.04),
}

flagged_df = df[df["flag_label"] != "full_report"].copy()
flagged_df["lat"] = flagged_df["NAME"].map(lambda n: flagged_centers.get(n, (None, None))[0])
flagged_df["lon"] = flagged_df["NAME"].map(lambda n: flagged_centers.get(n, (None, None))[1])

# circle colour: orange for partial reporting states, black for DC
flagged_df["circle_color"] = flagged_df["flag_label"].map({
    "NIBRS partial": "darkorange",
    "City-state (DC)": "black",
})

layers = {
    "crime_rate": {
        "col": "crime_rate_per_100k", "label": "Crime Rate (/100k)", "colorscale": "Viridis", "fmt": ":,.0f"},
    "cluster": {"col": "cluster_int", "label": "State Clusters", "colorscale": None, "fmt": ""},
    "income": {"col": "median_household_income", "label": "Median Income ($)", "colorscale": "Viridis", "fmt": ":$,.0f"},
    "poverty": {"col": "poverty_rate_pct", "label": "Poverty Rate (%)", "colorscale": "Viridis", "fmt": ":.1f%"},
    "unemployment": {"col": "unemployment_rate_pct", "label": "Unemployment (%)",     "colorscale": "Viridis", "fmt": ":.1f%"},
    "vacancy": {"col": "housing_vacancy_rate_pct", "label": "Housing Vacancy (%)",  "colorscale": "Viridis", "fmt": ":.1f%"},
}

# discrete colors for the cluster map
cluster_colorscale = [
    [0.00, '#440154'], [0.25, '#440154'], # C0
    [0.25, '#31688e'], [0.50, '#31688e'], # C1
    [0.50, '#35b779'], [0.75, '#35b779'], # C2
    [0.75, '#fde725'], [1.00, '#fde725']  # DC
]

# cluster legend items
cluster_legend = [
    dict(x=0.01, y=0.28, text="● C0: Mainstream", font=dict(color="#440154", size=11)),
    dict(x=0.01, y=0.23, text="● C1: Large-state", font=dict(color="#31688e", size=11)),
    dict(x=0.01, y=0.18, text="● C2: High-risk Southern", font=dict(color="#35b779", size=11)),
    dict(x=0.01, y=0.13, text="● DC: Outlier", font=dict(color="#b8860b", size=11)),
]

# Starting the Dash app
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H2("2022 US Crime & Socioeconomic Map", style={"textAlign": "center"}),
    html.Div([
        dcc.Dropdown(
            id = "layer-dropdown",
            options = [{"label": v["label"], "value": k} for k, v in layers.items()],
            value = "crime_rate",
            clearable = False,
        ),
    ], style={"width": "40%", "margin": "0 auto", "paddingBottom": "10px"}),
    dcc.Graph(id="main-map", style={"height": "85vh"}),
])

@app.callback(Output("main-map", "figure"), [Input("layer-dropdown", "value")])
def update_map(selected_layer):
    info = layers[selected_layer]
    is_cluster = selected_layer == "cluster"

    fig = go.Figure()

    fig.add_trace(go.Choropleth(
        locations = df["abbr"],
        z = df[info["col"]],
        locationmode = "USA-states",
        colorscale = cluster_colorscale if is_cluster else info["colorscale"],
        showscale = not is_cluster,
        colorbar = dict(
            title = info["label"],
            len = 0.55,
            thickness = 14,
            tickvals = [0, 1, 2, 3] if is_cluster else None,
            ticktext = ["C0","C1","C2","DC"] if is_cluster else None,
        ) if not is_cluster else None,
        hovertemplate=(
            "<b>%{customdata[0]} (%{locations})</b><br>"
            "%{customdata[1]}<br>"
            "Cluster: %{customdata[2]}<br>"
            "<b>" + info['label'] + ": %{z" + info['fmt'] + "}</b><br>"
            "<extra></extra>"
        ),
        customdata = df[["NAME", "status_msg", "cluster_short"]].values,
        marker_line_color = "white",
        marker_line_width = 0.5,
    ))

    annotations = []
    if is_cluster:
        annotations = [
            dict(
                xref="paper", yref="paper", showarrow=False,
                align="left",
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="#ccc", borderwidth=1,
                **leg,
            )
            for leg in cluster_legend
        ]

    fig.add_trace(go.Scattergeo(
        lat = flagged_df["lat"],
        lon = flagged_df["lon"],
        mode = "markers",
        marker = dict(
            size = 16,
            color = "rgba(0,0,0,0)",
            line = dict(
                color = flagged_df["circle_color"].tolist(),
                width = 3,
            ),
            symbol = "circle",
        ),
        hovertemplate= (
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]}<br>"
            "<extra></extra>"
        ),
        customdata = flagged_df[["NAME", "status_msg"]].values,
        showlegend = False,
    ))

    fig.update_layout(
        annotations = annotations,
        geo = dict(
            scope = "usa",
            projection_type = "albers usa",
            showlakes = True,
            lakecolor = "rgb(255, 255, 255)",
        ),
        margin = {"r": 0, "t": 40, "l": 0, "b": 0},
    )
    return fig

if __name__ == "__main__":
    app.run()
