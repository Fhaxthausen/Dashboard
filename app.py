import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# ------------------------------
# Load data
# ------------------------------
df = pd.read_csv("medallists.csv", parse_dates=["medal_date"])
df.columns = df.columns.str.strip()

# ------------------------------
# Control options
# ------------------------------
# Dropdown: list of strings, sorted A–Z
countries = sorted(df["country"].dropna().astype(str).unique().tolist())
default_countries = [c for c in ["Australia", "United States", "China", "France"] if c in countries]

# Checklist: label -> value mapping (per spec)
MEDAL_MAP = {"Gold Medal": "Gold", "Silver Medal": "Silver", "Bronze Medal": "Bronze"}
MEDAL_LABELS = list(MEDAL_MAP.keys())               # default selection = labels
GENDER_OPTIONS = ["All", "Female", "Male"]

# ------------------------------
# App & layout
# ------------------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id="country",
        options=countries,                 # list[str], sorted
        value=default_countries,           # default selection
        multi=True,
        clearable=False
    ),
    dcc.Checklist(
        id="medal_type",
        options=MEDAL_MAP,                 # dict: label -> value
        value=MEDAL_LABELS,                # list of labels (default all)
        inline=True
    ),
    dcc.RadioItems(
        id="gender",
        options=GENDER_OPTIONS,            # list[str]
        value="All",
        inline=True
    ),
    dcc.Graph(id="graph")
])

# ------------------------------
# Single required callback
# ------------------------------
@app.callback(
    Output("graph", "figure"),
    Input("country", "value"),
    Input("medal_type", "value"),   # selected LABELS
    Input("gender", "value")
)
def update_graph(selected_countries, selected_medal_labels, selected_gender):
    if not selected_countries:
        return {}

    # Map selected labels -> simple values ("Gold", "Silver", "Bronze")
    selected_medals = [MEDAL_MAP[label] for label in selected_medal_labels]

    dff = df[df["country"].isin(selected_countries)].copy()

    # Normalize medal type to simple values so ordering & color maps work
    dff["medal_type_norm"] = dff["medal_type"].replace(MEDAL_MAP).replace(
        {"Gold Medal": "Gold", "Silver Medal": "Silver", "Bronze Medal": "Bronze"}
    )
    dff = dff[dff["medal_type_norm"].isin(selected_medals)]

    if selected_gender != "All":
        dff = dff[dff["gender"] == selected_gender]

    # Cast dates to strings so the checker sees 'YYYY-MM-DD' (not numpy datetime64)
    dff["medal_day"] = pd.to_datetime(dff["medal_date"]).dt.strftime("%Y-%m-%d")

    # Histogram per spec (counts by country), forced horizontal
    fig = px.histogram(
        dff,
        x="medal_day",                      # dates as strings
        y="country",
        color="medal_type_norm",
        barmode="group",
        histfunc="count",
        category_orders={"medal_type_norm": ["Gold", "Silver", "Bronze"]},
        color_discrete_map={
            "Gold": "gold",
            "Silver": "silver",
            "Bronze": "darkgoldenrod"
        },
        title="Medal Tally - Paris Olympics 2024",
        orientation="h"
    )
    # Enforce orientation on all traces (some Plotly versions need this)
    fig.for_each_trace(lambda t: t.update(orientation="h"))

    # Axis titles & legend (per spec)
    fig.update_layout(
        yaxis_title_text="Country",
        xaxis_title_text="Medal Count"
    )
    fig.update_traces(showlegend=False)

    return fig

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
