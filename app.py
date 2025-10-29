# app.py
import dash
from dash import dcc, html, Input, Output, State, MATCH
import plotly.express as px
from plotly import graph_objects as go
import pandas as pd
from pathlib import Path
from datetime import date
from functools import lru_cache
import re
import numpy as np

# ==============================================================
# Shared box style (ensures equal widths + alignment)
# ==============================================================
BOX = {
    "border": "2px dashed #cfcfcf",
    "borderRadius": "14px",
    "padding": "12px",
    "background": "white",
    "width": "100%",
    "boxSizing": "border-box",
}

# ==============================================================
# Player database (bio cards under each search)
# ==============================================================
DB_LOCAL = Path("ATP_Database.csv")
DB_MOUNT = Path("/mnt/data/ATP_Database.csv")
CSV_PATH_PLAYERS = DB_LOCAL if DB_LOCAL.exists() else DB_MOUNT

PLAYERS_USECOLS = {
    "id", "player", "atpname", "birthdate", "weight", "height",
    "turnedpro", "birthplace", "coaches", "hand", "backhand", "ioc"
}
df_players = pd.read_csv(
    CSV_PATH_PLAYERS,
    encoding="latin1",
    usecols=lambda c: c.strip().lower() in {x.lower() for x in PLAYERS_USECOLS}
)
df_players.columns = df_players.columns.str.strip().str.lower()

def pick_display_name(row):
    a = str(row.get("atpname", "")).strip()
    p = str(row.get("player", "")).strip()
    return a if a and a.lower() != "nan" else p

df_players["display_name"] = df_players.apply(pick_display_name, axis=1)
df_players = df_players.sort_values("display_name").reset_index(drop=True)

def yyyymmdd_to_date(v):
    try:
        s = str(int(v))
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except Exception:
        return None

def compute_age(bd):
    if not bd:
        return None
    today = date.today()
    return today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))

def fmt(val, suffix=""):
    if pd.isna(val) or str(val).strip().lower() in {"", "nan"}:
        return "—"
    return f"{val}{suffix}" if suffix else str(val)

def build_info_list(row):
    bd = yyyymmdd_to_date(row.get("birthdate"))
    age = compute_age(bd) if bd else None
    birthdate_str = bd.isoformat() if bd else "—"
    return [
        ("ATP name", row.get("atpname") or row.get("player")),
        ("Birthdate", birthdate_str),
        ("Age", "—" if age is None else f"{age}"),
        ("Handed", fmt(row.get("hand"))),
        ("Backhand", fmt(row.get("backhand"))),
        ("Turned pro", "—" if pd.isna(row.get("turnedpro")) else str(int(row.get("turnedpro")))),
        ("Weight", "—" if pd.isna(row.get("weight")) else f"{int(row.get('weight'))} kg"),
        ("Height", "—" if pd.isna(row.get("height")) else f"{int(row.get('height'))} cm"),
        ("Birthplace", fmt(row.get("birthplace"))),
        ("Coaches", fmt(row.get("coaches"))),
        ("Country (IOC)", fmt(row.get("ioc"))),
    ]

def avatar_from_name(name: str):
    initials = "".join([p[:1].upper() for p in str(name).split()[:2]]) or "?"
    return html.Div(
        initials,
        style={
            "width": "140px", "height": "140px",
            "borderRadius": "50%", "display": "grid", "placeItems": "center",
            "fontWeight": 800, "fontSize": "42px", "color": "#fff",
            "background": "linear-gradient(135deg, #4c78ff, #2ecc71)",
            "boxShadow": "inset 0 0 0 4px rgba(255,255,255,.25)",
        }
    )

def card_from_row(row):
    info_items = build_info_list(row)
    name = row["display_name"]
    return html.Div(
        [
            html.Div(
                [avatar_from_name(name),
                 html.Div(name, style={"fontWeight": 800, "fontSize": "22px", "marginTop": "10px"})],
                style={"display": "grid", "placeItems": "center", "gap": "8px"}
            ),
            html.Div(
                html.Ul(
                    [
                        html.Li(
                            html.Span([
                                html.Span(f"{label}: ", style={"fontWeight": 600}),
                                html.Span(value)
                            ])
                        ) for label, value in info_items
                    ],
                    style={"listStyleType": "none", "padding": 0, "margin": 0, "fontSize": "13px", "lineHeight": "1.25"}
                ),
                style={
                    "border": "2px solid #d8d8d8",
                    "borderRadius": "16px",
                    "padding": "12px 14px",
                    "background": "white",
                    "marginTop": "10px",
                },
            ),
        ],
        style=BOX  # <- equal width with radar blocks
    )

def find_player_row_by_name(name: str):
    if not name: return None
    n = str(name).strip().casefold()
    m = df_players[df_players["display_name"].astype(str).str.casefold() == n]
    if not m.empty: return m.iloc[0].to_dict()
    for col in ["player", "atpname"]:
        if col in df_players.columns:
            m = df_players[df_players[col].astype(str).str.casefold() == n]
            if not m.empty: return m.iloc[0].to_dict()
    m = df_players[df_players["display_name"].astype(str).str.casefold().str.contains(n, na=False)]
    return m.iloc[0].to_dict() if not m.empty else None

# ==============================================================
# Multi-year match analytics (2000–2025)
# ==============================================================

SURF_MAP = {"grass": "Grass", "clay": "Clay", "hard": "Hard court", "hard court": "Hard court"}
SURF_ORDER = ["Grass", "Clay", "Hard court"]

# Expose ATP 250 and ATP 500 separately
PREFERRED_TIERS = ["ATP 250", "ATP 500", "Masters 1000", "Grand Slams", "ATP Finals"]
TIER_SHORT = {
    "ATP 250": "ATP 250",
    "ATP 500": "ATP 500",
    "Masters 1000": "ATP 1000",
    "Grand Slams": "Slams",
    "ATP Finals": "ATP Finals (1500)",
}
TIERS_LONG = PREFERRED_TIERS
TIERS_TICK = [TIER_SHORT[t] for t in TIERS_LONG]

A_COLOR = "#4c78ff"; B_COLOR = "#2ecc71"
COLOR_MAP = {"Grass": B_COLOR, "Clay": "#f1c40f", "Hard court": A_COLOR}

# ---------- CSV mapping for A-level tournaments (ATP 250 vs ATP 500)
MAP_PATHS = [Path("atp_tier_map.csv"), Path("/mnt/data/atp_tier_map.csv")]

def _load_tier_map() -> pd.DataFrame:
    for p in MAP_PATHS:
        if p.exists():
            dfm = pd.read_csv(p)
            dfm["pattern"] = dfm["pattern"].astype(str).str.strip().str.lower()
            dfm["category"] = dfm["category"].astype(str).str.strip()
            # Prioritize 500 for overlaps
            dfm["__order"] = dfm["category"].map({"ATP 500": 0, "ATP 250": 1}).fillna(2)
            return dfm.sort_values(["__order"]).drop(columns="__order")
    return pd.DataFrame(columns=["pattern", "category"])

TIER_MAP_DF = _load_tier_map()

def _compile_regex_from_map(df_map: pd.DataFrame, cat: str):
    pats = df_map.loc[df_map["category"] == cat, "pattern"].dropna().unique().tolist()
    if not pats:
        return None
    pats = [re.escape(p) for p in pats if p]
    return re.compile("|".join(pats))

RE_500 = _compile_regex_from_map(TIER_MAP_DF, "ATP 500")
RE_250 = _compile_regex_from_map(TIER_MAP_DF, "ATP 250")

# --------- Data loading & normalization (vectorized)
READ_USECOLS = [
    "winner_name", "loser_name", "surface", "tourney_level",
    "tourney_name", "draw_size"
]

def _read_one_csv(path: Path, year_hint: int | None) -> pd.DataFrame:
    df_ = pd.read_csv(path, usecols=lambda c: c.strip().lower() in {x.lower() for x in READ_USECOLS})
    df_.columns = df_.columns.str.strip()
    lower_map = {c.lower(): c for c in df_.columns}

    req = ["winner_name", "loser_name", "surface", "tourney_level"]
    alias = {
        "winner_name": ["winner", "winner name"],
        "loser_name":  ["loser", "loser name"],
        "surface":     ["court_surface", "court surface"],
        "tourney_level": ["tourney level", "level"],
    }
    opt = {
        "tourney_name": ["tournament", "tourney", "tournament_name", "tourney name", "event"],
        "draw_size": ["draw_size", "draw size", "draw", "size"]
    }
    for need in list(req):
        if need not in lower_map:
            for cand in alias.get(need, []):
                if cand in lower_map:
                    lower_map[need] = lower_map[cand]
                    break
    if any(k not in lower_map for k in ["winner_name", "loser_name", "surface", "tourney_level"]):
        return pd.DataFrame(columns=[*req, "tourney_name", "draw_size", "source_year"])

    tname_col = lower_map.get("tourney_name")
    if not tname_col:
        for cand in opt["tourney_name"]:
            if cand in lower_map:
                tname_col = lower_map[cand]; break
    dsize_col = lower_map.get("draw_size")
    if not dsize_col:
        for cand in opt["draw_size"]:
            if cand in lower_map:
                dsize_col = lower_map[cand]; break

    year = year_hint
    if year is None:
        m = re.search(r"(\d{4})", str(path.name))
        if m: year = int(m.group(1))

    out = pd.DataFrame({
        "winner_name": df_[lower_map["winner_name"]].astype(str),
        "loser_name":  df_[lower_map["loser_name"]].astype(str),
        "surface":     df_[lower_map["surface"]].astype(str),
        "tourney_level": df_[lower_map["tourney_level"]].astype(str),
        "tourney_name": (df_[tname_col].astype(str) if tname_col else ""),
        "draw_size": pd.to_numeric(df_[dsize_col], errors="coerce").astype("Int64") if dsize_col else pd.Series([pd.NA]*len(df_)).astype("Int64"),
        "source_year": year,
    })
    return out

@lru_cache(maxsize=1)
def load_years(start: int = 2000, end: int = 2025) -> pd.DataFrame:
    paths = []
    for y in range(start, end + 1):
        p = Path(f"{y}.csv")
        if not p.exists():
            p = Path(f"/mnt/data/{y}.csv")
        if p.exists():
            paths.append((p, y))
    if not paths:
        raise FileNotFoundError("Ingen årsfiler fundet (2000–2025).")

    frames = []
    for p, y in paths:
        try:
            frames.append(_read_one_csv(p, y))
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=READ_USECOLS + ["source_year"])

    df_all = pd.concat(frames, ignore_index=True)

    # normalization (vectorized)
    df_all["tourney_level"] = df_all["tourney_level"].str.strip().str.upper()
    df_all["surface_norm"] = df_all["surface"].str.strip().str.lower()
    df_all["surface_label"] = df_all["surface_norm"].map(SURF_MAP).fillna("Other")
    df_all["tname_norm"] = df_all["tourney_name"].fillna("").astype(str).str.lower()

    # vectorized tier mapping
    lvl = df_all["tourney_level"].to_numpy()
    tname = df_all["tname_norm"].to_numpy()
    tier = np.full(len(df_all), "", dtype=object)

    # Slams
    mask_G = (lvl == "G")
    tier[mask_G] = "Grand Slams"

    # Masters vs Finals
    mask_MF = (lvl == "M") | (lvl == "F")
    finals_by_name = np.char.find(tname.astype(str), "finals") >= 0
    tier[mask_MF & ~finals_by_name] = "Masters 1000"
    tier[mask_MF & finals_by_name] = "ATP Finals"

    # A level -> 250/500 via regex
    mask_A = (lvl == "A")
    a_idx = np.where(mask_A)[0]
    if len(a_idx):
        sub = np.full(len(a_idx), "", dtype=object)
        subnames = tname[a_idx]
        if RE_500 is not None:
            sub[np.fromiter((bool(RE_500.search(s)) for s in subnames), dtype=bool, count=len(subnames))] = "ATP 500"
        if RE_250 is not None:
            empty = (sub == "")
            hits_250 = np.fromiter((bool(RE_250.search(s)) for s in subnames), dtype=bool, count=len(subnames))
            sub[empty & hits_250] = "ATP 250"
        sub[sub == ""] = "ATP 250"  # default for unknown A
        tier[a_idx] = sub

    df_all["tier"] = pd.Categorical(tier, categories=PREFERRED_TIERS, ordered=True)

    # keep only necessary columns downstream
    df_all = df_all.loc[
        (df_all["winner_name"].notna()) & (df_all["loser_name"].notna()),
        ["winner_name","loser_name","surface_label","tier","source_year"]
    ]

    # restrict to known surfaces early
    df_all = df_all[df_all["surface_label"].isin(SURF_ORDER)]

    # categoricals for perf
    df_all["surface_label"] = pd.Categorical(df_all["surface_label"], categories=SURF_ORDER, ordered=True)
    df_all["source_year"] = df_all["source_year"].astype("int16")

    return df_all

df = load_years(2000, 2025)
YEAR_MIN = int(df["source_year"].min()) if not df.empty else 2000
YEAR_MAX = int(df["source_year"].max()) if not df.empty else 2025

# ==============================================================
# Radar styling
# ==============================================================
def _style_radar(fig, theta_labels):
    short = [TIER_SHORT.get(t, t) for t in theta_labels]
    fig.update_traces(hoverinfo="skip", fill="toself", opacity=0.38, line=dict(width=2.2), marker=dict(size=3.5))
    fig.update_layout(
        height=450, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(t=60, l=72, r=72, b=64), showlegend=True,
        legend=dict(orientation="h", y=-0.08, yanchor="top", x=0.5, xanchor="center", font=dict(size=11)),
        title=dict(y=0.98)
    )
    try:
        fig.update_polars(
            radialaxis=dict(range=[0, 1], dtick=0.2, gridcolor="#ececec", tickfont=dict(size=11), angle=90),
            angularaxis=dict(tickmode="array", tickvals=theta_labels, ticktext=short,
                             tickfont=dict(size=12), rotation=90, direction="clockwise",
                             tickpadding=8, ticklen=5),
        )
    except Exception:
        pass
    return fig

def style_overlay_radar(fig):
    fig.update_traces(fill="toself", opacity=0.38, line=dict(width=2.2), marker=dict(size=3.5), hoverinfo="skip")
    fig.update_layout(
        height=430, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(t=52, l=56, r=56, b=64), showlegend=True,
        legend=dict(orientation="h", y=-0.10, yanchor="top", x=0.5, xanchor="center", font=dict(size=11)),
        title=dict(y=0.96)
    )
    try:
        fig.update_polars(
            radialaxis=dict(range=[0, 1], dtick=0.2, gridcolor="#ececec", tickfont=dict(size=11), angle=90),
            angularaxis=dict(tickfont=dict(size=12), rotation=90, direction="clockwise", tickpadding=6, ticklen=5),
        )
    except Exception:
        pass
    return fig

def empty_radar_layout(_title_ignored: str):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[0]*len(PREFERRED_TIERS), theta=PREFERRED_TIERS, mode="lines",
                                  line=dict(color="rgba(0,0,0,0)"), fill="toself", showlegend=False))
    return _style_radar(fig, PREFERRED_TIERS)

# ==============================================================
# Helpers for stats (operate on pre-normalized df)
# ==============================================================
def _filter_years(dff: pd.DataFrame, years: tuple[int, int] | None):
    if years:
        y0, y1 = years
        return dff[(dff["source_year"] >= y0) & (dff["source_year"] <= y1)]
    return dff

def all_players():
    return sorted(pd.Index(df["winner_name"].astype(str)).union(df["loser_name"].astype(str)).unique().tolist())

ALL_PLAYERS = all_players()

@lru_cache(maxsize=4096)
def _player_surface_tier_wr_cached(player_name: str):
    dff = df[(df["winner_name"] == player_name) | (df["loser_name"] == player_name)]
    if dff.empty:
        return pd.DataFrame(columns=["surface_label","tier","win_rate","matches"])
    is_win = (dff["winner_name"] == player_name).astype("int8")
    g = dff.assign(is_win=is_win).groupby(["surface_label","tier"], observed=True)["is_win"] \
            .agg(wins="sum", matches="count").reset_index()
    g["win_rate"] = g["wins"] / g["matches"]
    return g

def player_surface_tier_wr(player_name: str) -> pd.DataFrame:
    g = _player_surface_tier_wr_cached(player_name)
    if g.empty:
        return g
    return g[["surface_label","tier","win_rate","matches"]]

@lru_cache(maxsize=4096)
def _player_record_cached(player: str, surface_label: str | None):
    dff = df if surface_label is None else df[df["surface_label"] == surface_label]
    wins = int((dff["winner_name"] == player).sum())
    losses = int((dff["loser_name"] == player).sum())
    pct = (wins / (wins + losses)) if (wins + losses) else 0.0
    return wins, losses, pct

def player_record(player, surface_label=None):
    return _player_record_cached(player, surface_label)

@lru_cache(maxsize=4096)
def _h2h_cached(a: str, b: str):
    dff = df[((df["winner_name"] == a) & (df["loser_name"] == b)) |
             ((df["winner_name"] == b) & (df["loser_name"] == a))]
    wins_a = int((dff["winner_name"] == a).sum())
    wins_b = int((dff["winner_name"] == b).sum())
    pct_a = (wins_a / (wins_a + wins_b)) if (wins_a + wins_b) else 0.0
    return wins_a, wins_b, pct_a

def head_to_head(a, b):
    return _h2h_cached(a, b)

@lru_cache(maxsize=4096)
def _win_rate_by_tiers_cached(player: str, surface: str | None, years: tuple[int, int] | None):
    dff = df
    if years:
        dff = _filter_years(dff, years)
    if surface:
        dff = dff[dff["surface_label"] == surface]
    dff = dff[(dff["winner_name"] == player) | (dff["loser_name"] == player)]
    if dff.empty:
        return pd.DataFrame({"tier": TIERS_LONG, "win_rate": 0.0, "matches": 0})
    is_win = (dff["winner_name"] == player).astype("int8")
    g = dff.assign(is_win=is_win).groupby("tier", observed=True)["is_win"] \
           .agg(wins="sum", matches="count").reindex(TIERS_LONG, fill_value=0).reset_index()
    g["win_rate"] = np.divide(g["wins"], g["matches"], out=np.zeros_like(g["wins"], dtype=float), where=g["matches"]>0)
    return g[["tier","win_rate","matches"]]

def win_rate_by_tiers(player, surface=None, years: tuple[int, int] | None = None):
    return _win_rate_by_tiers_cached(player, surface, years)

@lru_cache(maxsize=4096)
def _win_rate_by_surfaces_cached(player: str, tier: str | None, years: tuple[int, int] | None):
    dff = df
    if years:
        dff = _filter_years(dff, years)
    if tier:
        dff = dff[dff["tier"] == tier]
    dff = dff[(dff["winner_name"] == player) | (dff["loser_name"] == player)]
    if dff.empty:
        return pd.DataFrame({"surface_label": SURF_ORDER, "win_rate": 0.0, "matches": 0})
    is_win = (dff["winner_name"] == player).astype("int8")
    g = dff.assign(is_win=is_win).groupby("surface_label", observed=True)["is_win"] \
           .agg(wins="sum", matches="count").reindex(SURF_ORDER, fill_value=0).reset_index()
    g["win_rate"] = np.divide(g["wins"], g["matches"], out=np.zeros_like(g["wins"], dtype=float), where=g["matches"]>0)
    return g[["surface_label","win_rate","matches"]]

def win_rate_by_surfaces(player, tier=None, years: tuple[int, int] | None = None):
    return _win_rate_by_surfaces_cached(player, tier, years)

# ==============================================================
# Head-to-head metric row (needed by H2H panel)
# ==============================================================
def metric_row(title, a_rec, b_rec, a_color=A_COLOR, b_color=B_COLOR):
    a_w, a_l, a_pct = a_rec
    b_w, b_l, b_pct = b_rec
    total = (a_pct or 0.0) + (b_pct or 0.0)
    frac_a = 0.5 if total == 0 else (a_pct or 0.0) / total
    split_pct = int(round(frac_a * 100))
    a_pct_100 = int(round((a_pct or 0.0) * 100))
    b_pct_100 = int(round((b_pct or 0.0) * 100))

    labels = html.Div(
        [
            html.Div(f"{a_pct_100}%", style={"color": a_color, "fontWeight": 700, "fontSize": "12px"}),
            html.Div(f"{b_pct_100}%", style={"color": b_color, "fontWeight": 700, "fontSize": "12px", "textAlign": "right"}),
        ],
        style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px", "minWidth": "300px"},
    )
    pill_style = {
        "height": "16px", "width": "100%", "minWidth": "300px", "borderRadius": "999px",
        "background": f"linear-gradient(90deg, {a_color} 0%, {a_color} {split_pct}%, {b_color} {split_pct}%, {b_color} 100%)"
    }
    center = html.Div([labels, html.Div(style=pill_style)], style={"flex": 1, "display": "flex", "flexDirection": "column"})

    return html.Div([
        html.Div(title, style={"fontWeight": 700, "marginBottom": "6px", "fontSize": "14px"}),
        html.Div([
            html.Div(f"{a_w}/{a_l}", style={"minWidth": "70px", "textAlign": "left", "fontWeight": 600}),
            center,
            html.Div(f"{b_w}/{b_l}", style={"minWidth": "70px", "textAlign": "right", "fontWeight": 600}),
        ], style={"display": "flex", "alignItems": "center", "gap": "12px"})
    ], style={"marginBottom": "18px"})

# ==============================================================
# UI pieces
# ==============================================================
def empty_overlay_radar():
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[0]*len(TIERS_TICK), theta=TIERS_TICK, mode="lines",
                                  line=dict(color="rgba(0,0,0,0)"), showlegend=False))
    return style_overlay_radar(fig)

app = dash.Dash(__name__)
app.title = "ATP — H2H & Player Cards (2000–2025)"

def radar_block(index, title_caption):
    return html.Div(
        [
            html.Div(title_caption, style={"fontWeight": 700, "margin": "4px 0 6px 4px", "fontSize": "14px"}),
            dcc.Graph(
                id={"type": "radar_graph", "index": index},
                figure=empty_radar_layout(""),
                config={"displayModeBar": False, "responsive": True},
                style={"height": "450px", "width": "100%"},
            ),
        ],
        style=BOX  # <- equal width with the player card
    )

def _year_marks(start, end):
    span = end - start
    step = 1 if span <= 8 else 2 if span <= 16 else 5
    return {y: str(y) for y in range(start, end + 1, step)}

def overlay_card():
    return html.Div(
        [
            html.Div("Comparison by surface or tournament", style={"fontWeight": 800, "fontSize": "18px", "marginBottom": "6px"}),
            html.Div(
                [
                    html.Div(
                        [html.Div("Surface", style={"fontWeight": 700, "marginBottom": 6}),
                         dcc.Dropdown(
                             id="surface_dd",
                             options=[{"label": "Any", "value": "Any"}] + [{"label": s, "value": s} for s in SURF_ORDER],
                             value="Any", clearable=False)]
                    ),
                    html.Div(
                        [html.Div("Tournament", style={"fontWeight": 700, "marginBottom": 6}),
                         dcc.Dropdown(
                             id="tier_dd",
                             options=[{"label": "Any", "value": "Any"}] + [{"label": t, "value": t} for t in PREFERRED_TIERS],
                             value="Any", clearable=False)]
                    ),
                ],
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px", "marginBottom": "8px"},
            ),
            html.Div(
                [
                    html.Div("Years", style={"fontWeight": 700, "marginBottom": 6}),
                    dcc.RangeSlider(
                        id="years_rs",
                        min=YEAR_MIN, max=YEAR_MAX, value=[YEAR_MIN, YEAR_MAX], step=1,
                        marks=_year_marks(YEAR_MIN, YEAR_MAX), allowCross=False, tooltip={"placement": "bottom", "always_visible": False}
                    ),
                ],
                style={"marginBottom": "4px"},
            ),
            dcc.Graph(id="overlay_radar", figure=empty_overlay_radar(), config={"displayModeBar": False}),
        ],
        style=BOX
    )

app.layout = html.Div(
    style={
        "display": "grid",
        "gridTemplateColumns": "340px minmax(580px, 1fr) 340px",
        "gap": "28px",                 # equal gaps left|middle|right
        "justifyContent": "center",
        "maxWidth": "1320px",
        "margin": "32px auto",
        "alignItems": "start",
        "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif",
    },
    children=[
        dcc.Store(id="selected_players", data={"a": None, "b": None}),

        # Left column (grid so children share width & line up)
        html.Div(
            [
                dcc.Dropdown(id={"type": "player_dd", "index": 0}, options=[], value=None, placeholder="Search player", clearable=True),
                html.Div(id={"type": "player_card", "index": 0}, style={"marginTop": "12px"}),
                radar_block(0, "Overall performance by surface and tournament size"),
            ],
            style={
                "display": "grid",
                "gridTemplateRows": "auto auto 1fr",
                "alignItems": "start",
                "gap": "12px",
                "width": "100%",
                "boxSizing": "border-box",
            },
        ),

        # Middle column
        html.Div(
            [
                overlay_card(),
                html.Div(
                    id="h2h_panel",
                    style={
                        **BOX,
                        "padding": "16px",
                        "minHeight": "420px",
                        "maxWidth": "600px",
                        "margin": "12px auto 0",
                    },
                ),
                html.Div(
                    f"Data: {df['source_year'].min() if not df.empty else '—'}–{df['source_year'].max() if not df.empty else '—'} "
                    f"({df['source_year'].nunique() if not df.empty else 0} years found)",
                    style={"color": "#555", "fontSize": "12px", "textAlign": "center", "marginTop": "4px"}
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateRows": "auto auto auto",
                "alignItems": "start",
                "gap": "12px",
                "width": "100%",
                "boxSizing": "border-box",
            },
        ),

        # Right column (mirrors left)
        html.Div(
            [
                dcc.Dropdown(id={"type": "player_dd", "index": 1}, options=[], value=None, placeholder="Search player", clearable=True),
                html.Div(id={"type": "player_card", "index": 1}, style={"marginTop": "12px"}),
                radar_block(1, "Overall performance by surface and tournament size"),
            ],
            style={
                "display": "grid",
                "gridTemplateRows": "auto auto 1fr",
                "alignItems": "start",
                "gap": "12px",
                "width": "100%",
                "boxSizing": "border-box",
            },
        ),
    ],
)

# ==============================================================
# Callbacks
# ==============================================================
@app.callback(
    Output({"type": "player_dd", "index": MATCH}, "options"),
    Input({"type": "player_dd", "index": MATCH}, "search_value"),
    State({"type": "player_dd", "index": MATCH}, "value")
)
def suggest_players(search_value, current_value):
    base = ALL_PLAYERS
    if not search_value:
        candidates = base[:50]
    else:
        s = str(search_value).strip().lower()
        starts = [p for p in base if p.lower().startswith(s)]
        inside = [p for p in base if s in p.lower() and not p.lower().startswith(s)]
        candidates = (starts + inside)[:50]
    if current_value and current_value not in candidates and current_value in base:
        candidates = ([current_value] + candidates)[:50]
    return [{"label": p, "value": p} for p in candidates]

@app.callback(
    Output({"type": "player_card", "index": MATCH}, "children"),
    Input({"type": "player_dd", "index": MATCH}, "value"),
    prevent_initial_call=False
)
def show_inline_card(selected_name):
    if not selected_name:
        return html.Div("Search a player to see profile.", style={"color": "#666"})
    row = find_player_row_by_name(selected_name)
    if not row:
        return html.Div("No player info found.", style={"color": "crimson"})
    return card_from_row(row)

@app.callback(
    Output("selected_players", "data"),
    Input({"type": "player_dd", "index": 0}, "value"),
    Input({"type": "player_dd", "index": 1}, "value"),
    prevent_initial_call=False
)
def sync_store(a, b):
    return {"a": a, "b": b}

@app.callback(
    Output({"type": "radar_graph", "index": MATCH}, "figure"),
    Input({"type": "player_dd", "index": MATCH}, "value")
)
def update_radar(selected_player):
    if not selected_player or df.empty:
        return empty_radar_layout("")
    data = player_surface_tier_wr(selected_player)
    if data.empty:
        return empty_radar_layout("")
    data = data[data["matches"] > 0]
    if data.empty:
        return empty_radar_layout("")
    theta_order = [t for t in PREFERRED_TIERS if t in data["tier"].unique().tolist()]
    full = (
        data.set_index(["surface_label", "tier"])
            .reindex(pd.MultiIndex.from_product([SURF_ORDER, theta_order], names=["surface_label", "tier"]))
            .reset_index()
    )
    full["win_rate"] = full["win_rate"].fillna(0)
    fig = px.line_polar(
        full,
        r="win_rate", theta="tier", color="surface_label",
        category_orders={"tier": theta_order, "surface_label": SURF_ORDER},
        line_close=True, markers=True, color_discrete_map=COLOR_MAP, template="none", height=450
    )
    fig = _style_radar(fig, theta_order)
    return fig

# ---- Overlay radar with YEAR FILTER ----
@app.callback(
    Output("overlay_radar", "figure"),
    Input("selected_players", "data"),
    Input("surface_dd", "value"),
    Input("tier_dd", "value"),
    Input("years_rs", "value"),
)
def update_overlay(sel, surface_val, tier_val, years_range):
    a = (sel or {}).get("a")
    b = (sel or {}).get("b")
    if not a or not b or a == b or df.empty:
        return empty_overlay_radar()

    surface_val = None if (not surface_val or surface_val == "Any") else surface_val
    tier_val    = None if (not tier_val or tier_val == "Any") else tier_val
    years = tuple(years_range) if years_range else None

    # Axis mode
    show_axes_as_tiers = True
    if surface_val is None and tier_val is not None:
        show_axes_as_tiers = False

    if show_axes_as_tiers:
        a_df = win_rate_by_tiers(a, surface=surface_val, years=years)
        b_df = win_rate_by_tiers(b, surface=surface_val, years=years)
        theta = TIERS_TICK
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=a_df["win_rate"], theta=theta, name=a, line=dict(color=A_COLOR)))
        fig.add_trace(go.Scatterpolar(r=b_df["win_rate"], theta=theta, name=b, line=dict(color=B_COLOR)))
        yr_text = f" ({years[0]}–{years[1]})" if years else ""
        title = "Win rate by tournament size" + (f" — {surface_val}" if surface_val else "") + yr_text
        fig.update_layout(title=dict(text=title, x=0.5, y=0.98, font=dict(size=16)))
        return style_overlay_radar(fig)
    else:
        a_df = win_rate_by_surfaces(a, tier=tier_val, years=years)
        b_df = win_rate_by_surfaces(b, tier=tier_val, years=years)
        theta = SURF_ORDER
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=a_df["win_rate"], theta=theta, name=a, line=dict(color=A_COLOR)))
        fig.add_trace(go.Scatterpolar(r=b_df["win_rate"], theta=theta, name=b, line=dict(color=B_COLOR)))
        yr_text = f" ({years[0]}–{years[1]})" if years else ""
        fig.update_layout(title=dict(text=f"Win rate by surface — {tier_val}{yr_text}", x=0.5, y=0.98, font=dict(size=16)))
        return style_overlay_radar(fig)

@app.callback(
    Output("h2h_panel", "children"),
    Input("selected_players", "data"),
    prevent_initial_call=False
)
def update_h2h(sel):
    a = sel.get("a"); b = sel.get("b")
    if not a or not b or a == b or df.empty:
        return html.Div("Pick two different players to see the head-to-head panel.",
                        style={"color": "#666", "textAlign": "center", "paddingTop": "20px"})

    wins_a, wins_b, pct_a = head_to_head(a, b)
    total_h2h = wins_a + wins_b
    pct_text = f"{int(round(pct_a*100))}%" if total_h2h else "—"
    left_pct = int(round(pct_a*100)) if total_h2h else 50

    career_a = player_record(a, None); career_b = player_record(b, None)
    clay_a   = player_record(a, "Clay");  clay_b  = player_record(b, "Clay")
    grass_a  = player_record(a, "Grass"); grass_b = player_record(b, "Grass")
    hard_a   = player_record(a, "Hard court"); hard_b = player_record(b, "Hard court")

    header = html.Div([
        html.Div("Head-to-head Performance", style={"fontWeight": 800, "textAlign": "center", "fontSize": "20px", "marginBottom": "4px"}),
        html.Div("VS", style={"textAlign": "center", "fontWeight": 700, "letterSpacing": "1px", "marginBottom": "6px"}),
        html.Div([
            html.Div(style={
                "width": "240px", "height": "120px",
                "borderTopLeftRadius": "240px", "borderTopRightRadius": "240px",
                "background": f"linear-gradient(90deg, {A_COLOR} 0%, {A_COLOR} {left_pct}%, {B_COLOR} {left_pct}%, {B_COLOR} 100%)",
                "margin": "0 auto", "position": "relative"
            }),
            html.Div(pct_text, style={"position": "relative", "top": "-78px", "textAlign": "center", "fontSize": "26px", "fontWeight": 800}),
        ]),
        html.Div([
            html.Div(str(wins_a), style={"fontWeight": 800, "color": A_COLOR}),
            html.Div("VS", style={"fontWeight": 800, "opacity": 0.75}),
            html.Div(str(wins_b), style={"fontWeight": 800, "color": B_COLOR}),
        ], style={
            "display": "flex", "justifyContent": "center", "gap": "12px",
            "position": "relative", "top": "10px",   # move the "5 VS 4" line slightly lower
            "marginBottom": "8px"
        })
    ])

    rows = [
        metric_row("Career win/loss",     career_a, career_b),
        metric_row("Clay win/loss",       clay_a,   clay_b),
        metric_row("Grass win/loss",      grass_a,  grass_b),
        metric_row("Hard court win/loss", hard_a,   hard_b),
    ]
    return html.Div([header, html.Div(rows, style={"marginTop": "8px"})])

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
