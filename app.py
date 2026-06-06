"""
app.py — HouseWise: a professional Streamlit dashboard for House Price Prediction.

Run with:
    streamlit run app.py
"""

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st
from sklearn.model_selection import learning_curve
import scipy.stats as stats

from eda import HousingEDA
from train import engineer_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DIR = Path(__file__).resolve().parent
SCALER_PATH = DIR / "scaler.pkl"
METRICS_PATH = DIR / "metrics.json"
FEAT_NAMES_PATH = DIR / "feature_names.json"
FEAT_IMP_PATH = DIR / "feature_importance.json"

MODEL_PATHS = {
    "Linear Regression": DIR / "model_lr.pkl",
    "Ridge Regression": DIR / "model_ridge.pkl",
    "Random Forest": DIR / "model_rf.pkl",
    "Gradient Boosting": DIR / "model_gb.pkl",
    "Stacking Ensemble": DIR / "model_stack.pkl",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HouseWise",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Google Fonts
# ---------------------------------------------------------------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Marcellus&family=Josefin+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Art Deco — Full CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* GLOBAL */
html, body, [class*="css"] {
    font-family: 'Josefin Sans', sans-serif !important;
    background-color: #0A0A0A !important;
    color: #F2F0E4 !important;
}
#MainMenu, footer, header { visibility: hidden; }

/* DIAGONAL CROSSHATCH BACKGROUND PATTERN */
[data-testid="stAppViewContainer"] {
    background-image:
        repeating-linear-gradient(
            45deg,
            rgba(212,175,55,0.03) 0px, rgba(212,175,55,0.03) 1px,
            transparent 1px, transparent 40px
        ),
        repeating-linear-gradient(
            -45deg,
            rgba(212,175,55,0.03) 0px, rgba(212,175,55,0.03) 1px,
            transparent 1px, transparent 40px
        ) !important;
    background-color: #0A0A0A !important;
}

/* HEADINGS */
h1, h2, h3, h4 {
    font-family: 'Marcellus', serif !important;
    color: #D4AF37 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #0A0A0A !important;
    border-right: 1px solid rgba(212,175,55,0.3) !important;
}

/* BUTTONS */
.stButton > button {
    font-family: 'Josefin Sans', sans-serif !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
    font-size: 0.8rem !important;
    background: transparent !important;
    color: #D4AF37 !important;
    border: 1px solid #D4AF37 !important;
    border-radius: 0px !important;
    padding: 0.7rem 2rem !important;
    min-height: 48px !important;
    box-shadow: none !important;
    transition: all 400ms ease-out !important;
}
.stButton > button:hover {
    background: #D4AF37 !important;
    color: #0A0A0A !important;
    box-shadow: 0 0 20px rgba(212,175,55,0.35) !important;
}
.stButton > button:active {
    background: #F2E8C4 !important;
    color: #0A0A0A !important;
}

/* PRIMARY BUTTON */
.stButton > button[kind="primary"] {
    background: #D4AF37 !important;
    color: #0A0A0A !important;
    border: 1px solid #D4AF37 !important;
    box-shadow: 0 0 15px rgba(212,175,55,0.25) !important;
}
.stButton > button[kind="primary"]:hover {
    background: #F2E8C4 !important;
    box-shadow: 0 0 25px rgba(212,175,55,0.45) !important;
}

/* DOWNLOAD BUTTON */
.stDownloadButton > button {
    font-family: 'Josefin Sans', sans-serif !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
    font-size: 0.8rem !important;
    background: transparent !important;
    color: #D4AF37 !important;
    border: 1px solid #D4AF37 !important;
    border-radius: 0px !important;
    padding: 0.7rem 2rem !important;
    transition: all 400ms ease-out !important;
}
.stDownloadButton > button:hover {
    background: #D4AF37 !important;
    color: #0A0A0A !important;
    box-shadow: 0 0 20px rgba(212,175,55,0.35) !important;
}

/* INPUTS */
.stTextArea textarea, .stTextInput input {
    font-family: 'Josefin Sans', sans-serif !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid #D4AF37 !important;
    border-radius: 0px !important;
    color: #F2F0E4 !important;
    padding: 0.75rem 0.5rem !important;
    min-height: 48px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-bottom-color: #F2E8C4 !important;
    box-shadow: 0 4px 12px rgba(212,175,55,0.2) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder {
    color: #888888 !important;
    font-style: italic !important;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 1px solid rgba(212,175,55,0.3) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Josefin Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #888888 !important;
    background: transparent !important;
    border: none !important;
    border-radius: 0px !important;
    padding: 0.75rem 1.5rem !important;
    transition: all 300ms ease-out !important;
}
.stTabs [aria-selected="true"] {
    color: #D4AF37 !important;
    border-bottom: 2px solid #D4AF37 !important;
    box-shadow: 0 4px 12px rgba(212,175,55,0.15) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    color: #F2E8C4 !important;
}

/* METRIC CARDS */
[data-testid="stMetric"] {
    background: #141414 !important;
    border: 1px solid rgba(212,175,55,0.25) !important;
    border-radius: 0px !important;
    padding: 1.25rem !important;
    position: relative !important;
    transition: all 400ms ease !important;
}
[data-testid="stMetric"]:hover {
    border-color: #D4AF37 !important;
    box-shadow: 0 0 20px rgba(212,175,55,0.15) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Josefin Sans', sans-serif !important;
    font-size: 0.65rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: #888888 !important;
    font-weight: 700 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Marcellus', serif !important;
    color: #D4AF37 !important;
    font-size: 1.8rem !important;
}

/* EXPANDERS */
details > summary {
    font-family: 'Josefin Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    background: #141414 !important;
    border: 1px solid rgba(212,175,55,0.3) !important;
    border-radius: 0px !important;
    padding: 0.75rem 1rem !important;
    color: #D4AF37 !important;
}

/* SLIDERS */
[data-testid="stSlider"] > div > div > div {
    background: #D4AF37 !important;
}

/* SELECT BOX */
.stSelectbox [data-baseweb="select"] {
    background: #141414 !important;
    border: 1px solid rgba(212,175,55,0.4) !important;
    border-radius: 0px !important;
    font-family: 'Josefin Sans', sans-serif !important;
    color: #F2F0E4 !important;
}

/* DATAFRAME */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(212,175,55,0.3) !important;
    border-radius: 0px !important;
}

/* PROGRESS BAR */
[data-testid="stProgressBar"] > div {
    background: #D4AF37 !important;
    box-shadow: 0 0 8px rgba(212,175,55,0.5) !important;
}

/* ALERTS */
.stAlert {
    background: #141414 !important;
    border: 1px solid rgba(212,175,55,0.3) !important;
    border-radius: 0px !important;
    color: #F2F0E4 !important;
}

/* DIVIDERS */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(
        90deg, transparent, #D4AF37, transparent
    ) !important;
    margin: 2rem 0 !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: #141414 !important;
    border: 1px dashed rgba(212,175,55,0.4) !important;
    border-radius: 0px !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Art Deco — UI helper functions
# ---------------------------------------------------------------------------

def render_header():
    """Render the main Art Deco page header with corner brackets and gold lines."""
    st.markdown("""
    <div style="
        background:#141414;padding:2rem;
        border-bottom:1px solid rgba(212,175,55,0.3);
        margin-bottom:1.5rem;position:relative;
        box-shadow:0 4px 30px rgba(212,175,55,0.08);
    ">
        <!-- Decorative corner brackets -->
        <div style="position:absolute;top:12px;left:12px;
            width:20px;height:20px;
            border-top:2px solid #D4AF37;border-left:2px solid #D4AF37;">
        </div>
        <div style="position:absolute;top:12px;right:12px;
            width:20px;height:20px;
            border-top:2px solid #D4AF37;border-right:2px solid #D4AF37;">
        </div>
        <div style="position:absolute;bottom:12px;left:12px;
            width:20px;height:20px;
            border-bottom:2px solid #D4AF37;border-left:2px solid #D4AF37;">
        </div>
        <div style="position:absolute;bottom:12px;right:12px;
            width:20px;height:20px;
            border-bottom:2px solid #D4AF37;
            border-right:2px solid #D4AF37;">
        </div>
        <!-- Decorative gold line above title -->
        <div style="
            display:flex;align-items:center;gap:1rem;
            margin-bottom:0.75rem;
        ">
            <div style="flex:1;height:1px;
                background:linear-gradient(90deg,transparent,#D4AF37);">
            </div>
            <div style="
                font-family:'Josefin Sans',sans-serif;
                font-size:0.6rem;font-weight:700;
                text-transform:uppercase;letter-spacing:0.3em;
                color:#888888;
            ">EST. 2025</div>
            <div style="flex:1;height:1px;
                background:linear-gradient(90deg,#D4AF37,transparent);">
            </div>
        </div>
        <h1 style="
            font-family:'Marcellus',serif;
            font-size:2.75rem;font-weight:400;
            color:#D4AF37;margin:0;
            text-align:center;
            text-transform:uppercase;letter-spacing:0.2em;
            text-shadow:0 0 40px rgba(212,175,55,0.3);
        ">🏠 HouseWise</h1>
        <!-- Decorative gold line below title -->
        <div style="
            display:flex;align-items:center;gap:1rem;
            margin-top:0.75rem;
        ">
            <div style="flex:1;height:1px;
                background:linear-gradient(90deg,transparent,#D4AF37);">
            </div>
            <div style="
                font-family:'Josefin Sans',sans-serif;
                font-size:0.6rem;font-weight:700;
                text-transform:uppercase;letter-spacing:0.3em;
                color:#888888;
            ">PROPERTY PRICE PREDICTOR</div>
            <div style="flex:1;height:1px;
                background:linear-gradient(90deg,#D4AF37,transparent);">
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_header():
    """Render the Art Deco sidebar branded header."""
    st.sidebar.markdown("""
    <div style="
        background:#141414;padding:1.75rem;
        border:1px solid rgba(212,175,55,0.3);
        margin-bottom:1rem;position:relative;
    ">
        <div style="position:absolute;top:8px;left:8px;
            width:12px;height:12px;
            border-top:1px solid #D4AF37;
            border-left:1px solid #D4AF37;"></div>
        <div style="position:absolute;bottom:8px;right:8px;
            width:12px;height:12px;
            border-bottom:1px solid #D4AF37;
            border-right:1px solid #D4AF37;"></div>
        <div style="display:flex;justify-content:center;margin-bottom:0.5rem;">
            <div style="width:24px;height:24px;background:#D4AF37;transform:rotate(45deg);box-shadow:0 0 10px rgba(212,175,55,0.5);"></div>
        </div>
        <div style="
            font-family:'Josefin Sans',sans-serif;
            font-size:0.55rem;font-weight:700;
            text-transform:uppercase;letter-spacing:0.3em;
            color:#888888;text-align:center;
        ">CONTROL PANEL</div>
        <h2 style="
            font-family:'Marcellus',serif;font-size:1.25rem;
            color:#D4AF37;margin:0.25rem 0 0;
            text-transform:uppercase;letter-spacing:0.12em;
            text-align:center;
        ">🏠 HouseWise</h2>
    </div>
    """, unsafe_allow_html=True)


def render_price_result(price: float, delta: float,
                        category: str, rmse: float):
    """Render the Art Deco price prediction result card."""
    cat_colors = {
        "Budget": "#60A5FA", "Mid-range": "#34D399",
        "Premium": "#D4AF37", "Luxury": "#F472B6"
    }
    cat_color = cat_colors.get(category, "#D4AF37")
    delta_sign = "+" if delta >= 0 else ""

    st.markdown(f"""
    <div style="
        background:#141414;padding:2rem;
        border:1px solid rgba(212,175,55,0.4);
        position:relative;
        box-shadow:0 0 40px rgba(212,175,55,0.1);
        transition:all 500ms ease;
    ">
        <!-- Corner brackets -->
        <div style="position:absolute;top:8px;left:8px;
            width:16px;height:16px;
            border-top:2px solid #D4AF37;
            border-left:2px solid #D4AF37;"></div>
        <div style="position:absolute;bottom:8px;right:8px;
            width:16px;height:16px;
            border-bottom:2px solid #D4AF37;
            border-right:2px solid #D4AF37;"></div>
        <!-- Category badge -->
        <div style="text-align:center;margin-bottom:0.75rem;">
            <span style="
                font-family:'Josefin Sans',sans-serif;
                font-size:0.65rem;font-weight:700;
                text-transform:uppercase;letter-spacing:0.2em;
                color:{cat_color};
                border:1px solid {cat_color};
                padding:0.25rem 1rem;
                box-shadow:0 0 12px {cat_color}44;
            ">{category}</span>
        </div>
        <!-- Price -->
        <div style="
            font-family:'Marcellus',serif;
            font-size:3rem;font-weight:400;
            color:#D4AF37;text-align:center;
            text-shadow:0 0 30px rgba(212,175,55,0.4);
            letter-spacing:0.05em;
        ">${price:,.0f}</div>
        <!-- Delta -->
        <div style="
            font-family:'Josefin Sans',sans-serif;
            font-size:0.8rem;text-align:center;
            color:#888888;text-transform:uppercase;
            letter-spacing:0.1em;margin-top:0.25rem;
        ">{delta_sign}${abs(delta):,.0f} vs median</div>
        <!-- Gold divider -->
        <div style="
            height:1px;margin:1rem 2rem;
            background:linear-gradient(90deg,
                transparent,#D4AF37 30%,#D4AF37 70%,transparent);
        "></div>
        <!-- Confidence interval -->
        <div style="
            font-family:'Josefin Sans',sans-serif;
            font-size:0.7rem;text-align:center;
            color:#888888;text-transform:uppercase;
            letter-spacing:0.1em;
        ">± ${rmse:,.0f} CONFIDENCE INTERVAL</div>
    </div>
    """, unsafe_allow_html=True)


def ad_section_heading(title: str, subtitle: str = ""):
    """Render an Art Deco section heading with gold divider lines."""
    subtitle_html = f'<p style="font-family:Josefin Sans,sans-serif;font-size:0.8rem;color:#888;text-transform:uppercase;letter-spacing:0.12em;">{subtitle}</p>' if subtitle else ''
    st.markdown(f"""
    <div style="text-align:center;margin:2rem 0 1.5rem;">
        <div style="
            display:flex;align-items:center;gap:1rem;
            margin-bottom:0.75rem;
        ">
            <div style="flex:1;height:1px;
                background:linear-gradient(
                    90deg,transparent,rgba(212,175,55,0.6));
            "></div>
            <h2 style="
                font-family:'Marcellus',serif;
                font-size:1.5rem;font-weight:400;
                color:#D4AF37;margin:0;
                text-transform:uppercase;letter-spacing:0.2em;
            ">{title}</h2>
            <div style="flex:1;height:1px;
                background:linear-gradient(
                    90deg,rgba(212,175,55,0.6),transparent);
            "></div>
        </div>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)


def ad_divider():
    """Render an Art Deco divider with a rotated diamond centerpiece."""
    st.markdown("""
    <div style="
        display:flex;align-items:center;
        gap:0.75rem;margin:1.5rem 0;
    ">
        <div style="flex:1;height:1px;
            background:linear-gradient(
                90deg,transparent,rgba(212,175,55,0.4));
        "></div>
        <div style="
            width:8px;height:8px;
            border:1px solid #D4AF37;
            transform:rotate(45deg);
            box-shadow:0 0 8px rgba(212,175,55,0.4);
        "></div>
        <div style="flex:1;height:1px;
            background:linear-gradient(
                90deg,rgba(212,175,55,0.4),transparent);
        "></div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_metric(label: str, value: str):
    """Render an Art Deco metric card in the sidebar."""
    st.sidebar.markdown(f"""
    <div style="
        background:#141414;
        border:1px solid rgba(212,175,55,0.25);
        padding:0.875rem 1rem;margin-bottom:0.6rem;
        transition:all 400ms ease;
    ">
        <div style="font-family:'Josefin Sans',sans-serif;
            font-size:0.6rem;color:#888888;
            text-transform:uppercase;letter-spacing:0.15em;
            font-weight:700;">{label}</div>
        <div style="font-family:'Marcellus',serif;
            font-size:1.5rem;color:#D4AF37;
            margin-top:2px;">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def render_eda_metric(label: str, value: str):
    """Render an Art Deco styled metric card for EDA."""
    return f"""
    <div style="
        background:#141414;
        border:1px solid rgba(212,175,55,0.25);
        padding:1rem 1.25rem;text-align:center;
        position:relative;
        transition:all 400ms ease;
    ">
        <div style="position:absolute;top:6px;left:6px;
            width:8px;height:8px;
            border-top:1px solid #D4AF37;
            border-left:1px solid #D4AF37;"></div>
        <div style="position:absolute;bottom:6px;right:6px;
            width:8px;height:8px;
            border-bottom:1px solid #D4AF37;
            border-right:1px solid #D4AF37;"></div>
        <div style="font-family:'Josefin Sans',sans-serif;
            font-size:0.6rem;color:#888888;
            text-transform:uppercase;letter-spacing:0.15em;
            font-weight:700;margin-bottom:4px;">{label}</div>
        <div style="font-family:'Marcellus',serif;
            font-size:1.4rem;color:#D4AF37;">{value}</div>
    </div>"""


def render_feat_card(name: str, importance: float,
                     explanation: str, correlation: float):
    """Render an Art Deco feature explanation card."""
    corr_text = "Positive — Higher values = higher prices" if correlation > 0 \
        else "Negative — Higher values = lower prices"
    st.markdown(f"""
    <div style="
        background:#141414;
        border:1px solid rgba(212,175,55,0.25);
        border-left:3px solid #D4AF37;
        padding:1.25rem 1.5rem;margin-bottom:1rem;
        position:relative;
        transition:all 400ms ease;
    ">
        <div style="position:absolute;top:8px;right:8px;
            width:10px;height:10px;
            border-top:1px solid rgba(212,175,55,0.4);
            border-right:1px solid rgba(212,175,55,0.4);"></div>
        <h4 style="
            font-family:'Marcellus',serif;font-size:1.1rem;
            color:#D4AF37;margin:0 0 0.25rem;
            text-transform:uppercase;letter-spacing:0.1em;
        ">{name} <span style="
            font-family:'Josefin Sans',sans-serif;font-size:0.75rem;
            color:#888888;font-weight:400;letter-spacing:0.05em;
        ">({importance*100:.1f}% importance)</span></h4>
        <p style="font-family:'Josefin Sans',sans-serif;
            font-size:0.85rem;color:#F2F0E4;
            margin:0.4rem 0;">{explanation}</p>
        <p style="font-family:'Josefin Sans',sans-serif;
            font-size:0.8rem;color:#888888;margin:0;">
            <strong style="color:#D4AF37;">Correlation:</strong> {correlation:.3f}
            — {corr_text}</p>
    </div>
    """, unsafe_allow_html=True)


def apply_ad_plotly_theme(fig):
    """Apply Art Deco theme to a Plotly figure."""
    fig.update_layout(
        plot_bgcolor='#141414',
        paper_bgcolor='#0A0A0A',
        font=dict(family='Josefin Sans', color='#F2F0E4', size=11),
        title_font=dict(family='Marcellus', size=15, color='#D4AF37'),
        hoverlabel=dict(bgcolor="#141414", bordercolor="#D4AF37", font_family="Josefin Sans", font_color="#F2F0E4"),
        colorway=['#D4AF37', '#F2E8C4', '#888888',
                  '#1E3D59', '#60A5FA', '#34D399'],
        legend=dict(
            bgcolor='#141414',
            bordercolor='rgba(212,175,55,0.3)',
            borderwidth=1,
            font=dict(family='Josefin Sans', color='#888888')
        ),
        margin=dict(t=50, b=30, l=30, r=30),
        xaxis=dict(
            gridcolor='rgba(212,175,55,0.08)',
            linecolor='rgba(212,175,55,0.3)',
            tickfont=dict(family='Josefin Sans', color='#888888', size=10),
            title_font=dict(family='Josefin Sans', color='#888888', size=10)
        ),
        yaxis=dict(
            gridcolor='rgba(212,175,55,0.08)',
            linecolor='rgba(212,175,55,0.3)',
            tickfont=dict(family='Josefin Sans', color='#888888', size=10),
            title_font=dict(family='Josefin Sans', color='#888888', size=10)
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Cached loaders & functions
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    eda = HousingEDA()
    df = eda.load_data()
    return df

@st.cache_resource
def load_model(model_name: str):
    path = MODEL_PATHS.get(model_name)
    if path and path.exists():
        return joblib.load(path)
    return None

@st.cache_resource
def load_scaler():
    if SCALER_PATH.exists():
        return joblib.load(SCALER_PATH)
    return None

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as fh:
            return json.load(fh)
    return None

@st.cache_data
def load_feature_names():
    if FEAT_NAMES_PATH.exists():
        with open(FEAT_NAMES_PATH) as fh:
            return json.load(fh)
    return None

@st.cache_data
def load_feature_importances():
    if FEAT_IMP_PATH.exists():
        with open(FEAT_IMP_PATH) as fh:
            return json.load(fh)
    return None

@st.cache_data(show_spinner=False)
def compute_learning_curve(model_name):
    # Retrieve data
    df = load_data()
    df_eng = engineer_features(df)

    feature_cols = load_feature_names()
    X = df_eng[feature_cols].values
    y = df_eng["Price"].values

    # Cap outlier
    cap = np.percentile(y, 99)
    y = np.clip(y, a_min=None, a_max=cap)

    scaler = load_scaler()
    if scaler:
        X = scaler.transform(X)

    model = load_model(model_name)
    if model is None:
        return None, None, None

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=3, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring="neg_mean_squared_error"
    )

    train_errors = np.mean(-train_scores, axis=1)
    test_errors = np.mean(-test_scores, axis=1)
    return train_sizes, train_errors, test_errors

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    render_sidebar_header()
    st.divider()

    metrics = load_metrics()

    if metrics is None:
        st.warning("Models not trained yet.")
        if st.button("🚀 Train Models Now"):
            with st.spinner("Training models (takes ~10 seconds)..."):
                result = subprocess.run([sys.executable, str(DIR / "train.py")], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("Training complete! Reloading...")
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error(f"Error: {result.stderr}")
        st.stop()

    model_choice = st.selectbox(
        "Select Model",
        list(MODEL_PATHS.keys()),
        index=list(MODEL_PATHS.keys()).index(metrics["best_model"]) if metrics["best_model"] in MODEL_PATHS else 0
    )

    model = load_model(model_choice)
    scaler = load_scaler()
    feat_names = load_feature_names()

    # Show selected model's metrics
    mod_metrics = metrics["models"][model_choice]

    ad_divider_sidebar_html = """
    <div style="display:flex;align-items:center;gap:0.5rem;margin:0.75rem 0;">
        <div style="flex:1;height:1px;background:linear-gradient(90deg,transparent,rgba(212,175,55,0.3));"></div>
        <div style="width:5px;height:5px;border:1px solid rgba(212,175,55,0.5);transform:rotate(45deg);"></div>
        <div style="flex:1;height:1px;background:linear-gradient(90deg,rgba(212,175,55,0.3),transparent);"></div>
    </div>"""
    st.sidebar.markdown(ad_divider_sidebar_html, unsafe_allow_html=True)

    render_sidebar_metric("R²", f"{mod_metrics['r2']:.3f}")
    render_sidebar_metric("RMSE", f"{mod_metrics['rmse']:.3f}")
    render_sidebar_metric("MAE", f"{mod_metrics['mae']:.3f}")

    with st.expander("📖 Prediction Guide"):
        st.markdown("""
        **MedInc**: Median income in block group ($10,000s)  
        **HouseAge**: Median house age in block group  
        **AveRooms**: Average rooms per household  
        **AveBedrms**: Average bedrooms per household  
        **Population**: Block group population  
        **AveOccup**: Average house occupancy  
        **Latitude / Longitude**: Geo coordinates  
        """)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
render_header()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Predict Price",
    "📊 EDA Dashboard",
    "🤖 Model Comparison",
    "🔍 Feature Importance"
])

# ===== TAB 1 — Predict Price =============================================
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        ad_section_heading("Property Details", "Configure your property parameters")
        med_inc = st.slider("Median Income (in $10,000s)", 0.5, 15.0, 4.0, step=0.1)
        house_age = st.slider("House Age (years)", 1, 52, 20)
        ave_rooms = st.slider("Average Rooms", 1.0, 20.0, 5.0, step=0.1)
        ave_bedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0, step=0.1)
        population = st.slider("Population", 100, 5000, 1000)
        ave_occup = st.slider("Average Occupancy", 1.0, 10.0, 3.0, step=0.1)
        latitude = st.slider("Latitude", 32.0, 42.0, 37.0, step=0.1)
        longitude = st.slider("Longitude", -125.0, -114.0, -120.0, step=0.1)

        calc_btn = st.button("Calculate Price", type="primary", use_container_width=True)

    with col2:
        ad_section_heading("Price Estimate", "AI-powered valuation")

        # Prepare input
        input_data = pd.DataFrame([{
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": latitude,
            "Longitude": longitude
        }])

        # Engineer features
        input_eng = engineer_features(input_data)

        # Predict
        if scaler and model and feat_names:
            # Ensure correct order
            X_input = input_eng[feat_names].values
            X_scaled = scaler.transform(X_input)
            pred_price = model.predict(X_scaled)[0]

            # Convert to actual dollars (target is in $100,000s)
            pred_dollars = pred_price * 100000

            # Metric with delta
            median_val = metrics["price_stats"]["median"] * 100000
            delta_val = pred_dollars - median_val

            # Category
            if pred_dollars < 150000:
                badge_text = "Budget"
            elif pred_dollars < 300000:
                badge_text = "Mid-range"
            elif pred_dollars < 450000:
                badge_text = "Premium"
            else:
                badge_text = "Luxury"

            # Confidence interval in dollars
            rmse_dollars = mod_metrics["rmse"] * 100000

            # Art Deco price result card
            render_price_result(
                price=pred_dollars,
                delta=delta_val,
                category=badge_text,
                rmse=rmse_dollars,
            )

            ad_divider()

            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_price,
                title={'text': "PRICE IN $100K vs DATASET RANGE",
                       'font': {'family': 'Josefin Sans', 'size': 12, 'color': '#888888'}},
                number={'font': {'family': 'Marcellus', 'size': 36, 'color': '#D4AF37'}},
                gauge={
                    'axis': {'range': [metrics["price_stats"]["min"], metrics["price_stats"]["max"]],
                             'tickfont': {'family': 'Josefin Sans', 'size': 10, 'color': '#888888'}},
                    'bar': {'color': '#D4AF37'},
                    'bgcolor': '#141414',
                    'bordercolor': 'rgba(212,175,55,0.3)',
                    'steps': [
                        {'range': [0, 1.5], 'color': 'rgba(30,61,89,0.2)'},
                        {'range': [1.5, 3.0], 'color': 'rgba(30,61,89,0.4)'},
                        {'range': [3.0, 4.5], 'color': 'rgba(212,175,55,0.2)'},
                        {'range': [4.5, 6.0], 'color': 'rgba(212,175,55,0.4)'}
                    ],
                }
            ))
            fig_gauge.update_layout(
                height=250,
                paper_bgcolor='#0A0A0A',
                plot_bgcolor='#0A0A0A',
                font=dict(family='Josefin Sans', color='#F2F0E4'),
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.info("Comparable homes in this area might vary based on local amenities, exact condition, and market trends.")
        else:
            st.warning("Model or scaler not loaded properly.")

    # Map below inputs
    ad_divider()
    ad_section_heading("📍 Location Preview")

    # Show map with marker
    map_df = pd.DataFrame({'lat': [latitude], 'lon': [longitude], 'Price Estimate': [f"${pred_dollars:,.0f}"]})
    fig_map = px.scatter_mapbox(
        map_df, lat="lat", lon="lon", hover_name="Price Estimate",
        zoom=5, height=400
    )
    fig_map.update_layout(
        mapbox_style="carto-darkmatter",
        margin=dict(l=0, r=0, t=0, b=0)
    )
    st.plotly_chart(fig_map, use_container_width=True)


# ===== TAB 2 — EDA Dashboard =============================================
with tab2:
    ad_section_heading("Exploratory Data Analysis", "Dataset insights & distributions")
    df = load_data()

    if df is not None:
        # Row 1: Metrics
        c1, c2, c3, c4 = st.columns(4)
        avg_price = df['Price'].mean() * 100000
        min_price = df['Price'].min() * 100000
        max_price = df['Price'].max() * 100000
        avg_inc = df['MedInc'].mean() * 10000

        c1.markdown(render_eda_metric("Total Properties", f"{len(df):,}"), unsafe_allow_html=True)
        c2.markdown(render_eda_metric("Avg Price", f"${avg_price:,.0f}"), unsafe_allow_html=True)
        c3.markdown(render_eda_metric("Price Range", f"${min_price:,.0f} – ${max_price:,.0f}"), unsafe_allow_html=True)
        c4.markdown(render_eda_metric("Avg Median Income", f"${avg_inc:,.0f}"), unsafe_allow_html=True)

        ad_divider()

        # Row 2
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            prices = df["Price"]
            fig_hist = ff.create_distplot([prices], ['Price ($100k)'], bin_size=0.2, show_rug=False)
            fig_hist.update_layout(title="Price Distribution")
            apply_ad_plotly_theme(fig_hist)
            st.plotly_chart(fig_hist, use_container_width=True)

        with r2c2:
            df_temp = df.copy()
            df_temp['AgeGroup'] = pd.cut(df_temp['HouseAge'], bins=[0, 10, 20, 30, 40, 60], labels=['0-10', '11-20', '21-30', '31-40', '41+'])
            fig_box = px.box(df_temp, x='AgeGroup', y='Price', color='AgeGroup', title="Price by House Age Groups")
            apply_ad_plotly_theme(fig_box)
            st.plotly_chart(fig_box, use_container_width=True)

        ad_divider()

        # Row 3
        r3c1, r3c2 = st.columns(2)
        with r3c1:
            sample_df = df.sample(min(2000, len(df)), random_state=42)
            fig_scatter = px.scatter(
                sample_df, x="MedInc", y="Price", color="HouseAge",
                trendline="ols", title="Income vs Price (Sample of 2000)",
                color_continuous_scale=["#141414", "#D4AF37", "#F2F0E4"]
            )
            apply_ad_plotly_theme(fig_scatter)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with r3c2:
            corr = df.corr()
            fig_corr = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                title="Feature Correlation Matrix",
                color_continuous_scale=[[0, '#1E3D59'], [0.5, '#0A0A0A'], [1, '#D4AF37']]
            )
            apply_ad_plotly_theme(fig_corr)
            st.plotly_chart(fig_corr, use_container_width=True)

        ad_divider()

        # Row 4
        r4c1, r4c2 = st.columns(2)
        with r4c1:
            map_sample = df.sample(min(5000, len(df)), random_state=42)
            fig_map2 = px.scatter_mapbox(
                map_sample, lat="Latitude", lon="Longitude", color="Price",
                size_max=15, zoom=4.5, mapbox_style="carto-darkmatter",
                title="Geographical Price Distribution",
                color_continuous_scale=[[0, '#1E3D59'], [0.5, '#D4AF37'], [1, '#F2E8C4']]
            )
            fig_map2.update_layout(
                paper_bgcolor='#0A0A0A',
                margin=dict(l=10, r=10, t=40, b=10),
                title_font=dict(family='Marcellus', size=15, color='#D4AF37'),
            )
            st.plotly_chart(fig_map2, use_container_width=True)

        with r4c2:
            fig_latlon = px.scatter(
                map_sample, x="Longitude", y="Latitude", color="Price",
                title="Longitude & Latitude vs Price",
                color_continuous_scale=[[0, '#1E3D59'], [0.5, '#D4AF37'], [1, '#F2E8C4']]
            )
            apply_ad_plotly_theme(fig_latlon)
            st.plotly_chart(fig_latlon, use_container_width=True)


# ===== TAB 3 — Model Comparison ==========================================
with tab3:
    ad_section_heading("Model Performance", "Comparative analysis")

    if metrics:
        # Table
        mod_dict = metrics["models"]
        comp_df = pd.DataFrame.from_dict(mod_dict, orient="index")
        comp_df = comp_df[["r2", "rmse", "mae", "cv_r2_mean", "cv_r2_std"]].reset_index()
        comp_df.rename(columns={
            "index": "Model", "r2": "R²", "rmse": "RMSE", "mae": "MAE",
            "cv_r2_mean": "CV R² Mean", "cv_r2_std": "CV R² Std"
        }, inplace=True)

        st.dataframe(comp_df.style.highlight_max(subset=["R²", "CV R² Mean"], color="#D4AF3733")
                                   .highlight_min(subset=["RMSE", "MAE", "CV R² Std"], color="#D4AF3733")
                                   .format({c: "{:.4f}" for c in comp_df.columns if c != "Model"}),
                     use_container_width=True, hide_index=True)

        ad_divider()

        # Grouped Bar Chart
        melted = comp_df.melt(id_vars="Model", value_vars=["RMSE", "R²"], var_name="Metric", value_name="Score")
        fig_bar = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group", title="RMSE & R² Comparison")
        apply_ad_plotly_theme(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

        ad_divider()

        # Row for scatter and residuals
        r1, r2 = st.columns(2)

        actual = metrics.get("best_actual_sample", [])
        pred = metrics.get("best_pred_sample", [])

        if actual and pred:
            with r1:
                fig_act_pred = px.scatter(x=actual, y=pred, labels={'x': "Actual Price", 'y': "Predicted Price"},
                                          title=f"Actual vs Predicted ({metrics['best_model']})", opacity=0.6)
                fig_act_pred.add_shape(type="line", x0=min(actual), y0=min(actual), x1=max(actual), y1=max(actual),
                                       line=dict(color="#D4AF37", dash="dash"))
                apply_ad_plotly_theme(fig_act_pred)
                st.plotly_chart(fig_act_pred, use_container_width=True)

            with r2:
                residuals = np.array(actual) - np.array(pred)
                fig_res = px.scatter(x=pred, y=residuals, labels={'x': "Predicted Price", 'y': "Residuals"},
                                     title="Residual Plot", opacity=0.6)
                fig_res.add_shape(type="line", x0=min(pred), y0=0, x1=max(pred), y1=0,
                                  line=dict(color="#D4AF37", dash="dash"))
                apply_ad_plotly_theme(fig_res)
                st.plotly_chart(fig_res, use_container_width=True)

        ad_divider()

        # Learning Curves
        ad_section_heading(f"Learning Curves", f"{metrics['best_model']}")
        with st.spinner("Computing learning curves..."):
            train_sizes, train_errors, test_errors = compute_learning_curve(metrics["best_model"])
            if train_sizes is not None:
                fig_lc = go.Figure()
                fig_lc.add_trace(go.Scatter(x=train_sizes, y=train_errors, mode='lines+markers',
                                            name='Training Error', line=dict(color='#D4AF37')))
                fig_lc.add_trace(go.Scatter(x=train_sizes, y=test_errors, mode='lines+markers',
                                            name='Validation Error', line=dict(color='#888888')))
                fig_lc.update_layout(
                    title="Learning Curve (MSE)",
                    xaxis_title="Training Set Size",
                    yaxis_title="Negative Mean Squared Error (closer to 0 is better)",
                )
                apply_ad_plotly_theme(fig_lc)
                st.plotly_chart(fig_lc, use_container_width=True)


# ===== TAB 4 — Feature Importance ========================================
with tab4:
    ad_section_heading("Feature Importance", f"{model_choice}")

    feat_importances = load_feature_importances()
    if feat_importances:
        imps = feat_importances.get(model_choice)
        if imps:
            # Top 8 bar chart
            top_8 = imps[:8]
            fig_fi = px.bar(
                x=[x["importance"] for x in top_8][::-1],
                y=[x["feature"] for x in top_8][::-1],
                orientation="h",
                title=f"Top 8 Features ({model_choice})",
                labels={'x': "Importance / Coefficient Weight", 'y': "Feature"}
            )
            apply_ad_plotly_theme(fig_fi)
            st.plotly_chart(fig_fi, use_container_width=True)

            ad_divider()
            ad_section_heading("Feature Insights", "What these features mean")

            df_full = load_data()
            df_eng = engineer_features(df_full)
            corr_with_price = df_eng.corr()["Price"].to_dict()

            # Plain English mapping
            explanations = {
                "MedInc": "Median income of households in the block group. Higher income usually correlates with higher property values.",
                "HouseAge": "Median age of houses in the block group. Can influence price depending on historic value vs degradation.",
                "AveRooms": "Average number of rooms per household.",
                "AveBedrms": "Average number of bedrooms per household.",
                "Population": "Total population in the block group.",
                "AveOccup": "Average number of household members.",
                "Latitude": "Geographical latitude. Determines location, e.g., NorCal vs SoCal.",
                "Longitude": "Geographical longitude. Determines location, e.g., coastal vs inland.",
                "rooms_per_household": "Derived: Average total rooms divided by average occupancy. Indicates spaciousness.",
                "bedrooms_ratio": "Derived: Proportion of rooms that are bedrooms.",
                "population_per_household": "Derived: Population density per household.",
                "income_per_room": "Derived: Median income relative to the number of rooms. A proxy for wealth density."
            }

            top_5 = imps[:5]
            for item in top_5:
                f_name = item["feature"]
                f_imp = item["importance"]
                f_corr = corr_with_price.get(f_name, 0)
                expl = explanations.get(f_name, "No explanation available.")
                render_feat_card(f_name, f_imp, expl, f_corr)
        else:
            st.info("No feature importance available for the selected model.")
