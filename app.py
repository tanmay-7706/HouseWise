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
    "Lasso Regression": DIR / "model_lasso.pkl",
    "Random Forest": DIR / "model_rf.pkl",
    "Gradient Boosting": DIR / "model_gb.pkl",
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
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.metric-card {
    background: #f7f8fa;
    border: 1px solid #e1e4e8;
    border-radius: 10px;
    padding: 16px 18px;
    text-align: center;
    margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.metric-card .label { font-size: 0.9rem; color: #6a737d; margin-bottom: 4px; }
.metric-card .value { font-size: 1.6rem; font-weight: 700; color: #24292e; }

.badge-budget { background:#28a745; color:#fff; padding:6px 16px; border-radius:20px; font-weight:600; font-size:1rem; }
.badge-mid { background:#17a2b8; color:#fff; padding:6px 16px; border-radius:20px; font-weight:600; font-size:1rem; }
.badge-premium { background:#ffc107; color:#212529; padding:6px 16px; border-radius:20px; font-weight:600; font-size:1rem; }
.badge-luxury { background:#dc3545; color:#fff; padding:6px 16px; border-radius:20px; font-weight:600; font-size:1rem; }

.feat-card {
    background: #ffffff;
    border-left: 5px solid #1f77b4;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
.feat-card h4 { margin-top: 0; margin-bottom: 5px; color: #333; }
.feat-card p { margin-bottom: 5px; color: #555; }
</style>
""", unsafe_allow_html=True)


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
    st.markdown("## 🏠 HouseWise")
    st.caption("AI-powered Property Price Predictor")
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
    st.markdown("### Selected Model Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{mod_metrics['r2']:.3f}")
    c2.metric("RMSE", f"{mod_metrics['rmse']:.3f}")
    c3.metric("MAE", f"{mod_metrics['mae']:.3f}")
    
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
        st.subheader("🏡 Property Details")
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
        st.subheader("💰 Price Estimate")
        
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
            
            # Display Large Price
            st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>${pred_dollars:,.0f}</h1>", unsafe_allow_html=True)
            
            # Metric with delta
            median_val = metrics["price_stats"]["median"] * 100000
            delta_val = pred_dollars - median_val
            st.metric("Compared to Median House", f"${pred_dollars:,.0f}", f"{delta_val:+,.0f}")
            
            # Confidence interval
            rmse_dollars = mod_metrics["rmse"] * 100000
            st.caption(f"Estimated Range: **${max(0, pred_dollars - rmse_dollars):,.0f} — ${pred_dollars + rmse_dollars:,.0f}**")
            
            # Category Badge
            if pred_dollars < 150000:
                badge_class, badge_text = "badge-budget", "Budget"
            elif pred_dollars < 300000:
                badge_class, badge_text = "badge-mid", "Mid-range"
            elif pred_dollars < 450000:
                badge_class, badge_text = "badge-premium", "Premium"
            else:
                badge_class, badge_text = "badge-luxury", "Luxury"
                
            st.markdown(f'<div style="text-align:center; margin-top:10px;"><span class="{badge_class}">{badge_text} Tier</span></div>', unsafe_allow_html=True)
            
            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_price,
                title={'text': "Price in $100k's vs Dataset Range"},
                gauge={
                    'axis': {'range': [metrics["price_stats"]["min"], metrics["price_stats"]["max"]]},
                    'bar': {'color': "#1f77b4"},
                    'steps': [
                        {'range': [0, 1.5], 'color': "#d4edda"},
                        {'range': [1.5, 3.0], 'color': "#c3e6cb"},
                        {'range': [3.0, 4.5], 'color': "#ffc107"},
                        {'range': [4.5, 6.0], 'color': "#f8d7da"}
                    ],
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.info("Comparable homes in this area might vary based on local amenities, exact condition, and market trends.")
        else:
            st.warning("Model or scaler not loaded properly.")
            
    # Map below inputs
    st.markdown("---")
    st.subheader("📍 Location Preview")
    
    # Show map with marker
    map_df = pd.DataFrame({'lat': [latitude], 'lon': [longitude], 'Price Estimate': [f"${pred_dollars:,.0f}"]})
    fig_map = px.scatter_mapbox(
        map_df, lat="lat", lon="lon", hover_name="Price Estimate",
        zoom=5, height=400
    )
    fig_map.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=0, b=0)
    )
    st.plotly_chart(fig_map, use_container_width=True)


# ===== TAB 2 — EDA Dashboard =============================================
with tab2:
    st.subheader("📊 Exploratory Data Analysis")
    df = load_data()
    
    if df is not None:
        # Row 1: Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="label">Total Properties</div><div class="value">{len(df):,}</div></div>', unsafe_allow_html=True)
        
        avg_price = df['Price'].mean() * 100000
        c2.markdown(f'<div class="metric-card"><div class="label">Avg Price</div><div class="value">${avg_price:,.0f}</div></div>', unsafe_allow_html=True)
        
        min_price = df['Price'].min() * 100000
        max_price = df['Price'].max() * 100000
        c3.markdown(f'<div class="metric-card"><div class="label">Price Range</div><div class="value">${min_price:,.0f} - ${max_price:,.0f}</div></div>', unsafe_allow_html=True)
        
        avg_inc = df['MedInc'].mean() * 10000
        c4.markdown(f'<div class="metric-card"><div class="label">Avg Median Income</div><div class="value">${avg_inc:,.0f}</div></div>', unsafe_allow_html=True)
        
        # Row 2
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            # Histogram with normal curve overlay
            prices = df["Price"]
            fig_hist = ff.create_distplot([prices], ['Price ($100k)'], bin_size=0.2, show_rug=False)
            fig_hist.update_layout(title="Price Distribution", template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with r2c2:
            # Box plot by House Age group
            df_temp = df.copy()
            df_temp['AgeGroup'] = pd.cut(df_temp['HouseAge'], bins=[0, 10, 20, 30, 40, 60], labels=['0-10', '11-20', '21-30', '31-40', '41+'])
            fig_box = px.box(df_temp, x='AgeGroup', y='Price', color='AgeGroup', title="Price by House Age Groups")
            fig_box.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_box, use_container_width=True)
            
        # Row 3
        r3c1, r3c2 = st.columns(2)
        with r3c1:
            # Scatter plot MedInc vs Price
            # Sample to prevent browser crash
            sample_df = df.sample(min(2000, len(df)), random_state=42)
            fig_scatter = px.scatter(
                sample_df, x="MedInc", y="Price", color="HouseAge", 
                trendline="ols", title="Income vs Price (Sample of 2000)",
                color_continuous_scale="Viridis"
            )
            fig_scatter.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with r3c2:
            # Heatmap correlation
            corr = df.corr()
            fig_corr = px.imshow(
                corr, text_auto=".2f", aspect="auto", 
                title="Feature Correlation Matrix",
                color_continuous_scale="RdBu_r"
            )
            fig_corr.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_corr, use_container_width=True)
            
        # Row 4
        r4c1, r4c2 = st.columns(2)
        with r4c1:
            # Map of California
            # Sample down for map performance
            map_sample = df.sample(min(5000, len(df)), random_state=42)
            fig_map2 = px.scatter_mapbox(
                map_sample, lat="Latitude", lon="Longitude", color="Price",
                size_max=15, zoom=4.5, mapbox_style="open-street-map",
                title="Geographical Price Distribution",
                color_continuous_scale="Spectral_r"
            )
            fig_map2.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_map2, use_container_width=True)
            
        with r4c2:
            # Scatter Lat/Lon vs Price
            fig_latlon = px.scatter(
                map_sample, x="Longitude", y="Latitude", color="Price",
                title="Longitude & Latitude vs Price",
                color_continuous_scale="Spectral_r"
            )
            fig_latlon.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_latlon, use_container_width=True)


# ===== TAB 3 — Model Comparison ==========================================
with tab3:
    st.subheader("🤖 Model Performance Comparison")
    
    if metrics:
        # Table
        mod_dict = metrics["models"]
        comp_df = pd.DataFrame.from_dict(mod_dict, orient="index")
        comp_df = comp_df[["r2", "rmse", "mae", "cv_r2_mean", "cv_r2_std"]].reset_index()
        comp_df.rename(columns={
            "index": "Model", "r2": "R²", "rmse": "RMSE", "mae": "MAE", 
            "cv_r2_mean": "CV R² Mean", "cv_r2_std": "CV R² Std"
        }, inplace=True)
        
        st.dataframe(comp_df.style.highlight_max(subset=["R²", "CV R² Mean"], color="lightgreen")
                                   .highlight_min(subset=["RMSE", "MAE", "CV R² Std"], color="lightgreen")
                                   .format({c: "{:.4f}" for c in comp_df.columns if c != "Model"}), 
                     use_container_width=True, hide_index=True)
        
        # Grouped Bar Chart
        melted = comp_df.melt(id_vars="Model", value_vars=["RMSE", "R²"], var_name="Metric", value_name="Score")
        fig_bar = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group", title="RMSE & R² Comparison")
        fig_bar.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Row for scatter and residuals
        r1, r2 = st.columns(2)
        
        actual = metrics.get("best_actual_sample", [])
        pred = metrics.get("best_pred_sample", [])
        
        if actual and pred:
            with r1:
                # Actual vs Predicted
                fig_act_pred = px.scatter(x=actual, y=pred, labels={'x': "Actual Price", 'y': "Predicted Price"}, 
                                          title=f"Actual vs Predicted ({metrics['best_model']})", opacity=0.6)
                fig_act_pred.add_shape(type="line", x0=min(actual), y0=min(actual), x1=max(actual), y1=max(actual),
                                       line=dict(color="red", dash="dash"))
                fig_act_pred.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_act_pred, use_container_width=True)
                
            with r2:
                # Residuals
                residuals = np.array(actual) - np.array(pred)
                fig_res = px.scatter(x=pred, y=residuals, labels={'x': "Predicted Price", 'y': "Residuals"},
                                     title="Residual Plot", opacity=0.6)
                fig_res.add_shape(type="line", x0=min(pred), y0=0, x1=max(pred), y1=0,
                                  line=dict(color="red", dash="dash"))
                fig_res.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_res, use_container_width=True)
                
        # Learning Curves
        st.subheader(f"📈 Learning Curves ({metrics['best_model']})")
        with st.spinner("Computing learning curves..."):
            train_sizes, train_errors, test_errors = compute_learning_curve(metrics["best_model"])
            if train_sizes is not None:
                fig_lc = go.Figure()
                fig_lc.add_trace(go.Scatter(x=train_sizes, y=train_errors, mode='lines+markers', name='Training Error'))
                fig_lc.add_trace(go.Scatter(x=train_sizes, y=test_errors, mode='lines+markers', name='Validation Error'))
                fig_lc.update_layout(
                    title="Learning Curve (MSE)",
                    xaxis_title="Training Set Size",
                    yaxis_title="Negative Mean Squared Error (closer to 0 is better)",
                    template="plotly_white",
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_lc, use_container_width=True)


# ===== TAB 4 — Feature Importance ========================================
with tab4:
    st.subheader("🔍 Feature Importance")
    
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
            fig_fi.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_fi, use_container_width=True)
            
            st.subheader("💡 What do these features mean?")
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
                
                st.markdown(f"""
                <div class="feat-card">
                    <h4>{f_name} <span style="font-weight:normal; font-size:0.9em; color:#666;">({f_imp*100:.1f}% importance)</span></h4>
                    <p>{expl}</p>
                    <p><b>Correlation with Price:</b> {f_corr:.3f} 
                    ({ 'Positive (Higher values = higher prices)' if f_corr > 0 else 'Negative (Higher values = lower prices)' })</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No feature importance available for the selected model.")
