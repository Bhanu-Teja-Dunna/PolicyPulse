# ============================================================
#  GMP OMNISCIENT RESEARCH TERMINAL (vFINAL - STABLE)
#  Features: Robust Data Patcher, Glass UI, Stress Testing
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# 1. GLASSMORPHISM UI CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="GMP Research Terminal",
    layout="wide",
    page_icon="🧿",
    initial_sidebar_state="expanded"
)

# Deep Space Glass CSS
st.markdown("""
<style>
    .stApp { background: radial-gradient(circle at center, #14171f 0%, #0a0c10 100%); }
    .css-1r6slb0, .css-12oz5g7, .stExpander, div[data-testid="stMetric"] {
        background-color: rgba(20, 25, 35, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    h1, h2, h3, h4 { font-family: 'Inter', sans-serif; color: #F0F0F0; font-weight: 300; }
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 12, 16, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 2. DATA & MODEL ENGINE (AGGRESSIVE HEALING)
# ============================================================

def safe_read_csv(path):
    """Reads CSV without crashing, returns empty DF on failure."""
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except:
        return pd.DataFrame()

@st.cache_data
def load_all_data():
    """
    Loads data and FORCEFULLY ensures all required columns exist.
    This prevents KeyErrors even if the CSVs are partial or corrupt.
    """
    data = {}
    
    # 1. Load Raw
    data['us_macro'] = safe_read_csv("us_monetary_data_phase1.csv")
    data['us_event'] = safe_read_csv("phase2_event_study_output.csv")
    data['us_nlp'] = safe_read_csv("phase3_fomc_nlp_features_2010plus.csv")
    data['uk_model'] = safe_read_csv("uk_phase4_model_data.csv")
    
    # 2. DEFINE REQUIRED COLUMNS
    req_uk = ["Rate_Change_bp", "is_hike", "is_cut", "FTSE_CAR", "FTSE_Day0", "BoE_Rate_Current", "UK10Y_Current"]
    req_us = ["sentiment_score", "count_inflation", "count_employment", "count_risk", "FEDFUNDS", "DGS10", "SP500_Day0"]

    # 3. PATCH UK DATA
    if data['uk_model'].empty or len(data['uk_model'].columns) < 2:
        # Create Dummy if missing/empty
        dates = pd.date_range("2010-01-01", "2023-01-01", freq="Q")
        data['uk_model'] = pd.DataFrame(0.0, index=dates, columns=req_uk)
    else:
        # Fill missing cols with 0
        for c in req_uk:
            if c not in data['uk_model'].columns:
                data['uk_model'][c] = 0.0
    
    # Ensure float types
    for c in req_uk:
        data['uk_model'][c] = pd.to_numeric(data['uk_model'][c], errors='coerce').fillna(0.0)

    # 4. PATCH USA DATA
    # Create US Full
    if not data['us_event'].empty and not data['us_nlp'].empty:
        combined = data['us_event'].join(data['us_nlp'], how="inner")
        if not data['us_macro'].empty:
            macro_aligned = data['us_macro'].reindex(combined.index, method="ffill")
            data['us_full'] = combined.join(macro_aligned, how="left")
        else:
            data['us_full'] = combined
    else:
        # Create Dummy
        dates = pd.date_range("2010-01-01", "2023-01-01", freq="M")
        data['us_full'] = pd.DataFrame(0.0, index=dates, columns=req_us)

    # Fill missing US cols
    for c in req_us:
        if c not in data['us_full'].columns:
            data['us_full'][c] = 0.0
            
    # Clean up NaNs
    data['us_full'] = data['us_full'].fillna(0.0)
    data['uk_model'] = data['uk_model'].fillna(0.0)

    return data

DATA = load_all_data()

@st.cache_resource
def get_model(region="USA"):
    """Trains Random Forest on the fly using the Patched Data."""
    if region == "USA":
        df = DATA['us_full']
        features = ["sentiment_score", "count_inflation", "count_employment", "count_risk", "FEDFUNDS", "DGS10"]
        target = "SP500_Day0"
    else:
        df = DATA['uk_model']
        features = ["Rate_Change_bp", "is_hike", "is_cut", "FTSE_CAR"]
        target = "FTSE_Day0"
        
    # Validation
    if df.empty or len(df) < 5: 
        return None, None, features 
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    y = df[target]
    
    model = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
    model.fit(X, y)
    
    return model, scaler, features

# ============================================================
# 3. PLOTTING FUNCTIONS
# ============================================================

def apply_glass(fig):
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#E0E0E0", family="Inter"),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', showline=True, linecolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', showline=True, linecolor='rgba(255,255,255,0.1)'),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig

# ============================================================
# 4. SIDEBAR CONTROL ROOM
# ============================================================

st.sidebar.title("🎛️ Control Room")
st.sidebar.markdown("---")

# REGION SELECTOR
region_mode = st.sidebar.radio("📡 Active Market Uplink", ["🇺🇸 USA (FOMC)", "🇬🇧 UK (BoE MPC)"])

# Map pretty label -> model region key
region_key = "USA" if region_mode.startswith("🇺🇸") else "UK"

st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Stress Test Lab")

overrides = {}

if region_mode.startswith("🇺🇸"):
    # USA CONTROLS
    with st.sidebar.expander("📉 CRASH & RISK", expanded=True):
        s_crash = st.slider("Financial Stress", 0, 10, 2)
    with st.sidebar.expander("💸 INFLATION & PRICES", expanded=True):
        s_inf = st.slider("Inflation Intensity", 0, 25, 5)
    with st.sidebar.expander("🔨 LABOR MARKET", expanded=True):
        s_emp = st.slider("Unemployment Intensity", 0, 25, 5)
    with st.sidebar.expander("🏦 MONETARY POLICY", expanded=True):
        s_rate = st.slider("Fed Funds Rate (%)", 0.0, 12.0, 5.25)
        s_yield = st.slider("10Y Treasury Yield (%)", 0.0, 12.0, 4.50)
    
    base_sent = 0.5 - (s_crash * 0.15) 
    base_risk = 5 + (s_crash * 3)
    
    overrides = {
        "sentiment_score": max(-1.0, min(1.0, base_sent)),
        "count_risk": base_risk,
        "count_inflation": s_inf,
        "count_employment": s_emp,
        "FEDFUNDS": s_rate,
        "DGS10": s_yield
    }

else:
    # UK CONTROLS
    with st.sidebar.expander("🇬🇧 BOE POLICY", expanded=True):
        s_bp = st.slider("Rate Shock (bps)", -100, 100, 0)
        s_car = st.slider("FTSE Trend (CAR %)", -5.0, 5.0, 0.0)
        
    overrides = {
        "Rate_Change_bp": s_bp,
        "is_hike": 1 if s_bp > 0 else 0,
        "is_cut": 1 if s_bp < 0 else 0,
        "FTSE_CAR": s_car
    }

# ============================================================
# 5. MAIN DASHBOARD CONTENT
# ============================================================

st.title(f"🔭 {region_mode} // OMNISCIENT VIEW")

tabs = st.tabs(["📈 Phase 1: Macro", "💥 Phase 2: Events", "🗣️ Phase 3: NLP", "🧠 Phase 4: Diagnostics", "🔮 Phase 5: Scenario Sim", "🧊 3D Lab"])

# TAB 1: MACRO
with tabs[0]:
    if region_mode.startswith("🇺🇸"):
        df_macro = DATA['us_macro']
    else:
        df_macro = DATA['uk_model']
        
    # Get numeric cols only
    cols = df_macro.select_dtypes(include=np.number).columns.tolist()
    if cols:
        selected = st.multiselect("Select Indicators", cols, default=cols[:2])
        if selected:
            fig = px.line(df_macro, y=selected, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(apply_glass(fig), use_container_width=True)
    else:
        st.info("No numeric macro data available.")

# TAB 2: EVENTS
with tabs[1]:
    if region_mode.startswith("🇺🇸"):
        df_ev = DATA['us_event']
        metric = "SP500_Day0" if "SP500_Day0" in df_ev.columns else df_ev.columns[0]
    else:
        df_ev = DATA['uk_model']
        metric = "FTSE_Day0"
    
    if not df_ev.empty and metric in df_ev.columns:
        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(df_ev, x=metric, nbins=40, title="Distribution", color_discrete_sequence=['#00E676'])
            st.plotly_chart(apply_glass(fig_hist), use_container_width=True)
        with c2:
            fig_box = px.box(df_ev, y=metric, title="Regime Boxplot", color_discrete_sequence=['#2979FF'])
            st.plotly_chart(apply_glass(fig_box), use_container_width=True)

# TAB 3: NLP
with tabs[2]:
    if region_mode.startswith("🇺🇸"):
        df_nlp = DATA['us_nlp']
        kw_cols = ["count_inflation", "count_employment", "count_risk"]
        avail = [c for c in kw_cols if c in df_nlp.columns]
        if avail:
            fig_stream = px.area(df_nlp, y=avail, title="Semantic Density", color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(apply_glass(fig_stream), use_container_width=True)
    else:
        st.info("NLP Lab unavailable for UK Model (Switch to USA).")

# TAB 4: DIAGNOSTICS
with tabs[3]:
    model, scaler, feats = get_model(region_key)
    if model:
        imp = pd.DataFrame({'Feature': feats, 'Importance': model.feature_importances_}).sort_values("Importance")
        fig_imp = px.bar(imp, x="Importance", y="Feature", orientation='h', color="Importance", color_continuous_scale="Viridis")
        st.plotly_chart(apply_glass(fig_imp), use_container_width=True)

# TAB 5: SCENARIO LAB
with tabs[4]:
    st.subheader("⚡ Monte Carlo & Stress Test")
    
    model, scaler, feats = get_model(region_key)
    
    if model:
        # Prepare Base
        if region_mode.startswith("🇺🇸"):
            base = DATA['us_full'].iloc[-1].copy()
        else:
            base = DATA['uk_model'].iloc[-1].copy()
            
        # Apply Overrides
        for k, v in overrides.items():
            base[k] = v
            
        # Predict
        # Use pandas DataFrame to keep feature names valid for scaler
        X_in = pd.DataFrame([base[feats].values], columns=feats)
        X_scaled = scaler.transform(X_in)
        pred = model.predict(X_scaled)[0]
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric(f"Predicted Return", f"{pred:+.2f}%")
        with c2:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = pred,
                title = {'text': "Market Reaction"},
                gauge = {
                    'axis': {'range': [-5, 5]},
                    'bar': {'color': "#00E676" if pred > 0 else "#FF5252"},
                    'bgcolor': "rgba(0,0,0,0)"
                }
            ))
            st.plotly_chart(apply_glass(fig_gauge), use_container_width=True)

# TAB 6: 3D LAB
with tabs[5]:
    st.subheader("🧊 Multidimensional Lab")
    if region_mode.startswith("🇺🇸"):
        df_3d = DATA['us_full']
        default_x, default_y, default_z = "FEDFUNDS", "sentiment_score", "SP500_Day0"
    else:
        df_3d = DATA['uk_model']
        default_x, default_y, default_z = "Rate_Change_bp", "FTSE_CAR", "FTSE_Day0"
        
    cols_3d = df_3d.select_dtypes(include=np.number).columns.tolist()
    if len(cols_3d) >= 3:
        c1, c2, c3 = st.columns(3)
        x_ax = c1.selectbox("X-Axis", cols_3d, index=cols_3d.index(default_x) if default_x in cols_3d else 0)
        y_ax = c2.selectbox("Y-Axis", cols_3d, index=cols_3d.index(default_y) if default_y in cols_3d else 1)
        z_ax = c3.selectbox("Z-Axis", cols_3d, index=cols_3d.index(default_z) if default_z in cols_3d else 2)
        
        fig_3d = px.scatter_3d(df_3d, x=x_ax, y=y_ax, z=z_ax, color=z_ax, opacity=0.7, color_continuous_scale="Turbo")
        fig_3d.update_layout(scene=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=0,r=0,b=0,t=0), height=600)
        st.plotly_chart(apply_glass(fig_3d), use_container_width=True)

st.markdown("---")
st.caption("GMP OMNISCIENT TERMINAL | vFINAL-ROBUST")
