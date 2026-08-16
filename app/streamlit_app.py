import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Energy Demand Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Load model (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

model = load_model()

# ---------------------------------------------------------------------------
# Light styling (the .streamlit/config.toml sets the base theme so native
# widgets — sidebar, dataframe, charts — stay consistent; this CSS only
# adds a few extra visual touches on top).
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .section-card {
        background-color: #ffffff;
        border-radius: 14px;
        padding: 22px 26px;
        border: 1px solid #e2e8f0;
        margin-bottom: 18px;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 15px 20px;
        border: 1px solid #e2e8f0;
    }
    div.stButton > button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6em 1.4em;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Field definitions
# ---------------------------------------------------------------------------
field_specs = [
    ("UJVNL Generation", 0.0, 500.0, 1.0, "Hydro generation from UJVNL."),
    ("Central Share & VP Share", 0.0, 300.0, 1.0, "Power allocated from central/VP share."),
    ("Energy Availability in UK", 0.0, 500.0, 1.0, "Total energy available within Uttarakhand."),
    ("Energy Surplus/Shortage", -100.0, 100.0, 0.5, "Positive = surplus, negative = shortage."),
    ("IEX/PXIL Drawl", 0.0, 200.0, 1.0, "Power drawn from exchange markets (IEX/PXIL)."),
    ("Banking Power to/from Other States", -50.0, 50.0, 0.5, "Positive = banked out, negative = drawn in."),
    ("Planned/Emergency Rostering", 0.0, 50.0, 0.5, "Load-shedding / rostering in effect."),
    ("Energy Overdrawl/Underdrawl", -50.0, 50.0, 0.5, "Deviation from scheduled drawl."),
    ("U.I. Rate (Rs.)", 0.0, 20.0, 0.1, "Unscheduled Interchange rate."),
    ("Appx. Amount (Crores)", -50.0, 100.0, 0.5, "Approximate financial amount involved."),
]

# ---------------------------------------------------------------------------
# Session state init (runs once)
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

for i, (label, lo, hi, step, help_text) in enumerate(field_specs):
    if f"feature_{i}" not in st.session_state:
        st.session_state[f"feature_{i}"] = (lo + hi) / 2

# Handle a pending reset BEFORE the number_input widgets below are created.
# (Streamlit blocks writing to a widget's session_state key after that
# widget has already been instantiated in the same run, so this has to
# happen here, not down near the button.)
if st.session_state.get("do_reset"):
    for i, (label, lo, hi, step, help_text) in enumerate(field_specs):
        st.session_state[f"feature_{i}"] = (lo + hi) / 2
    st.session_state.do_reset = False

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    if st.button("🗑️ Clear prediction history"):
        st.session_state.history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("⚡ Energy Demand Predictor")
st.caption("Enter today's grid indicators to forecast gross energy demand.")

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.container():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📥 Grid Indicators")

    inputs = []
    cols = st.columns(2)
    for i, (label, lo, hi, step, help_text) in enumerate(field_specs):
        with cols[i % 2]:
            # No `value=` here on purpose — the widget reads its value from
            # st.session_state[key], which is what lets presets update it.
            val = st.number_input(
                label,
                min_value=lo,
                max_value=hi,
                step=step,
                help=help_text,
                key=f"feature_{i}",
            )
            inputs.append(val)

    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        predict_clicked = st.button("🔮 Predict Demand", use_container_width=True)
    with c2:
        reset_clicked = st.button("↺ Reset to midpoint values", use_container_width=True)

if reset_clicked:
    st.session_state.do_reset = True
    st.rerun()

# ---------------------------------------------------------------------------
# Prediction + gauge visualization
# ---------------------------------------------------------------------------
if predict_clicked:
    try:
        with st.spinner("Running model..."):
            features = np.array(inputs).reshape(1, -1)
            prediction = float(model.predict(features)[0])

        st.session_state.history.append(
            {"time": datetime.now().strftime("%H:%M:%S"), "prediction": round(prediction, 2)}
        )

        st.success("Prediction complete")

        col_a, col_b = st.columns([1, 1.3])

        with col_a:
            st.metric("Predicted Gross Energy Demand", f"{prediction:,.2f}")

        with col_b:
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prediction,
                    title={"text": "Demand level"},
                    gauge={
                        "axis": {"range": [None, max(300, prediction * 1.3)]},
                        "bar": {"color": "#2563eb"},
                        "steps": [
                            {"range": [0, max(prediction * 0.5, 1)], "color": "#dbeafe"},
                            {"range": [max(prediction * 0.5, 1), max(prediction, 1)], "color": "#93c5fd"},
                        ],
                    },
                )
            )
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=40, b=10),
                paper_bgcolor="#ffffff",
                font_color="#0f172a",
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------------------------
# Prediction history
# ---------------------------------------------------------------------------
if st.session_state.history:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🕓 Prediction history (this session)")
    hist_df = pd.DataFrame(st.session_state.history[::-1])
    st.dataframe(hist_df, use_container_width=True, hide_index=True)
    st.line_chart(hist_df.set_index("time")["prediction"])
    st.markdown("</div>", unsafe_allow_html=True)
