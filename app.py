# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Import our custom MLOps data loader
from src.data_loader import WWTPDataLoader

# --- Page Configuration ---
st.set_page_config(page_title="WWTP Anomaly Radar", layout="wide")
st.title("🌊 Wastewater SCADA Anomaly Radar")
st.markdown("""
Welcome to the interactive anomaly detection dashboard. This tool uses an **Unsupervised Isolation Forest** to monitor 16 interacting cyber-physical sensors and flag severe operational shock loads.
""")

# --- Data Loading (Cached for speed) ---
@st.cache_data
def load_and_prep_data():
    loader = WWTPDataLoader(config_path="config/config.yaml")
    df = loader.load_raw_data()
    return df.dropna().copy()

try:
    df_clean = load_and_prep_data()
except Exception as e:
    st.error(f"Error loading data: {e}. Please check your config.yaml and data folder.")
    st.stop()

# --- Sidebar UI ---
st.sidebar.header("⚙️ Model Parameters")
st.sidebar.markdown("Adjust the algorithm's sensitivity:")
# Allow the user to dynamically change the anomaly percentage!
contamination = st.sidebar.slider("Expected Anomaly Rate", min_value=0.01, max_value=0.10, value=0.05, step=0.01)

st.sidebar.header("📊 Sensor Selection")
numeric_cols = df_clean.select_dtypes(include='number').columns.tolist()
# Let the user choose which sensor to look at
feature_to_plot = st.sidebar.selectbox("Select Sensor to Visualize", numeric_cols, index=numeric_cols.index('Average Inflow') if 'Average Inflow' in numeric_cols else 0)

# --- Model Training (Live!) ---
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_clean[numeric_cols])

# Train the model with the user's selected contamination rate
iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
df_clean['Anomaly_Label'] = iso_forest.fit_predict(scaled_data)

anomalies = df_clean[df_clean['Anomaly_Label'] == -1]

# --- Dashboard Metrics ---
st.subheader("Operational Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Days Analyzed", len(df_clean))
col2.metric("Anomalies Detected", len(anomalies))
col3.metric("Current Sensor", feature_to_plot)

# --- Plotting ---
st.subheader(f"Live Sensor Feed: {feature_to_plot}")
fig, ax = plt.subplots(figsize=(14, 5))

# Plot normal data and anomalies
ax.plot(df_clean.index, df_clean[feature_to_plot], color='#1f77b4', linewidth=1, label='Normal Operation', alpha=0.7)
ax.scatter(anomalies.index, anomalies[feature_to_plot], color='red', label='Detected Anomaly', zorder=5)

ax.set_ylabel(feature_to_plot, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.5)

# Render the plot in Streamlit
st.pyplot(fig)