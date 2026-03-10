import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Customer Segmentation AI", layout="wide")

# Load Pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load('model_pipeline.pkl')

pipeline = load_pipeline()

st.title("📊 Customer Segmentation AI")
st.markdown("Use this tool to segment customers based on Recency, Frequency, and Monetary (RFM) values.")

# --- SIDEBAR: Individual Prediction ---
with st.sidebar:
    st.header("Individual Prediction")
    r = st.number_input("Recency (days)", 0, 365, 30)
    f = st.number_input("Frequency (transactions)", 1, 1000, 5)
    m = st.number_input("Monetary ($)", 0.0, 50000.0, 500.0)
    
    if st.button("Predict Segment"):
        data = pd.DataFrame([[r, f, m]], columns=['Recency', 'Frequency', 'Monetary'])
        data_log = np.log1p(data)
        cluster = pipeline.predict(data_log)
        st.subheader(f"Assigned Cluster: {cluster[0]}")

# --- MAIN: Bulk Analysis ---
st.divider()
st.subheader("Bulk Testing & Cluster Profiling")
uploaded_file = st.file_uploader("Upload CSV (Columns: Recency, Frequency, Monetary)", type=['csv'])

if uploaded_file:
    test_df = pd.read_csv(uploaded_file)
    
    # Predict
    features_log = np.log1p(test_df[['Recency', 'Frequency', 'Monetary']])
    test_df['Cluster'] = pipeline.predict(features_log)
    
    # 1. Visualization
    st.write("### 3D Cluster Visualization")
    
    fig = px.scatter_3d(test_df, x='Recency', y='Frequency', z='Monetary', color='Cluster')
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. Statistical Analysis
    st.write("### Cluster Profiles")
    analysis = test_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    st.table(analysis)
    
    # 3. Distribution Charts
    st.write("### Feature Distribution by Cluster")
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, col in enumerate(['Recency', 'Frequency', 'Monetary']):
        sns.boxplot(x='Cluster', y=col, data=test_df, ax=axes[i])
    st.pyplot(fig2)
    
    # 4. Download
    csv = test_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Segmented Results", csv, "results.csv", "text/csv")
