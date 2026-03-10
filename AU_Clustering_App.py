import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Customer Segmentation AI", layout="wide")

@st.cache_resource
def load_pipeline():
    return joblib.load('model_pipeline.pkl')

pipeline = load_pipeline()

st.title("📊 Customer Segmentation AI")

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
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file:
    try:
        # Robust loading
        test_df = None
        for enc in ['utf-8', 'ISO-8859-1', 'cp1252']:
            try:
                uploaded_file.seek(0)
                test_df = pd.read_csv(uploaded_file, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        
        if test_df is not None:
            # --- AUTO-FIX COLUMN NAMES ---
            test_df.columns = test_df.columns.str.strip().str.capitalize()
            # Mapping common variations
            mapping = {'Recency': 'Recency', 'Frequency': 'Frequency', 'Monetary': 'Monetary'}
            test_df = test_df.rename(columns=mapping)
            
            required = ['Recency', 'Frequency', 'Monetary']
            if not all(col in test_df.columns for col in required):
                st.error(f"Missing columns. Found: {list(test_df.columns)}. Need: {required}")
            else:
                # Prediction
                features_log = np.log1p(test_df[required])
                test_df['Cluster'] = pipeline.predict(features_log)
                
                # Visuals
                
                fig = px.scatter_3d(test_df, x='Recency', y='Frequency', z='Monetary', color='Cluster')
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("### Cluster Profiles")
                analysis = test_df.groupby('Cluster')[required].mean()
                st.table(analysis)
                
                # Distribution
                
                fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
                for i, col in enumerate(required):
                    sns.boxplot(x='Cluster', y=col, data=test_df, ax=axes[i])
                st.pyplot(fig2)
                
                csv = test_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results", csv, "results.csv", "text/csv")
                
    except Exception as e:
        st.error(f"An error occurred: {e}")
