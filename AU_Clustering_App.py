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
uploaded_file = st.file_uploader("Upload Raw Transaction CSV", type=['csv'])

if uploaded_file:
    try:
        # Load File
        test_df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        
        # Aggregate to RFM
        test_df['TotalSum'] = test_df['Quantity'] * test_df['UnitPrice']
        snapshot = pd.to_datetime(test_df['InvoiceDate']).max() + pd.Timedelta(days=1)
        rfm = test_df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (snapshot - pd.to_datetime(x).max()).days,
            'InvoiceNo': 'count',
            'TotalSum': 'sum'
        })
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        # --- CRITICAL: Data Sanitization ---
        rfm = rfm.replace([np.inf, -np.inf], np.nan)
        rfm = rfm.dropna(subset=['Recency', 'Frequency', 'Monetary'])
        rfm = rfm.fillna(rfm.median())
        
        # Predict
        features_log = np.log1p(rfm)
        rfm['Cluster'] = pipeline.predict(features_log)
        
        # --- ANALYSIS & VISUALIZATION ---
        st.write("### Segmentation Results", rfm.head())
        
        
        fig = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary', color='Cluster')
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("### Cluster Profiles")
        st.table(rfm.groupby('Cluster').mean())
        
        
        fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
        for i, col in enumerate(['Recency', 'Frequency', 'Monetary']):
            sns.boxplot(x='Cluster', y=col, data=rfm, ax=axes[i])
        st.pyplot(fig2)
        
    except Exception as e:
        st.error(f"Error: {e}")
