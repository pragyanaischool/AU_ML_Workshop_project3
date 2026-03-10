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

uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

if uploaded_file:
    try:
        # 1. Robust Loading with Encoding Fix
        test_df = None
        for enc in ['utf-8', 'ISO-8859-1', 'cp1252']:
            try:
                uploaded_file.seek(0)
                test_df = pd.read_csv(uploaded_file, encoding=enc)
                break
            except UnicodeDecodeError: continue
        
        if test_df is not None:
            # --- THE CRITICAL FIX FOR KAGGLE DATA ---
            # Remove rows where CustomerID is missing immediately
            if 'CustomerID' in test_df.columns:
                test_df = test_df.dropna(subset=['CustomerID'])
            
            # Remove negative quantities (returns) which cause log(negative) = NaN
            if 'Quantity' in test_df.columns:
                test_df = test_df[test_df['Quantity'] > 0]

            # 2. Transform Raw Data to RFM
            if 'InvoiceNo' in test_df.columns:
                test_df['TotalSum'] = test_df['Quantity'] * test_df['UnitPrice']
                test_df['InvoiceDate'] = pd.to_datetime(test_df['InvoiceDate'])
                snapshot = test_df['InvoiceDate'].max() + pd.Timedelta(days=1)
                
                rfm = test_df.groupby('CustomerID').agg({
                    'InvoiceDate': lambda x: (snapshot - x.max()).days,
                    'InvoiceNo': 'count',
                    'TotalSum': 'sum'
                })
                rfm.columns = ['Recency', 'Frequency', 'Monetary']
            else:
                rfm = test_df[['Recency', 'Frequency', 'Monetary']]

            # 3. FINAL AGGRESSIVE CLEANING
            # This ensures absolutely no NaNs or Infs reach the model
            rfm = rfm.replace([np.inf, -np.inf], np.nan).dropna()
            
            # 4. Predict
            # np.log1p handles 0, but we ensure no negative values exist
            features_log = np.log1p(rfm.clip(lower=0)) 
            rfm['Cluster'] = pipeline.predict(features_log)
            
            # 5. UI and Visualization
            st.success(f"Successfully segmented {len(rfm)} customers!")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                
                fig = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary', color='Cluster', title="Customer Clusters")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("### Cluster Averages")
                st.table(rfm.groupby('Cluster').mean())

            # Download
            csv = rfm.to_csv(index=True).encode('utf-8')
            st.download_button("Download Results", csv, "segmented_customers.csv", "text/csv")
            
    except Exception as e:
        st.error(f"Error: {e}")
