import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Customer Segmentation AI", layout="wide")

# Load Pipeline (Ensure model_pipeline.pkl is in the same folder)
@st.cache_resource
def load_pipeline():
    return joblib.load('model_pipeline.pkl')

pipeline = load_pipeline()

st.title("📊 Customer Segmentation AI")

# --- BULK ANALYSIS ---
st.subheader("Upload Transactional CSV for Segmentation")
uploaded_file = st.file_uploader("Upload Raw Transaction CSV", type=['csv'])

if uploaded_file:
    try:
        # 1. Robust Loading
        test_df = None
        for enc in ['utf-8', 'ISO-8859-1', 'cp1252']:
            try:
                uploaded_file.seek(0)
                test_df = pd.read_csv(uploaded_file, encoding=enc)
                break
            except UnicodeDecodeError: continue
        
        if test_df is not None:
            # 2. Convert Raw Transactions to RFM
            if 'InvoiceNo' in test_df.columns:
                test_df['InvoiceDate'] = pd.to_datetime(test_df['InvoiceDate'])
                test_df['TotalSum'] = test_df['Quantity'] * test_df['UnitPrice']
                snapshot = test_df['InvoiceDate'].max() + pd.Timedelta(days=1)
                
                rfm = test_df.groupby('CustomerID').agg({
                    'InvoiceDate': lambda x: (snapshot - x.max()).days,
                    'InvoiceNo': 'count',
                    'TotalSum': 'sum'
                })
                rfm.columns = ['Recency', 'Frequency', 'Monetary']
            else:
                rfm = test_df[['Recency', 'Frequency', 'Monetary']]

            # 3. CRITICAL: Clean Data to prevent ValueError (NaNs, Infs)
            rfm = rfm.replace([np.inf, -np.inf], np.nan).dropna()
            
            # 4. Predict
            features_log = np.log1p(rfm)
            rfm['Cluster'] = pipeline.predict(features_log)
            
            # 5. Output and Visualize
            st.write("### Segmentation Results (Preview)", rfm.head())
            
            
            fig = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary', color='Cluster')
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("### Cluster Profiles")
            st.table(rfm.groupby('Cluster').mean())
            
            
            fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
            for i, col in enumerate(['Recency', 'Frequency', 'Monetary']):
                sns.boxplot(x='Cluster', y=col, data=rfm, ax=axes[i])
            st.pyplot(fig2)
            
            # Download
            csv = rfm.to_csv(index=True).encode('utf-8')
            st.download_button("Download Results", csv, "results.csv", "text/csv")
            
    except Exception as e:
        st.error(f"Error processing file: {e}")
