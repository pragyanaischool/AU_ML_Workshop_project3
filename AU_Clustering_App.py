import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="Customer Segmentation AI", layout="wide")

# Load the Pipeline (includes Imputer, Scaler, and KMeans)
@st.cache_resource
def load_pipeline():
    return joblib.load('model_pipeline.pkl')

pipeline = load_pipeline()

st.title("📊 Customer Segmentation AI")
st.markdown("Upload your raw transaction data to automatically segment customers based on their RFM behavior.")

# File Uploader
uploaded_file = st.file_uploader("Upload Raw Transaction CSV", type=['csv'])

if uploaded_file:
    try:
        # 1. Load Data
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        
        # 2. Transform Raw Data to RFM (Automated)
        if 'InvoiceNo' in df.columns:
            df['TotalSum'] = df['Quantity'] * df['UnitPrice']
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            snapshot = df['InvoiceDate'].max() + pd.Timedelta(days=1)
            
            rfm = df.groupby('CustomerID').agg({
                'InvoiceDate': lambda x: (snapshot - x.max()).days,
                'InvoiceNo': 'count',
                'TotalSum': 'sum'
            })
            rfm.columns = ['Recency', 'Frequency', 'Monetary']
        else:
            rfm = df[['Recency', 'Frequency', 'Monetary']]

        # 3. Predict (Pipeline automatically handles imputation and scaling)
        rfm_log = np.log1p(rfm)
        rfm['Cluster'] = pipeline.predict(rfm_log)
        
        # 4. Display Results
        st.write("### Segmentation Results (First 5 Customers)", rfm.head())
        
        # 5. Visualization
        
        fig = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary', color='Cluster')
        st.plotly_chart(fig, use_container_width=True)
        
        # 6. Statistical Profiling
        st.write("### Cluster Profiles (Average Values)")
        st.table(rfm.groupby('Cluster').mean())
        
        # 7. Distribution Analysis
        
        fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
        for i, col in enumerate(['Recency', 'Frequency', 'Monetary']):
            sns.boxplot(x='Cluster', y=col, data=rfm, ax=axes[i])
        st.pyplot(fig2)
        
        # 8. Download
        csv = rfm.to_csv(index=True).encode('utf-8')
        st.download_button("Download Segmented Results", csv, "results.csv", "text/csv")
            
    except Exception as e:
        st.error(f"Error processing file: {e}")
