import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

# Load model
@st.cache_resource
def load_pipeline():
    return joblib.load('model_pipeline.pkl')

pipeline = load_pipeline()

st.title("📊 Customer Segmentation AI")

uploaded_file = st.file_uploader("Upload Raw Transaction CSV", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    
    # Auto-convert Raw Transactions to RFM
    if 'InvoiceNo' in df.columns:
        df['TotalSum'] = df['Quantity'] * df['UnitPrice']
        snapshot = pd.to_datetime(df['InvoiceDate']).max() + pd.Timedelta(days=1)
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (snapshot - pd.to_datetime(x).max()).days,
            'InvoiceNo': 'count',
            'TotalSum': 'sum'
        })
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
    else:
        rfm = df[['Recency', 'Frequency', 'Monetary']]

    # Prediction
    rfm['Cluster'] = pipeline.predict(np.log1p(rfm))
    
    # Analysis
    st.write("### Segmentation Results")
    st.dataframe(rfm.head())
    
    
    fig = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary', color='Cluster')
    st.plotly_chart(fig, use_container_width=True)
    
    st.table(rfm.groupby('Cluster').mean())
