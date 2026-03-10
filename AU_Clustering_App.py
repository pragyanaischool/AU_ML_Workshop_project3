import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

# Load the saved pipeline (scaler + model)
@st.cache_resource
def load_pipeline():
    return joblib.load('model_pipeline.pkl')

pipeline = load_pipeline()

st.title(" Customer Segmentation AI")
st.markdown("Use this tool to segment customers based on Recency, Frequency, and Monetary (RFM) values.")

# Sidebar for individual testing
with st.sidebar:
    st.header("Individual Prediction")
    recency = st.number_input("Recency (days)", 0, 365, 30)
    frequency = st.number_input("Frequency (transactions)", 1, 1000, 5)
    monetary = st.number_input("Monetary ($)", 0.0, 50000.0, 500.0)
    
    if st.button("Predict Segment"):
        input_data = pd.DataFrame([[recency, frequency, monetary]], 
                                  columns=['Recency', 'Frequency', 'Monetary'])
        input_log = np.log1p(input_data)
        cluster = pipeline.predict(input_log)
        st.subheader(f"Assigned Cluster: {cluster[0]}")

# Bulk testing section
st.divider()
st.subheader("Bulk Testing & Analysis")
uploaded_file = st.file_uploader("Upload a CSV file for batch segmentation", type=['csv'])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    
    # Preprocessing
    features = test_df[['Recency', 'Frequency', 'Monetary']]
    features_log = np.log1p(features)
    
    # Prediction
    test_df['Cluster'] = pipeline.predict(features_log)
    
    # Display results
    st.write("### Processed Data")
    st.dataframe(test_df.head())
    
    # Visualization
    st.write("### Cluster Visualization")
    fig = px.scatter_3d(test_df, x='Recency', y='Frequency', z='Monetary', 
                        color='Cluster', title="Customer Segments in 3D Space")
    st.plotly_chart(fig)
    
    # Download results
    csv = test_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results", csv, "segmentation_results.csv", "text/csv")
