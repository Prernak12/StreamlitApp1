import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from model import load_and_prepare_data, train_anomaly_detection_model

# Title
st.title('Credit Card Fraud Detection')

# Upload file
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file:
    try:
        # Load and prepare data
        df = load_and_prepare_data(uploaded_file)

        # Train model and predict anomalies
        df, y_pred, y_pred_lof = train_anomaly_detection_model(df)

        # Display results
        st.write(f"Total anomalies detected: {sum(y_pred)}")

        # Visualization
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x='PC1', y='PC2', hue='Anomaly', palette={0: 'blue', 1: 'red'}, alpha=0.5)
        plt.title('Anomaly Detection: PCA Transformed Features')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Anomaly', loc='upper right', labels=['Normal', 'Fraudulent'])
        st.pyplot(plt)

        # Show the DataFrame
        st.write(df)

        # Additional details
        st.write(f"Length of anomaly predictions: {len(y_pred)}")
        st.write(f"Length of transactions: {len(df)}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
