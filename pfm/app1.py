import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import joblib
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load Model and Scaler ---
model = load_model('lstm_sales_model_plus.h5', custom_objects={'mse': MeanSquaredError()})
scaler = joblib.load('scaler.pkl')  # <-- Load pre-trained scaler

# Page configuration


st.set_page_config(page_title="Sales Forecasting Dashboard",  
                   page_icon="🏪",  # Best match for "store"
                   layout="wide")
st.title("📈 Sales Forecasting Dashboard")

# Upload CSV
st.sidebar.header("📤 Upload Test CSV File")
uploaded_file = st.sidebar.file_uploader("Upload your test dataset", type=["csv"])

# If a file is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.write(df.head())

    required_columns = {'date', 'store', 'item'}
    if not required_columns.issubset(df.columns):
        st.error(f"Your file must contain the columns: {required_columns}")
    else:
        # Preprocess
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

        # Generate lag and rolling features (same as training)
        df['lag_1'] = df.groupby(['store', 'item'])['sales'].shift(1)
        df['rolling_mean_7'] = df.groupby(['store', 'item'])['sales'].shift(1).rolling(7).mean()

        # Fill NaN values for the features
        df.fillna(0, inplace=True)

        # Feature selection for model prediction
        features = ['store', 'item', 'year', 'month', 'day', 'dayofweek', 'is_weekend', 'lag_1', 'rolling_mean_7']
        X = df[features]

        # Use loaded scaler instead of fitting again
        X_scaled = scaler.transform(X)

        # Reshape for LSTM: [samples, timesteps, features]
        X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

        # Make predictions (log-transformed output)
        predictions_log = model.predict(X_scaled)

        # Inverse log1p to get real values
        df['predicted_sales'] = np.expm1(predictions_log.flatten())

        # Display only selected columns
        display_columns = ['date', 'store', 'item', 'predicted_sales']
        if 'id' in df.columns:
            display_columns.insert(0, 'id')

        df_display = df[display_columns]

        # Show predictions
        st.subheader("🧾 Prediction Table")
        st.dataframe(df_display)

        # Visualizations
        st.subheader("📊 Visualizations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📈 Line Chart")
            line_fig = px.line(df, x='date', y='predicted_sales', title='Predicted Sales Over Time')
            st.plotly_chart(line_fig, use_container_width=True)

        with col2:
            st.markdown("### 📦 Boxplot")
            box_fig = px.box(df, y='predicted_sales', title='Boxplot of Predicted Sales')
            st.plotly_chart(box_fig, use_container_width=True)

        st.markdown("### 📉 Histogram")
        hist_fig = plt.figure(figsize=(10, 5))
        sns.histplot(df['predicted_sales'], kde=True)
        st.pyplot(hist_fig)

        # Download link
        st.sidebar.subheader("⬇️ Download Predictions")
        csv_data = df_display.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="predicted_sales.csv",
            mime="text/csv"
        )

else:
    st.info("👈 Please upload a test CSV file to get started.")
