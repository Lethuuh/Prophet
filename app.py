import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# Load data
customer_features = pd.read_csv("customer_features.csv")
forecast = pd.read_csv("forecast.csv")  # Prophet output
daily_sales = pd.read_csv("daily_sales.csv")

# App title
st.title("🧠 Retail Intelligence Dashboard")

# Tabs
tab1, tab2, tab3 = st.tabs(["Customer Classification", "Revenue Forecast", "Raw Data"])

# Tab 1: Classification
with tab1:
    st.subheader("🔍 High Spender Classification")
    st.write("This tab shows customer behavior and predicted high spenders.")

    fig1 = px.scatter(customer_features, x='TotalQuantity', y='TotalSpent',
                      color='HighSpender', title="Customer Spending Behavior")
    st.plotly_chart(fig1)

    st.dataframe(customer_features[['CustomerID', 'Age', 'Country', 'TotalSpent', 'HighSpender']])

# Tab 2: Forecast
with tab2:
    st.subheader("📈 Revenue Forecast (Prophet)")
    st.write("Forecasting daily revenue using Facebook Prophet.")

    fig2 = px.line(forecast, x='ds', y='yhat', title="30-Day Revenue Forecast")
    st.plotly_chart(fig2)

    st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# Tab 3: Raw Data
with tab3:
    st.subheader("📅 Daily Sales Data")
    st.line_chart(daily_sales.set_index('ds')['y'])
    st.dataframe(daily_sales)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Nonjabulo using Streamlit + Prophet")