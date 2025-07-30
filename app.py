import streamlit as st

st.set_page_config(
    page_title="Stock Market Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Market Predictor")
st.write("Minimal test deployment")

# Add a simple interactive element
user_input = st.text_input("Enter a stock symbol (e.g., AAPL):", "AAPL")
st.write(f"You entered: {user_input}")

# Add a simple plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
data = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'Price': np.random.normal(100, 10, 100).cumsum()
})

# Plot
st.line_chart(data.set_index('Date'))

st.write("If you can see this chart, the basic deployment is working!")
