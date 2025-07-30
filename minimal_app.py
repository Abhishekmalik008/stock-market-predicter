import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta

def main():
    st.set_page_config(
        page_title="Stock Market Predictor",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Stock Market Predictor")
    st.write("Minimal version for deployment testing")
    
    # Stock selection
    stock_symbol = st.text_input("Enter stock symbol (e.g., AAPL, MSFT, TCS.NS):", "TCS.NS")
    
    # Date range selection
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    try:
        # Fetch stock data
        df = yf.download(stock_symbol, start=start_date, end=end_date)
        
        if df.empty:
            st.error("No data found for the given stock symbol. Please try another symbol.")
            return
            
        # Display basic stock info
        st.subheader(f"Stock Data for {stock_symbol}")
        st.dataframe(df.tail())
        
        # Plot closing price
        fig = px.line(df, x=df.index, y='Close', title=f"{stock_symbol} Closing Price")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
