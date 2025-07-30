import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

def main():
    st.set_page_config(
        page_title="Stock Market Predictor",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Stock Market Predictor (Minimal Version)")
    
    # Sidebar for user input
    st.sidebar.header('Stock Data')
    ticker = st.sidebar.text_input('Enter Stock Ticker (e.g., AAPL, MSFT, GOOGL)', 'AAPL')
    start_date = st.sidebar.date_input('Start Date', datetime.now() - timedelta(days=365))
    end_date = st.sidebar.date_input('End Date', datetime.now())
    
    # Fetch stock data
    @st.cache_data(ttl=3600)  # Cache data for 1 hour
    def load_data(ticker, start, end):
        try:
            data = yf.download(ticker, start=start, end=end)
            return data
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None
    
    data = load_data(ticker, start_date, end_date)
    
    if data is not None and not data.empty:
        # Display basic stock info
        st.subheader(f"{ticker} Stock Price")
        
        # Create interactive candlestick chart
        fig = go.Figure(data=[
            go.Candlestick(x=data.index,
                         open=data['Open'],
                         high=data['High'],
                         low=data['Low'],
                         close=data['Close'],
                         name='OHLC')
        ])
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_dark',
            height=600,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show recent data
        st.subheader("Recent Data")
        st.dataframe(data.tail().style.format({
            'Open': '{:.2f}',
            'High': '{:.2f}',
            'Low': '{:.2f}',
            'Close': '{:.2f}',
            'Adj Close': '{:.2f}',
            'Volume': '{:,}'
        }))
        
    else:
        st.warning("No data available for the selected ticker and date range.")

if __name__ == "__main__":
    main()
