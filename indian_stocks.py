"""
Indian Stock Market Symbols and Utilities
"""

# Popular Indian stocks with NSE symbols
INDIAN_STOCKS = {
    # IT Sector
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys Limited",
    "HCLTECH.NS": "HCL Technologies",
    "WIPRO.NS": "Wipro Limited",
    "TECHM.NS": "Tech Mahindra",
    
    # Banking & Finance
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "State Bank of India",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "AXISBANK.NS": "Axis Bank",
    
    # FMCG
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC Limited",
    "NESTLEIND.NS": "Nestle India",
    "BRITANNIA.NS": "Britannia Industries",
    
    # Automotive
    "MARUTI.NS": "Maruti Suzuki",
    "TATAMOTORS.NS": "Tata Motors",
    "M&M.NS": "Mahindra & Mahindra",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    
    # Pharma
    "SUNPHARMA.NS": "Sun Pharmaceutical",
    "DRREDDY.NS": "Dr. Reddy's Laboratories",
    "CIPLA.NS": "Cipla Limited",
    
    # Energy & Oil
    "RELIANCE.NS": "Reliance Industries",
    "ONGC.NS": "Oil & Natural Gas Corporation",
    "NTPC.NS": "NTPC Limited",
    "POWERGRID.NS": "Power Grid Corporation",
    
    # Metals & Mining
    "TATASTEEL.NS": "Tata Steel",
    "HINDALCO.NS": "Hindalco Industries",
    "JSWSTEEL.NS": "JSW Steel",
    
    # Telecom
    "BHARTIARTL.NS": "Bharti Airtel",
    "IDEA.NS": "Vodafone Idea",
    
    # Consumer Goods
    "ASIANPAINT.NS": "Asian Paints",
    "ULTRACEMCO.NS": "UltraTech Cement",
    
    # Indices
    "^NSEI": "Nifty 50",
    "^BSESN": "BSE Sensex"
}

def get_indian_stock_suggestions():
    """Get list of popular Indian stocks for dropdown"""
    return list(INDIAN_STOCKS.keys())

def get_stock_name(symbol):
    """Get full company name from symbol"""
    return INDIAN_STOCKS.get(symbol, symbol)

def is_indian_stock(symbol):
    """Check if symbol is an Indian stock"""
    return symbol.endswith('.NS') or symbol.endswith('.BO') or symbol in ['^NSEI', '^BSESN']

def format_indian_currency(amount):
    """Format amount in Indian currency (Rupees)"""
    if amount >= 10000000:  # 1 crore
        return f"₹{amount/10000000:.2f} Cr"
    elif amount >= 100000:  # 1 lakh
        return f"₹{amount/100000:.2f} L"
    else:
        return f"₹{amount:.2f}"
