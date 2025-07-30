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
    "LTIM.NS": "LTI Mindtree",
    "MPHASIS.NS": "Mphasis Limited",
    "LTTS.NS": "L&T Technology Services",
    "COFORGE.NS": "Coforge Limited",
    "PERSISTENT.NS": "Persistent Systems",
    "MINDTREE.NS": "Mindtree Limited",
    "LTI.NS": "Larsen & Toubro Infotech",
    "ZENSARTECH.NS": "Zensar Technologies",
    "CYIENT.NS": "Cyient Limited",
    "HEXAWARE.NS": "Hexaware Technologies",
    
    # Banking & Finance
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "State Bank of India",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "AXISBANK.NS": "Axis Bank",
    "INDUSINDBK.NS": "IndusInd Bank",
    "BANDHANBNK.NS": "Bandhan Bank",
    "FEDERALBNK.NS": "Federal Bank",
    "PNB.NS": "Punjab National Bank",
    "HDFCLIFE.NS": "HDFC Life Insurance",
    "SBILIFE.NS": "SBI Life Insurance",
    "ICICIPRULI.NS": "ICICI Prudential Life",
    "BAJFINANCE.NS": "Bajaj Finance",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "HDFCAMC.NS": "HDFC AMC",
    "SBICARD.NS": "SBI Cards",
    
    # FMCG
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC Limited",
    "NESTLEIND.NS": "Nestle India",
    "BRITANNIA.NS": "Britannia Industries",
    "DABUR.NS": "Dabur India",
    "GODREJCP.NS": "Godrej Consumer",
    "MARICO.NS": "Marico Limited",
    "COLPAL.NS": "Colgate Palmolive",
    "EMAMILTD.NS": "Emami Limited",
    "VBL.NS": "Varun Beverages",
    "JUBLFOOD.NS": "Jubilant FoodWorks",
    "UBL.NS": "United Breweries",
    "RADICO.NS": "Radico Khaitan",
    "PGHL.NS": "Procter & Gamble",
    "TATACONSUM.NS": "Tata Consumer",
    "GILLETTE.NS": "Gillette India",
    "BBTC.NS": "Bombay Burmah",
    "CCL.NS": "CCL Products",
    "DFMFOODS.NS": "DFM Foods",
    "DODLA.NS": "Dodla Dairy",
    
    # Automotive & Auto Ancillaries
    "MARUTI.NS": "Maruti Suzuki",
    "TATAMOTORS.NS": "Tata Motors",
    "M&M.NS": "Mahindra & Mahindra",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "EICHERMOT.NS": "Eicher Motors",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "ASHOKLEY.NS": "Ashok Leyland",
    "TVSMOTOR.NS": "TVS Motor Company",
    "BHARATFORG.NS": "Bharat Forge",
    "BOSCHLTD.NS": "Bosch Limited",
    "MOTHERSUMI.NS": "Motherson Sumi",
    "APOLLOTYRE.NS": "Apollo Tyres",
    "MRF.NS": "MRF Limited",
    "EXIDEIND.NS": "Exide Industries",
    "BALKRISIND.NS": "Balkrishna Industries",
    "ESCORTS.NS": "Escorts Kubota",
    "AMARAJABAT.NS": "Amara Raja Batteries",
    "JKTYRE.NS": "JK Tyre & Industries",
    "TIINDIA.NS": "Tube Investments",
    "BHARATFORG.NS": "Bharat Forge",
    
    # Pharma & Healthcare
    "SUNPHARMA.NS": "Sun Pharmaceutical",
    "DRREDDY.NS": "Dr. Reddy's Laboratories",
    "CIPLA.NS": "Cipla Limited",
    "DIVISLAB.NS": "Divis Laboratories",
    "BIOCON.NS": "Biocon Limited",
    "LUPIN.NS": "Lupin Limited",
    "AUROPHARMA.NS": "Aurobindo Pharma",
    "TORNTPHARMA.NS": "Torrent Pharma",
    "ALKEM.NS": "Alkem Laboratories",
    "GLAND.NS": "Gland Pharma",
    "LAURUSLABS.NS": "Laurus Labs",
    "FORTIS.NS": "Fortis Healthcare",
    "APOLLOHOSP.NS": "Apollo Hospitals",
    "NARAYANKHAND.NS": "Narayana Hrudayalaya",
    "METROPOLIS.NS": "Metropolis Healthcare",
    "GRANULES.NS": "Granules India",
    "IPCALAB.NS": "IPCA Laboratories",
    "GLENMARK.NS": "Glenmark Pharma",
    "AJANTPHARM.NS": "Ajanta Pharma",
    "NATCOPHARM.NS": "Natco Pharma",
    
    # Energy & Oil & Gas
    "RELIANCE.NS": "Reliance Industries",
    "ONGC.NS": "Oil & Natural Gas Corporation",
    "NTPC.NS": "NTPC Limited",
    "POWERGRID.NS": "Power Grid Corporation",
    "IOC.NS": "Indian Oil Corporation",
    "BPCL.NS": "Bharat Petroleum",
    "HINDPETRO.NS": "Hindustan Petroleum",
    "GAIL.NS": "GAIL India",
    "TATAPOWER.NS": "Tata Power",
    "ADANIPOWER.NS": "Adani Power",
    "TORNTPOWER.NS": "Torrent Power",
    "NHPC.NS": "NHPC Limited",
    "SJVN.NS": "SJVN Limited",
    "ADANIGAS.NS": "Adani Total Gas",
    "GUJGASLTD.NS": "Gujarat Gas",
    "PETRONET.NS": "Petronet LNG",
    "MGL.NS": "Mahanagar Gas",
    "IGL.NS": "Indraprastha Gas",
    "CASTROLIND.NS": "Castrol India",
    "AEGISCHEM.NS": "Aegis Logistics",
    
    # Metals & Mining
    "TATASTEEL.NS": "Tata Steel",
    "HINDALCO.NS": "Hindalco Industries",
    "JSWSTEEL.NS": "JSW Steel",
    "VEDL.NS": "Vedanta Limited",
    "JINDALSTEL.NS": "Jindal Steel & Power",
    "SAIL.NS": "Steel Authority of India",
    "NMDC.NS": "NMDC Limited",
    "COALINDIA.NS": "Coal India",
    "HINDZINC.NS": "Hindustan Zinc",
    "NATIONALUM.NS": "National Aluminium",
    "APLAPOLLO.NS": "APL Apollo Tubes",
    "RATNAMANI.NS": "Ratnamani Metals",
    "WELCORP.NS": "Welspun Corp",
    "JSWISPAT.NS": "JSW Ispat Special Products",
    "MAHSEAMLES.NS": "Maharashtra Seamless",
    
    # Telecom & Media
    "BHARTIARTL.NS": "Bharti Airtel",
    "IDEA.NS": "Vodafone Idea",
    "RELIANCE.NS": "Reliance Jio (via RIL)",
    "ZOMATO.NS": "Zomato Limited",
    "NAUKRI.NS": "Info Edge (Naukri.com)",
    "NAZARA.NS": "Nazara Technologies",
    "INOXLEISUR.NS": "INOX Leisure",
    "PVRINOX.NS": "PVR INOX",
    "SUNTV.NS": "Sun TV Network",
    "ZEEL.NS": "Zee Entertainment",
    "DBCORP.NS": "DB Corp",
    "TV18BRDCST.NS": "TV18 Broadcast",
    "DISHTV.NS": "Dish TV India",
    "HATHWAY.NS": "Hathway Cable",
    "SITINET.NS": "Siti Networks",
    
    # Consumer Goods & Retail
    "ASIANPAINT.NS": "Asian Paints",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "TITAN.NS": "Titan Company",
    "DABUR.NS": "Dabur India",
    "MARICO.NS": "Marico Limited",
    "GODREJCP.NS": "Godrej Consumer",
    "HAVELLS.NS": "Havells India",
    "CROMPTON.NS": "Crompton Greaves",
    "VOLTAS.NS": "Voltas Limited",
    "BLUESTARCO.NS": "Blue Star",
    "KAJARIACER.NS": "Kajaria Ceramics",
    "JUBLFOOD.NS": "Jubilant FoodWorks",
    "VBL.NS": "Varun Beverages",
    "BBTC.NS": "Bombay Burmah",
    "WHIRLPOOL.NS": "Whirlpool India",
    
    # Indices
    "^NSEI": "Nifty 50",
    "^BSESN": "BSE Sensex",
    "^NSEBANK": "Nifty Bank",
    "^CNXIT": "Nifty IT",
    "^CNXAUTO": "Nifty Auto",
    "^CNXFMCG": "Nifty FMCG",
    "^CNXPHARMA": "Nifty Pharma",
    "^CNXMETAL": "Nifty Metal",
    "^CNXREALTY": "Nifty Realty",
    "^CNXMEDIA": "Nifty Media",
    "^CNXFINANCE": "Nifty Financial Services"
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
