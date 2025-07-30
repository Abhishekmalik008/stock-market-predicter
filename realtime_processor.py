import asyncio
import json
import websockets
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any
import yfinance as yf
from datetime import datetime, timedelta
import pytz
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataType(Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    FUTURES = "futures"

@dataclass
class MarketData:
    """Class to store market data"""
    symbol: str
    data_type: MarketDataType
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'type': self.data_type.value,
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'vwap': self.vwap
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketData':
        """Create from dictionary"""
        return cls(
            symbol=data['symbol'],
            data_type=MarketDataType(data['type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            open=float(data['open']),
            high=float(data['high']),
            low=float(data['low']),
            close=float(data['close']),
            volume=float(data['volume']),
            vwap=float(data.get('vwap', 0)) if data.get('vwap') else None
        )

class DataFeed(Enum):
    """Available data feeds"""
    YAHOO_FINANCE = "yahoo"
    ALPACA = "alpaca"
    BINANCE = "binance"
    COINBASE = "coinbase"
    
class RealTimeProcessor:
    """Real-time market data processor"""
    
    def __init__(self, feed: DataFeed = DataFeed.YAHOO_FINANCE, **feed_kwargs):
        """
        Initialize the real-time processor
        
        Args:
            feed: Data feed to use
            **feed_kwargs: Additional arguments for the data feed
        """
        self.feed = feed
        self.feed_kwargs = feed_kwargs
        self.subscriptions: Dict[str, List[Callable]] = {}
        self.running = False
        self.websocket = None
        self.last_prices: Dict[str, MarketData] = {}
        self.historical_data: Dict[str, pd.DataFrame] = {}
        
    async def connect(self) -> None:
        """Connect to the data feed"""
        if self.feed == DataFeed.YAHOO_FINANCE:
            # Yahoo Finance uses REST API, no WebSocket needed
            logger.info("Connected to Yahoo Finance")
            return
            
        elif self.feed == DataFeed.ALPACA:
            # Initialize Alpaca WebSocket connection
            key_id = self.feed_kwargs.get('key_id')
            secret_key = self.feed_kwargs.get('secret_key')
            
            if not key_id or not secret_key:
                raise ValueError("Alpaca key_id and secret_key are required")
                
            self.websocket = await websockets.connect(
                'wss://stream.data.alpaca.markets/v2/iex',
                extra_headers={
                    'APCA-API-KEY-ID': key_id,
                    'APCA-API-SECRET-KEY': secret_key
                }
            )
            
            # Authenticate
            auth_msg = {
                'action': 'auth',
                'key': key_id,
                'secret': secret_key
            }
            await self.websocket.send(json.dumps(auth_msg))
            
        elif self.feed == DataFeed.BINANCE:
            # Initialize Binance WebSocket connection
            self.websocket = await websockets.connect('wss://stream.binance.com:9443/ws')
            
        elif self.feed == DataFeed.COINBASE:
            # Initialize Coinbase WebSocket connection
            self.websocket = await websockets.connect('wss://ws-feed.pro.coinbase.com')
            
        logger.info(f"Connected to {self.feed.value} data feed")
    
    async def subscribe(self, symbols: List[str], data_type: MarketDataType = MarketDataType.STOCK) -> None:
        """Subscribe to market data for the given symbols"""
        if not symbols:
            return
            
        if self.feed == DataFeed.YAHOO_FINANCE:
            # For Yahoo Finance, we'll use yfinance's download function
            # No actual subscription needed, we'll poll in the background
            for symbol in symbols:
                if symbol not in self.subscriptions:
                    self.subscriptions[symbol] = []
                    
        elif self.feed == DataFeed.ALPACA and self.websocket:
            # Subscribe to Alpaca WebSocket
            subscribe_msg = {
                'action': 'subscribe',
                'trades': symbols,
                'quotes': [],
                'bars': ['1Min']
            }
            await self.websocket.send(json.dumps(subscribe_msg))
            
        elif self.feed == DataFeed.BINANCE and self.websocket:
            # Subscribe to Binance WebSocket
            streams = [f"{symbol.lower()}@ticker" for symbol in symbols]
            subscribe_msg = {
                'method': 'SUBSCRIBE',
                'params': streams,
                'id': 1
            }
            await self.websocket.send(json.dumps(subscribe_msg))
            
        elif self.feed == DataFeed.COINBASE and self.websocket:
            # Subscribe to Coinbase WebSocket
            subscribe_msg = {
                'type': 'subscribe',
                'product_ids': symbols,
                'channels': ['ticker']
            }
            await self.websocket.send(json.dumps(subscribe_msg))
            
        logger.info(f"Subscribed to {len(symbols)} symbols on {self.feed.value}")
    
    async def fetch_historical_data(
        self, 
        symbol: str, 
        period: str = '1y',
        interval: str = '1d',
        data_type: MarketDataType = MarketDataType.STOCK
    ) -> pd.DataFrame:
        """Fetch historical market data"""
        if data_type == MarketDataType.CRYPTO and self.feed == DataFeed.YAHOO_FINANCE:
            # For crypto on Yahoo Finance, we need to add '-USD' suffix
            if not symbol.endswith('-USD'):
                symbol = f"{symbol}-USD"
        
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
                
            # Store historical data
            self.historical_data[symbol] = df
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def add_callback(self, symbol: str, callback: Callable[[MarketData], None]) -> None:
        """Add a callback function for a symbol"""
        if symbol not in self.subscriptions:
            self.subscriptions[symbol] = []
        self.subscriptions[symbol].append(callback)
        logger.info(f"Added callback for {symbol}")
    
    async def _process_message(self, message: str) -> None:
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            if self.feed == DataFeed.ALPACA:
                await self._process_alpaca_message(data)
            elif self.feed == DataFeed.BINANCE:
                await self._process_binance_message(data)
            elif self.feed == DataFeed.COINBASE:
                await self._process_coinbase_message(data)
                
        except json.JSONDecodeError:
            logger.error(f"Failed to parse message: {message}")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    async def _process_alpaca_message(self, data: Dict) -> None:
        """Process Alpaca WebSocket message"""
        if 'T' in data and 'S' in data:  # Trade message
            symbol = data['S']
            market_data = MarketData(
                symbol=symbol,
                data_type=MarketDataType.STOCK,
                timestamp=pd.to_datetime(data['t']),
                open=float(data['o']),
                high=float(data['h']),
                low=float(data['l']),
                close=float(data['c']),
                volume=float(data['v']),
                vwap=float(data['vw']) if 'vw' in data else None
            )
            await self._notify_subscribers(market_data)
    
    async def _process_binance_message(self, data: Dict) -> None:
        """Process Binance WebSocket message"""
        if 'e' in data and data['e'] == '24hrTicker':
            symbol = data['s']
            market_data = MarketData(
                symbol=symbol,
                data_type=MarketDataType.CRYPTO,
                timestamp=datetime.fromtimestamp(data['E'] / 1000, tz=pytz.UTC),
                open=float(data['o']),
                high=float(data['h']),
                low=float(data['l']),
                close=float(data['c']),
                volume=float(data['v']),
                vwap=float(data['w']) if 'w' in data else None
            )
            await self._notify_subscribers(market_data)
    
    async def _process_coinbase_message(self, data: Dict) -> None:
        """Process Coinbase WebSocket message"""
        if 'type' in data and data['type'] == 'ticker':
            symbol = data['product_id']
            market_data = MarketData(
                symbol=symbol,
                data_type=MarketDataType.CRYPTO,
                timestamp=datetime.now(pytz.UTC),  # Coinbase doesn't provide timestamp in ticker
                open=float(data['open_24h']) if 'open_24h' in data else 0,
                high=float(data['high_24h']) if 'high_24h' in data else 0,
                low=float(data['low_24h']) if 'low_24h' in data else 0,
                close=float(data['price']),
                volume=float(data['volume_24h']) if 'volume_24h' in data else 0,
                vwap=float(data['vwap_24h']) if 'vwap_24h' in data else None
            )
            await self._notify_subscribers(market_data)
    
    async def _notify_subscribers(self, market_data: MarketData) -> None:
        """Notify all subscribers of new market data"""
        self.last_prices[market_data.symbol] = market_data
        
        if market_data.symbol in self.subscriptions:
            for callback in self.subscriptions[market_data.symbol]:
                try:
                    await callback(market_data)
                except Exception as e:
                    logger.error(f"Error in callback for {market_data.symbol}: {str(e)}")
    
    async def start(self) -> None:
        """Start the real-time data feed"""
        if self.running:
            return
            
        self.running = True
        
        try:
            await self.connect()
            
            if self.feed == DataFeed.YAHOO_FINANCE:
                # For Yahoo Finance, we'll poll for updates
                while self.running:
                    for symbol in list(self.subscriptions.keys()):
                        try:
                            df = await self.fetch_historical_data(symbol, period='1d', interval='1m')
                            if not df.empty:
                                latest = df.iloc[-1]
                                market_data = MarketData(
                                    symbol=symbol,
                                    data_type=MarketDataType.STOCK,
                                    timestamp=df.index[-1].to_pydatetime(),
                                    open=latest['Open'],
                                    high=latest['High'],
                                    low=latest['Low'],
                                    close=latest['Close'],
                                    volume=latest['Volume']
                                )
                                await self._notify_subscribers(market_data)
                        except Exception as e:
                            logger.error(f"Error fetching data for {symbol}: {str(e)}")
                    
                    # Wait for 1 minute before next update
                    await asyncio.sleep(60)
                    
            else:
                # For WebSocket feeds, listen for messages
                while self.running and self.websocket:
                    try:
                        message = await self.websocket.recv()
                        await self._process_message(message)
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WebSocket connection closed. Reconnecting...")
                        await self.connect()
                    except Exception as e:
                        logger.error(f"Error in WebSocket: {str(e)}")
                        await asyncio.sleep(5)  # Wait before reconnecting
                        
        except Exception as e:
            logger.error(f"Fatal error in data feed: {str(e)}")
            self.running = False
            raise
            
    async def stop(self) -> None:
        """Stop the real-time data feed"""
        self.running = False
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        logger.info("Stopped real-time data feed")

# Example usage
async def example_callback(market_data: MarketData) -> None:
    print(f"{market_data.symbol}: ${market_data.close:.2f} (Volume: {market_data.volume:.2f})")

async def main():
    # Example usage with Yahoo Finance
    processor = RealTimeProcessor(feed=DataFeed.YAHOO_FINANCE)
    
    # Add a callback for a symbol
    processor.add_callback("AAPL", example_callback)
    
    # Start the processor
    await processor.start()

if __name__ == "__main__":
    import asyncio
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopping...")
