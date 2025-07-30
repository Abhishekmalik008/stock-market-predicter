import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

class TradeType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    trade_type: TradeType = TradeType.BUY
    size: float = 1.0
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    
    def close(self, exit_time: pd.Timestamp, exit_price: float) -> None:
        """Close the trade"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        
        if self.trade_type == TradeType.BUY:
            self.pnl = (exit_price - self.entry_price) * self.size
        else:  # SELL
            self.pnl = (self.entry_price - exit_price) * self.size
            
        self.pnl_pct = (self.pnl / self.entry_price) * 100 if self.entry_price else 0

class BacktestResult:
    """Class to store backtest results"""
    
    def __init__(self):
        self.trades: List[Trade] = []
        self.equity_curve: pd.Series = None
        self.metrics: Dict[str, float] = {}
        
    def add_trade(self, trade: Trade) -> None:
        """Add a completed trade to the results"""
        self.trades.append(trade)
    
    def calculate_metrics(self, risk_free_rate: float = 0.0) -> Dict[str, float]:
        """Calculate performance metrics"""
        if not self.trades:
            return {}
            
        # Calculate trade metrics
        pnls = [trade.pnl for trade in self.trades if trade.pnl is not None]
        pnl_pcts = [trade.pnl_pct for trade in self.trades if trade.pnl_pct is not None]
        
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p <= 0]
        
        # Basic metrics
        total_return = sum(pnls)
        total_return_pct = (sum(pnl_pcts) / 100) * 100  # Convert to percentage
        
        # Risk metrics
        returns_std = np.std(pnl_pcts) if pnl_pcts else 0
        sharpe_ratio = ((np.mean(pnl_pcts) - risk_free_rate) / returns_std) if returns_std > 0 else 0
        
        # Trade metrics
        win_rate = (len(winning_trades) / len(pnls)) * 100 if pnls else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0
        profit_factor = (sum(winning_trades) / abs(sum(losing_trades))) if losing_trades else float('inf')
        
        # Store metrics
        self.metrics = {
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'num_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self._calculate_max_drawdown(),
            'avg_trade_duration': self._calculate_avg_trade_duration()
        }
        
        return self.metrics
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if not self.trades:
            return 0.0
            
        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0
        
        for trade in self.trades:
            equity += trade.pnl if trade.pnl else 0
            if equity > peak:
                peak = equity
            
            drawdown = (peak - equity) / peak if peak != 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
        return max_drawdown * 100  # Convert to percentage
    
    def _calculate_avg_trade_duration(self) -> float:
        """Calculate average trade duration in days"""
        if not self.trades:
            return 0.0
            
        durations = []
        for trade in self.trades:
            if trade.exit_time and trade.entry_time:
                duration = (trade.exit_time - trade.entry_time).days
                durations.append(duration)
                
        return np.mean(durations) if durations else 0.0


class Backtester:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001):
        """
        Initialize the backtester
        
        Args:
            initial_capital: Initial capital in base currency
            commission: Commission per trade as a fraction (e.g., 0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = BacktestResult()
        
    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        price_col: str = 'Close',
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> BacktestResult:
        """
        Run backtest on historical data
        
        Args:
            data: DataFrame with OHLCV data and signals
            signals: Series of trading signals (-1, 0, 1)
            price_col: Name of the price column
            stop_loss: Stop loss as a percentage (e.g., 0.02 for 2%)
            take_profit: Take profit as a percentage (e.g., 0.04 for 4%)
            
        Returns:
            BacktestResult object with trades and metrics
        """
        self.results = BacktestResult()
        current_trade = None
        
        for i in range(1, len(data)):
            current_time = data.index[i]
            current_price = data[price_col].iloc[i]
            prev_signal = signals.iloc[i-1]
            
            # Check if we need to close the current trade
            if current_trade:
                close_trade = False
                exit_price = current_price
                
                # Check stop loss
                if stop_loss:
                    if current_trade.trade_type == TradeType.BUY:
                        sl_price = current_trade.entry_price * (1 - stop_loss)
                        if current_price <= sl_price:
                            close_trade = True
                            exit_price = sl_price
                    else:  # SELL
                        sl_price = current_trade.entry_price * (1 + stop_loss)
                        if current_price >= sl_price:
                            close_trade = True
                            exit_price = sl_price
                
                # Check take profit
                if not close_trade and take_profit:
                    if current_trade.trade_type == TradeType.BUY:
                        tp_price = current_trade.entry_price * (1 + take_profit)
                        if current_price >= tp_price:
                            close_trade = True
                            exit_price = tp_price
                    else:  # SELL
                        tp_price = current_trade.entry_price * (1 - take_profit)
                        if current_price <= tp_price:
                            close_trade = True
                            exit_price = tp_price
                
                # Close the trade if needed
                if close_trade:
                    current_trade.close(current_time, exit_price)
                    self.results.add_trade(current_trade)
                    current_trade = None
            
            # Check for new trade signals
            if prev_signal != 0 and current_trade is None:
                trade_type = TradeType.BUY if prev_signal > 0 else TradeType.SELL
                current_trade = Trade(
                    entry_time=current_time,
                    entry_price=current_price * (1 + self.commission if trade_type == TradeType.BUY else 1 - self.commission),
                    trade_type=trade_type
                )
        
        # Close any open trade at the end
        if current_trade:
            current_trade.close(data.index[-1], data[price_col].iloc[-1])
            self.results.add_trade(current_trade)
        
        # Calculate performance metrics
        self.results.calculate_metrics()
        
        return self.results
    
    def optimize_parameters(
        self,
        data: pd.DataFrame,
        signal_generator: callable,
        param_grid: Dict[str, List[float]],
        metric: str = 'sharpe_ratio'
    ) -> Dict:
        """
        Optimize strategy parameters using grid search
        
        Args:
            data: DataFrame with OHLCV data
            signal_generator: Function that generates signals given data and parameters
            param_grid: Dictionary of parameter ranges to test
            metric: Metric to optimize for (default: 'sharpe_ratio')
            
        Returns:
            Dictionary of best parameters and their performance
        """
        from itertools import product
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(product(*param_grid.values()))
        
        best_score = -float('inf')
        best_params = {}
        
        for values in param_values:
            params = dict(zip(param_names, values))
            
            # Generate signals with current parameters
            signals = signal_generator(data, **params)
            
            # Run backtest
            result = self.run_backtest(data, signals)
            
            # Get metric
            score = result.metrics.get(metric, -float('inf'))
            
            # Update best parameters
            if score > best_score:
                best_score = score
                best_params = params
                best_params[metric] = score
        
        return best_params
