import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class FutureAnalytics:
    def __init__(self):
        self.economic_events = {
            'earnings': {'impact': 0.15, 'duration': 3},
            'fed_meeting': {'impact': 0.08, 'duration': 5},
            'gdp_release': {'impact': 0.06, 'duration': 2},
            'inflation_data': {'impact': 0.07, 'duration': 3},
            'employment_data': {'impact': 0.05, 'duration': 2}
        }
    
    def extended_price_prediction(self, model, data, days_ahead=90, confidence_interval=0.95):
        """Generate extended predictions with confidence intervals"""
        predictions = []
        confidence_bands = []
        
        # Base prediction using existing model
        base_pred = self._get_base_predictions(model, data, days_ahead)
        
        # Calculate volatility for confidence intervals
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Generate confidence intervals
        z_score = stats.norm.ppf((1 + confidence_interval) / 2)
        
        for i, pred in enumerate(base_pred):
            # Increase uncertainty over time
            time_factor = np.sqrt(i + 1) / np.sqrt(30)  # Normalize to 30 days
            uncertainty = pred * volatility * time_factor * z_score
            
            predictions.append(pred)
            confidence_bands.append({
                'lower': pred - uncertainty,
                'upper': pred + uncertainty,
                'prediction': pred
            })
        
        return predictions, confidence_bands
    
    def scenario_analysis(self, data, scenarios=None):
        """Perform what-if scenario analysis"""
        if scenarios is None:
            scenarios = {
                'bull_market': {'trend': 0.15, 'volatility': 0.8},
                'bear_market': {'trend': -0.20, 'volatility': 1.5},
                'sideways_market': {'trend': 0.02, 'volatility': 1.0},
                'high_volatility': {'trend': 0.05, 'volatility': 2.0},
                'recession': {'trend': -0.30, 'volatility': 2.5}
            }
        
        current_price = data['Close'].iloc[-1]
        results = {}
        
        for scenario_name, params in scenarios.items():
            # Generate scenario-based predictions
            days = 90
            daily_return = params['trend'] / 252  # Convert annual to daily
            daily_vol = params['volatility'] * data['Close'].pct_change().std()
            
            # Monte Carlo simulation for scenario
            simulations = []
            for _ in range(1000):
                prices = [current_price]
                for day in range(days):
                    random_shock = np.random.normal(0, daily_vol)
                    next_price = prices[-1] * (1 + daily_return + random_shock)
                    prices.append(max(next_price, 0.01))  # Prevent negative prices
                simulations.append(prices[1:])
            
            # Calculate statistics
            sim_array = np.array(simulations)
            results[scenario_name] = {
                'mean_path': np.mean(sim_array, axis=0),
                'percentile_5': np.percentile(sim_array, 5, axis=0),
                'percentile_25': np.percentile(sim_array, 25, axis=0),
                'percentile_75': np.percentile(sim_array, 75, axis=0),
                'percentile_95': np.percentile(sim_array, 95, axis=0),
                'final_price_range': {
                    'min': np.min(sim_array[:, -1]),
                    'max': np.max(sim_array[:, -1]),
                    'mean': np.mean(sim_array[:, -1])
                }
            }
        
        return results
    
    def long_term_trend_forecast(self, data, periods=['3M', '6M', '1Y', '2Y']):
        """Generate long-term trend forecasts"""
        forecasts = {}
        current_price = data['Close'].iloc[-1]
        
        # Calculate historical trends
        returns_1m = data['Close'].pct_change(21).dropna()
        returns_3m = data['Close'].pct_change(63).dropna()
        returns_6m = data['Close'].pct_change(126).dropna()
        returns_1y = data['Close'].pct_change(252).dropna()
        
        # Trend analysis
        price_series = data['Close'].values
        time_index = np.arange(len(price_series))
        
        # Linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, price_series)
        
        for period in periods:
            if period == '3M':
                days = 90
                base_return = returns_3m.mean() if len(returns_3m) > 0 else 0
            elif period == '6M':
                days = 180
                base_return = returns_6m.mean() if len(returns_6m) > 0 else 0
            elif period == '1Y':
                days = 365
                base_return = returns_1y.mean() if len(returns_1y) > 0 else 0
            elif period == '2Y':
                days = 730
                base_return = returns_1y.mean() * 2 if len(returns_1y) > 0 else 0
            else:
                days = 30
                base_return = returns_1m.mean() if len(returns_1m) > 0 else 0
            
            # Project trend forward
            future_trend = slope * days
            trend_based_price = current_price + future_trend
            
            # Return-based projection
            return_based_price = current_price * (1 + base_return)
            
            # Combined forecast (weighted average)
            forecast_price = 0.6 * trend_based_price + 0.4 * return_based_price
            
            # Calculate confidence based on R-squared
            confidence = min(abs(r_value) * 100, 95)
            
            forecasts[period] = {
                'target_price': forecast_price,
                'current_price': current_price,
                'expected_return': (forecast_price - current_price) / current_price,
                'confidence': confidence,
                'trend_strength': abs(slope),
                'days_ahead': days
            }
        
        return forecasts
    
    def volatility_forecast(self, data, days_ahead=30):
        """Forecast future volatility using GARCH-like approach"""
        returns = data['Close'].pct_change().dropna()
        
        # Calculate rolling volatilities
        vol_5d = returns.rolling(5).std()
        vol_20d = returns.rolling(20).std()
        vol_60d = returns.rolling(60).std()
        
        # Current volatilities
        current_vol_5d = vol_5d.iloc[-1] if not pd.isna(vol_5d.iloc[-1]) else returns.std()
        current_vol_20d = vol_20d.iloc[-1] if not pd.isna(vol_20d.iloc[-1]) else returns.std()
        current_vol_60d = vol_60d.iloc[-1] if not pd.isna(vol_60d.iloc[-1]) else returns.std()
        
        # Weighted volatility forecast
        weights = [0.5, 0.3, 0.2]  # More weight to recent volatility
        forecast_vol = (weights[0] * current_vol_5d + 
                       weights[1] * current_vol_20d + 
                       weights[2] * current_vol_60d)
        
        # Volatility regime detection
        vol_percentile = (vol_20d.iloc[-1] / vol_60d.mean()) * 100
        
        if vol_percentile > 150:
            regime = "High Volatility"
            regime_adjustment = 1.2
        elif vol_percentile < 70:
            regime = "Low Volatility"
            regime_adjustment = 0.8
        else:
            regime = "Normal Volatility"
            regime_adjustment = 1.0
        
        adjusted_forecast = forecast_vol * regime_adjustment
        
        return {
            'forecast_volatility': adjusted_forecast,
            'annualized_volatility': adjusted_forecast * np.sqrt(252),
            'volatility_regime': regime,
            'current_vs_historical': vol_percentile,
            'volatility_trend': self._get_volatility_trend(vol_20d)
        }
    
    def economic_event_impact(self, data, events=None, days_ahead=30):
        """Model impact of economic events on stock price"""
        if events is None:
            events = ['earnings', 'fed_meeting']
        
        current_price = data['Close'].iloc[-1]
        base_volatility = data['Close'].pct_change().std()
        
        impact_scenarios = {}
        
        for event in events:
            if event in self.economic_events:
                event_data = self.economic_events[event]
                
                # Positive and negative scenarios
                positive_impact = current_price * (1 + event_data['impact'])
                negative_impact = current_price * (1 - event_data['impact'])
                
                # Duration of impact
                impact_duration = event_data['duration']
                
                # Generate impact timeline
                timeline = []
                for day in range(days_ahead):
                    if day < impact_duration:
                        # Full impact during event period
                        impact_factor = 1.0
                    else:
                        # Decay impact over time
                        decay_factor = np.exp(-(day - impact_duration) / 5)
                        impact_factor = decay_factor
                    
                    timeline.append({
                        'day': day + 1,
                        'positive_scenario': current_price + (positive_impact - current_price) * impact_factor,
                        'negative_scenario': current_price + (negative_impact - current_price) * impact_factor,
                        'impact_strength': impact_factor
                    })
                
                impact_scenarios[event] = {
                    'base_impact': event_data['impact'],
                    'duration_days': impact_duration,
                    'timeline': timeline,
                    'expected_volatility_increase': base_volatility * 1.5
                }
        
        return impact_scenarios
    
    def portfolio_future_analysis(self, symbols, weights=None, days_ahead=60):
        """Analyze future performance of a portfolio"""
        if weights is None:
            weights = [1.0 / len(symbols)] * len(symbols)
        
        portfolio_data = {}
        correlations = {}
        
        # Fetch data for all symbols
        for symbol in symbols:
            try:
                stock_data = yf.Ticker(symbol).history(period="1y")
                if not stock_data.empty:
                    portfolio_data[symbol] = stock_data['Close']
            except:
                continue
        
        if len(portfolio_data) < 2:
            return {"error": "Insufficient data for portfolio analysis"}
        
        # Calculate correlations
        price_df = pd.DataFrame(portfolio_data)
        correlation_matrix = price_df.pct_change().corr()
        
        # Portfolio volatility calculation
        returns_df = price_df.pct_change().dropna()
        portfolio_returns = (returns_df * weights).sum(axis=1)
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)
        
        # Future scenarios
        scenarios = {
            'optimistic': 0.15,
            'realistic': 0.08,
            'pessimistic': -0.10
        }
        
        portfolio_forecasts = {}
        current_portfolio_value = sum(price_df.iloc[-1] * weights)
        
        for scenario, annual_return in scenarios.items():
            daily_return = annual_return / 252
            future_values = []
            
            for day in range(days_ahead):
                # Add some randomness
                random_factor = np.random.normal(0, portfolio_vol / np.sqrt(252))
                future_value = current_portfolio_value * (1 + daily_return + random_factor) ** (day + 1)
                future_values.append(future_value)
            
            portfolio_forecasts[scenario] = {
                'final_value': future_values[-1],
                'total_return': (future_values[-1] - current_portfolio_value) / current_portfolio_value,
                'path': future_values
            }
        
        return {
            'current_value': current_portfolio_value,
            'portfolio_volatility': portfolio_vol,
            'correlations': correlation_matrix.to_dict(),
            'forecasts': portfolio_forecasts,
            'diversification_ratio': self._calculate_diversification_ratio(correlation_matrix, weights)
        }
    
    def _get_base_predictions(self, model, data, days_ahead):
        """Helper method to get base predictions from existing model"""
        # This would integrate with your existing prediction models
        # For now, using a simple trend-based approach
        recent_prices = data['Close'].tail(30).values
        trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        
        predictions = []
        last_price = data['Close'].iloc[-1]
        
        for i in range(days_ahead):
            # Simple trend projection with some noise
            pred = last_price + (trend * (i + 1))
            predictions.append(pred)
        
        return predictions
    
    def _get_volatility_trend(self, vol_series):
        """Determine if volatility is increasing or decreasing"""
        if len(vol_series) < 10:
            return "Insufficient data"
        
        recent_vol = vol_series.tail(5).mean()
        older_vol = vol_series.tail(20).head(15).mean()
        
        if recent_vol > older_vol * 1.1:
            return "Increasing"
        elif recent_vol < older_vol * 0.9:
            return "Decreasing"
        else:
            return "Stable"
    
    def _calculate_diversification_ratio(self, correlation_matrix, weights):
        """Calculate portfolio diversification ratio"""
        n_assets = len(weights)
        avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
        
        # Diversification ratio: 1 means perfectly diversified, 0 means no diversification
        diversification_ratio = (1 - avg_correlation) * (n_assets - 1) / n_assets
        return max(0, min(1, diversification_ratio))
