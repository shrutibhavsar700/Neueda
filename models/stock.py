
import pandas as pd
import numpy as np

class Stock:
    def __init__(self, symbol, data):
        self.symbol = symbol
        self.data = data.copy()
        self._preprocess_data()
    
    def _preprocess_data(self):
        """Preprocess stock data"""
        # Ensure Date column is datetime and set as index
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data = self.data.sort_values('Date')
            self.data.set_index('Date', inplace=True)
        
        # Remove any duplicate dates
        self.data = self.data[~self.data.index.duplicated(keep='first')]
    
    def calculate_metrics(self, price_column='Close'):
        """Calculate various stock metrics"""
        if price_column not in self.data.columns:
            return None
        
        prices = self.data[price_column]
        
        # Daily returns
        daily_returns = prices.pct_change().dropna()
        
        # Cumulative returns
        cumulative_returns = (1 + daily_returns).cumprod() - 1
        
        # Calculate metrics
        metrics = {
            'daily_returns': daily_returns,
            'cumulative_returns': cumulative_returns,
            'total_return': cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else 0,
            'volatility': daily_returns.std() * np.sqrt(252),  # Annualized
            'sharpe_ratio': self._calculate_sharpe_ratio(daily_returns),
            'max_drawdown': self._calculate_max_drawdown(cumulative_returns),
            'var_95': np.percentile(daily_returns, 5) if len(daily_returns) > 0 else 0,
            'current_price': prices.iloc[-1] if len(prices) > 0 else 0,
            'price_change': self._calculate_price_change(prices),
            'moving_averages': self._calculate_moving_averages(prices)
        }
        
        return metrics
    
    def _calculate_sharpe_ratio(self, daily_returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        if len(daily_returns) == 0 or daily_returns.std() == 0:
            return 0
        
        excess_returns = daily_returns.mean() * 252 - risk_free_rate
        return excess_returns / (daily_returns.std() * np.sqrt(252))
    
    def _calculate_max_drawdown(self, cumulative_returns):
        """Calculate maximum drawdown"""
        if len(cumulative_returns) == 0:
            return 0
        
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown.min()
    
    def _calculate_price_change(self, prices):
        """Calculate price change metrics"""
        if len(prices) < 2:
            return {'daily_change': 0, 'daily_change_pct': 0}
        
        daily_change = prices.iloc[-1] - prices.iloc[-2]
        daily_change_pct = (daily_change / prices.iloc[-2]) * 100
        
        return {
            'daily_change': daily_change,
            'daily_change_pct': daily_change_pct
        }
    
    def _calculate_moving_averages(self, prices):
        """Calculate moving averages"""
        return {
            'ma_20': prices.rolling(window=20).mean(),
            'ma_50': prices.rolling(window=50).mean(),
            'ma_200': prices.rolling(window=200).mean()
        }
    
    def get_ohlcv_data(self):
        """Get OHLCV data"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in required_columns if col in self.data.columns]
        return self.data[available_columns]