import pandas as pd
import numpy as np
from typing import Dict, Any

class FinancialCalculations:
    @staticmethod
    def calculate_stock_returns(df: pd.DataFrame, price_column: str = 'Close') -> Dict[str, Any]:
        """Calculate returns and metrics for a single stock"""
        # Daily returns
        daily_returns = df[price_column].pct_change().dropna()
        
        # Cumulative returns
        cumulative_returns = (1 + daily_returns).cumprod() - 1
        
        # Metrics
        total_return = cumulative_returns.iloc[-1]
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized
        mean_return = daily_returns.mean() * 252  # Annualized
        sharpe_ratio = mean_return / volatility if volatility != 0 else 0
        
        return {
            'daily_returns': daily_returns,
            'cumulative_returns': cumulative_returns,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'mean_return': mean_return
        }
    
    @staticmethod
    def calculate_portfolio_performance(weights: Dict[str, float], returns_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate portfolio performance metrics"""
        # Align all daily returns
        daily_returns_df = pd.DataFrame()
        for company in weights.keys():
            if company in returns_data:
                daily_returns_df[company] = returns_data[company]['daily_returns']
        
        daily_returns_df = daily_returns_df.dropna()
        
        if daily_returns_df.empty:
            return None
        
        # Calculate portfolio returns
        portfolio_weights_array = np.array([weights[col] for col in daily_returns_df.columns])
        portfolio_daily_returns = (daily_returns_df * portfolio_weights_array).sum(axis=1)
        portfolio_cumulative_returns = (1 + portfolio_daily_returns).cumprod() - 1
        
        # Portfolio metrics
        total_return = portfolio_cumulative_returns.iloc[-1]
        volatility = portfolio_daily_returns.std() * np.sqrt(252)
        mean_return = portfolio_daily_returns.mean() * 252
        sharpe_ratio = mean_return / volatility if volatility != 0 else 0
        max_drawdown = FinancialCalculations.calculate_max_drawdown(portfolio_cumulative_returns)
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'daily_returns': portfolio_daily_returns,
            'cumulative_returns': portfolio_cumulative_returns
        }
    
    @staticmethod
    def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown.min()
    
    @staticmethod
    def calculate_correlation_matrix(returns_data: Dict[str, Dict]) -> pd.DataFrame:
        """Calculate correlation matrix of returns"""
        daily_returns_df = pd.DataFrame()
        for company, data in returns_data.items():
            daily_returns_df[company] = data['daily_returns']
        
        return daily_returns_df.corr()
    
    @staticmethod
    def calculate_risk_metrics(returns_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """Calculate comprehensive risk metrics"""
        risk_metrics = {}
        
        for company, data in returns_data.items():
            daily_returns = data['daily_returns']
            
            # Value at Risk (VaR)
            var_95 = np.percentile(daily_returns, 5)
            var_99 = np.percentile(daily_returns, 1)
            
            # Conditional VaR (Expected Shortfall)
            cvar_95 = daily_returns[daily_returns <= var_95].mean()
            
            risk_metrics[company] = {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'skewness': daily_returns.skew(),
                'kurtosis': daily_returns.kurtosis()
            }
        
        return risk_metrics