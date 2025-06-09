import pandas as pd
import numpy as np
import streamlit as st
from .stock import Stock

class PortfolioManager:
    def __init__(self):
        self.data = {}
        self.portfolio_weights = {}
        self.stocks = {}
        
    def add_stock(self, company_name, dataframe):
        """Add a stock to the portfolio"""
        try:
            stock = Stock(company_name, dataframe)
            self.stocks[company_name] = stock
            self.data[company_name] = dataframe
            return True
        except Exception as e:
            st.error(f"Error adding stock {company_name}: {str(e)}")
            return False
    
    def remove_stock(self, company_name):
        """Remove a stock from the portfolio"""
        if company_name in self.stocks:
            del self.stocks[company_name]
            del self.data[company_name]
            if company_name in self.portfolio_weights:
                del self.portfolio_weights[company_name]
    
    def calculate_returns(self, price_column='Close'):
        """Calculate daily and cumulative returns for all stocks"""
        returns_data = {}
        
        for company, df in self.data.items():
            if price_column in df.columns:
                stock = self.stocks[company]
                returns_data[company] = stock.calculate_metrics(price_column)
        
        return returns_data
    
    def portfolio_performance(self, weights, returns_data):
        """Calculate portfolio performance metrics"""
        if not weights or abs(sum(weights.values()) - 1.0) > 0.01:
            return None
        
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
        portfolio_metrics = {
            'total_return': portfolio_cumulative_returns.iloc[-1] if len(portfolio_cumulative_returns) > 0 else 0,
            'volatility': portfolio_daily_returns.std() * np.sqrt(252),
            'sharpe_ratio': (portfolio_daily_returns.mean() * 252) / (portfolio_daily_returns.std() * np.sqrt(252)) if portfolio_daily_returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_cumulative_returns),
            'daily_returns': portfolio_daily_returns,
            'cumulative_returns': portfolio_cumulative_returns
        }
        
        return portfolio_metrics
    
    def _calculate_max_drawdown(self, cumulative_returns):
        """Calculate maximum drawdown"""
        if len(cumulative_returns) == 0:
            return 0
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown.min()
    
    def correlation_matrix(self, returns_data):
        """Calculate correlation matrix of returns"""
        daily_returns_df = pd.DataFrame()
        for company, data in returns_data.items():
            daily_returns_df[company] = data['daily_returns']
        
        return daily_returns_df.corr()
    
    def get_stock_list(self):
        """Get list of available stocks"""
        return list(self.stocks.keys())
    
    def get_stock_data(self, company_name):
        """Get data for a specific stock"""
        return self.data.get(company_name, None)