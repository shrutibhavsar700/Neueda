import streamlit as st
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from models.portfolio import PortfolioManager
from components.sidebar import render_sidebar
from components.overview import render_overview
from components.individual_stocks import render_individual_stocks
from components.portfolio_analysis import render_portfolio_analysis
from components.correlation_analysis import render_correlation_analysis
from components.risk_metrics import render_risk_metrics

# Set page config
st.set_page_config(
    page_title="Portfolio Management Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown("<h1 class='main-header'>📊 Portfolio Management Dashboard</h1>", unsafe_allow_html=True)
    
    # Initialize portfolio manager
    if 'portfolio_manager' not in st.session_state:
        st.session_state.portfolio_manager = PortfolioManager()
    
    pm = st.session_state.portfolio_manager
    
    # Render sidebar
    render_sidebar(pm)
    
    if not pm.data:
        st.info("👆 Please upload your stock data files using the sidebar to get started!")
        st.markdown("""
        ### Expected Data Format:
        Your files should contain the following columns:
        - **Date**: Date of the stock price
        - **Open**: Opening price
        - **High**: Highest price
        - **Low**: Lowest price
        - **Close**: Closing price
        - **Volume**: Trading volume
        """)
        return
    
    # Calculate returns for all stocks
    returns_data = pm.calculate_returns()
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "📊 Individual Stocks", 
        "🎯 Portfolio Analysis", 
        "🔄 Correlation Analysis",
        "📋 Risk Metrics"
    ])
    
    with tab1:
        render_overview(pm, returns_data)
    
    with tab2:
        render_individual_stocks(pm, returns_data)
    
    with tab3:
        render_portfolio_analysis(pm, returns_data)
    
    with tab4:
        render_correlation_analysis(pm, returns_data)
    
    with tab5:
        render_risk_metrics(pm, returns_data)

if __name__ == "__main__":
    main()

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import seaborn as sns
# import matplotlib.pyplot as plt
# from datetime import datetime, timedelta
# import warnings
# warnings.filterwarnings('ignore')

# # Set page config
# st.set_page_config(
#     page_title="Portfolio Management Dashboard",
#     page_icon="📊",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .metric-card {
#         background-color: #f0f2f6;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         border-left: 5px solid #1f77b4;
#     }
#     .sidebar .sidebar-content {
#         background-color: #f8f9fa;
#     }
# </style>
# """, unsafe_allow_html=True)

# class PortfolioManager:
#     def __init__(self):
#         self.data = {}
#         self.portfolio_weights = {}
        
#     def load_data(self, file_paths):
#         """Load stock data from multiple files"""
#         for file_path in file_paths:
#             try:
#                 # Handle both CSV and Excel files
#                 if file_path.endswith('.csv'):
#                     df = pd.read_csv(file_path)
#                 else:
#                     df = pd.read_excel(file_path)
                
#                 # Extract company name from filename
#                 company_name = file_path.split('/')[-1].split('.')[0]
                
#                 # Ensure Date column is datetime
#                 if 'Date' in df.columns:
#                     df['Date'] = pd.to_datetime(df['Date'])
#                     df = df.sort_values('Date')
#                     df.set_index('Date', inplace=True)
                
#                 self.data[company_name] = df
                
#             except Exception as e:
#                 st.error(f"Error loading {file_path}: {str(e)}")
    
#     def calculate_returns(self, price_column='Close'):
#         """Calculate daily and cumulative returns"""
#         returns_data = {}
        
#         for company, df in self.data.items():
#             if price_column in df.columns:
#                 # Daily returns
#                 daily_returns = df[price_column].pct_change().dropna()
                
#                 # Cumulative returns
#                 cumulative_returns = (1 + daily_returns).cumprod() - 1
                
#                 returns_data[company] = {
#                     'daily_returns': daily_returns,
#                     'cumulative_returns': cumulative_returns,
#                     'total_return': cumulative_returns.iloc[-1],
#                     'volatility': daily_returns.std() * np.sqrt(252),  # Annualized
#                     'sharpe_ratio': (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
#                 }
        
#         return returns_data
    
#     def portfolio_performance(self, weights, returns_data):
#         """Calculate portfolio performance metrics"""
#         if not weights or sum(weights.values()) != 1.0:
#             st.error("Portfolio weights must sum to 1.0")
#             return None
        
#         # Align all daily returns
#         daily_returns_df = pd.DataFrame()
#         for company in weights.keys():
#             if company in returns_data:
#                 daily_returns_df[company] = returns_data[company]['daily_returns']
        
#         daily_returns_df = daily_returns_df.dropna()
        
#         # Calculate portfolio returns
#         portfolio_weights_array = np.array([weights[col] for col in daily_returns_df.columns])
#         portfolio_daily_returns = (daily_returns_df * portfolio_weights_array).sum(axis=1)
#         portfolio_cumulative_returns = (1 + portfolio_daily_returns).cumprod() - 1
        
#         # Portfolio metrics
#         portfolio_metrics = {
#             'total_return': portfolio_cumulative_returns.iloc[-1],
#             'volatility': portfolio_daily_returns.std() * np.sqrt(252),
#             'sharpe_ratio': (portfolio_daily_returns.mean() * 252) / (portfolio_daily_returns.std() * np.sqrt(252)),
#             'max_drawdown': self.calculate_max_drawdown(portfolio_cumulative_returns),
#             'daily_returns': portfolio_daily_returns,
#             'cumulative_returns': portfolio_cumulative_returns
#         }
        
#         return portfolio_metrics
    
#     def calculate_max_drawdown(self, cumulative_returns):
#         """Calculate maximum drawdown"""
#         rolling_max = cumulative_returns.expanding().max()
#         drawdown = (cumulative_returns - rolling_max) / rolling_max
#         return drawdown.min()
    
#     def correlation_matrix(self, returns_data):
#         """Calculate correlation matrix of returns"""
#         daily_returns_df = pd.DataFrame()
#         for company, data in returns_data.items():
#             daily_returns_df[company] = data['daily_returns']
        
#         return daily_returns_df.corr()

# def main():
#     st.markdown("<h1 class='main-header'>📊 Portfolio Management Dashboard</h1>", unsafe_allow_html=True)
    
#     # Initialize portfolio manager
#     if 'portfolio_manager' not in st.session_state:
#         st.session_state.portfolio_manager = PortfolioManager()
    
#     pm = st.session_state.portfolio_manager
    
#     # Sidebar for file upload and settings
#     with st.sidebar:
#         st.header("📁 Data Upload")
        
#         # File uploader for multiple files
#         uploaded_files = st.file_uploader(
#             "Upload your stock data files",
#             type=['csv', 'xlsx', 'xls'],
#             accept_multiple_files=True,
#             help="Upload CSV or Excel files containing stock data with Date, Open, High, Low, Close, Volume columns"
#         )
        
#         if uploaded_files:
#             # Process uploaded files
#             file_data = {}
#             for uploaded_file in uploaded_files:
#                 try:
#                     if uploaded_file.name.endswith('.csv'):
#                         df = pd.read_csv(uploaded_file)
#                     else:
#                         df = pd.read_excel(uploaded_file)
                    
#                     company_name = uploaded_file.name.split('.')[0]
                    
#                     if 'Date' in df.columns:
#                         df['Date'] = pd.to_datetime(df['Date'])
#                         df = df.sort_values('Date')
#                         df.set_index('Date', inplace=True)
                    
#                     pm.data[company_name] = df
#                     file_data[company_name] = df
                    
#                 except Exception as e:
#                     st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
#         # Portfolio settings
#         if pm.data:
#             st.header("⚖️ Portfolio Weights")
#             weights = {}
#             companies = list(pm.data.keys())
            
#             # Equal weight default
#             default_weight = 1.0 / len(companies)
            
#             for company in companies:
#                 weights[company] = st.slider(
#                     f"{company}",
#                     min_value=0.0,
#                     max_value=1.0,
#                     value=default_weight,
#                     step=0.01,
#                     key=f"weight_{company}"
#                 )
            
#             # Show total weight
#             total_weight = sum(weights.values())
#             if abs(total_weight - 1.0) > 0.01:
#                 st.warning(f"Total weight: {total_weight:.2f} (should be 1.0)")
#             else:
#                 st.success(f"Total weight: {total_weight:.2f} ✓")
            
#             pm.portfolio_weights = weights
    
#     if not pm.data:
#         st.info("👆 Please upload your stock data files using the sidebar to get started!")
#         st.markdown("""
#         ### Expected Data Format:
#         Your files should contain the following columns:
#         - **Date**: Date of the stock price
#         - **Open**: Opening price
#         - **High**: Highest price
#         - **Low**: Lowest price
#         - **Close**: Closing price
#         - **Volume**: Trading volume
#         """)
#         return
    
#     # Main dashboard
#     tab1, tab2, tab3, tab4, tab5 = st.tabs([
#         "📈 Overview", 
#         "📊 Individual Stocks", 
#         "🎯 Portfolio Analysis", 
#         "🔄 Correlation Analysis",
#         "📋 Risk Metrics"
#     ])
    
#     # Calculate returns for all stocks
#     returns_data = pm.calculate_returns()
    
#     with tab1:
#         st.header("Portfolio Overview")
        
#         # Key metrics in columns
#         col1, col2, col3, col4 = st.columns(4)
        
#         with col1:
#             st.metric("Total Companies", len(pm.data))
        
#         with col2:
#             if returns_data:
#                 avg_return = np.mean([data['total_return'] for data in returns_data.values()])
#                 st.metric("Avg Total Return", f"{avg_return:.2%}")
        
#         with col3:
#             if returns_data:
#                 avg_volatility = np.mean([data['volatility'] for data in returns_data.values()])
#                 st.metric("Avg Volatility", f"{avg_volatility:.2%}")
        
#         with col4:
#             if returns_data:
#                 avg_sharpe = np.mean([data['sharpe_ratio'] for data in returns_data.values()])
#                 st.metric("Avg Sharpe Ratio", f"{avg_sharpe:.2f}")
        
#         # Portfolio composition pie chart
#         if pm.portfolio_weights and sum(pm.portfolio_weights.values()) > 0:
#             st.subheader("Portfolio Composition")
#             fig_pie = px.pie(
#                 values=list(pm.portfolio_weights.values()),
#                 names=list(pm.portfolio_weights.keys()),
#                 title="Portfolio Allocation"
#             )
#             st.plotly_chart(fig_pie, use_container_width=True)
    
#     with tab2:
#         st.header("Individual Stock Analysis")
        
#         # Stock selector
#         selected_stock = st.selectbox("Select a stock for detailed analysis:", list(pm.data.keys()))
        
#         if selected_stock and selected_stock in pm.data:
#             df = pm.data[selected_stock]
            
#             # Stock metrics
#             col1, col2, col3, col4 = st.columns(4)
            
#             if selected_stock in returns_data:
#                 stock_data = returns_data[selected_stock]
                
#                 with col1:
#                     st.metric("Total Return", f"{stock_data['total_return']:.2%}")
#                 with col2:
#                     st.metric("Volatility", f"{stock_data['volatility']:.2%}")
#                 with col3:
#                     st.metric("Sharpe Ratio", f"{stock_data['sharpe_ratio']:.2f}")
#                 with col4:
#                     current_price = df['Close'].iloc[-1]
#                     st.metric("Current Price", f"₹{current_price:.2f}")
            
#             # Price chart
#             st.subheader(f"{selected_stock} - Price Chart")
#             fig_price = go.Figure()
#             fig_price.add_trace(go.Scatter(
#                 x=df.index,
#                 y=df['Close'],
#                 mode='lines',
#                 name='Close Price',
#                 line=dict(color='#1f77b4', width=2)
#             ))
#             fig_price.update_layout(
#                 title=f"{selected_stock} Stock Price",
#                 xaxis_title="Date",
#                 yaxis_title="Price (₹)",
#                 hovermode='x unified'
#             )
#             st.plotly_chart(fig_price, use_container_width=True)
            
#             # Volume chart
#             st.subheader(f"{selected_stock} - Volume Chart")
#             fig_volume = px.bar(
#                 x=df.index,
#                 y=df['Volume'],
#                 title=f"{selected_stock} Trading Volume"
#             )
#             st.plotly_chart(fig_volume, use_container_width=True)
    
#     with tab3:
#         st.header("Portfolio Analysis")
        
#         if pm.portfolio_weights and abs(sum(pm.portfolio_weights.values()) - 1.0) <= 0.01:
#             portfolio_metrics = pm.portfolio_performance(pm.portfolio_weights, returns_data)
            
#             if portfolio_metrics:
#                 # Portfolio metrics
#                 col1, col2, col3, col4 = st.columns(4)
                
#                 with col1:
#                     st.metric("Portfolio Return", f"{portfolio_metrics['total_return']:.2%}")
#                 with col2:
#                     st.metric("Portfolio Volatility", f"{portfolio_metrics['volatility']:.2%}")
#                 with col3:
#                     st.metric("Sharpe Ratio", f"{portfolio_metrics['sharpe_ratio']:.2f}")
#                 with col4:
#                     st.metric("Max Drawdown", f"{portfolio_metrics['max_drawdown']:.2%}")
                
#                 # Portfolio vs individual stocks comparison
#                 st.subheader("Portfolio vs Individual Stocks - Cumulative Returns")
                
#                 fig_comparison = go.Figure()
                
#                 # Add portfolio
#                 fig_comparison.add_trace(go.Scatter(
#                     x=portfolio_metrics['cumulative_returns'].index,
#                     y=portfolio_metrics['cumulative_returns'].values * 100,
#                     mode='lines',
#                     name='Portfolio',
#                     line=dict(color='red', width=3)
#                 ))
                
#                 # Add individual stocks
#                 for company, data in returns_data.items():
#                     fig_comparison.add_trace(go.Scatter(
#                         x=data['cumulative_returns'].index,
#                         y=data['cumulative_returns'].values * 100,
#                         mode='lines',
#                         name=company,
#                         opacity=0.7
#                     ))
                
#                 fig_comparison.update_layout(
#                     title="Cumulative Returns Comparison",
#                     xaxis_title="Date",
#                     yaxis_title="Cumulative Return (%)",
#                     hovermode='x unified'
#                 )
#                 st.plotly_chart(fig_comparison, use_container_width=True)
                
#         else:
#             st.warning("Please ensure portfolio weights sum to 1.0 for portfolio analysis.")
    
#     with tab4:
#         st.header("Correlation Analysis")
        
#         if len(returns_data) > 1:
#             corr_matrix = pm.correlation_matrix(returns_data)
            
#             # Correlation heatmap
#             st.subheader("Returns Correlation Matrix")
#             fig_corr = px.imshow(
#                 corr_matrix,
#                 text_auto=True,
#                 color_continuous_scale='RdBu_r',
#                 aspect="auto",
#                 title="Stock Returns Correlation"
#             )
#             st.plotly_chart(fig_corr, use_container_width=True)
            
#             # Correlation insights
#             st.subheader("Correlation Insights")
            
#             # Find highest and lowest correlations
#             corr_values = []
#             for i in range(len(corr_matrix.columns)):
#                 for j in range(i+1, len(corr_matrix.columns)):
#                     corr_values.append({
#                         'Stock 1': corr_matrix.columns[i],
#                         'Stock 2': corr_matrix.columns[j],
#                         'Correlation': corr_matrix.iloc[i, j]
#                     })
            
#             corr_df = pd.DataFrame(corr_values).sort_values('Correlation', ascending=False)
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.write("**Highest Correlations:**")
#                 st.dataframe(corr_df.head(3))
            
#             with col2:
#                 st.write("**Lowest Correlations:**")
#                 st.dataframe(corr_df.tail(3))
        
#         else:
#             st.info("Upload at least 2 stocks to see correlation analysis.")
    
#     with tab5:
#         st.header("Risk Metrics")
        
#         if returns_data:
#             # Risk metrics table
#             risk_df = pd.DataFrame({
#                 'Company': list(returns_data.keys()),
#                 'Total Return': [f"{data['total_return']:.2%}" for data in returns_data.values()],
#                 'Volatility': [f"{data['volatility']:.2%}" for data in returns_data.values()],
#                 'Sharpe Ratio': [f"{data['sharpe_ratio']:.2f}" for data in returns_data.values()]
#             })
            
#             st.subheader("Risk-Return Summary")
#             st.dataframe(risk_df, use_container_width=True)
            
#             # Risk-Return scatter plot
#             st.subheader("Risk-Return Scatter Plot")
            
#             returns_list = [data['total_return'] * 100 for data in returns_data.values()]
#             volatility_list = [data['volatility'] * 100 for data in returns_data.values()]
            
#             fig_scatter = px.scatter(
#                 x=volatility_list,
#                 y=returns_list,
#                 text=list(returns_data.keys()),
#                 title="Risk-Return Profile",
#                 labels={'x': 'Volatility (%)', 'y': 'Total Return (%)'}
#             )
#             fig_scatter.update_traces(textposition="top center")
#             st.plotly_chart(fig_scatter, use_container_width=True)
            
#             # Value at Risk (VaR) calculation
#             st.subheader("Value at Risk (VaR) - 95% Confidence")
            
#             var_data = []
#             for company, data in returns_data.items():
#                 daily_returns = data['daily_returns']
#                 var_95 = np.percentile(daily_returns, 5)
#                 var_data.append({
#                     'Company': company,
#                     'Daily VaR (95%)': f"{var_95:.2%}",
#                     'Monthly VaR (95%)': f"{var_95 * np.sqrt(21):.2%}"
#                 })
            
#             var_df = pd.DataFrame(var_data)
#             st.dataframe(var_df, use_container_width=True)

# if __name__ == "__main__":
#     main()