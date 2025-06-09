import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

def render_risk_metrics(portfolio_manager, returns_data):
    """Render the risk metrics tab"""
    
    st.header("Risk Metrics")
    
    if returns_data:
        # Risk metrics table
        risk_data = []
        for company, data in returns_data.items():
            risk_data.append({
                'Company': company,
                'Total Return': f"{data['total_return']:.2%}",
                'Volatility': f"{data['volatility']:.2%}",
                'Sharpe Ratio': f"{data['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{data.get('max_drawdown', 0):.2%}",
                'VaR (95%)': f"{data.get('var_95', 0):.2%}"
            })
        
        risk_df = pd.DataFrame(risk_data)
        
        st.subheader("Risk-Return Summary")
        st.dataframe(risk_df, use_container_width=True)
        
        # Risk-Return scatter plot
        st.subheader("Risk-Return Scatter Plot")
        
        returns_list = [data['total_return'] * 100 for data in returns_data.values()]
        volatility_list = [data['volatility'] * 100 for data in returns_data.values()]
        
        fig_scatter = px.scatter(
            x=volatility_list,
            y=returns_list,
            text=list(returns_data.keys()),
            title="Risk-Return Profile",
            labels={'x': 'Volatility (%)', 'y': 'Total Return (%)'}
        )
        fig_scatter.update_traces(textposition="top center")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Value at Risk (VaR) analysis
        st.subheader("Value at Risk (VaR) Analysis")
        
        var_data = []
        for company, data in returns_data.items():
            daily_returns = data['daily_returns']
            if len(daily_returns) > 0:
                var_95 = np.percentile(daily_returns, 5)
                var_99 = np.percentile(daily_returns, 1)
                cvar_95 = daily_returns[daily_returns <= var_95].mean()
                
                var_data.append({
                    'Company': company,
                    'Daily VaR (95%)': f"{var_95:.2%}",
                    'Daily VaR (99%)': f"{var_99:.2%}",
                    'Monthly VaR (95%)': f"{var_95 * np.sqrt(21):.2%}",
                    'CVaR (95%)': f"{cvar_95:.2%}"
                })
        
        if var_data:
            var_df = pd.DataFrame(var_data)
            st.dataframe(var_df, use_container_width=True)
            
            st.info("**VaR Interpretation:** VaR represents the maximum expected loss over a given time period at a specified confidence level. CVaR (Conditional VaR) shows the expected loss beyond the VaR threshold.")
        
        # Risk comparison chart
        st.subheader("Risk Metrics Comparison")
        
        if len(returns_data) > 1:
            # Create comparison chart for different risk metrics
            companies = list(returns_data.keys())
            volatilities = [returns_data[company]['volatility'] * 100 for company in companies]
            max_drawdowns = [abs(returns_data[company].get('max_drawdown', 0)) * 100 for company in companies]
            
            comparison_df = pd.DataFrame({
                'Company': companies,
                'Volatility (%)': volatilities,
                'Max Drawdown (%)': max_drawdowns
            })
            
            fig_risk = px.bar(
                comparison_df, 
                x='Company', 
                y=['Volatility (%)', 'Max Drawdown (%)'],
                title="Risk Metrics Comparison",
                barmode='group'
            )
            st.plotly_chart(fig_risk, use_container_width=True)
    
    else:
        st.info("No data available for risk analysis.")