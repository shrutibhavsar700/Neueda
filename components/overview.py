
import streamlit as st
import numpy as np
import plotly.express as px

def render_overview(portfolio_manager, returns_data):
    """Render the overview tab"""
    
    st.header("Portfolio Overview")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Companies", len(portfolio_manager.data))
    
    with col2:
        if returns_data:
            avg_return = np.mean([data['total_return'] for data in returns_data.values()])
            st.metric("Avg Total Return", f"{avg_return:.2%}")
    
    with col3:
        if returns_data:
            avg_volatility = np.mean([data['volatility'] for data in returns_data.values()])
            st.metric("Avg Volatility", f"{avg_volatility:.2%}")
    
    with col4:
        if returns_data:
            avg_sharpe = np.mean([data['sharpe_ratio'] for data in returns_data.values()])
            st.metric("Avg Sharpe Ratio", f"{avg_sharpe:.2f}")
    
    # Portfolio composition pie chart
    if portfolio_manager.portfolio_weights and sum(portfolio_manager.portfolio_weights.values()) > 0:
        st.subheader("Portfolio Composition")
        fig_pie = px.pie(
            values=list(portfolio_manager.portfolio_weights.values()),
            names=list(portfolio_manager.portfolio_weights.keys()),
            title="Portfolio Allocation"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Quick stats table
    if returns_data:
        st.subheader("Stock Performance Summary")
        
        summary_data = []
        for company, data in returns_data.items():
            summary_data.append({
                'Company': company,
                'Current Price': f"₹{data.get('current_price', 0):.2f}",
                'Total Return': f"{data['total_return']:.2%}",
                'Volatility': f"{data['volatility']:.2%}",
                'Sharpe Ratio': f"{data['sharpe_ratio']:.2f}"
            })
        
        import pandas as pd
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
