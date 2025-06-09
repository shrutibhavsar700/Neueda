import streamlit as st
import plotly.graph_objects as go

def render_portfolio_analysis(portfolio_manager, returns_data):
    """Render the portfolio analysis tab"""
    
    st.header("Portfolio Analysis")
    
    if portfolio_manager.portfolio_weights and abs(sum(portfolio_manager.portfolio_weights.values()) - 1.0) <= 0.01:
        portfolio_metrics = portfolio_manager.portfolio_performance(portfolio_manager.portfolio_weights, returns_data)
        
        if portfolio_metrics:
            # Portfolio metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Portfolio Return", f"{portfolio_metrics['total_return']:.2%}")
            with col2:
                st.metric("Portfolio Volatility", f"{portfolio_metrics['volatility']:.2%}")
            with col3:
                st.metric("Sharpe Ratio", f"{portfolio_metrics['sharpe_ratio']:.2f}")
            with col4:
                st.metric("Max Drawdown", f"{portfolio_metrics['max_drawdown']:.2%}")
            
            # Portfolio vs individual stocks comparison
            st.subheader("Portfolio vs Individual Stocks - Cumulative Returns")
            
            fig_comparison = go.Figure()
            
            # Add portfolio
            fig_comparison.add_trace(go.Scatter(
                x=portfolio_metrics['cumulative_returns'].index,
                y=portfolio_metrics['cumulative_returns'].values * 100,
                mode='lines',
                name='Portfolio',
                line=dict(color='red', width=3)
            ))
            
            # Add individual stocks
            for company, data in returns_data.items():
                fig_comparison.add_trace(go.Scatter(
                    x=data['cumulative_returns'].index,
                    y=data['cumulative_returns'].values * 100,
                    mode='lines',
                    name=company,
                    opacity=0.7
                ))
            
            fig_comparison.update_layout(
                title="Cumulative Returns Comparison",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                hovermode='x unified'
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Portfolio allocation vs performance
            st.subheader("Portfolio Allocation Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Current Allocation:**")
                for company, weight in portfolio_manager.portfolio_weights.items():
                    if weight > 0:
                        individual_return = returns_data[company]['total_return'] if company in returns_data else 0
                        st.write(f"• {company}: {weight:.1%} (Return: {individual_return:.2%})")
            
            with col2:
                st.write("**Contribution to Portfolio Return:**")
                total_portfolio_return = portfolio_metrics['total_return']
                for company, weight in portfolio_manager.portfolio_weights.items():
                    if weight > 0 and company in returns_data:
                        contribution = weight * returns_data[company]['total_return']
                        contribution_pct = (contribution / total_portfolio_return * 100) if total_portfolio_return != 0 else 0
                        st.write(f"• {company}: {contribution:.2%} ({contribution_pct:.1f}% of total)")
            
        else:
            st.error("Unable to calculate portfolio metrics. Please check your data and weights.")
    else:
        st.warning("Please ensure portfolio weights sum to 1.0 for portfolio analysis.")
