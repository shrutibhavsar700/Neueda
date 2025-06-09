import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

def render_individual_stocks(portfolio_manager, returns_data):
    """Render the individual stocks analysis tab"""
    
    st.header("Individual Stock Analysis")
    
    # Stock selector
    selected_stock = st.selectbox("Select a stock for detailed analysis:", list(portfolio_manager.data.keys()))
    
    if selected_stock and selected_stock in portfolio_manager.data:
        df = portfolio_manager.data[selected_stock]
        
        # Stock metrics
        col1, col2, col3, col4 = st.columns(4)
        
        if selected_stock in returns_data:
            stock_data = returns_data[selected_stock]
            
            with col1:
                st.metric("Total Return", f"{stock_data['total_return']:.2%}")
            with col2:
                st.metric("Volatility", f"{stock_data['volatility']:.2%}")
            with col3:
                st.metric("Sharpe Ratio", f"{stock_data['sharpe_ratio']:.2f}")
            with col4:
                current_price = stock_data.get('current_price', df['Close'].iloc[-1] if 'Close' in df.columns else 0)
                st.metric("Current Price", f"₹{current_price:.2f}")
        
        # Price chart with moving averages
        st.subheader(f"{selected_stock} - Price Chart with Moving Averages")
        
        fig_price = go.Figure()
        
        # Add price line
        if 'Close' in df.columns:
            fig_price.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Add moving averages if available
            if selected_stock in returns_data and 'moving_averages' in returns_data[selected_stock]:
                ma_data = returns_data[selected_stock]['moving_averages']
                
                if 'ma_20' in ma_data:
                    fig_price.add_trace(go.Scatter(
                        x=df.index,
                        y=ma_data['ma_20'],
                        mode='lines',
                        name='MA 20',
                        line=dict(color='orange', width=1)
                    ))
                
                if 'ma_50' in ma_data:
                    fig_price.add_trace(go.Scatter(
                        x=df.index,
                        y=ma_data['ma_50'],
                        mode='lines',
                        name='MA 50',
                        line=dict(color='red', width=1)
                    ))
        
        fig_price.update_layout(
            title=f"{selected_stock} Stock Price with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            hovermode='x unified'
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Volume chart
        if 'Volume' in df.columns:
            st.subheader(f"{selected_stock} - Volume Chart")
            fig_volume = px.bar(
                x=df.index,
                y=df['Volume'],
                title=f"{selected_stock} Trading Volume"
            )
            st.plotly_chart(fig_volume, use_container_width=True)
        
        # OHLC Chart
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            st.subheader(f"{selected_stock} - OHLC Chart")
            
            fig_ohlc = go.Figure(data=go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=selected_stock
            ))
            
            fig_ohlc.update_layout(
                title=f"{selected_stock} OHLC Chart",
                xaxis_title="Date",
                yaxis_title="Price (₹)"
            )
            st.plotly_chart(fig_ohlc, use_container_width=True)
