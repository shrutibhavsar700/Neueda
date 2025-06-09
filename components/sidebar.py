import streamlit as st
import pandas as pd

def render_sidebar(portfolio_manager):
    """Render the sidebar with file upload and portfolio settings"""
    
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # File uploader for multiple files
        uploaded_files = st.file_uploader(
            "Upload your stock data files",
            type=['csv', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="Upload CSV or Excel files containing stock data with Date, Open, High, Low, Close, Volume columns"
        )
        
        if uploaded_files:
            # Process uploaded files
            for uploaded_file in uploaded_files:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    company_name = uploaded_file.name.split('.')[0]
                    portfolio_manager.add_stock(company_name, df)
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        # Portfolio settings
        if portfolio_manager.data:
            st.header("⚖️ Portfolio Weights")
            weights = {}
            companies = list(portfolio_manager.data.keys())
            
            # Equal weight default
            default_weight = 1.0 / len(companies)
            
            for company in companies:
                weights[company] = st.slider(
                    f"{company}",
                    min_value=0.0,
                    max_value=1.0,
                    value=default_weight,
                    step=0.01,
                    key=f"weight_{company}"
                )
            
            # Show total weight
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                st.warning(f"Total weight: {total_weight:.2f} (should be 1.0)")
            else:
                st.success(f"Total weight: {total_weight:.2f} ✓")
            
            portfolio_manager.portfolio_weights = weights
            
            # Portfolio rebalancing options
            st.header("🔄 Quick Actions")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Equal Weights", use_container_width=True):
                    equal_weight = 1.0 / len(companies)
                    for company in companies:
                        st.session_state[f"weight_{company}"] = equal_weight
                    st.rerun()
            
            with col2:
                if st.button("Reset Weights", use_container_width=True):
                    for company in companies:
                        st.session_state[f"weight_{company}"] = 0.0
                    st.rerun()
