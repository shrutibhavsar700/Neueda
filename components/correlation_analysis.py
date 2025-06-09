import streamlit as st
import plotly.express as px
import pandas as pd

def render_correlation_analysis(portfolio_manager, returns_data):
    """Render the correlation analysis tab"""
    
    st.header("Correlation Analysis")
    
    if len(returns_data) > 1:
        corr_matrix = portfolio_manager.correlation_matrix(returns_data)
        
        # Correlation heatmap
        st.subheader("Returns Correlation Matrix")
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            aspect="auto",
            title="Stock Returns Correlation"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Correlation insights
        st.subheader("Correlation Insights")
        
        # Find highest and lowest correlations
        corr_values = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_values.append({
                    'Stock 1': corr_matrix.columns[i],
                    'Stock 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
        
        if corr_values:
            corr_df = pd.DataFrame(corr_values).sort_values('Correlation', ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Highest Correlations:**")
                st.dataframe(corr_df.head(3))
            
            with col2:
                st.write("**Lowest Correlations:**")
                st.dataframe(corr_df.tail(3))
            
            # Diversification insights
            st.subheader("Diversification Insights")
            
            avg_correlation = corr_df['Correlation'].mean()
            high_corr_pairs = len(corr_df[corr_df['Correlation'] > 0.7])
            low_corr_pairs = len(corr_df[corr_df['Correlation'] < 0.3])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Correlation", f"{avg_correlation:.3f}")
            with col2:
                st.metric("High Correlation Pairs (>0.7)", high_corr_pairs)
            with col3:
                st.metric("Low Correlation Pairs (<0.3)", low_corr_pairs)
            
            # Recommendations
            if avg_correlation > 0.6:
                st.warning("⚠️ High average correlation suggests limited diversification benefits")
            elif avg_correlation < 0.3:
                st.success("✅ Low average correlation indicates good diversification")
            else:
                st.info("ℹ️ Moderate correlation - consider reviewing portfolio composition")
    
    else:
        st.info("Upload at least 2 stocks to see correlation analysis.")
