import pandas as pd
import streamlit as st
from typing import Dict, List

class DataLoader:
    """Utility class for loading and preprocessing stock data"""
    
    @staticmethod
    def load_file(file_path: str) -> pd.DataFrame:
        """Load a single stock data file"""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            return DataLoader.preprocess_dataframe(df)
            
        except Exception as e:
            st.error(f"Error loading file {file_path}: {str(e)}")
            return None
    
    @staticmethod
    def load_uploaded_file(uploaded_file) -> pd.DataFrame:
        """Load data from Streamlit uploaded file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {uploaded_file.name}")
            
            return DataLoader.preprocess_dataframe(df)
            
        except Exception as e:
            st.error(f"Error loading uploaded file {uploaded_file.name}: {str(e)}")
            return None
    
    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataframe"""
        df_processed = df.copy()
        
        # Handle Date column
        if 'Date' in df_processed.columns:
            df_processed['Date'] = pd.to_datetime(df_processed['Date'], errors='coerce')
            df_processed = df_processed.dropna(subset=['Date'])
            df_processed = df_processed.sort_values('Date')
            df_processed.set_index('Date', inplace=True)
        
        # Remove duplicates
        df_processed = df_processed[~df_processed.index.duplicated(keep='first')]
        
        # Ensure numeric columns
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        return df_processed
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> Dict[str, bool]:
        """Validate if the dataframe has required columns"""
        required_columns = ['Close']
        optional_columns = ['Open', 'High', 'Low', 'Volume']
        
        validation = {
            'has_close': 'Close' in df.columns,
            'has_ohlc': all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']),
            'has_volume': 'Volume' in df.columns,
            'has_date_index': isinstance(df.index, pd.DatetimeIndex),
            'sufficient_data': len(df) >= 30  # At least 30 data points
        }
        
        return validation
