# src/data/clean.py

import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the raw accident dataframe."""
    logger.info(f"Starting cleaning. Initial shape: {df.shape}")
    
    # 1. Drop columns with >90% missing data (like Precipitation)
    # This leaves columns that actually have signal.
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > 0.90].index.tolist()
    if cols_to_drop:
        logger.info(f"Dropping columns with >90% missing: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
        
    # 2. Convert time columns to datetime
    logger.info("Converting Start_Time and End_Time to datetime...")
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
    
    # Drop rows where Start_Time failed to parse (corrupt timestamps)
    df = df.dropna(subset=['Start_Time'])
    
    # 3. Handle missing weather values (Impute)
    # We fill numeric weather with the median
    numeric_cols = ['Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            
    # Fill categorical weather with 'Unknown'
    if 'Weather_Condition' in df.columns:
        df['Weather_Condition'] = df['Weather_Condition'].fillna('Unknown')
        
    # 4. Drop rows missing coordinates or target
    df = df.dropna(subset=['Start_Lat', 'Start_Lng', 'Severity'])
    
    logger.info(f"Cleaning complete. Final shape: {df.shape}")
    return df