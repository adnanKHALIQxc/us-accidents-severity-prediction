# src/features/build.py

import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates new features from existing ones."""
    logger.info("Building features...")
    
    # 1. Extract Time Features
    if 'Start_Time' in df.columns:
        df['Hour'] = df['Start_Time'].dt.hour
        df['Day_of_Week'] = df['Start_Time'].dt.dayofweek
        df['Month'] = df['Start_Time'].dt.month
        df['Is_Weekend'] = df['Day_of_Week'].isin([5, 6]).astype(int)
        
        # Drop original time columns to save memory
        df = df.drop(columns=['Start_Time', 'End_Time'], errors='ignore')
        
    # 2. Convert Booleans to ints (0/1)
    bool_cols = ['Crossing', 'Junction', 'Traffic_Signal']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
            
    # 3. Handle Categorical Weather and Sunrise_Sunset
    # For tree-based models, we can just convert them to category codes
    cat_cols = ['Weather_Condition', 'Sunrise_Sunset']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category').cat.codes
            
    # 4. Drop ID and coordinates for now (we don't want the model memorizing coordinates)
    # In a real app, you'd use clustering for coordinates, but let's keep it simple.
    df = df.drop(columns=['ID', 'Start_Lat', 'Start_Lng', 'Distance(mi)'], errors='ignore')
    
    logger.info(f"Feature engineering complete. Final shape: {df.shape}")
    return df