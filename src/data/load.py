# src/data/load.py

import pandas as pd
from src.config import RAW_CSV_PATH, USECOLS, RANDOM_SEED, INTERIM_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_raw_data(mode: str = "full") -> pd.DataFrame:
    """
    Loads the raw CSV. 
    mode="test" loads 50,000 rows for quick testing.
    mode="full" loads the entire 3GB dataset.
    """
    logger.info(f"Loading data from {RAW_CSV_PATH} in mode: {mode}")
    
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {RAW_CSV_PATH}. Please ensure it is in data/raw/")

    try:
        if mode == "test":
            df = pd.read_csv(RAW_CSV_PATH, usecols=USECOLS, nrows=50000)
        else:
            # For the full 3GB dataset
            df = pd.read_csv(RAW_CSV_PATH, usecols=USECOLS)
            
        logger.info(f"Loaded DataFrame shape: {df.shape}")
        return df
        
    except ValueError as e:
        logger.error(f"Column mismatch error. Check your USECOLS list. Error: {e}")
        raise

def save_interim_data(df: pd.DataFrame, filename: str = "sampled.parquet"):
    """Saves intermediate data as parquet for fast loading later."""
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERIM_DATA_DIR / filename
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved interim data to {out_path}")