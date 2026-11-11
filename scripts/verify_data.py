import sys
from pathlib import Path
import pandas as pd  # <-- Make sure this is here!

# Add project root to path FIRST, before importing src
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.load import load_raw_data
from src.utils.logger import get_logger

logger = get_logger("verify_script")

def main():
    logger.info("Loading 50,000 rows to verify raw data quality...")
    
    # Load only 50k rows for a quick check
    df = load_raw_data(mode="test")
    
    print("\n" + "="*50)
    print(" 1. DATA SHAPE (Rows, Columns) ")
    print("="*50)
    print(df.shape)
    
    print("\n" + "="*50)
    print(" 2. MISSING VALUES PER COLUMN (Raw/Messy) ")
    print("="*50)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage (%)': missing_pct})
    print(missing_df[missing_df['Missing Count'] > 0].sort_values(by='Percentage (%)', ascending=False))
    
    print("\n" + "="*50)
    print(" 3. DUPLICATE IDs ")
    print("="*50)
    dupes = df['ID'].duplicated().sum()
    print(f"Duplicate IDs found: {dupes} out of {len(df)} rows.")
    
    print("\n" + "="*50)
    print(" 4. DATA TYPES ")
    print("="*50)
    print(df.dtypes)

if __name__ == "__main__":
    import pandas as pd # Import here so the script works if pandas is installed
    main()