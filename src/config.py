# src/config.py

from pathlib import Path

# Project root directory (goes up two levels from src/config.py)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Exact filename of the Kaggle dataset
# MAKE SURE THIS MATCHES YOUR ACTUAL FILE NAME IN data/raw/
RAW_CSV_FILENAME = "US_accidents_march23.csv" 
RAW_CSV_PATH = RAW_DATA_DIR / RAW_CSV_FILENAME

# Model output directory
MODELS_DIR = ROOT_DIR / "models"

# Global seed for reproducibility
RANDOM_SEED = 42

# Columns we want to load initially (saves RAM vs loading all 46)
USECOLS = [
    "ID", "Severity", "Start_Time", "End_Time", "Start_Lat", "Start_Lng",
    "Distance(mi)", "Temperature(F)", "Humidity(%)", "Visibility(mi)", 
    "Wind_Speed(mph)", "Precipitation(in)", "Weather_Condition", 
    "Junction", "Traffic_Signal", "Crossing", "Sunrise_Sunset"
]

# Target variable
TARGET_COL = "Severity"