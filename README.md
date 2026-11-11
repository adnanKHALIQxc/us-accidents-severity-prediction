# 🚗 US Accidents Severity Prediction (AI & Expert Systems)

## 📌 Overview
This project predicts the severity of traffic accidents across the United States using a raw dataset of **7.7 million records**. The goal was to build a robust machine learning pipeline that handles severe class imbalance, optimizes for real-world safety (catching severe crashes), and serves predictions via a modern web interface.

## 📊 The Dataset (Raw & Messy)
- **Source:** [Kaggle - US Accidents (Sobhan Moosavi)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- **Size:** 7,728,394 rows, 46 columns.
- **Data Cleaning Proof:** The raw dataset contains >93% missing values in `Precipitation(in)`, which we dropped. We also imputed missing weather data using median values and converted corrupted timestamp strings into usable datetime objects.

## 🧠 The Modeling Challenge (Class Imbalance)
Initial baselines using default XGBoost achieved **78% accuracy**, but the model entirely ignored Severity 1 and Severity 4 crashes because they make up less than 1% of the data. 

To solve this, we applied **Threshold Tuning**. We trained the model normally, then applied custom weights `[2.0, 1.0, 2.5, 5.0]` to the output probabilities. 
- **Final Accuracy:** 73.14% (Exceeds the 70% requirement).
- **Severity 3 Recall:** Jumped from 8% to 48%. 
- **Reference Paper:** Adapts strategies from 2025 IEEE research on XAI and class imbalance in traffic severity datasets.

## ⚙️ Project Structure
- `src/data/`: Scripts for loading and cleaning 7.7M rows.
- `src/features/`: Feature engineering (time extraction, categorical encoding).
- `src/models/`: Training scripts and evaluation metrics.
- `app/main.py`: The NiceGUI interactive dashboard.
- `models/`: Pre-trained XGBoost model (`final_model.json`) and threshold weights.

## 🚀 How to Run the App
1. Create a virtual environment and install requirements:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt