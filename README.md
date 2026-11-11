# US Accidents Severity Prediction 🚗

## 📌 Project Overview
This project uses machine learning to predict the severity of traffic accidents across the United States. The dataset comprises **7.7 million** raw traffic accident records (from Kaggle) containing weather conditions, road features, and temporal data.

This project specifically addresses **class imbalance** (severe crashes are rare) and uses **Optuna** for hyperparameter tuning to build a robust, explainable model.

## 👥 Team Members
- Adnan Khalique (and [Add your team members' names here])

## 📊 Dataset Source & Messiness
- **Source:** [US Accidents (2016 - 2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) (CC BY-NC-SA 4.0).
- **Raw Data Quality:** The dataset was intentionally kept raw to demonstrate real-world preprocessing. It contains >93% missing values in `Precipitation(in)`, duplicate entries, and unparsed datetime strings. 

## ⚙️ Pipeline Architecture
1. **Ingestion (`src/data/load.py`):** Loads specified columns directly from the 3GB CSV to save RAM.
2. **Cleaning (`src/data/clean.py`):** Drops high-missingness columns, parses datetimes, imputes missing weather data, and removes rows lacking target labels.
3. **Feature Engineering (`src/features/build.py`):** Extracts `Hour`, `Day_of_Week`, `Is_Weekend`, and converts categorical weather into numerical codes.
4. **Modeling (`src/models/train.py`):** Uses XGBoost with `compute_class_weight` to handle class imbalance, optimized via Optuna.

## 🚀 How to Run
1. Clone the repo: `git clone https://github.com/adnanKHALIQxc/us-accidents-severity-prediction.git`
2. Create environment: `python -m venv .venv`
3. Activate environment: `.\.venv\Scripts\activate` (Windows)
4. Install requirements: `pip install -r requirements.txt`
5. Download the raw dataset and place it in `data/raw/`
6. Run training: `python src/models/train.py`

## 📈 Results
*(To be filled after final model run)*

- **Accuracy:** 70.XX%
- **Macro F1 Score:** 0.XX
- **Reference Paper:** Adapting class weight strategies inspired by 2025 XAI literature on accident severity.