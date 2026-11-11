import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from src.data.load import load_raw_data
from src.data.clean import clean_data
from src.features.build import build_features
from src.config import TARGET_COL, RANDOM_SEED, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

def get_prepared_data(mode="test"):
    df = load_raw_data(mode=mode)
    df = clean_data(df)
    df = build_features(df)
    
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    y = y - 1 
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    
    # Calculate weights to handle class imbalance
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    weights_dict = dict(enumerate(class_weights))
    
    # Map weights to the training data
    sample_weights = np.array([weights_dict[y_val_i] for y_val_i in y_train])
    
    return X_train, X_val, y_train, y_val, sample_weights

def objective(trial, X_train, y_train, X_val, y_val, sample_weights):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'objective': 'multi:softprob',
        'num_class': 4,
        'random_state': RANDOM_SEED,
        'n_jobs': 1 
    }
    
    model = xgb.XGBClassifier(**param)
    # Pass sample_weights here
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average='macro')
    return f1

def train_model():
    logger.info("Preparing data for training...")
    # Now returns sample_weights too
    X_train, X_val, y_train, y_val, sample_weights = get_prepared_data(mode="full") 
    
    logger.info("Starting Optuna hyperparameter tuning...")
    study = optuna.create_study(direction='maximize')
    
    # Pass sample_weights to Optuna
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, sample_weights), n_trials=10)
    
    logger.info(f"Best trial F1: {study.best_value}")
    
    logger.info("Training final model with best parameters...")
    best_params = study.best_params
    best_params['objective'] = 'multi:softprob'
    best_params['num_class'] = 4
    best_params['random_state'] = RANDOM_SEED
    
    final_model = xgb.XGBClassifier(**best_params)
    # Pass sample_weights here too
    final_model.fit(X_train, y_train, sample_weight=sample_weights)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "xgboost_severity_model.json"
    final_model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    preds = final_model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    
    print("\n" + "="*50)
    print(" FINAL MODEL EVALUATION (Validation Set) ")
    print("="*50)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1 Score: {f1:.4f}")
    print("\nClassification Report:\n")
    # Added zero_division=0 to fix the warnings
    print(classification_report(y_val, preds, target_names=['Severity 1', 'Severity 2', 'Severity 3', 'Severity 4'], zero_division=0))

if __name__ == "__main__":
    print("STARTING TRAINING SCRIPT...")
    try:
        train_model()
    except Exception as e:
        print(f"CRASH DETECTED: {e}")
        import traceback
        traceback.print_exc()