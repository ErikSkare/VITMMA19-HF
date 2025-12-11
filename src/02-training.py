# Model training script
# This script defines the model architecture and runs the training loop.
import string
import re
import joblib
import pandas as pd
import numpy as np
from utils import setup_logger
from config import TRAIN_DATA_PATH, BASELINE_MODEL_SAVE_PATH
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

logger = setup_logger()

# -- BASELINE MODEL --
def extract_baseline_features(df: pd.DataFrame):
    result = pd.DataFrame()
    result["char_count"] = df.apply(lambda x: len(str(x)))
    result["word_count"] = df.apply(lambda x: len(str(x).split()))
    result["sentence_count"] = df.apply(lambda x: len(re.findall(r'[.!?]+', x)))
    result["avg_word_length"] = df.apply(lambda x: sum(len(word) for word in str(x).split()) / max(len(str(x).split()), 1))
    result["punctuation_count"] = df.apply(lambda x: sum(1 for c in str(x) if c in string.punctuation))
    return result

def train_baseline_model(train_df: pd.DataFrame):
    logger.info("Training baseline model...")

    param_grid = {"model__C": np.logspace(-2, 2, 20)}
    model_pipeline = Pipeline([
        ("extractor", FunctionTransformer(extract_baseline_features)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(penalty="l2"))
    ])

    logger.info("Running hyperparameter optimization on regularization strength...")
    grid = GridSearchCV(model_pipeline, param_grid, cv=10, scoring='balanced_accuracy')
    grid.fit(train_df["text"], train_df["rating"])

    logger.info(f"Best params: {grid.best_params_}")
    logger.info(f"Best CV score (10 folds): {grid.best_score_} (balanced accuracy)")

    joblib.dump(grid.best_estimator_, BASELINE_MODEL_SAVE_PATH)
    logger.info(f"Saved baseline model (refitted) to {BASELINE_MODEL_SAVE_PATH}")

def train():
    logger.info("Starting training process...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)

    train_baseline_model(train_df)
    
    logger.info("Training complete.")

if __name__ == "__main__":
    train()
