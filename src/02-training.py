# Model training script
# This script defines the model architecture and runs the training loop.
import string
import re
import joblib
import pandas as pd
import numpy as np
from utils import setup_logger
from config import TRAIN_DATA_PATH, BASELINE_MODEL_SAVE_PATH

logger = setup_logger()

# -- BASELINE MODEL -- #
def train_baseline_model(train_df: pd.DataFrame):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    
    logger.info("Training baseline model (logistic regression)...")

    param_grid = {"C": np.logspace(-2, 2, 20)}
    model_pipeline = LogisticRegression(penalty="l2")

    logger.info("Running hyperparameter optimization on regularization strength...")
    grid = GridSearchCV(model_pipeline, param_grid, cv=10, scoring='balanced_accuracy')
    grid.fit(train_df["text"], train_df["rating"])

    logger.info(f"Best params: {grid.best_params_}")
    logger.info(f"Best CV score (10 folds): {grid.best_score_} (balanced accuracy)")

    joblib.dump(grid.best_estimator_, BASELINE_MODEL_SAVE_PATH)
    logger.info(f"Saved baseline model (refitted) to {BASELINE_MODEL_SAVE_PATH}")

def train_dl_model(train_df: pd.DataFrame):
    logger.info("Training deep learning model...")

def train():
    logger.info("#####################")
    logger.info("Starting training process...")

    logger.info(f"Load in: {TRAIN_DATA_PATH}")
    train_df = pd.read_csv(TRAIN_DATA_PATH)

    train_dl_model(train_df)
    
    logger.info("Training complete.")

if __name__ == "__main__":
    train()
