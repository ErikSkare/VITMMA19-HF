# Model evaluation script
# This script evaluates the trained model on the test set and generates metrics.
import pandas as pd
import joblib
import torch
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, mean_absolute_error
from config import TEST_PATH, BASELINE_MODEL_PATH, FINAL_MODEL_PATH
from utils import setup_logger, compute_baseline_features, tokenize_dataset, collate_fn, to_labels
from models import build_cnn_model

logger = setup_logger()

def cm_to_df(cm):
    return pd.DataFrame(cm, index=[f"true_{l}" for l in range(1, 6)], columns=[f"pred_{l}" for l in range(1, 6)])

def evaluate_baseline(df: pd.DataFrame):
    logger.info("> Evaluating baseline model")

    logger.info("Loading best baseline model")
    model: Pipeline = joblib.load(BASELINE_MODEL_PATH)

    logger.info("Calculating prediction labels")
    X_test, y_test = compute_baseline_features(df["text"]), df["rating"]
    preds = model.predict(X_test)

    logger.info("> Results")
    cm = confusion_matrix(y_test, preds)
    logger.info(f"MAE: {mean_absolute_error(y_test, preds)}")
    logger.info(f"Confusion matrix:\n {cm_to_df(cm)}")

@torch.no_grad()
def evaluate_final(df: pd.DataFrame):
    logger.info("> Evaluating final model")

    logger.info("Loading best final model")
    state_dict = torch.load(FINAL_MODEL_PATH)
    model = build_cnn_model(embedding_dim=100)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info("Tokenize test dataset")
    X_test, y_test = tokenize_dataset(df["text"]), torch.tensor(df["rating"].values - 1, dtype=torch.long)
    X_test, y_test = collate_fn(list(zip(X_test, y_test)))

    logger.info(": Results")
    preds = to_labels(model(X_test))
    cm = confusion_matrix(y_test, preds)
    logger.info(f"MAE: {mean_absolute_error(y_test, preds)}")
    logger.info(f"Confusion matrix:\n {cm_to_df(cm)}")

def evaluate():
    logger.info("#####################")
    logger.info("Evaluating model...")
    logger.info("#####################")

    test_df = pd.read_csv(TEST_PATH)
    logger.info(f": Read in: {TEST_PATH}")

    evaluate_baseline(test_df)
    evaluate_final(test_df)

if __name__ == "__main__":
    evaluate()
