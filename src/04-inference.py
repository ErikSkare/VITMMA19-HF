# Inference script
# This script runs the model on new, unseen data.
import pandas as pd
import joblib
import torch
from sklearn.pipeline import Pipeline
from torch.nn.utils.rnn import pad_sequence
from utils import setup_logger, compute_baseline_features, tokenize_dataset, to_labels
from models import build_cnn_model
from config import BASELINE_MODEL_PATH, FINAL_MODEL_PATH

logger = setup_logger()

paragraph1 = """
A Felek az Egyedi Szerződésben határozzák meg képzés jelen ÁSZF-hez képest egyedi
feltételeit, így különösen a Képzési Szerződéssel érintett képzést, a képzés munkarendjét, a
képzésért felelős kart, a képzés helyét, a költségtérítés első tanévre megállapított mértékét,
"""
paragraph2 = """
Az egy tanévre meghatározott költségtérítés két részletben esedékes. A fix díj 50%-a
tanulmányi félévenként, míg a felvett kreditek után fizetendő díj szintén tanulmányi félévente, a
regisztrációs időszakban ténylegesen felvett kreditek alapján fizetendő, az Egyetem által
kibocsátott számla alapján.
"""

def predict_baseline(paragraphs: list[str]):
    logger.info("> Predicting with baseline model")

    model: Pipeline = joblib.load(BASELINE_MODEL_PATH)

    X_test = compute_baseline_features(pd.Series(paragraphs))
    preds = model.predict(X_test)

    for paragraph, pred in zip(paragraphs, preds):
        logger.info("-----------------------------")
        logger.info(f"Paragraph: {paragraph}")
        logger.info(f"Prediction: {int(pred)}")
        logger.info("-----------------------------")

@torch.no_grad()
def predict_final(paragraphs: list[str]):
    logger.info("> Predicting with final model")

    state_dict = torch.load(FINAL_MODEL_PATH)
    model = build_cnn_model(embedding_dim=100)
    model.load_state_dict(state_dict)
    model.eval()

    X_test = tokenize_dataset(pd.Series(paragraphs))
    X_test = pad_sequence(X_test, batch_first=True)
    X_test = X_test.permute(0, 2, 1)
    preds = to_labels(model(X_test)) + 1

    for paragraph, pred in zip(paragraphs, preds):
        logger.info("-----------------------------")
        logger.info(f"Paragraph: {paragraph}")
        logger.info(f"Prediction: {pred}")
        logger.info("-----------------------------")

def predict():
    logger.info("#####################")
    logger.info("Running inference...")
    logger.info("#####################")

    predict_baseline([paragraph1, paragraph2])
    predict_final([paragraph1, paragraph2])

if __name__ == "__main__":
    predict()
