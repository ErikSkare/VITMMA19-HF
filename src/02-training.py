# Model training script
# This script defines the model architecture and runs the training loop.
import config
import pandas as pd
from utils import setup_logger

logger = setup_logger()

# For baseline I use K-nearest neighbors with simple hand-crafted features.
def train_baseline_model(train_df: pd.DataFrame):
    logger.info('Training baseline model...')
    
def train():
    logger.info("Starting training process...")
    logger.info(f"Loaded configuration. Epochs: {config.EPOCHS}")
    
    # Simulation of training loop
    for epoch in range(1, config.EPOCHS + 1):
        logger.info(f"Epoch {epoch}/{config.EPOCHS} - Training...")
    
    logger.info("Training complete.")

if __name__ == "__main__":
    train()
