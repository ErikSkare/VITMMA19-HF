import torch.nn as nn

def build_cnn_model(embedding_dim: int = 100):
    model = nn.Sequential(
        nn.Conv1d(in_channels=embedding_dim, out_channels=32, kernel_size=5), 
        nn.BatchNorm1d(32), 
        nn.ReLU(), 
        nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5), 
        nn.BatchNorm1d(16), 
        nn.ReLU(), 
        nn.AdaptiveMaxPool1d(4), 
        nn.Flatten(), 
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 4)
    )
    return model
