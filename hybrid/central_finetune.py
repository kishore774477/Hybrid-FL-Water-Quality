import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy

def centralized_finetune(model, df, target_col, epochs=5, lr=0.0005):
    """
    Fine-tunes the global model on the provided dataframe (Centralized Simulation).
    Adapted for Multi-Class (CrossEntropyLoss).
    """
    print("  -> Fine-tuning model...")
    # Deep copy to ensure we don't mutate if failed (though here we want mutation)
    # Actually, we should just train the model passed.
    model.train()

    X = torch.tensor(
        df.drop(columns=[target_col]).values,
        dtype=torch.float32
    )
    y = torch.tensor(
        df[target_col].values,
        dtype=torch.long # Adapted for Multi-Class
    )

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=32,
        shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() # Adapted

    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    return model
