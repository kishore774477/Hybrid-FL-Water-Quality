import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import copy

def local_train(model, df, target_col, epochs=5, lr=0.001):
    model = copy.deepcopy(model)
    model.train()

    X = torch.tensor(
        df.drop(columns=[target_col]).values,
        dtype=torch.float32
    )
    y = torch.tensor(
        df[target_col].values,
        dtype=torch.long # Changed to long for CrossEntropyLoss
    )
    # Note: CrossEntropyLoss expects 1D target, so we do NOT unsqueeze

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=32,
        shuffle=True
    )

    criterion = nn.CrossEntropyLoss() # Changed to CrossEntropyLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    return model.state_dict()
