import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_pos_weight(y_train_np: np.ndarray, device: torch.device):
    # y_train_np expected 0/1
    pos = float((y_train_np == 1).sum())
    neg = float((y_train_np == 0).sum())
    w = neg / max(pos, 1.0)
    return torch.tensor([w], dtype=torch.float32, device=device)

@torch.no_grad()
def predict_proba(model, loader, device: torch.device) -> np.ndarray:
    model.eval()
    probs = []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            xb = batch[0]
        else:
            xb = batch
        xb = xb.to(device)
        logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        probs.append(p)
    return np.concatenate(probs, axis=0) if probs else np.array([], dtype=np.float32)

def train_one_epoch(model, loader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        n += bs
    return total_loss / max(n, 1)

def fit_with_early_stopping(
    model,
    train_loader,
    val_loader,
    y_val_np: np.ndarray,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    max_epochs: int = 50,
):
    # loss with pos_weight for imbalance
    pos_weight = make_pos_weight(y_train_np=_extract_labels_from_loader(train_loader), device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_ap = -1.0
    pat_left = patience

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        val_proba = predict_proba(model, val_loader, device)
        val_auc = roc_auc_score(y_val_np, val_proba) if len(np.unique(y_val_np)) > 1 else float("nan")
        val_ap  = average_precision_score(y_val_np, val_proba) if len(np.unique(y_val_np)) > 1 else float("nan")

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.5f} | val_auc={val_auc:.5f} | val_ap={val_ap:.5f}")

        if val_ap > best_val_ap + 1e-5:
            best_val_ap = val_ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            pat_left = patience
        else:
            pat_left -= 1
            if pat_left <= 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return {"best_val_ap": best_val_ap}

def _extract_labels_from_loader(loader) -> np.ndarray:
    # used only to compute pos_weight consistently
    ys = []
    for _, yb in loader:
        ys.append(yb.detach().cpu().numpy().reshape(-1))
    return np.concatenate(ys, axis=0) if ys else np.array([], dtype=np.float32)
