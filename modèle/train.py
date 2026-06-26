#!/usr/bin/env python3
"""
train.py – Version finale robuste (avec --num_layers)
"""

import sys, csv, json, copy, random, logging, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Modèle avec pooling moyen et LSTM empilable (num_layers paramétrable)
# ----------------------------------------------------------------------
class SignLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        pooled = torch.mean(out, dim=1)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

# ----------------------------------------------------------------------
# Chargement
# ----------------------------------------------------------------------
def load_data(data_dir: Path, batch_size: int, device: torch.device):
    X_train = torch.from_numpy(np.load(data_dir / "X_train.npy")).float().to(device)
    y_train = torch.from_numpy(np.load(data_dir / "y_train.npy")).long().to(device)
    X_val   = torch.from_numpy(np.load(data_dir / "X_val.npy")).float().to(device)
    y_val   = torch.from_numpy(np.load(data_dir / "y_val.npy")).long().to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    with open(data_dir / "config.json", "r") as f:
        cfg = json.load(f)
    log.info(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
    log.info(f"T={cfg['seq_length']}, dim={cfg['input_dim']}, classes={cfg['num_classes']}")
    return train_loader, val_loader, cfg

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_samples = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * len(X)
        total_samples += len(X)
    return total_loss / total_samples

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * len(X)
        total_correct += (preds == y).sum().item()
        total_samples += len(X)
    return total_loss / total_samples, total_correct / total_samples * 100.0

def save_training_curves(log_path, out_path, best_epoch):
    epochs, loss_train, loss_val, acc_val = [], [], [], []
    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            loss_train.append(float(row["loss_train"]))
            loss_val.append(float(row["loss_val"]))
            acc_val.append(float(row["acc_val"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
    ax1.plot(epochs, loss_train, label="Train")
    ax1.plot(epochs, loss_val, label="Val")
    ax1.axvline(best_epoch, color='r', linestyle='--')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax2.plot(epochs, acc_val, label="Val accuracy")
    ax2.axvline(best_epoch, color='r', linestyle='--')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Courbes sauvegardées : {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data_ready")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_layers", type=int, default=2, help="Nombre de couches LSTM empilées")  # ← réintégré
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--scheduler_patience", type=int, default=5)
    args = parser.parse_args()

    set_seed(SEED)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device : {device}")

    # Vérification des fichiers
    for f in ["X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy", "config.json"]:
        if not (data_dir / f).exists():
            log.error(f"Fichier manquant : {data_dir / f}")
            sys.exit(1)

    train_loader, val_loader, cfg = load_data(data_dir, args.batch_size, device)

    # Modèle avec num_layers provenant de l'argument CLI
    model = SignLSTM(
        input_dim=cfg["input_dim"],
        hidden_size=cfg["hidden_size"],   # ← depuis le checkpoint
        num_layers=cfg.get("num_layers", 2),
        num_classes=cfg["num_classes"],
        dropout=cfg["dropout"]            # ← depuis le checkpoint
    ).to(device)

    log.info(f"Paramètres : {sum(p.numel() for p in model.parameters()):,}")
    log.info(f"hidden_size={args.hidden_size}, dropout={args.dropout}, num_layers={args.num_layers}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=args.scheduler_patience, factor=0.5)

    log_path = out_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss_train", "loss_val", "acc_val", "lr"])

    best_val_loss = float("inf")
    best_epoch = 1
    no_improve = 0
    best_state = None
    model_path = out_dir / "best_model.pth"

    pbar = tqdm(range(1, args.max_epochs + 1), desc="Entraînement", unit="epoch")
    for epoch in pbar:
        loss_train = train_one_epoch(model, train_loader, criterion, optimizer, device)
        loss_val, acc_val = evaluate(model, val_loader, criterion, device)
        scheduler.step(loss_val)
        lr = optimizer.param_groups[0]["lr"]

        pbar.set_postfix(train_loss=f"{loss_train:.4f}", val_loss=f"{loss_val:.4f}", val_acc=f"{acc_val:.2f}%")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{loss_train:.4f}", f"{loss_val:.4f}", f"{acc_val:.4f}", f"{lr:.6f}"])

        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_epoch = epoch
            no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "input_dim": cfg["input_dim"],
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "num_classes": cfg["num_classes"],
                    "dropout": args.dropout,
                    "seq_length": cfg["seq_length"]
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss
            }, model_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                log.info(f"Early stopping à l'epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
        log.info(f"Meilleurs poids restaurés (epoch {best_epoch}, val_loss={best_val_loss:.4f})")

    save_training_curves(log_path, out_dir / "training_curves.png", best_epoch)
    with open(out_dir / "training_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    log.info(f"✅ Terminé. Meilleur val_loss={best_val_loss:.4f} à epoch {best_epoch}")
    log.info(f"Modèle sauvegardé : {model_path}")

if __name__ == "__main__":
    main()
