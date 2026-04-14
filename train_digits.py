"""Train a small MNIST CNN for gate-digit recognition.

Usage: ./venv/bin/python train_digits.py
Saves models/digits.pt.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from digits import DigitCNN


DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "digits.pt"

EPOCHS = 3
BATCH = 128
LR = 1e-3


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    # Augmentation: leichte Rotation/Scale/Shift — robuster gegen Crop-Ungenauigkeit
    train_tf = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.08, 0.08),
                                scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST(str(DATA_DIR), train=True, download=True,
                              transform=train_tf)
    test_ds = datasets.MNIST(str(DATA_DIR), train=False, download=True,
                             transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False,
                             num_workers=2)

    device = "cpu"
    model = DigitCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * xb.size(0)
            total += xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
        tr_acc = correct / total
        tr_loss = loss_sum / total

        model.eval()
        t_total, t_correct = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                t_total += xb.size(0)
                t_correct += (logits.argmax(1) == yb).sum().item()
        te_acc = t_correct / t_total
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f} "
              f"train_acc={tr_acc:.4f} test_acc={te_acc:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
