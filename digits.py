from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def preprocess(crop_bgr_or_gray: np.ndarray) -> np.ndarray:
    """Black-on-white digit crop → MNIST-style (white-on-black) 28x28 float32."""
    if crop_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(crop_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_bgr_or_gray
    # Otsu-Schwelle + Invert (MNIST: weiß=Ziffer, schwarz=Hintergrund)
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Größte zusammenhängende Komponente = Ziffer (isoliert von Rand-Noise)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        k = 1 + int(np.argmax(areas))
        x, y, w, h = (stats[k, cv2.CC_STAT_LEFT],
                      stats[k, cv2.CC_STAT_TOP],
                      stats[k, cv2.CC_STAT_WIDTH],
                      stats[k, cv2.CC_STAT_HEIGHT])
        mask = (labels[y:y + h, x:x + w] == k).astype(np.uint8) * 255
        bw = mask
    # In 20x20 einpassen, auf 28x28 zentriert padden (MNIST-Konvention)
    h, w = bw.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((28, 28), dtype=np.float32)
    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(bw, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    img = canvas.astype(np.float32) / 255.0
    img = (img - 0.1307) / 0.3081
    return img


def preprocess_canvas(crop_bgr_or_gray: np.ndarray) -> np.ndarray:
    """Wie preprocess, aber als 28x28 uint8 (MNIST-artig) zur Visualisierung."""
    img = preprocess(crop_bgr_or_gray)
    out = (img * 0.3081 + 0.1307) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


class DigitClassifier:
    def __init__(self, model: DigitCNN, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "DigitClassifier":
        model = DigitCNN()
        state = torch.load(str(path), map_location=device)
        model.load_state_dict(state)
        return cls(model, device)

    def predict(self, crop: np.ndarray) -> Tuple[int, float]:
        """Return (digit, confidence)."""
        arr = preprocess(crop)
        x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0]
        conf, idx = probs.max(0)
        return int(idx.item()), float(conf.item())
