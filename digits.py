from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    """Lightweight CNN for MNIST-style digit recognition.

    Architecture: two conv-pool blocks followed by two fully connected layers.
    Input: single-channel 28×28 image normalized to MNIST statistics.
    Output: 10 logits (digits 0–9).
    """

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
    """Converts a black-on-white digit crop to MNIST-style 28×28 float32.

    Applies Otsu thresholding and inverts so the digit is white on black,
    isolates the largest connected component to remove border noise, fits
    it into a 20×20 box, and centers it on a 28×28 canvas — matching the
    MNIST preprocessing convention.

    Args:
        crop_bgr_or_gray: BGR or grayscale crop around the digit.

    Returns:
        Normalized float32 array of shape (28, 28).
    """
    if crop_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(crop_bgr_or_gray, cv2.COLOR_BGR2GRAY)  # drop color channels
    else:
        gray = crop_bgr_or_gray
    # Otsu threshold + invert: MNIST convention is white digit on black background
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # isolate the largest connected component to remove border noise
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
    # fit into a 20×20 box, then center-pad to 28×28 (MNIST convention)
    h, w = bw.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((28, 28), dtype=np.float32)
    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(bw, (new_w, new_h), interpolation=cv2.INTER_AREA)  # scale digit
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    img = canvas.astype(np.float32) / 255.0
    img = (img - 0.1307) / 0.3081  # MNIST mean/std normalization
    return img


def preprocess_canvas(crop_bgr_or_gray: np.ndarray) -> np.ndarray:
    """Returns a 28×28 uint8 version of preprocess() output for visualization.

    Args:
        crop_bgr_or_gray: BGR or grayscale crop around the digit.

    Returns:
        uint8 array of shape (28, 28) in the MNIST visual style.
    """
    img = preprocess(crop_bgr_or_gray)
    out = (img * 0.3081 + 0.1307) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


class DigitClassifier:
    """Wraps DigitCNN for inference on raw image crops.

    Attributes:
        model: Loaded DigitCNN in eval mode.
        device: Torch device string used for inference.
    """

    def __init__(self, model: DigitCNN, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.device = device

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "DigitClassifier":
        """Loads a saved DigitCNN state dict from disk.

        Args:
            path: Path to the .pt file.
            device: Torch device to load weights onto.

        Returns:
            Initialized DigitClassifier.
        """
        model = DigitCNN()
        state = torch.load(str(path), map_location=device)
        model.load_state_dict(state)
        return cls(model, device)

    def predict(self, crop: np.ndarray) -> Tuple[int, float]:
        """Classifies a digit crop.

        Args:
            crop: BGR or grayscale image containing the digit.

        Returns:
            Tuple (digit, confidence) where digit is in [0, 9] and
            confidence is the softmax probability of that class.
        """
        arr = preprocess(crop)
        x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0]
        conf, idx = probs.max(0)
        return int(idx.item()), float(conf.item())
