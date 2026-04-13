"""
2-Stage SOTA Predict Pipeline
==============================
Stage 1 — DMC-Gate Detector: Custom YOLOv8 (best_yolo_v2.pt)
           Detects bins, filters traps, extracts 150% halo crops

Stage 2 — Contextual Observer: Fine-tuned Swin Transformer
           Classifies each crop → action_required / no_action
           with softmax confidence thresholding + Test-Time Augmentation (TTA)

Aggregation — Confidence-gated MAX rule with TTA-averaged probabilities
"""

import os
import io
import cv2
import pickle
import tempfile
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import SwinConfig, SwinForImageClassification
from ultralytics import YOLO

# ──────────────────────────────────────────────
# HYPERPARAMETERS (tune these to dial in accuracy)
# ──────────────────────────────────────────────
YOLO_CONF         = 0.25   # Raised from 0.1 to filter noisy detections
ACTION_THRESHOLD   = 0.3   # Softmax P(action) must exceed this to flag
TTA_ENABLED        = True   # Test-Time Augmentation for robustness
INFERENCE_DEVICE   = "cpu"  # Force CPU inference in deployment environments

CLASS_NAMES = {
    0: "bin_caged",
    1: "bin_elevated",
    2: "bin_ground",
    3: "trap_object"
}


def load_model():
    """Load all models from model.pkl (expected in same directory)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_pkl_path = os.path.join(current_dir, "model.pkl")

    with open(model_pkl_path, "rb") as f:
        data = pickle.load(f)

    # Stage 1: YOLO
    yolo_temp_path = os.path.join(tempfile.gettempdir(), 'temp_yolo.pt')
    with open(yolo_temp_path, 'wb') as f:
        f.write(data['yolo'])
    yolo_model = YOLO(yolo_temp_path)

    # Stage 2: Swin Transformer (air-gap safe via bundled config)
    device = torch.device(INFERENCE_DEVICE)
    vit_config = SwinConfig.from_dict(data['vit_config'])
    vit_model = SwinForImageClassification(vit_config)

    vit_bytes_io = io.BytesIO(data['vit'])
    vit_model.load_state_dict(torch.load(vit_bytes_io, map_location=device))
    vit_model.to(device)
    vit_model.eval()

    # Pre-build transforms once (not per-image)
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return {
        "yolo": yolo_model,
        "vit": vit_model,
        "device": device,
        "transform": base_transform,
    }


def _tta_transforms(pil_img):
    """
    Test-Time Augmentation: generate multiple deterministic views of the crop.
    Returns a list of PIL images to run through the ViT independently.
    The predictions are averaged for a smoother, more robust score.
    """
    views = [pil_img]                                                        # 1. Original
    views.append(pil_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT))         # 2. H-flip
    views.append(pil_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM))         # 3. V-flip

    # 4-5. Slight crops (centre 85% and 90%) — simulates scale jitter
    w, h = pil_img.size
    for ratio in (0.85, 0.90):
        dw = int(w * (1 - ratio) / 2)
        dh = int(h * (1 - ratio) / 2)
        cropped = pil_img.crop((dw, dh, w - dw, h - dh))
        views.append(cropped)

    return views


def _get_action_probability(vit_model, pil_img, transform, device, use_tta=True):
    """
    Run the ViT on a single crop and return P(action_required).
    If TTA is enabled, averages softmax probabilities across augmented views.
    """
    if use_tta:
        views = _tta_transforms(pil_img)
    else:
        views = [pil_img]

    all_probs = []
    for view in views:
        tensor = transform(view).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = vit_model(tensor).logits
            probs = F.softmax(logits, dim=1)
            action_prob = probs[0, 0].item()  # class 0 = action_required
        all_probs.append(action_prob)

    # Average across TTA views
    return sum(all_probs) / len(all_probs)


def predict(models, image_path):
    """
    Full 2-stage pipeline on a single image.
    Returns 1 (intimate DMC) or 0 (no action).
    """
    yolo_model = models["yolo"]
    vit_model  = models["vit"]
    device     = models["device"]
    transform  = models["transform"]

    # ── STAGE 1: DMC-Gate Selective Detection ──────────────────
    results = yolo_model.predict(
        source=image_path,
        conf=YOLO_CONF,
        verbose=False,
        device=INFERENCE_DEVICE,
    )
    result  = results[0]

    original_img = result.orig_img
    if original_img is None:
        original_img = cv2.imread(image_path)
    if original_img is None:
        return 0

    if result.boxes is None or len(result.boxes) == 0:
        return 0

    img_h, img_w = original_img.shape[:2]
    vit_inputs = []

    for box in result.boxes:
        class_id   = int(box.cls[0].item())
        class_name = CLASS_NAMES.get(class_id, "unknown")

        # SINK LOGIC: skip trap objects
        if class_name == "trap_object":
            continue

        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # 150% Halo (kept as-is per user request)
        pad_x = bbox_w * 1.5
        pad_y = bbox_h * 1.5

        crop_x1 = int(max(0,     x1 - pad_x))
        crop_y1 = int(max(0,     y1 - pad_y))
        crop_x2 = int(min(img_w, x2 + pad_x))
        crop_y2 = int(min(img_h, y2 + pad_y))

        crop = original_img[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size > 0:
            vit_inputs.append(crop)

    # No authorized bin → DMC not concerned
    if not vit_inputs:
        return 0

    # ── STAGE 2: ViT with TTA + Confidence Thresholding ──────
    action_probs = []
    for crop in vit_inputs:
        # BGR → RGB → PIL
        img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb).convert("RGB")

        prob = _get_action_probability(
            vit_model, pil_img, transform, device, use_tta=TTA_ENABLED
        )
        action_probs.append(prob)

    # ── AGGREGATION: Confidence-gated MAX ─────────────────────
    # Only flag if the MOST confident crop exceeds the threshold.
    # This prevents weak/noisy crops (flower pots, manmade objects)
    # from triggering a false positive.
    max_prob = max(action_probs)

    if max_prob > ACTION_THRESHOLD:
        return 1

    return 0
