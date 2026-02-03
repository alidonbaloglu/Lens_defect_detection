import os
import json
import random
import time
from datetime import datetime
from collections import Counter
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, SequentialLR

import torchvision
from torchvision.transforms import functional as F
import torchvision.transforms as T
from torchvision.models.detection.mask_rcnn import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet101_Weights
from torch.utils.tensorboard import SummaryWriter

# Optional dependency: pycocotools is needed to decode polygons/RLE to masks
try:
    from pycocotools import mask as coco_mask
except Exception:
    coco_mask = None


# ----------------------------
# Configuration
# ----------------------------
COCO_ROOT = r"C:/Users/arge.ortak/Desktop/Lens/kabin_dataset_v1"
# Split folders (Roboflow-style): each contains images and a COCO JSON
TRAIN_SPLIT = "train"
VAL_SPLIT_NAME = "valid"  # change to "val" if that's your folder name
TEST_SPLIT = "test"
OUTPUT_DIR = os.path.join("results", "MaskRCNN")
MODEL_DIR = os.path.join("Modeller")
MODEL_PATH = os.path.join(MODEL_DIR, "MaskRCNN_COCO_kabin_v2.pt")
INPUT_MODEL_PATH = os.path.join(MODEL_DIR, "MaskRCNN_COCO_kabin_v1.pt")
# Optional extra checkpoints for ensemble evaluation
ENSEMBLE_MODEL_PATHS: List[str] = []

# Backbone config: "resnet50" or "resnet101"
BACKBONE_NAME = "resnet101"

# Training hyperparameters (Updated)
RANDOM_SEED = 42
NUM_EPOCHS = 100
BATCH_SIZE = 4
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PRINT_FREQ = 100

# Early stopping
EARLY_STOP_PATIENCE = 20
EARLY_STOP_MIN_DELTA = 0.001

# Mixed precision / optimization helpers
GRAD_CLIP_MAX_NORM = 5.0
WARMUP_EPOCHS = 5
COSINE_T0 = 10
COSINE_T_MULT = 2
COSINE_MIN_LR = 1e-7

# Optional analyses / logging
ANALYZE_DATASET = True


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


class TrainTransforms:
    """Lightweight color/blur augmentations executed on-the-fly."""

    def __init__(self) -> None:
        self.to_pil = T.ToPILImage()
        self.transforms = T.Compose(
            [
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomAdjustSharpness(sharpness_factor=2.0, p=0.5),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, image: Image.Image | torch.Tensor) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            image = self.to_pil(image)
        return self.transforms(image)


class EvalTransforms:
    """Deterministic transform for val/test splits."""

    def __init__(self) -> None:
        self.to_pil = T.ToPILImage()
        self.transforms = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, image: Image.Image | torch.Tensor) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            image = self.to_pil(image)
        return self.transforms(image)


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    boxes: List[List[float]] = []
    masks_np = masks.cpu().numpy()
    for m in masks_np:
        ys, xs = np.where(m > 0)
        if len(xs) == 0 or len(ys) == 0:
            boxes.append([0.0, 0.0, 0.0, 0.0])
            continue
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        boxes.append([x_min, y_min, x_max, y_max])
    return torch.as_tensor(boxes, dtype=torch.float32)


# ----------------------------
# COCO Dataset
# ----------------------------
class COCOSegmentationDataset(Dataset):
    """
    COCO-style instance segmentation dataset.
    - Reads images from IMAGES_ROOT
    - Reads annotations from COCO JSON (images, annotations, categories)
    - Builds per-image targets with masks, boxes, labels, area, iscrowd
    """

    def __init__(self, json_path: str, images_root: str, category_ids: Optional[List[int]] = None, transforms=None) -> None:
        if coco_mask is None:
            raise ImportError("pycocotools is required for COCO polygon/RLE decoding. Please install 'pycocotools'.")

        self.images_root = images_root
        self.transforms = transforms

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.images: List[Dict] = data.get("images", [])
        annotations: List[Dict] = data.get("annotations", [])
        categories: List[Dict] = data.get("categories", [])

        # Select categories if provided; else use all
        if category_ids is None:
            category_ids = [c["id"] for c in categories]

        # Map category_id -> contiguous label index [1..num_classes-1]
        self.category_id_to_label: Dict[int, int] = {cid: i + 1 for i, cid in enumerate(sorted(category_ids))}
        self.label_to_category_id: Dict[int, int] = {v: k for k, v in self.category_id_to_label.items()}
        self.num_classes: int = len(self.category_id_to_label) + 1  # +1 for background

        # Build image_id -> annotations list filtered by chosen categories
        self.image_id_to_anns: Dict[int, List[Dict]] = {}
        for ann in annotations:
            if ann.get("iscrowd", 0) not in [0, 1]:
                ann["iscrowd"] = 0
            if ann["category_id"] not in self.category_id_to_label:
                continue
            img_id = ann["image_id"]
            self.image_id_to_anns.setdefault(img_id, []).append(ann)

        # Keep only images that have at least one annotation (or keep all? here keep all)
        # We keep all, empty targets are allowed
        self.id_to_image: Dict[int, Dict] = {im["id"]: im for im in self.images}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_info = self.images[idx]
        img_path = os.path.join(self.images_root, img_info["file_name"])

        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(img)
        else:
            image = F.pil_to_tensor(img).float() / 255.0

        img_id = img_info["id"]
        anns = self.image_id_to_anns.get(img_id, [])

        masks_list: List[torch.Tensor] = []
        boxes_list: List[List[float]] = []
        labels_list: List[int] = []
        areas_list: List[float] = []
        iscrowd_list: List[int] = []

        width = img_info.get("width", image.shape[-1])
        height = img_info.get("height", image.shape[-2])

        for ann in anns:
            cat_id = ann["category_id"]
            label = self.category_id_to_label[cat_id]

            # bbox in COCO is [x, y, width, height]
            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            if w <= 0 or h <= 0:
                continue

            # segmentation can be polygons (list[list[float]]) or RLE
            seg = ann.get("segmentation")
            rle = None
            if isinstance(seg, list):
                # Polygon(s)
                rles = coco_mask.frPyObjects(seg, height, width)
                rle = coco_mask.merge(rles)
            elif isinstance(seg, dict) and {"counts", "size"}.issubset(seg.keys()):
                rle = seg
            else:
                # No segmentation; skip instance
                continue

            m = coco_mask.decode(rle)
            if m.ndim == 3:
                m = np.any(m, axis=2).astype(np.uint8)
            else:
                m = (m > 0).astype(np.uint8)

            masks_list.append(torch.from_numpy(m).to(torch.uint8))
            boxes_list.append([float(x), float(y), float(x + w), float(y + h)])
            labels_list.append(int(label))
            areas_list.append(float(ann.get("area", w * h)))
            iscrowd_list.append(int(ann.get("iscrowd", 0)))

        if len(masks_list) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, image.shape[-2], image.shape[-1]), dtype=torch.uint8),
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }
        else:
            masks = torch.stack(masks_list, dim=0).to(torch.uint8)
            boxes = torch.as_tensor(boxes_list, dtype=torch.float32)
            labels = torch.as_tensor(labels_list, dtype=torch.int64)
            area = torch.as_tensor(areas_list, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd_list, dtype=torch.int64)

            # filter degenerate boxes
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid]
            masks = masks[valid]
            labels = labels[valid]
            area = area[valid]
            iscrowd = iscrowd[valid]

            target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area": area,
                "iscrowd": iscrowd,
            }

        return image, target


def analyze_dataset(dataset: COCOSegmentationDataset) -> Counter:
    """Compute simple label distribution stats from stored annotations."""
    label_counts: Counter = Counter()
    for anns in dataset.image_id_to_anns.values():
        for ann in anns:
            cat_id = ann.get("category_id")
            if cat_id in dataset.category_id_to_label:
                label_idx = dataset.category_id_to_label[cat_id]
                label_counts.update([label_idx])
    print(f"Sınıf dağılımı: {dict(label_counts)}")
    return label_counts


def compute_class_weights(label_counts: Counter, num_classes: int) -> torch.Tensor:
    """Return normalized inverse-frequency weights (background index 0 untouched)."""
    weights = torch.ones(num_classes, dtype=torch.float32)
    total = sum(label_counts.values())
    if total == 0:
        return weights
    for label_idx, count in label_counts.items():
        if label_idx < len(weights):
            weights[label_idx] = total / max(count, 1)
    # Normalize to keep mean of weights ~1
    weights = weights / weights.mean()
    return weights


def build_image_sampling_weights(dataset: COCOSegmentationDataset, class_weights: torch.Tensor) -> List[float]:
    """Map per-image sampling weights using class weights."""
    weights: List[float] = []
    for img in dataset.images:
        anns = dataset.image_id_to_anns.get(img["id"], [])
        label_indices: List[int] = []
        for ann in anns:
            cid = ann.get("category_id")
            if cid in dataset.category_id_to_label:
                label_indices.append(dataset.category_id_to_label[cid])
        if len(label_indices) == 0:
            weights.append(1.0)
        else:
            label_tensor = torch.tensor(label_indices, dtype=torch.long)
            weights.append(float(class_weights[label_tensor].mean().item()))
    return weights

def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


def _find_split_paths(split_root: str) -> Tuple[str, str]:
    """Return (images_root, json_path) for a split folder.
    Tries common COCO filenames produced by tools like Roboflow.
    """
    # Candidates for JSON names
    json_candidates = [
        "_annotations.coco.json",
        "annotations.coco.json",
        "annotations.json",
    ]
    json_path = None
    for name in json_candidates:
        p = os.path.join(split_root, name)
        if os.path.isfile(p):
            json_path = p
            break
    if json_path is None:
        # Try to find any .json in split folder
        for fname in os.listdir(split_root):
            if fname.lower().endswith(".json"):
                json_path = os.path.join(split_root, fname)
                break
    if json_path is None:
        raise FileNotFoundError(f"No COCO JSON found in {split_root}")

    # Images root is the split folder itself in Roboflow exports
    images_root = split_root
    return images_root, json_path


# ----------------------------
# Training / Evaluation helpers
# ----------------------------
def get_model(num_classes: int, backbone_name: Optional[str] = None) -> torchvision.models.detection.MaskRCNN:
    backbone_to_use = (backbone_name or BACKBONE_NAME).lower()
    if backbone_to_use == "resnet101":
        backbone = resnet_fpn_backbone(
            backbone_name="resnet101",
            weights=ResNet101_Weights.DEFAULT,
            trainable_layers=4,
        )
        model = torchvision.models.detection.MaskRCNN(
            backbone=backbone,
            num_classes=num_classes,
            box_detections_per_img=100,
            box_score_thresh=0.05,
        )
    else:
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layers = 256
        model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            in_features_mask, hidden_layers, num_classes
        )

    # Add dropout regularization in ROI head if supported
    if hasattr(model.roi_heads, "box_head") and hasattr(model.roi_heads.box_head, "fc6"):
        model.roi_heads.box_head.fc6 = nn.Sequential(
            model.roi_heads.box_head.fc6,
            nn.Dropout(p=0.2),
        )

    return model


@torch.no_grad()
def evaluate_pixel_f1(model, data_loader, device) -> float:
    model.eval()
    f1_scores: List[float] = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            if len(out.get("masks", [])) == 0:
                pred_binary = np.zeros((images[0].shape[-2], images[0].shape[-1]), dtype=np.uint8)
            else:
                scores = out["scores"].detach().cpu().numpy()
                masks = out["masks"].detach().cpu().numpy()
                keep = scores >= 0.5
                if keep.sum() == 0:
                    pred_binary = np.zeros(masks.shape[-2:], dtype=np.uint8)
                else:
                    masks = masks[keep, 0]
                    pred_binary = (np.any(masks >= 0.5, axis=0)).astype(np.uint8)

            gt_masks = tgt.get("masks")
            if gt_masks is None or gt_masks.numel() == 0:
                gt_binary = np.zeros_like(pred_binary, dtype=np.uint8)
            else:
                gt_binary = (gt_masks.max(dim=0).values.detach().cpu().numpy() > 0).astype(np.uint8)

            tp = np.logical_and(pred_binary == 1, gt_binary == 1).sum()
            fp = np.logical_and(pred_binary == 1, gt_binary == 0).sum()
            fn = np.logical_and(pred_binary == 0, gt_binary == 1).sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1_scores.append(float(f1))

    return float(np.mean(f1_scores)) if len(f1_scores) > 0 else 0.0


@torch.no_grad()
def evaluate_detailed_metrics(model, data_loader, device, category_id_to_label: Optional[Dict[int, int]] = None) -> Dict[str, Dict[str, float]]:
    model.eval()
    eps = 1e-8

    # Accumulators for overall pixel metrics
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    overall_union = 0

    # Per-class accumulators (label index as keys, starting from 1)
    per_class_tp: Dict[int, int] = {}
    per_class_fp: Dict[int, int] = {}
    per_class_fn: Dict[int, int] = {}
    per_class_union: Dict[int, int] = {}

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            # Pred combined
            if len(out.get("masks", [])) == 0:
                pred_binary = np.zeros((images[0].shape[-2], images[0].shape[-1]), dtype=np.uint8)
                pred_labels = []
                pred_masks = []
            else:
                scores = out["scores"].detach().cpu().numpy()
                masks = out["masks"].detach().cpu().numpy()  # [N,1,H,W]
                labels = out["labels"].detach().cpu().numpy().tolist()
                keep = scores >= 0.5
                masks = masks[keep, 0]
                pred_labels = [int(l) for l, k in zip(labels, keep.tolist()) if k]
                pred_masks = [(m >= 0.5).astype(np.uint8) for m in masks]
                if len(pred_masks) == 0:
                    pred_binary = np.zeros(masks.shape[-2:], dtype=np.uint8)
                else:
                    pred_binary = (np.any(np.stack(pred_masks, axis=0) > 0, axis=0)).astype(np.uint8)

            # GT combined
            gt_masks_t = tgt.get("masks")
            gt_labels_t = tgt.get("labels")
            if gt_masks_t is None or gt_masks_t.numel() == 0:
                gt_binary = np.zeros_like(pred_binary, dtype=np.uint8)
                gt_labels = []
                gt_masks = []
            else:
                gt_masks = [m.detach().cpu().numpy().astype(np.uint8) for m in gt_masks_t]
                gt_binary = (np.any(np.stack(gt_masks, axis=0) > 0, axis=0)).astype(np.uint8)
                gt_labels = gt_labels_t.detach().cpu().numpy().tolist()

            # Overall pixel stats
            tp = np.logical_and(pred_binary == 1, gt_binary == 1).sum()
            fp = np.logical_and(pred_binary == 1, gt_binary == 0).sum()
            fn = np.logical_and(pred_binary == 0, gt_binary == 1).sum()
            union = np.logical_or(pred_binary == 1, gt_binary == 1).sum()
            overall_tp += int(tp)
            overall_fp += int(fp)
            overall_fn += int(fn)
            overall_union += int(union)

            # Per-class pixel stats: build per-class union masks
            # labels are contiguous starting from 1 per dataset construction
            if len(gt_labels) > 0:
                unique_labels = sorted(list(set(gt_labels + pred_labels)))
            else:
                unique_labels = sorted(list(set(pred_labels)))

            for lbl in unique_labels:
                # GT per-class
                gt_c = None
                if len(gt_masks) > 0:
                    gt_c = np.zeros_like(gt_binary, dtype=np.uint8)
                    for m, l in zip(gt_masks, gt_labels):
                        if l == lbl:
                            gt_c = np.logical_or(gt_c, m > 0)
                    gt_c = gt_c.astype(np.uint8)
                else:
                    gt_c = np.zeros_like(gt_binary, dtype=np.uint8)

                # Pred per-class
                if len(pred_masks) > 0:
                    pr_c = np.zeros_like(pred_binary, dtype=np.uint8)
                    for m, l in zip(pred_masks, pred_labels):
                        if l == lbl:
                            pr_c = np.logical_or(pr_c, m > 0)
                    pr_c = pr_c.astype(np.uint8)
                else:
                    pr_c = np.zeros_like(pred_binary, dtype=np.uint8)

                tp_c = np.logical_and(pr_c == 1, gt_c == 1).sum()
                fp_c = np.logical_and(pr_c == 1, gt_c == 0).sum()
                fn_c = np.logical_and(pr_c == 0, gt_c == 1).sum()
                union_c = np.logical_or(pr_c == 1, gt_c == 1).sum()

                per_class_tp[lbl] = per_class_tp.get(lbl, 0) + int(tp_c)
                per_class_fp[lbl] = per_class_fp.get(lbl, 0) + int(fp_c)
                per_class_fn[lbl] = per_class_fn.get(lbl, 0) + int(fn_c)
                per_class_union[lbl] = per_class_union.get(lbl, 0) + int(union_c)

    # Compute overall metrics
    overall_precision = overall_tp / (overall_tp + overall_fp + eps)
    overall_recall = overall_tp / (overall_tp + overall_fn + eps)
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall + eps)
    overall_iou = overall_tp / (overall_union + eps)

    # Per-class metrics
    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for lbl, tp_c in per_class_tp.items():
        fp_c = per_class_fp.get(lbl, 0)
        fn_c = per_class_fn.get(lbl, 0)
        union_c = per_class_union.get(lbl, 0)
        prec_c = tp_c / (tp_c + fp_c + eps)
        rec_c = tp_c / (tp_c + fn_c + eps)
        f1_c = 2 * prec_c * rec_c / (prec_c + rec_c + eps)
        iou_c = tp_c / (union_c + eps)
        key = f"class_{lbl}"
        per_class_metrics[key] = {
            "precision": float(prec_c),
            "recall": float(rec_c),
            "f1": float(f1_c),
            "iou": float(iou_c),
        }

    results = {
        "overall": {
            "precision": float(overall_precision),
            "recall": float(overall_recall),
            "f1": float(overall_f1),
            "iou": float(overall_iou),
        },
        "per_class": per_class_metrics,
    }
    return results


def _binary_mask_from_output(output: Dict[str, torch.Tensor], height: int, width: int, score_thresh: float = 0.5) -> torch.Tensor:
    """Helper to convert model outputs to a combined binary mask (torch tensor on CPU)."""
    masks = output.get("masks")
    scores = output.get("scores")

    if masks is None or len(masks) == 0:
        return torch.zeros((height, width), dtype=torch.uint8)

    if scores is not None:
        keep = scores.detach().cpu().numpy() >= score_thresh
    else:
        keep = np.ones(len(masks), dtype=bool)

    if keep.sum() == 0:
        return torch.zeros((height, width), dtype=torch.uint8)

    masks = masks[keep]
    if masks.dim() == 4:
        masks = masks[:, 0]
    binary = torch.any(masks.detach().cpu() >= 0.5, dim=0).to(torch.uint8)
    return binary


@torch.no_grad()
def evaluate_with_tta(model, data_loader, device) -> Dict[str, float]:
    """Perform simple TTA (identity + flips) and return aggregated pixel metrics."""
    model.eval()
    eps = 1e-8
    overall_tp = overall_fp = overall_fn = overall_union = 0

    def _combine_predictions(pred_masks: List[np.ndarray], template_shape: Tuple[int, int]) -> np.ndarray:
        if len(pred_masks) == 0:
            return np.zeros(template_shape, dtype=np.uint8)
        stacked = np.stack(pred_masks, axis=0)
        return np.any(stacked > 0, axis=0).astype(np.uint8)

    tta_ops = [
        ("identity", lambda x: x, lambda y: y),
        ("hflip", lambda x: torch.flip(x, dims=[-1]), lambda y: torch.flip(y, dims=[-1])),
        ("vflip", lambda x: torch.flip(x, dims=[-2]), lambda y: torch.flip(y, dims=[-2])),
    ]

    for images, targets in data_loader:
        for image, target in zip(images, targets):
            pred_masks_binary: List[np.ndarray] = []
            for _, forward_fn, inverse_fn in tta_ops:
                aug_image = forward_fn(image.to(device))
                outputs = model([aug_image])[0]
                binary = _binary_mask_from_output(outputs, image.shape[-2], image.shape[-1])
                binary = inverse_fn(binary)
                pred_masks_binary.append(binary.numpy().astype(np.uint8))

            pred_binary = _combine_predictions(pred_masks_binary, (image.shape[-2], image.shape[-1]))

            gt_masks = target.get("masks")
            if gt_masks is None or gt_masks.numel() == 0:
                gt_binary = np.zeros_like(pred_binary, dtype=np.uint8)
            else:
                gt_binary = (gt_masks.max(dim=0).values.detach().cpu().numpy() > 0).astype(np.uint8)

            tp = np.logical_and(pred_binary == 1, gt_binary == 1).sum()
            fp = np.logical_and(pred_binary == 1, gt_binary == 0).sum()
            fn = np.logical_and(pred_binary == 0, gt_binary == 1).sum()
            union = np.logical_or(pred_binary == 1, gt_binary == 1).sum()

            overall_tp += int(tp)
            overall_fp += int(fp)
            overall_fn += int(fn)
            overall_union += int(union)

    precision = overall_tp / (overall_tp + overall_fp + eps)
    recall = overall_tp / (overall_tp + overall_fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = overall_tp / (overall_union + eps)

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "iou": float(iou)}


@torch.no_grad()
def evaluate_model_ensemble(model_paths: List[str], num_classes: int, data_loader, device) -> Optional[Dict[str, float]]:
    """Evaluate an ensemble of checkpoints by combining their masks."""
    valid_paths = [p for p in model_paths if os.path.isfile(p)]
    if len(valid_paths) == 0:
        return None

    ensemble_models = []
    for path in valid_paths:
        try:
            ckpt = torch.load(path, map_location=device)
            n_cls = ckpt.get("num_classes", num_classes)
            backbone_name = ckpt.get("backbone", BACKBONE_NAME)
            mdl = get_model(n_cls, backbone_name=backbone_name)
            mdl.load_state_dict(ckpt["model_state"], strict=False)
            mdl.to(device)
            mdl.eval()
            ensemble_models.append(mdl)
        except Exception as exc:
            print(f"Ensemble modeli yüklenirken hata ({path}): {exc}")

    if len(ensemble_models) == 0:
        return None

    eps = 1e-8
    overall_tp = overall_fp = overall_fn = overall_union = 0

    for images, targets in data_loader:
        for image, target in zip(images, targets):
            image_device = image.to(device)
            pred_masks_binary: List[np.ndarray] = []
            for mdl in ensemble_models:
                output = mdl([image_device])[0]
                binary = _binary_mask_from_output(output, image.shape[-2], image.shape[-1])
                pred_masks_binary.append(binary.numpy().astype(np.uint8))

            if len(pred_masks_binary) == 0:
                pred_binary = np.zeros((image.shape[-2], image.shape[-1]), dtype=np.uint8)
            else:
                pred_binary = np.any(np.stack(pred_masks_binary, axis=0), axis=0).astype(np.uint8)

            gt_masks = target.get("masks")
            if gt_masks is None or gt_masks.numel() == 0:
                gt_binary = np.zeros_like(pred_binary, dtype=np.uint8)
            else:
                gt_binary = (gt_masks.max(dim=0).values.detach().cpu().numpy() > 0).astype(np.uint8)

            tp = np.logical_and(pred_binary == 1, gt_binary == 1).sum()
            fp = np.logical_and(pred_binary == 1, gt_binary == 0).sum()
            fn = np.logical_and(pred_binary == 0, gt_binary == 1).sum()
            union = np.logical_or(pred_binary == 1, gt_binary == 1).sum()

            overall_tp += int(tp)
            overall_fp += int(fp)
            overall_fn += int(fn)
            overall_union += int(union)

    # Free GPU memory
    for mdl in ensemble_models:
        mdl.to("cpu")
        del mdl

    precision = overall_tp / (overall_tp + overall_fp + eps)
    recall = overall_tp / (overall_tp + overall_fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = overall_tp / (overall_union + eps)

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "iou": float(iou)}

def train_one_epoch(model, optimizer, data_loader, device, epoch: int, scaler: Optional[GradScaler] = None) -> Dict[str, float]:
    model.train()
    epoch_losses = []
    epoch_loss_dict = {}
    use_amp = scaler is not None and device.type == "cuda"

    for step, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with autocast(enabled=True):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            optimizer.step()

        # Accumulate losses for epoch summary
        epoch_losses.append(float(losses.item()))
        for k, v in loss_dict.items():
            if k not in epoch_loss_dict:
                epoch_loss_dict[k] = []
            epoch_loss_dict[k].append(float(v.detach().item()))

        if (step + 1) % PRINT_FREQ == 0:
            loss_vals = {k: float(v.detach().item()) for k, v in loss_dict.items()}
            print(f"Epoch {epoch+1} | Step {step+1}/{len(data_loader)} | Loss: {float(losses.item()):.4f} | {loss_vals}")
    
    # Return average losses for this epoch
    avg_losses = {k: np.mean(v) for k, v in epoch_loss_dict.items()}
    avg_losses['total_loss'] = np.mean(epoch_losses)
    return avg_losses


def save_training_results(training_info: Dict, output_dir: str) -> None:
    """Save training results to both JSON and TXT files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results as JSON
    json_path = os.path.join(output_dir, f"training_results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)
    
    # Save human-readable summary as TXT
    txt_path = os.path.join(output_dir, f"training_summary_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MASK R-CNN EĞİTİM SONUÇLARI\n")
        f.write("=" * 80 + "\n\n")
        
        # Training configuration
        f.write("EĞİTİM KONFİGÜRASYONU:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Veri Seti: {training_info['config']['dataset_path']}\n")
        f.write(f"Backbone: {training_info['config']['backbone']}\n")
        f.write(f"Epoch Sayısı: {training_info['config']['num_epochs']}\n")
        f.write(f"Batch Size: {training_info['config']['batch_size']}\n")
        f.write(f"Learning Rate: {training_info['config']['learning_rate']}\n")
        f.write(f"Weight Decay: {training_info['config']['weight_decay']}\n")
        f.write(f"Sınıf Sayısı: {training_info['config']['num_classes']}\n")
        f.write(f"Train Örnek Sayısı: {training_info['config']['train_samples']}\n")
        f.write(f"Validation Örnek Sayısı: {training_info['config']['val_samples']}\n")
        if training_info['config']['test_samples'] > 0:
            f.write(f"Test Örnek Sayısı: {training_info['config']['test_samples']}\n")
        # Extra config
        if 'random_seed' in training_info['config']:
            f.write(f"Seed: {training_info['config']['random_seed']}\n")
        if 'device' in training_info['config']:
            f.write(f"Cihaz: {training_info['config']['device']}\n")
        if 'optimizer' in training_info['config']:
            opt = training_info['config']['optimizer']
            opt_line = f"Optimizer: {opt.get('name','-')} (lr={opt.get('lr','-')}, weight_decay={opt.get('weight_decay','-')}"
            if 'betas' in opt and opt['betas'] is not None:
                opt_line += f", betas={tuple(opt['betas'])}"
            if 'momentum' in opt and opt['momentum'] is not None:
                opt_line += f", momentum={opt['momentum']}"
            opt_line += ")\n"
            f.write(opt_line)
        if 'scheduler' in training_info['config']:
            sch = training_info['config']['scheduler']
            sch_line = f"LR Scheduler: {sch.get('name','-')}"
            # Print known params if present
            extras = []
            for key in ['mode', 'factor', 'patience', 'min_lr', 'T_max', 'gamma', 'step_size']:
                if key in sch and sch[key] is not None:
                    extras.append(f"{key}={sch[key]}")
            if len(extras) > 0:
                sch_line += " (" + ", ".join(extras) + ")"
            f.write(sch_line + "\n")
        if 'framework' in training_info['config']:
            fw = training_info['config']['framework']
            f.write(f"PyTorch: {fw.get('torch','-')}, Torchvision: {fw.get('torchvision','-')}\n")
        f.write("\n")
        
        # Training duration
        f.write("EĞİTİM SÜRESİ:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Başlangıç Zamanı: {training_info['training_time']['start_time']}\n")
        f.write(f"Bitiş Zamanı: {training_info['training_time']['end_time']}\n")
        f.write(f"Toplam Süre: {training_info['training_time']['total_duration']:.2f} dakika\n")
        f.write(f"Epoch Başına Ortalama: {training_info['training_time']['avg_epoch_time']:.2f} dakika\n")
        f.write("\n")
        
        # Training progress
        f.write("EĞİTİM İLERLEMESİ:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Tamamlanan Epoch: {training_info['training_progress']['completed_epochs']}\n")
        f.write(f"En İyi Validation F1: {training_info['training_progress']['best_val_f1']:.4f}\n")
        f.write(f"En İyi Epoch: {training_info['training_progress']['best_epoch']}\n")
        f.write(f"Early Stopping: {'Evet' if training_info['training_progress']['early_stopped'] else 'Hayır'}\n")
        f.write("\n")
        
        # Final metrics
        f.write("FİNAL METRİKLER:\n")
        f.write("-" * 40 + "\n")
        f.write("VALIDATION SET:\n")
        val_metrics = training_info['final_metrics'].get('validation', {})
        if val_metrics:
            f.write(f"  Precision: {val_metrics['precision']:.4f}\n")
            f.write(f"  Recall: {val_metrics['recall']:.4f}\n")
            f.write(f"  F1-Score: {val_metrics['f1']:.4f}\n")
            f.write(f"  IoU: {val_metrics['iou']:.4f}\n")
        else:
            f.write("  Ölçüm bulunamadı.\n")
        val_tta_metrics = training_info['final_metrics'].get('validation_tta')
        if val_tta_metrics:
            f.write(f"  (TTA) Precision: {val_tta_metrics['precision']:.4f}\n")
            f.write(f"  (TTA) Recall: {val_tta_metrics['recall']:.4f}\n")
            f.write(f"  (TTA) F1-Score: {val_tta_metrics['f1']:.4f}\n")
            f.write(f"  (TTA) IoU: {val_tta_metrics['iou']:.4f}\n")
        val_ensemble_metrics = training_info['final_metrics'].get('validation_ensemble')
        if val_ensemble_metrics:
            f.write(f"  (Ensemble) Precision: {val_ensemble_metrics['precision']:.4f}\n")
            f.write(f"  (Ensemble) Recall: {val_ensemble_metrics['recall']:.4f}\n")
            f.write(f"  (Ensemble) F1-Score: {val_ensemble_metrics['f1']:.4f}\n")
            f.write(f"  (Ensemble) IoU: {val_ensemble_metrics['iou']:.4f}\n")
        
        if 'test' in training_info['final_metrics']:
            f.write("\nTEST SET:\n")
            test_metrics = training_info['final_metrics']['test']
            f.write(f"  Precision: {test_metrics['precision']:.4f}\n")
            f.write(f"  Recall: {test_metrics['recall']:.4f}\n")
            f.write(f"  F1-Score: {test_metrics['f1']:.4f}\n")
            f.write(f"  IoU: {test_metrics['iou']:.4f}\n")
        if 'test_tta' in training_info['final_metrics']:
            test_tta_metrics = training_info['final_metrics']['test_tta']
            f.write(f"  (TTA) Precision: {test_tta_metrics['precision']:.4f}\n")
            f.write(f"  (TTA) Recall: {test_tta_metrics['recall']:.4f}\n")
            f.write(f"  (TTA) F1-Score: {test_tta_metrics['f1']:.4f}\n")
            f.write(f"  (TTA) IoU: {test_tta_metrics['iou']:.4f}\n")
        if 'test_ensemble' in training_info['final_metrics']:
            test_ens_metrics = training_info['final_metrics']['test_ensemble']
            f.write(f"  (Ensemble) Precision: {test_ens_metrics['precision']:.4f}\n")
            f.write(f"  (Ensemble) Recall: {test_ens_metrics['recall']:.4f}\n")
            f.write(f"  (Ensemble) F1-Score: {test_ens_metrics['f1']:.4f}\n")
            f.write(f"  (Ensemble) IoU: {test_ens_metrics['iou']:.4f}\n")
        
        # Loss history
        f.write("\nLOSS GEÇMİŞİ (Son 5 Epoch):\n")
        f.write("-" * 40 + "\n")
        loss_history = training_info['loss_history'][-5:]  # Son 5 epoch
        for i, epoch_loss in enumerate(loss_history):
            epoch_num = len(training_info['loss_history']) - 5 + i + 1
            f.write(f"Epoch {epoch_num}: Total Loss = {epoch_loss['total_loss']:.4f}\n")
            for loss_name, loss_value in epoch_loss.items():
                if loss_name != 'total_loss':
                    f.write(f"  {loss_name}: {loss_value:.4f}\n")
            f.write("\n")
        
        # Model file info
        f.write("MODEL DOSYASI:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Kayıt Yeri: {training_info['model_info']['model_path']}\n")
        f.write(f"Dosya Boyutu: {training_info['model_info']['file_size_mb']:.2f} MB\n")
        # Model meta
        if 'architecture' in training_info['model_info']:
            f.write(f"Mimari: {training_info['model_info']['architecture']}\n")
        if 'total_params' in training_info['model_info']:
            f.write(f"Toplam Parametre: {training_info['model_info']['total_params']:,}\n")
        if 'trainable_params' in training_info['model_info']:
            f.write(f"Eğitilebilir Parametre: {training_info['model_info']['trainable_params']:,}\n")
        if 'non_trainable_params' in training_info['model_info']:
            f.write(f"Eğitilemez Parametre: {training_info['model_info']['non_trainable_params']:,}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Eğitim tamamlandı!\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nEğitim sonuçları kaydedildi:")
    print(f"  JSON: {json_path}")
    print(f"  TXT: {txt_path}")


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    set_seed(RANDOM_SEED)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Training start time
    training_start_time = datetime.now()
    start_timestamp = time.time()

    # Prepare split paths
    train_root = os.path.join(COCO_ROOT, TRAIN_SPLIT)
    val_root = os.path.join(COCO_ROOT, VAL_SPLIT_NAME)
    test_root = os.path.join(COCO_ROOT, TEST_SPLIT)

    train_images_root, train_json = _find_split_paths(train_root)
    val_images_root, val_json = _find_split_paths(val_root)
    # Test split is optional; load if exists
    test_images_root, test_json = None, None
    try:
        test_images_root, test_json = _find_split_paths(test_root)
    except Exception:
        pass

    # Datasets + transforms
    train_transforms = TrainTransforms()
    eval_transforms = EvalTransforms()

    train_dataset = COCOSegmentationDataset(
        train_json,
        train_images_root,
        transforms=train_transforms
    )
    # Reuse same category mapping for val/test to keep labels aligned
    category_ids = list(train_dataset.category_id_to_label.keys())
    val_dataset = COCOSegmentationDataset(val_json, val_images_root, category_ids=category_ids, transforms=eval_transforms)
    test_dataset = None
    if test_json is not None:
        test_dataset = COCOSegmentationDataset(test_json, test_images_root, category_ids=category_ids, transforms=eval_transforms)
    num_classes = train_dataset.num_classes

    sampler = None
    class_weights = None
    label_counts = Counter()
    if ANALYZE_DATASET:
        label_counts = analyze_dataset(train_dataset)
        if len(label_counts) > 0:
            class_weights = compute_class_weights(label_counts, num_classes)
            print(f"Hesaplanan sınıf ağırlıkları: {class_weights.tolist()}")
            image_weights = build_image_sampling_weights(train_dataset, class_weights)
            sampler = WeightedRandomSampler(weights=image_weights, num_samples=len(image_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
        )

    # Model
    model = get_model(num_classes)
    model.to(device)

    # ✅ YENİ KOD - ÖNCEDEN EĞİTİLMİŞ AĞIRLIKLARI YÜKLEME (Fine-Tuning)
    if os.path.isfile(INPUT_MODEL_PATH):
        print(f"\nÖnceden eğitilmiş model {INPUT_MODEL_PATH} yükleniyor...")
        try:
            checkpoint = torch.load(INPUT_MODEL_PATH, map_location=device)
            # Model mimarisi uyuşmazsa sadece ortak katmanları yükle
            # veya sadece state_dict'i yükle
            model.load_state_dict(checkpoint["model_state"],strict=False)

            # Eğitime kalınan yerden devam etmek için isteğe bağlı olarak epoch ve F1 değerlerini alabiliriz
            start_epoch = checkpoint.get("epoch", 0)
            best_f1 = checkpoint.get("best_val_pixel_f1", -1.0)
            # Transfer öğrenme yapıldığı için başlangıç değerlerini sıfırlamak daha mantıklıdır.
            #start_epoch = 0 
            #best_f1 = -1.0
            
            print(f"Model başarıyla yüklendi. (Son Epoch: {start_epoch}, En İyi Val F1: {best_f1:.4f})")
            #print(f"Modelin sadece uyumlu ağırlıkları başarıyla yüklendi. (Son katmanlar yeniden eğitilecek)")
        except Exception as e:
            print(f"UYARI: Model yüklenirken bir hata oluştu: {e}")
            print("Sıfırdan eğitime başlanıyor.")
            start_epoch = 0 # Hata durumunda sıfırdan başla
            best_f1 = -1.0
    else:
        print("Kayıtlı model dosyası bulunamadı. Sıfırdan eğitime başlanıyor.")
        start_epoch = 0
        best_f1 = -1.0

    # ✅ YENİ KOD - FAZ 1 Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=WARMUP_EPOCHS,
    )
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=COSINE_T0,
        T_mult=COSINE_T_MULT,
        eta_min=COSINE_MIN_LR,
    )
    lr_sched = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS],
    )
    scaler = GradScaler() if device.type == "cuda" else None
    run_name = f"maskrcnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    # Training tracking variables
    # Bu kısmı yukarıdaki checkpoint'ten gelen değerlerle güncelleyeceğiz
    # best_f1 = -1.0  # --> Artık yukarıda belirleniyor
    best_epoch = start_epoch # İlk eğitim bittiyse, en iyi epoch da buradan başlar
    epochs_no_improve = 0
    loss_history = []
    val_f1_history = []

    print(f"\nEğitim başlıyor...")
    print(f"Train örnek sayısı: {len(train_dataset)}")
    print(f"Validation örnek sayısı: {len(val_dataset)}")
    if test_dataset is not None:
        print(f"Test örnek sayısı: {len(test_dataset)}")
    print(f"Sınıf sayısı: {num_classes}")
    print(f"Backbone: {BACKBONE_NAME}")
    print("-" * 50)

    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start = time.time()
        
        # Train one epoch and get loss information
        epoch_losses = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler=scaler)
        loss_history.append(epoch_losses)

        # ✅ YENİ KOD - FAZ 1
        # Validation
        val_f1 = evaluate_pixel_f1(model, val_loader, device)
        val_f1_history.append(val_f1)

        current_lr = optimizer.param_groups[0].get("lr", LEARNING_RATE)
        if writer is not None:
            writer.add_scalar("Loss/train", epoch_losses.get("total_loss", 0.0), epoch + 1)
            writer.add_scalar("F1/validation", val_f1, epoch + 1)
            writer.add_scalar("LR", current_lr, epoch + 1)
            for loss_name, loss_value in epoch_losses.items():
                if loss_name != "total_loss":
                    writer.add_scalar(f"Loss/{loss_name}", loss_value, epoch + 1)
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Val Pixel-F1: {val_f1:.4f} | Epoch Time: {epoch_time/60:.2f} min")

        if val_f1 > best_f1 + EARLY_STOP_MIN_DELTA:
            best_f1 = val_f1
            best_epoch = epoch + 1
            torch.save({
                "model_state": model.state_dict(),
                "best_val_pixel_f1": best_f1,
                "epoch": epoch + 1,
                "num_classes": num_classes,
                "backbone": BACKBONE_NAME,
            }, MODEL_PATH)
            print(f"Saved best model to {MODEL_PATH} (Val F1={best_f1:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}. No improvement for {EARLY_STOP_PATIENCE} epochs.")
                break

        # Step scheduler after epoch bookkeeping
        lr_sched.step()

    # Training end time
    training_end_time = datetime.now()
    total_training_time = time.time() - start_timestamp
    completed_epochs = len(loss_history)
    early_stopped = completed_epochs < NUM_EPOCHS

    print(f"\nEğitim tamamlandı!")
    print(f"Toplam süre: {total_training_time/60.0:.2f} dakika")
    print(f"En iyi Validation F1: {best_f1:.4f} (Epoch {best_epoch})")

    # Final test evaluation using best checkpoint
    val_metrics = None
    test_metrics = None
    val_tta_metrics = None
    test_tta_metrics = None
    ensemble_val_metrics = None
    ensemble_test_metrics = None
    
    if os.path.isfile(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=device)
        num_classes_ckpt = ckpt.get("num_classes", num_classes)
        backbone_ckpt = ckpt.get("backbone", BACKBONE_NAME)
        best_model = get_model(num_classes_ckpt, backbone_name=backbone_ckpt)
        best_model.load_state_dict(ckpt["model_state"])
        best_model.to(device)

        # Detailed metrics on validation
        val_metrics = evaluate_detailed_metrics(best_model, val_loader, device)
        print(f"Validation metrics: Precision={val_metrics['overall']['precision']:.4f}, Recall={val_metrics['overall']['recall']:.4f}, F1={val_metrics['overall']['f1']:.4f}, IoU={val_metrics['overall']['iou']:.4f}")
        val_tta_metrics = evaluate_with_tta(best_model, val_loader, device)
        print(f"Validation TTA metrics: Precision={val_tta_metrics['precision']:.4f}, Recall={val_tta_metrics['recall']:.4f}, F1={val_tta_metrics['f1']:.4f}, IoU={val_tta_metrics['iou']:.4f}")

        if ENSEMBLE_MODEL_PATHS:
            ensemble_val_metrics = evaluate_model_ensemble(ENSEMBLE_MODEL_PATHS, num_classes, val_loader, device)
            if ensemble_val_metrics:
                print(f"Validation Ensemble metrics: Precision={ensemble_val_metrics['precision']:.4f}, Recall={ensemble_val_metrics['recall']:.4f}, F1={ensemble_val_metrics['f1']:.4f}, IoU={ensemble_val_metrics['iou']:.4f}")

        # Detailed metrics on test (if available)
        if test_loader is not None:
            test_metrics = evaluate_detailed_metrics(best_model, test_loader, device)
            print(f"Test metrics: Precision={test_metrics['overall']['precision']:.4f}, Recall={test_metrics['overall']['recall']:.4f}, F1={test_metrics['overall']['f1']:.4f}, IoU={test_metrics['overall']['iou']:.4f}")
            test_tta_metrics = evaluate_with_tta(best_model, test_loader, device)
            print(f"Test TTA metrics: Precision={test_tta_metrics['precision']:.4f}, Recall={test_tta_metrics['recall']:.4f}, F1={test_tta_metrics['f1']:.4f}, IoU={test_tta_metrics['iou']:.4f}")
            if ENSEMBLE_MODEL_PATHS:
                ensemble_test_metrics = evaluate_model_ensemble(ENSEMBLE_MODEL_PATHS, num_classes, test_loader, device)
                if ensemble_test_metrics:
                    print(f"Test Ensemble metrics: Precision={ensemble_test_metrics['precision']:.4f}, Recall={ensemble_test_metrics['recall']:.4f}, F1={ensemble_test_metrics['f1']:.4f}, IoU={ensemble_test_metrics['iou']:.4f}")

    # Choose a model reference for reporting (best if exists, else current)
    model_for_report = None
    if 'best_model' in locals():
        model_for_report = best_model
    else:
        model_for_report = model

    # Compute parameter counts
    total_params = sum(p.numel() for p in model_for_report.parameters())
    trainable_params = sum(p.numel() for p in model_for_report.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # Framework and environment info
    torch_version = getattr(torch, "__version__", "unknown")
    tv_version = getattr(torchvision, "__version__", "unknown")

    # Optimizer and scheduler config (use first param_group)
    first_pg = optimizer.param_groups[0] if len(optimizer.param_groups) > 0 else {}
    optimizer_info = {
        "name": optimizer.__class__.__name__,
        "lr": first_pg.get("lr"),
        "weight_decay": first_pg.get("weight_decay"),
        "betas": first_pg.get("betas") if "betas" in first_pg else None,
        "momentum": first_pg.get("momentum") if "momentum" in first_pg else None,
    }
    scheduler_info = {
        "name": lr_sched.__class__.__name__,
        "warmup_epochs": WARMUP_EPOCHS,
        "cosine_T0": COSINE_T0,
        "cosine_T_mult": COSINE_T_MULT,
        "eta_min": COSINE_MIN_LR,
        "base_schedulers": [s.__class__.__name__ for s in getattr(lr_sched, "_schedulers", [])],
        "start_factor": getattr(warmup_scheduler, "start_factor", None),
    }

    # Prepare comprehensive training information
    model_file_size = 0
    if os.path.isfile(MODEL_PATH):
        model_file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB

    default_metric = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}
    final_metrics = {
        "validation": val_metrics['overall'] if val_metrics else default_metric,
    }
    if val_tta_metrics:
        final_metrics["validation_tta"] = val_tta_metrics
    if test_metrics:
        final_metrics["test"] = test_metrics['overall']
    if test_tta_metrics:
        final_metrics["test_tta"] = test_tta_metrics
    if ensemble_val_metrics:
        final_metrics["validation_ensemble"] = ensemble_val_metrics
    if ensemble_test_metrics:
        final_metrics["test_ensemble"] = ensemble_test_metrics

    training_info = {
        "config": {
            "dataset_path": COCO_ROOT,
            "backbone": BACKBONE_NAME,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "num_classes": num_classes,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset) if test_dataset else 0,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "early_stop_min_delta": EARLY_STOP_MIN_DELTA,
            "random_seed": RANDOM_SEED,
            "device": str(device),
            "optimizer": optimizer_info,
            "scheduler": scheduler_info,
            "framework": {"torch": torch_version, "torchvision": tv_version},
            "class_weights": class_weights.tolist() if class_weights is not None else None,
            "label_distribution": dict(label_counts),
            "sampler": "weighted" if sampler is not None else "shuffle",
            "tensorboard_run": run_name,
            "ensemble_paths": ENSEMBLE_MODEL_PATHS,
        },
        "training_time": {
            "start_time": training_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": training_end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration": total_training_time / 60.0,  # minutes
            "avg_epoch_time": (total_training_time / completed_epochs) / 60.0 if completed_epochs > 0 else 0.0,  # minutes
        },
        "training_progress": {
            "completed_epochs": completed_epochs,
            "best_val_f1": best_f1,
            "best_epoch": best_epoch,
            "early_stopped": early_stopped,
            "val_f1_history": val_f1_history,
        },
        "loss_history": loss_history,
        "final_metrics": final_metrics,
        "model_info": {
            "model_path": MODEL_PATH,
            "file_size_mb": model_file_size,
            "architecture": f"MaskRCNN({BACKBONE_NAME})",
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": non_trainable_params,
            "backbone": BACKBONE_NAME,
        }
    }

    # Save training results
    save_training_results(training_info, OUTPUT_DIR)

    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()


