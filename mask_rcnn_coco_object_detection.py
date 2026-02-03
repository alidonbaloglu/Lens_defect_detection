import os
import json
import random
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
# SummaryWriter removed — logging to JSON/TXT instead of TensorBoard

import torchvision
import torchvision.transforms as T
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet101_Weights

try:
    from pycocotools import mask as coco_mask
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except Exception:
    coco_mask = None
    COCO = None
    COCOeval = None


# ----------------------------
# Configuration - IMPROVED
# ----------------------------
COCO_ROOT = r"C:/Users/ali.donbaloglu/Desktop/Lens/datasetler/Kabin_object_detection_v1"
TRAIN_SPLIT = "train"
VAL_SPLIT_NAME = "valid"
TEST_SPLIT = "test"
OUTPUT_DIR = os.path.join("results", "MaskRCNN_v2")
MODEL_DIR = os.path.join("Modeller")
MODEL_PATH = os.path.join(MODEL_DIR, "MaskRCNN_COCO_kabin_v5.pt")
INPUT_MODEL_PATH = os.path.join(MODEL_DIR, "none.pt")
LOG_DIR = os.path.join("runs", f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

BACKBONE_NAME = "resnet101"

# ✅ IMPROVED HYPERPARAMETERS
RANDOM_SEED = 42
NUM_EPOCHS = 100
BATCH_SIZE = 1 # 2 → 4
NUM_WORKERS = 4
LEARNING_RATE = 0.0001  # 0.0002 → 0.0001
WEIGHT_DECAY = 1e-4  # 5e-5 → 1e-4
PRINT_FREQ = 100
EARLY_STOP_PATIENCE = 20  # 25 → 20
EARLY_STOP_MIN_DELTA = 0.001  # 0.0005 → 0.001
GRADIENT_CLIP_NORM = 5.0  # 1.0 → 5.0
USE_MIXED_PRECISION = True  # ✅ NEW
TRAINABLE_BACKBONE_LAYERS = 4  # 3 → 4
DROPOUT_RATE = 0.2 # ✅ NEW
IOU_THRESHOLD = 0.25  # use this IoU for COCO mAP evaluation


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# masks_to_boxes removed: switching to bbox-only detection


# ✅ NEW: Training Augmentation
class TrainTransforms:
    """Training time augmentation"""
    def __call__(self, image):
        transforms = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            #T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])
        return transforms(image)


# ----------------------------
# COCO Dataset
# ----------------------------
class COCOSegmentationDataset(Dataset):
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

        if category_ids is None:
            category_ids = [c["id"] for c in categories]

        self.category_id_to_label: Dict[int, int] = {cid: i + 1 for i, cid in enumerate(sorted(category_ids))}
        self.label_to_category_id: Dict[int, int] = {v: k for k, v in self.category_id_to_label.items()}
        self.num_classes: int = len(self.category_id_to_label) + 1

        self.image_id_to_anns: Dict[int, List[Dict]] = {}
        for ann in annotations:
            if ann.get("iscrowd", 0) not in [0, 1]:
                ann["iscrowd"] = 0
            if ann["category_id"] not in self.category_id_to_label:
                continue
            img_id = ann["image_id"]
            self.image_id_to_anns.setdefault(img_id, []).append(ann)

        self.id_to_image: Dict[int, Dict] = {im["id"]: im for im in self.images}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_info = self.images[idx]
        img_path = os.path.join(self.images_root, img_info["file_name"])

        img = Image.open(img_path).convert("RGB")
        image = F.pil_to_tensor(img).float() / 255.0

        img_id = img_info["id"]
        anns = self.image_id_to_anns.get(img_id, [])

        boxes_list: List[List[float]] = []
        labels_list: List[int] = []
        areas_list: List[float] = []
        iscrowd_list: List[int] = []

        width = img_info.get("width", image.shape[-1])
        height = img_info.get("height", image.shape[-2])

        for ann in anns:
            cat_id = ann["category_id"]
            label = self.category_id_to_label[cat_id]

            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            if w <= 0 or h <= 0:
                continue

            boxes_list.append([float(x), float(y), float(x + w), float(y + h)])
            labels_list.append(int(label))
            areas_list.append(float(ann.get("area", w * h)))
            iscrowd_list.append(int(ann.get("iscrowd", 0)))

        if len(boxes_list) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([int(img_info.get("id", idx))], dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }
        else:
            boxes = torch.as_tensor(boxes_list, dtype=torch.float32)
            labels = torch.as_tensor(labels_list, dtype=torch.int64)
            area = torch.as_tensor(areas_list, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd_list, dtype=torch.int64)

            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid]
            labels = labels[valid]
            area = area[valid]
            iscrowd = iscrowd[valid]

            target = {
                "boxes": boxes,
                "labels": labels,
                "image_id": torch.tensor([int(img_info.get("id", idx))], dtype=torch.int64),
                "area": area,
                "iscrowd": iscrowd,
            }

        if self.transforms:
            image = self.transforms(image)

        return image, target


def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


def _find_split_paths(split_root: str) -> Tuple[str, str]:
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
        for fname in os.listdir(split_root):
            if fname.lower().endswith(".json"):
                json_path = os.path.join(split_root, fname)
                break
    if json_path is None:
        raise FileNotFoundError(f"No COCO JSON found in {split_root}")

    images_root = split_root
    return images_root, json_path


# ✅ IMPROVED: Model with Dropout
def get_model(num_classes: int) -> torchvision.models.detection.FasterRCNN:
    if BACKBONE_NAME.lower() == "resnet101":
        backbone = resnet_fpn_backbone(
            backbone_name="resnet101",
            weights=ResNet101_Weights.DEFAULT,
            trainable_layers=TRAINABLE_BACKBONE_LAYERS,
        )
        model = torchvision.models.detection.FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            box_detections_per_img=100,
            box_score_thresh=0.05,
        )
        # Optional: Add Dropout to ROI Head if present
        if hasattr(model.roi_heads, 'box_head') and hasattr(model.roi_heads.box_head, 'fc6'):
            original_fc6 = model.roi_heads.box_head.fc6
            model.roi_heads.box_head.fc6 = nn.Sequential(original_fc6, nn.Dropout(DROPOUT_RATE))
            if hasattr(model.roi_heads.box_head, 'fc7'):
                original_fc7 = model.roi_heads.box_head.fc7
                model.roi_heads.box_head.fc7 = nn.Sequential(original_fc7, nn.Dropout(DROPOUT_RATE))
        
        return model
    else:
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
        return model


# ✅ NEW: Dataset Analysis
def analyze_dataset(dataset):
    """Analyze class distribution in dataset"""
    label_counts = Counter()
    for idx in range(len(dataset)):
        _, target = dataset[idx]
        labels = target['labels'].tolist()
        label_counts.update(labels)
    
    print("\n📊 Sınıf Dağılımı:")
    for label, count in sorted(label_counts.items()):
        print(f"  Sınıf {label}: {count} örnek")
    return label_counts


# evaluate_pixel_f1 removed: not applicable for object detection


@torch.no_grad()
def evaluate_detection_metrics(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    inference_time = 0.0
    images_seen = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        t0 = time.time()
        outputs = model(images)
        t1 = time.time()
        inference_time += (t1 - t0)
        images_seen += len(images)

    fps = images_seen / inference_time if inference_time > 0 else 0.0
    return { 'inference_fps': float(fps) }


@torch.no_grad()
def evaluate_coco_map(coco_json_path: str, model, data_loader, device, label_to_category_id: Optional[Dict[int, int]] = None) -> Dict:
    """Compute COCO mAP for bbox detection."""
    if COCO is None or COCOeval is None or coco_mask is None:
        print("⚠️ pycocotools not available; skipping COCO mAP computation.")
        return {
            'map_025': 0.0,
            'map_50_95': 0.0,
            'stats_025': {},
            'stats_50_95': {}
        }

    coco_gt = COCO(coco_json_path)
    results = []

    for images, targets in data_loader:
        images_cuda = [img.to(device) for img in images]
        outputs = model(images_cuda)

        for out, tgt in zip(outputs, targets):
            img_id_tensor = tgt.get("image_id")
            if isinstance(img_id_tensor, (list, tuple)):
                image_id = int(img_id_tensor[0])
            elif isinstance(img_id_tensor, torch.Tensor):
                image_id = int(img_id_tensor.cpu().numpy().flatten()[0])
            else:
                image_id = int(img_id_tensor)

            scores = out["scores"].detach().cpu().numpy()
            boxes = out["boxes"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy().tolist()

            keep = scores >= 0.05
            if keep.sum() == 0:
                continue

            for box, sc, lbl in zip(boxes[keep], scores[keep], [int(l) for l, k in zip(labels, keep.tolist()) if k]):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                if label_to_category_id is not None and lbl in label_to_category_id:
                    category_id = int(label_to_category_id[lbl])
                else:
                    category_id = int(lbl)
                results.append({
                    'image_id': int(image_id),
                    'category_id': int(category_id),
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(sc),
                })

    if len(results) == 0:
        return {
            'map_025': 0.0,
            'map_50_95': 0.0,
            'stats_025': {},
            'stats_50_95': {}
        }

    coco_dt = coco_gt.loadRes(results)

    def _summarize_stats(coco_gt, coco_dt, iou_thrs=None):
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        if iou_thrs is not None:
            try:
                coco_eval.params.iouThrs = np.array(iou_thrs)
            except Exception:
                pass
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = {}
        keys = [
            'AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
            'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large'
        ]
        for k, v in zip(keys, coco_eval.stats.tolist()):
            stats[k] = float(v)
        return stats

    # Standard COCO eval (IoU=0.50:0.95)
    stats_50_95 = _summarize_stats(coco_gt, coco_dt, iou_thrs=None)

    # Single IoU eval at IOU_THRESHOLD (e.g., 0.25)
    stats_025 = _summarize_stats(coco_gt, coco_dt, iou_thrs=[IOU_THRESHOLD])

    return {
        'map_025': stats_025.get('AP', 0.0),
        'map_50_95': stats_50_95.get('AP', 0.0),
        'stats_025': stats_025,
        'stats_50_95': stats_50_95,
    }


# ✅ IMPROVED: Training with Mixed Precision
def train_one_epoch(model, optimizer, data_loader, device, epoch: int, scaler=None, writer=None) -> Dict[str, float]:
    model.train()
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    epoch_losses = []
    epoch_loss_dict = {}
    
    for step, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        
        # ✅ Mixed Precision Training
        if scaler is not None and USE_MIXED_PRECISION:
            with autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        epoch_losses.append(float(losses.item()))
        for k, v in loss_dict.items():
            if k not in epoch_loss_dict:
                epoch_loss_dict[k] = []
            epoch_loss_dict[k].append(float(v.detach().item()))

        if (step + 1) % PRINT_FREQ == 0:
            loss_vals = {k: float(v.detach().item()) for k, v in loss_dict.items()}
            print(f"Epoch {epoch+1} | Step {step+1}/{len(data_loader)} | Loss: {float(losses.item()):.4f} | {loss_vals}")
    
    avg_losses = {k: np.mean(v) for k, v in epoch_loss_dict.items()}
    avg_losses['total_loss'] = np.mean(epoch_losses)
    
    # ✅ Log to tensorboard
    if writer is not None:
        for loss_name, loss_value in avg_losses.items():
            writer.add_scalar(f'Loss/{loss_name}', loss_value, epoch)
    
    return avg_losses


def save_training_results(training_info: Dict, output_dir: str) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_path = os.path.join(output_dir, f"training_results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)
    
    txt_path = os.path.join(output_dir, f"training_summary_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MASK R-CNN EĞİTİM SONUÇLARI (IMPROVED v3)\n")
        f.write("=" * 80 + "\n\n")
        
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
        # Epoch history summary
        f.write("EPOCH GEÇMİŞİ (Özet):\n")
        f.write("-" * 80 + "\n")
        epoch_hist = training_info.get('epoch_history', [])
        if epoch_hist:
            f.write(f"{'Epoch':>5}  {'TrainLoss':>10}  {'ValF1':>7}  {'Val_mAP':>8}  {'ValFPS':>7}  {'LR':>10}  {'Time(s)':>8}\n")
            for e in epoch_hist:
                ep = e.get('epoch', '-')
                tl = e.get('train_losses', {}).get('total_loss', 0.0)
                vf1 = e.get('val_f1', 0.0)
                vmap = e.get('val_map', None)
                vfps = e.get('val_fps', None)
                lr = e.get('learning_rate', 0.0)
                et = e.get('epoch_time', 0.0)
                vmap_str = f"{vmap:.4f}" if vmap is not None else "-"
                vfps_str = f"{vfps:.2f}" if vfps is not None else "-"
                f.write(f"{ep:>5}  {tl:10.4f}  {vf1:7.4f}  {vmap_str:>8}  {vfps_str:>7}  {lr:10.2e}  {et:8.1f}\n")
        else:
            f.write("(Epoch history not available)\n")
        f.write("\n")
        
        f.write("EĞİTİM SÜRESİ:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Başlangıç Zamanı: {training_info['training_time']['start_time']}\n")
        f.write(f"Bitiş Zamanı: {training_info['training_time']['end_time']}\n")
        f.write(f"Toplam Süre: {training_info['training_time']['total_duration']}\n\n")
        
        f.write("EN İYİ MODEL SONUÇLARI:\n")
        f.write("=" * 80 + "\n")
        best_val = training_info.get('best_validation', {})
        if best_val:
            f.write(f"En İyi Epoch: {best_val.get('epoch', '-')}\n")
            f.write(f"Validation F1 Score: {best_val.get('f1_score', 0):.4f}\n")
            if 'map_025' in best_val:
                f.write(f"Validation mAP (IoU={IOU_THRESHOLD}): {best_val.get('map_025', 0):.4f}\n")
            if 'map_50_95' in best_val:
                f.write(f"Validation mAP (IoU=0.50:0.95): {best_val.get('map_50_95', 0):.4f}\n")
            if 'inference_fps' in best_val:
                f.write(f"Validation Inference FPS: {best_val.get('inference_fps', 0):.2f}\n")
            if 'coco_stats' in best_val and best_val.get('coco_stats'):
                f.write("\nValidation COCOeval (IoU=0.25) and (IoU=0.50:0.95):\n")
                stats_025 = best_val.get('coco_stats', {}).get('stats_025', {})
                stats_50 = best_val.get('coco_stats', {}).get('stats_50_95', {})
                if stats_025:
                    f.write(f"  AP@0.25 (all): {stats_025.get('AP', 0):.4f}\n")
                    f.write(f"  AP@0.25 (small): {stats_025.get('AP_small', 0):.4f}\n")
                    f.write(f"  AP@0.25 (medium): {stats_025.get('AP_medium', 0):.4f}\n")
                    f.write(f"  AP@0.25 (large): {stats_025.get('AP_large', 0):.4f}\n")
                if stats_50:
                    f.write(f"  AP@[0.50:0.95] (all): {stats_50.get('AP', 0):.4f}\n")
                    f.write(f"  AP@0.50 (all): {stats_50.get('AP50', 0):.4f}\n")
                    f.write(f"  AP@0.75 (all): {stats_50.get('AP75', 0):.4f}\n")
                f.write("\n")
            overall = best_val.get('overall_metrics', {})
            if overall:
                f.write(f"Precision: {overall.get('precision', 0):.4f}\n")
                f.write(f"Recall: {overall.get('recall', 0):.4f}\n")
                f.write(f"IoU: {overall.get('iou', 0):.4f}\n")
            f.write("\n")
        
        f.write("SINIF BAZLI METRIKLER:\n")
        f.write("-" * 80 + "\n")
        per_class = best_val.get('per_class_metrics', {})
        if per_class:
            for class_name, metrics in sorted(per_class.items()):
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {metrics.get('precision', 0):.4f}\n")
                f.write(f"  Recall: {metrics.get('recall', 0):.4f}\n")
                f.write(f"  F1 Score: {metrics.get('f1', 0):.4f}\n")
                f.write(f"  IoU: {metrics.get('iou', 0):.4f}\n")
        
        if 'test_metrics' in training_info and training_info['test_metrics']:
            f.write("\n" + "=" * 80 + "\n")
            f.write("TEST SETİ SONUÇLARI:\n")
            f.write("=" * 80 + "\n")
            test_metrics = training_info['test_metrics']
            overall_test = test_metrics.get('overall', {})
            if overall_test:
                f.write(f"Test F1 Score: {overall_test.get('f1', 0):.4f}\n")
                f.write(f"Test Precision: {overall_test.get('precision', 0):.4f}\n")
                f.write(f"Test Recall: {overall_test.get('recall', 0):.4f}\n")
                f.write(f"Test IoU: {overall_test.get('iou', 0):.4f}\n\n")
            if 'map_025' in test_metrics:
                f.write(f"Test mAP (IoU={IOU_THRESHOLD}): {test_metrics.get('map_025', 0):.4f}\n\n")
            if 'map_50_95' in test_metrics:
                f.write(f"Test mAP (IoU=0.50:0.95): {test_metrics.get('map_50_95', 0):.4f}\n\n")
            if 'inference_fps' in test_metrics:
                f.write(f"Test Inference FPS: {test_metrics.get('inference_fps', 0):.2f}\n\n")
            if 'coco_stats' in test_metrics and test_metrics.get('coco_stats'):
                stats_025 = test_metrics.get('coco_stats', {}).get('stats_025', {})
                stats_50 = test_metrics.get('coco_stats', {}).get('stats_50_95', {})
                f.write("Test COCOeval summary:\n")
                if stats_025:
                    f.write(f"  AP@0.25 (all): {stats_025.get('AP', 0):.4f}\n")
                if stats_50:
                    f.write(f"  AP@[0.50:0.95] (all): {stats_50.get('AP', 0):.4f}\n")
                f.write("\n")
            
            test_per_class = test_metrics.get('per_class', {})
            if test_per_class:
                f.write("Test Seti Sınıf Bazlı Metrikler:\n")
                f.write("-" * 80 + "\n")
                for class_name, metrics in sorted(test_per_class.items()):
                    f.write(f"\n{class_name}:\n")
                    f.write(f"  Precision: {metrics.get('precision', 0):.4f}\n")
                    f.write(f"  Recall: {metrics.get('recall', 0):.4f}\n")
                    f.write(f"  F1 Score: {metrics.get('f1', 0):.4f}\n")
                    f.write(f"  IoU: {metrics.get('iou', 0):.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("NOT: Bu sonuçlar otomatik olarak oluşturulmuştur.\n")
        f.write("=" * 80 + "\n")
    
    print(f"✅ Eğitim sonuçları kaydedildi:")
    print(f"   JSON: {json_path}")
    print(f"   TXT:  {txt_path}")


def main():
    print("=" * 80)
    print("Faster R-CNN EĞİTİM (Object Detection)")
    print("=" * 80)
    
    set_seed(RANDOM_SEED)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(MODEL_DIR)
    ensure_dir(LOG_DIR)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Cihaz: {device}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Bellek: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # TensorBoard logging disabled — using JSON/TXT outputs instead
    writer = None
    
    # Dataset Loading
    print("\n" + "=" * 80)
    print("VERİ SETİ YÜKLEME")
    print("=" * 80)
    
    train_root = os.path.join(COCO_ROOT, TRAIN_SPLIT)
    val_root = os.path.join(COCO_ROOT, VAL_SPLIT_NAME)
    test_root = os.path.join(COCO_ROOT, TEST_SPLIT)
    
    train_images_root, train_json = _find_split_paths(train_root)
    val_images_root, val_json = _find_split_paths(val_root)
    
    print(f"📂 Train JSON: {train_json}")
    print(f"📂 Val JSON: {val_json}")
    
    # Create datasets
    train_dataset = COCOSegmentationDataset(
        train_json, 
        train_images_root,
        transforms=TrainTransforms()  # ✅ Training augmentation
    )
    val_dataset = COCOSegmentationDataset(val_json, val_images_root, transforms=None)
    
    print(f"\n✅ Train örnekleri: {len(train_dataset)}")
    print(f"✅ Validation örnekleri: {len(val_dataset)}")
    print(f"✅ Sınıf sayısı: {train_dataset.num_classes}")
    
    # ✅ Analyze dataset
    print("\n📊 Train Seti Analizi:")
    train_label_counts = analyze_dataset(train_dataset)
    print("\n📊 Validation Seti Analizi:")
    val_label_counts = analyze_dataset(val_dataset)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Test dataset (if exists)
    test_loader = None
    test_dataset = None
    if os.path.isdir(test_root):
        try:
            test_images_root, test_json = _find_split_paths(test_root)
            test_dataset = COCOSegmentationDataset(test_json, test_images_root, transforms=None)
            test_loader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=NUM_WORKERS,
                collate_fn=collate_fn,
                pin_memory=True,
            )
            print(f"✅ Test örnekleri: {len(test_dataset)}")
        except Exception as e:
            print(f"⚠️ Test seti yüklenemedi: {e}")
    
    # Model
    print("\n" + "=" * 80)
    print("MODEL OLUŞTURMA")
    print("=" * 80)
    
    model = get_model(train_dataset.num_classes)
    
    # Load pretrained weights if exists
    if os.path.isfile(INPUT_MODEL_PATH):
        print(f"📥 Önceki model yükleniyor: {INPUT_MODEL_PATH}")
        checkpoint = torch.load(INPUT_MODEL_PATH, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Model yüklendi!")
    else:
        print("ℹ️ Yeni model oluşturuldu (pretrained backbone)")
    
    model = model.to(device)
    
    # Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-7
    )
    
    # ✅ Mixed Precision Scaler
    scaler = GradScaler() if USE_MIXED_PRECISION else None
    if scaler:
        print("✅ Mixed Precision Training aktif")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in params)

    print(f"\n🔧 Optimizer: AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
    print(f"🔧 Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)")
    print(f"🔧 Total parameters: {total_params:,}")
    print(f"🔧 Trainable parameters: {trainable_params:,}")
    
    # Training Loop
    print("\n" + "=" * 80)
    print("EĞİTİM BAŞLIYOR")
    print("=" * 80)
    
    best_map = 0.0
    best_epoch = 0
    best_metrics = {}
    best_val_map_50_95 = 0.0
    best_val_coco_stats = {}
    best_val_fps = 0.0
    patience_counter = 0
    start_time = datetime.now()
    
    epoch_history = []
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*80}")
        
        # Training
        train_losses = train_one_epoch(
            model, optimizer, train_loader, device, epoch, scaler, writer
        )
        
        # Validation
        print(f"\n🔍 Validation başlıyor...")
        val_metrics = evaluate_detection_metrics(model, val_loader, device)

        # ✅ Compute COCO-style mAP and full COCOeval stats for validation (if possible)
        try:
            val_coco = evaluate_coco_map(val_json, model, val_loader, device, train_dataset.label_to_category_id)
            val_map_025 = val_coco.get('map_025', 0.0)
            val_map_50_95 = val_coco.get('map_50_95', 0.0)
            val_coco_stats = {
                'stats_025': val_coco.get('stats_025', {}),
                'stats_50_95': val_coco.get('stats_50_95', {})
            }
        except Exception as e:
            print(f"⚠️ Val mAP hesaplanamadı: {e}")
            val_map_025 = 0.0
            val_map_50_95 = 0.0
            val_coco_stats = {}
        
        # Optional TensorBoard logging (disabled by default)
        if writer is not None:
            writer.add_scalar('Metrics/val_map_025', val_map_025, epoch)
            writer.add_scalar('Metrics/val_map_50_95', val_map_50_95, epoch)
            writer.add_scalar('Metrics/val_fps', val_metrics.get('inference_fps', 0.0), epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n📊 Epoch {epoch + 1} Sonuçları:")
        print(f"   ⏱️  Süre: {epoch_time:.2f}s")
        print(f"   📉 Train Loss: {train_losses['total_loss']:.4f}")
        val_fps = val_metrics.get('inference_fps', 0.0)
        print(f"   🧾 Val mAP (IoU={IOU_THRESHOLD}): {val_map_025:.4f}")
        print(f"   🧾 Val mAP (IoU=0.50:0.95): {val_map_50_95:.4f}")
        print(f"   ⚡ Val Inference FPS: {val_fps:.2f}")
        print(f"   🔧 LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Epoch history
        epoch_info = {
            'epoch': epoch + 1,
            'train_losses': train_losses,
            'val_metrics': val_metrics,
            'val_map_025': val_map_025,
            'val_map_50_95': val_map_50_95,
            'val_fps': val_fps,
            'val_coco_stats': val_coco_stats,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time
        }
        epoch_history.append(epoch_info)
        
        # Scheduler step
        scheduler.step(val_map_50_95)
        
        # Save best model
        if val_map_50_95 > best_map + EARLY_STOP_MIN_DELTA:
            improvement = val_map_50_95 - best_map
            best_map = val_map_50_95
            best_epoch = epoch + 1
            best_metrics = val_metrics
            best_val_map_50_95 = val_map_50_95
            best_val_coco_stats = val_coco_stats
            best_val_fps = val_fps
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_map': best_map,
                'val_metrics': val_metrics,
                'val_map_025': val_map_025,
                'val_map_50_95': val_map_50_95,
                'val_coco_stats': val_coco_stats,
                'val_fps': val_fps,
                'train_losses': train_losses,
            }
            
            torch.save(checkpoint, MODEL_PATH)
            print(f"   ✅ YENİ EN İYİ MODEL! (mAP@50-95: {best_map:.4f}, +{improvement:.4f})")
        else:
            patience_counter += 1
            print(f"   ⚠️  Gelişme yok. Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")
        
        # Early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n⏹️  Early stopping! {EARLY_STOP_PATIENCE} epoch boyunca gelişme olmadı.")
            break
    
    end_time = datetime.now()
    total_duration = str(end_time - start_time)
    
    print("\n" + "=" * 80)
    print("EĞİTİM TAMAMLANDI!")
    print("=" * 80)
    print(f"✅ En iyi model epoch {best_epoch}'de kaydedildi")
    print(f"✅ En iyi mAP@50-95: {best_map:.4f}")
    print(f"✅ Toplam süre: {total_duration}")
    
    # Test evaluation
    test_metrics = {}
    if test_loader is not None:
        print("\n" + "=" * 80)
        print("TEST SETİ DEĞERLENDİRME")
        print("=" * 80)
        
        # Load best model
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        test_metrics = evaluate_detection_metrics(model, test_loader, device)
        try:
            test_coco = evaluate_coco_map(test_json, model, test_loader, device, train_dataset.label_to_category_id)
            test_map_025 = test_coco.get('map_025', 0.0)
            test_map_50_95 = test_coco.get('map_50_95', 0.0)
            test_metrics['map_025'] = test_map_025
            test_metrics['map_50_95'] = test_map_50_95
            test_metrics['coco_stats'] = {
                'stats_025': test_coco.get('stats_025', {}),
                'stats_50_95': test_coco.get('stats_50_95', {})
            }
        except Exception as e:
            print(f"⚠️ Test mAP hesaplanamadı: {e}")
            test_metrics['map_025'] = 0.0
            test_metrics['map_50_95'] = 0.0
            test_metrics['coco_stats'] = {}
        
        print(f"\n📊 Test Seti Sonuçları:")
        # Only detection metrics and mAP shown
        print(f"   🧾 Test mAP (IoU={IOU_THRESHOLD}): {test_metrics.get('map_025', 0.0):.4f}")
        print(f"   🧾 Test mAP (IoU=0.50:0.95): {test_metrics.get('map_50_95', 0.0):.4f}")
        print(f"   ⚡ Test Inference FPS: {test_metrics.get('inference_fps', 0.0):.2f}")
    
    # Save training results
    training_info = {
        'config': {
            'dataset_path': COCO_ROOT,
            'backbone': BACKBONE_NAME,
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'num_classes': train_dataset.num_classes,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset) if test_dataset else 0,
            'random_seed': RANDOM_SEED,
            'device': str(device),
            'trainable_backbone_layers': TRAINABLE_BACKBONE_LAYERS,
            'dropout_rate': DROPOUT_RATE,
            'gradient_clip_norm': GRADIENT_CLIP_NORM,
            'use_mixed_precision': USE_MIXED_PRECISION,
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'optimizer': {
                'name': 'AdamW',
                'lr': LEARNING_RATE,
                'weight_decay': WEIGHT_DECAY,
                'betas': [0.9, 0.999],
            },
            'scheduler': {
                'name': 'ReduceLROnPlateau',
                'mode': 'max',
                'factor': 0.5,
                'patience': 5,
                'min_lr': 1e-7,
            },
            'framework': {
                'torch': torch.__version__,
                'torchvision': torchvision.__version__,
            }
        },
        'evaluation': {
            'iou_threshold': IOU_THRESHOLD,
        },
        'training_time': {
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_duration': total_duration,
        },
        'best_validation': {
            'epoch': best_epoch,
            'best_map': best_map,
            'map_50_95': best_val_map_50_95,
            'coco_stats': best_val_coco_stats,
            'inference_fps': best_val_fps,
        },
        'test_metrics': test_metrics if test_metrics else None,
        'epoch_history': epoch_history,
        'model_path': MODEL_PATH,
    }
    save_training_results(training_info, OUTPUT_DIR)
    print("\n" + "=" * 80)
    print("🎉 TÜM İŞLEMLER TAMAMLANDI!")
    print("=" * 80)


if __name__ == "__main__":
    main()