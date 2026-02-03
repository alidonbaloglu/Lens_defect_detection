"""
Mask R-CNN Training with Swin Transformer Backbone and Tiling Strategy
=======================================================================

ÖNERİLER:
1. Omurga Değişikliği: ResNet101 yerine Swin Transformer (Swin-T veya Swin-S) kullanılıyor
2. Karolama Stratejisi (Tiling): Yüksek çözünürlüklü görüntüler küçük karolara bölünüyor
   - Milimetrik kusurların piksel detaylarını korur
   - GPU bellek kullanımını optimize eder
   - Her karo bağımsız olarak işlenir ve sonuçlar birleştirilir
"""

import os
import json
import random
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.ops import MultiScaleRoIAlign

# Swin Transformer imports
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("WARNING: timm library not found. Install with: pip install timm")

# Optional dependency: pycocotools
try:
    from pycocotools import mask as coco_mask
except Exception:
    coco_mask = None


# ----------------------------
# Configuration
# ----------------------------
COCO_ROOT = r"C:/Users/ali.donbaloglu/Desktop/Lens/datasetler/merged_coco_dataset_v2"
TRAIN_SPLIT = "train"
VAL_SPLIT_NAME = "valid"
TEST_SPLIT = "test"
OUTPUT_DIR = os.path.join("results", "MaskRCNN_Swin_Tiling")
MODEL_DIR = os.path.join("Modeller")
MODEL_PATH = os.path.join(MODEL_DIR, "MaskRCNN_Swin_Tiling_v1.pt")

# Swin Transformer Configuration
SWIN_MODEL_NAME = "swin_small_patch4_window7_224"  # swin_tiny veya swin_small_patch4_window7_224
SWIN_PRETRAINED = True

# Tiling Configuration
ENABLE_TILING = True
TILE_SIZE = 640  # Her karonun boyutu (piksel)
TILE_OVERLAP = 160  # Karolar arası örtüşme (piksel) - keskin geçişleri yumuşatır
MIN_TILE_SIZE = 320  # Minimum karo boyutu

# Training Hyperparameters
RANDOM_SEED = 42
NUM_EPOCHS = 75
BATCH_SIZE = 2  # Swin Transformer daha fazla bellek kullanır, batch size'ı düşürün
NUM_WORKERS = 4
LEARNING_RATE = 0.0003  # Swin için biraz daha düşük LR
WEIGHT_DECAY = 1e-4
PRINT_FREQ = 50

# Early Stopping
EARLY_STOP_PATIENCE = 15
EARLY_STOP_MIN_DELTA = 0.0005


# ----------------------------
# Swin Transformer Backbone with FPN
# ----------------------------
class SwinTransformerBackboneWithFPN(torch.nn.Module):
    """
    Swin Transformer backbone with Feature Pyramid Network (FPN).
    Compatible with torchvision's Mask R-CNN implementation.
    """
    def __init__(self, model_name: str = "swin_tiny_patch4_window7_224", pretrained: bool = True):
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm library is required for Swin Transformer. Install with: pip install timm")
        
        # Load Swin Transformer from timm with dynamic image size support
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            img_size=(640, 640),
            strict_img_size=False,
        )
        
        # Get feature dimensions
        feature_info = self.backbone.feature_info
        in_channels_list = [info['num_chs'] for info in feature_info]
        
        print(f"Swin Transformer feature levels: {len(in_channels_list)}")
        print(f"Feature channels: {in_channels_list}")
        
        # FPN output channels
        self.out_channels = 256
        
        # Build custom FPN that outputs standard feature map names
        from torch import nn
        from collections import OrderedDict
        
        # Inner blocks: 1x1 conv to reduce channels to 256
        self.inner_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            self.inner_blocks.append(
                nn.Conv2d(in_channels, self.out_channels, 1)
            )
        
        # Layer blocks: 3x3 conv for smoothing
        self.layer_blocks = nn.ModuleList()
        for _ in range(len(in_channels_list)):
            self.layer_blocks.append(
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
            )
    
    def forward(self, x):
        """Forward pass through Swin + FPN"""
        # Get multi-scale features from Swin [B, H, W, C]
        features = self.backbone(x)
        
        # Convert from [B, H, W, C] to [B, C, H, W]
        features_converted = []
        for feat in features:
            if feat.dim() == 4 and feat.shape[-1] in [96, 192, 384, 768, 1024]:
                # Channels last to channels first
                feat = feat.permute(0, 3, 1, 2).contiguous()
            features_converted.append(feat)
        
        # Apply FPN
        # Start from the deepest layer
        results = []
        last_inner = self.inner_blocks[-1](features_converted[-1])
        results.append(self.layer_blocks[-1](last_inner))
        
        # Iterate from second-to-last to first
        for idx in range(len(features_converted) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](features_converted[idx])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = torch.nn.functional.interpolate(
                last_inner, size=feat_shape, mode="nearest"
            )
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))
        
        # Return as OrderedDict with standard names for Mask R-CNN
        from collections import OrderedDict
        out = OrderedDict()
        for idx, feat in enumerate(results):
            out[str(idx)] = feat
        
        return out


def get_swin_maskrcnn_model(num_classes: int, model_name: str = "swin_tiny_patch4_window7_224", 
                            pretrained: bool = True) -> MaskRCNN:
    """
    Create Mask R-CNN model with Swin Transformer backbone
    """
    # Create Swin backbone with FPN
    backbone = SwinTransformerBackboneWithFPN(model_name=model_name, pretrained=pretrained)
    
    # Anchor generator - match the number of feature maps from FPN
    # Swin typically outputs 4 feature levels
    anchor_sizes = ((32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    
    anchor_generator = AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=aspect_ratios
    )
    
    # ROI pooler for box head
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )
    
    # ROI pooler for mask head
    mask_roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=14,
        sampling_ratio=2
    )
    
    # Create Mask R-CNN model
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler,
    )
    
    return model


# ----------------------------
# Tiling Strategy
# ----------------------------
class TilingProcessor:
    """
    Yüksek çözünürlüklü görüntüleri küçük karolara böler ve sonuçları birleştirir.
    
    Avantajları:
    - Milimetrik kusurların detaylarını korur
    - GPU bellek kullanımını optimize eder
    - Büyük görüntülerde daha iyi performans
    """
    
    def __init__(self, tile_size: int = 640, overlap: int = 160, min_size: int = 320):
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_size = min_size
        
    def split_image_to_tiles(self, image: torch.Tensor) -> List[Dict]:
        """
        Görüntüyü karolara böler
        
        Args:
            image: [C, H, W] tensor
            
        Returns:
            List of dicts containing:
                - tile: [C, tile_h, tile_w] tensor
                - x_offset: x koordinat başlangıcı
                - y_offset: y koordinat başlangıcı
        """
        C, H, W = image.shape
        tiles = []
        
        # Eğer görüntü zaten küçükse, tek karo olarak döndür
        if H <= self.tile_size and W <= self.tile_size:
            return [{
                'tile': image,
                'x_offset': 0,
                'y_offset': 0,
                'tile_h': H,
                'tile_w': W,
            }]
        
        # Stride hesapla (overlap kadar örtüşme olacak şekilde)
        stride = self.tile_size - self.overlap
        
        # Y ekseni boyunca karolara böl
        y_positions = []
        y = 0
        while y < H:
            y_end = min(y + self.tile_size, H)
            # Son karo çok küçükse, bir önceki karo ile birleştir
            if H - y_end < self.min_size and len(y_positions) > 0:
                y_positions[-1] = (y_positions[-1][0], H)
                break
            y_positions.append((y, y_end))
            y += stride
            if y_end >= H:
                break
                
        # X ekseni boyunca karolara böl
        x_positions = []
        x = 0
        while x < W:
            x_end = min(x + self.tile_size, W)
            if W - x_end < self.min_size and len(x_positions) > 0:
                x_positions[-1] = (x_positions[-1][0], W)
                break
            x_positions.append((x, x_end))
            x += stride
            if x_end >= W:
                break
        
        # Karoları oluştur
        for y_start, y_end in y_positions:
            for x_start, x_end in x_positions:
                tile = image[:, y_start:y_end, x_start:x_end]
                tiles.append({
                    'tile': tile,
                    'x_offset': x_start,
                    'y_offset': y_start,
                    'tile_h': y_end - y_start,
                    'tile_w': x_end - x_start,
                })
        
        return tiles
    
    def merge_predictions(self, tile_predictions: List[Dict], 
                         original_h: int, original_w: int) -> Dict:
        """
        Karo tahminlerini birleştirir
        
        Args:
            tile_predictions: Her karo için tahmin sonuçları
            original_h, original_w: Orijinal görüntü boyutu
            
        Returns:
            Birleştirilmiş tahminler
        """
        if len(tile_predictions) == 0:
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'scores': torch.zeros((0,), dtype=torch.float32),
                'masks': torch.zeros((0, original_h, original_w), dtype=torch.uint8),
            }
        
        # Eğer tek karo varsa, direkt döndür
        if len(tile_predictions) == 1:
            pred = tile_predictions[0]['prediction']
            if 'masks' in pred and len(pred['masks']) > 0:
                # Mask boyutunu kontrol et ve gerekirse yeniden boyutlandır
                if pred['masks'].shape[-2:] != (original_h, original_w):
                    # Burada maske zaten tile boyutunda, koordinatlara göre yerleştir
                    pass
            return pred
        
        all_boxes = []
        all_labels = []
        all_scores = []
        all_masks = []
        
        for tile_pred in tile_predictions:
            pred = tile_pred['prediction']
            x_offset = tile_pred['x_offset']
            y_offset = tile_pred['y_offset']
            
            if len(pred.get('boxes', [])) == 0:
                continue
            
            # Kutu koordinatlarını global koordinatlara dönüştür
            boxes = pred['boxes'].clone()
            boxes[:, [0, 2]] += x_offset  # x koordinatları
            boxes[:, [1, 3]] += y_offset  # y koordinatları
            
            all_boxes.append(boxes)
            all_labels.append(pred['labels'])
            all_scores.append(pred['scores'])
            
            # Maskeleri global koordinatlara yerleştir
            if 'masks' in pred and len(pred['masks']) > 0:
                masks = pred['masks']
                for mask in masks:
                    # full mask'ı, indeksleme yapılacak tensörlerle aynı cihaza (CPU/GPU) oluştur
                    full_mask = torch.zeros((original_h, original_w), dtype=torch.float32, device=mask.device)
                    tile_h, tile_w = mask.shape[-2:]
                    y_end = min(y_offset + tile_h, original_h)
                    x_end = min(x_offset + tile_w, original_w)
                    actual_h = y_end - y_offset
                    actual_w = x_end - x_offset
                    full_mask[y_offset:y_end, x_offset:x_end] = mask[0, :actual_h, :actual_w]
                    all_masks.append(full_mask)
        
        if len(all_boxes) == 0:
            return {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'scores': torch.zeros((0,), dtype=torch.float32),
                'masks': torch.zeros((0, original_h, original_w), dtype=torch.uint8),
            }
        
        # Tüm tahminleri birleştir
        boxes = torch.cat(all_boxes, dim=0)
        labels = torch.cat(all_labels, dim=0)
        scores = torch.cat(all_scores, dim=0)
        masks = torch.stack(all_masks, dim=0) if len(all_masks) > 0 else torch.zeros((0, original_h, original_w))
        
        # NMS ile çakışan kutuları filtrele
        keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
        
        merged_prediction = {
            'boxes': boxes[keep_indices],
            'labels': labels[keep_indices],
            'scores': scores[keep_indices],
            'masks': (masks[keep_indices] > 0.5).to(torch.uint8) if len(masks) > 0 else masks,
        }
        
        return merged_prediction


# ----------------------------
# Augmentation
# ----------------------------
class AugmentationTransforms:
    """Training augmentation transforms for Mask R-CNN"""
    def __init__(self, apply_prob=0.5):
        self.apply_prob = apply_prob
    
    def __call__(self, image):
        # Random Horizontal Flip
        if torch.rand(1).item() > self.apply_prob:
            image = F.hflip(image)
        
        # Color Jitter - Brightness
        if torch.rand(1).item() > self.apply_prob:
            brightness_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.3
            image = F.adjust_brightness(image, brightness_factor=brightness_factor)
        
        # Color Jitter - Contrast
        if torch.rand(1).item() > self.apply_prob:
            contrast_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.3
            image = F.adjust_contrast(image, contrast_factor=contrast_factor)
        
        # Color Jitter - Saturation
        if torch.rand(1).item() > self.apply_prob:
            saturation_factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.3
            image = F.adjust_saturation(image, saturation_factor=saturation_factor)
        
        image = torch.clamp(image, 0.0, 1.0)
        return image


# ----------------------------
# Utility Functions
# ----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


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
    """COCO-style instance segmentation dataset"""

    def __init__(self, json_path: str, images_root: str, category_ids: Optional[List[int]] = None, transforms=None) -> None:
        if coco_mask is None:
            raise ImportError("pycocotools is required. Install with: pip install pycocotools")

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

            x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
            if w <= 0 or h <= 0:
                continue

            seg = ann.get("segmentation")
            rle = None
            if isinstance(seg, list):
                rles = coco_mask.frPyObjects(seg, height, width)
                rle = coco_mask.merge(rles)
            elif isinstance(seg, dict) and {"counts", "size"}.issubset(seg.keys()):
                rle = seg
            else:
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

        if self.transforms:
            image = self.transforms(image)

        return image, target


def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


def _find_split_paths(split_root: str) -> Tuple[str, str]:
    """Return (images_root, json_path) for a split folder"""
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


# ----------------------------
# Training / Evaluation with Tiling
# ----------------------------
@torch.no_grad()
def evaluate_pixel_f1_with_tiling(model, data_loader, device, tiling_processor=None) -> float:
    """Evaluate with tiling support"""
    model.eval()
    f1_scores: List[float] = []

    for images, targets in data_loader:
        for img, tgt in zip(images, targets):
            img = img.to(device)
            original_h, original_w = img.shape[-2:]
            
            # Tiling kullanılacaksa
            if tiling_processor and ENABLE_TILING:
                tiles = tiling_processor.split_image_to_tiles(img)
                tile_predictions = []
                
                for tile_info in tiles:
                    tile = tile_info['tile'].to(device)  # unsqueeze(0) kaldırıldı
                    pred = model([tile])[0]
                    tile_predictions.append({
                        'prediction': pred,
                        'x_offset': tile_info['x_offset'],
                        'y_offset': tile_info['y_offset'],
                    })
                
                out = tiling_processor.merge_predictions(tile_predictions, original_h, original_w)
            else:
                # Normal tahmin
                out = model([img])[0]

            # Prediction binary mask
            if len(out.get("masks", [])) == 0:
                pred_binary = np.zeros((original_h, original_w), dtype=np.uint8)
            else:
                scores = out["scores"].detach().cpu().numpy()
                masks = out["masks"].detach().cpu().numpy()
                keep = scores >= 0.5
                if keep.sum() == 0:
                    pred_binary = np.zeros(masks.shape[-2:], dtype=np.uint8)
                else:
                    masks = masks[keep, 0] if masks.ndim == 4 else masks[keep]
                    pred_binary = (np.any(masks >= 0.5, axis=0)).astype(np.uint8)

            # Ground truth binary mask
            gt_masks = tgt.get("masks")
            if gt_masks is None or gt_masks.numel() == 0:
                gt_binary = np.zeros_like(pred_binary, dtype=np.uint8)
            else:
                gt_binary = (gt_masks.max(dim=0).values.detach().cpu().numpy() > 0).astype(np.uint8)

            # Calculate F1
            tp = np.logical_and(pred_binary == 1, gt_binary == 1).sum()
            fp = np.logical_and(pred_binary == 1, gt_binary == 0).sum()
            fn = np.logical_and(pred_binary == 0, gt_binary == 1).sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1_scores.append(float(f1))

    return float(np.mean(f1_scores)) if len(f1_scores) > 0 else 0.0


def train_one_epoch(model, optimizer, data_loader, device, epoch: int, tiling_processor=None) -> Dict[str, float]:
    """Train one epoch with tiling support for inference (training on original images)"""
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

        # Training sırasında tiling yapılmaz, orijinal görüntüler kullanılır
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    return avg_losses


def save_training_results(training_info: Dict, output_dir: str) -> None:
    """Save training results to JSON and TXT files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_path = os.path.join(output_dir, f"training_results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)
    
    txt_path = os.path.join(output_dir, f"training_summary_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("MASK R-CNN + SWIN TRANSFORMER + TILING EĞİTİM SONUÇLARI\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("EĞİTİM KONFİGÜRASYONU:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Veri Seti: {training_info['config']['dataset_path']}\n")
        f.write(f"Backbone: {training_info['config']['backbone']}\n")
        f.write(f"Tiling Enabled: {training_info['config']['tiling_enabled']}\n")
        if training_info['config']['tiling_enabled']:
            f.write(f"Tile Size: {training_info['config']['tile_size']}\n")
            f.write(f"Tile Overlap: {training_info['config']['tile_overlap']}\n")
        f.write(f"Epoch Sayısı: {training_info['config']['num_epochs']}\n")
        f.write(f"Batch Size: {training_info['config']['batch_size']}\n")
        f.write(f"Learning Rate: {training_info['config']['learning_rate']}\n")
        f.write(f"Sınıf Sayısı: {training_info['config']['num_classes']}\n")
        f.write("\n")
        
        f.write("EĞİTİM SÜRESİ:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Toplam Süre: {training_info['training_time']['total_duration']:.2f} dakika\n")
        f.write(f"Epoch Başına Ortalama: {training_info['training_time']['avg_epoch_time']:.2f} dakika\n")
        f.write("\n")
        
        f.write("FİNAL METRİKLER:\n")
        f.write("-" * 40 + "\n")
        val_metrics = training_info['final_metrics']['validation']
        f.write(f"Validation F1: {val_metrics['f1']:.4f}\n")
        f.write(f"Validation Precision: {val_metrics['precision']:.4f}\n")
        f.write(f"Validation Recall: {val_metrics['recall']:.4f}\n")
        f.write("\n")
        
        f.write("MODEL DOSYASI:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Kayıt Yeri: {training_info['model_info']['model_path']}\n")
        f.write(f"Dosya Boyutu: {training_info['model_info']['file_size_mb']:.2f} MB\n")
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"\nEğitim sonuçları kaydedildi:")
    print(f"  JSON: {json_path}")
    print(f"  TXT: {txt_path}")


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    print("=" * 80)
    print("MASK R-CNN + SWIN TRANSFORMER + TILING STRATEGY")
    print("=" * 80)
    print("\nÖzellikler:")
    print("✓ Swin Transformer backbone (ResNet yerine)")
    print("✓ Tiling strategy (Yüksek çözünürlük desteği)")
    print("✓ FPN (Feature Pyramid Network)")
    print("✓ Data augmentation")
    print("✓ Early stopping")
    print("=" * 80 + "\n")
    
    set_seed(RANDOM_SEED)
    ensure_dir(OUTPUT_DIR)
    ensure_dir(MODEL_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not TIMM_AVAILABLE:
        print("\n⚠️  WARNING: timm library not installed!")
        print("Install with: pip install timm")
        print("Cannot proceed without timm.\n")
        return

    training_start_time = datetime.now()
    start_timestamp = time.time()

    # Prepare splits
    train_root = os.path.join(COCO_ROOT, TRAIN_SPLIT)
    val_root = os.path.join(COCO_ROOT, VAL_SPLIT_NAME)
    test_root = os.path.join(COCO_ROOT, TEST_SPLIT)

    train_images_root, train_json = _find_split_paths(train_root)
    val_images_root, val_json = _find_split_paths(val_root)
    
    test_images_root, test_json = None, None
    try:
        test_images_root, test_json = _find_split_paths(test_root)
    except Exception:
        pass

    # Datasets
    train_dataset = COCOSegmentationDataset(
        train_json, 
        train_images_root, 
        transforms=AugmentationTransforms(apply_prob=0.5)
    )
    category_ids = list(train_dataset.category_id_to_label.keys())
    val_dataset = COCOSegmentationDataset(val_json, val_images_root, category_ids=category_ids, transforms=None)
    
    test_dataset = None
    if test_json is not None:
        test_dataset = COCOSegmentationDataset(test_json, test_images_root, category_ids=category_ids, transforms=None)
    
    num_classes = train_dataset.num_classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
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

    # Tiling processor
    tiling_processor = TilingProcessor(
        tile_size=TILE_SIZE,
        overlap=TILE_OVERLAP,
        min_size=MIN_TILE_SIZE
    ) if ENABLE_TILING else None

    # Model with Swin Transformer
    print(f"\nCreating Mask R-CNN with Swin Transformer backbone: {SWIN_MODEL_NAME}")
    model = get_swin_maskrcnn_model(
        num_classes=num_classes,
        model_name=SWIN_MODEL_NAME,
        pretrained=SWIN_PRETRAINED
    )
    model.to(device)

    # Optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    # Training tracking
    best_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    loss_history = []
    val_f1_history = []

    print(f"\nEğitim başlıyor...")
    print(f"Train örnekleri: {len(train_dataset)}")
    print(f"Validation örnekleri: {len(val_dataset)}")
    if test_dataset:
        print(f"Test örnekleri: {len(test_dataset)}")
    print(f"Sınıf sayısı: {num_classes}")
    print(f"Backbone: Swin Transformer ({SWIN_MODEL_NAME})")
    print(f"Tiling: {'Enabled' if ENABLE_TILING else 'Disabled'}")
    if ENABLE_TILING:
        print(f"  Tile size: {TILE_SIZE}x{TILE_SIZE}")
        print(f"  Overlap: {TILE_OVERLAP}px")
    print("-" * 50)

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        epoch_losses = train_one_epoch(model, optimizer, train_loader, device, epoch, tiling_processor)
        loss_history.append(epoch_losses)

        val_f1 = evaluate_pixel_f1_with_tiling(model, val_loader, device, tiling_processor)
        val_f1_history.append(val_f1)

        lr_sched.step()
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Val F1: {val_f1:.4f} | Time: {epoch_time/60:.2f} min")

        if val_f1 > best_f1 + EARLY_STOP_MIN_DELTA:
            best_f1 = val_f1
            best_epoch = epoch + 1
            torch.save({
                "model_state": model.state_dict(),
                "best_val_pixel_f1": best_f1,
                "epoch": epoch + 1,
                "num_classes": num_classes,
                "backbone": SWIN_MODEL_NAME,
                "tiling_config": {
                    "enabled": ENABLE_TILING,
                    "tile_size": TILE_SIZE,
                    "overlap": TILE_OVERLAP,
                }
            }, MODEL_PATH)
            print(f"✓ Model kaydedildi: {MODEL_PATH} (Val F1={best_f1:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"\n⚠ Early stopping: {EARLY_STOP_PATIENCE} epoch boyunca gelişme yok.")
                break

    # Training end
    training_end_time = datetime.now()
    total_training_time = time.time() - start_timestamp
    completed_epochs = len(loss_history)

    print(f"\n{'='*80}")
    print(f"Eğitim tamamlandı!")
    print(f"Toplam süre: {total_training_time/60.0:.2f} dakika")
    print(f"En iyi Validation F1: {best_f1:.4f} (Epoch {best_epoch})")
    print(f"{'='*80}\n")

    # Final evaluation
    val_metrics = {"precision": 0, "recall": 0, "f1": best_f1, "iou": 0}
    
    model_file_size = 0
    if os.path.isfile(MODEL_PATH):
        model_file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)

    # Prepare training info
    training_info = {
        "config": {
            "dataset_path": COCO_ROOT,
            "backbone": f"Swin Transformer ({SWIN_MODEL_NAME})",
            "tiling_enabled": ENABLE_TILING,
            "tile_size": TILE_SIZE if ENABLE_TILING else None,
            "tile_overlap": TILE_OVERLAP if ENABLE_TILING else None,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "num_classes": num_classes,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset) if test_dataset else 0,
        },
        "training_time": {
            "start_time": training_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": training_end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration": total_training_time / 60.0,
            "avg_epoch_time": (total_training_time / completed_epochs) / 60.0,
        },
        "training_progress": {
            "completed_epochs": completed_epochs,
            "best_val_f1": best_f1,
            "best_epoch": best_epoch,
            "val_f1_history": val_f1_history,
        },
        "loss_history": loss_history,
        "final_metrics": {
            "validation": val_metrics,
        },
        "model_info": {
            "model_path": MODEL_PATH,
            "file_size_mb": model_file_size,
            "architecture": f"MaskRCNN + Swin Transformer ({SWIN_MODEL_NAME})",
        }
    }

    save_training_results(training_info, OUTPUT_DIR)
    
    print("\n✓ Eğitim tamamlandı ve sonuçlar kaydedildi!")
    print(f"Model: {MODEL_PATH}")
    print(f"Sonuçlar: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
