import os
import json
import random
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection import MaskRCNN
# --- HATA ÇÖZÜMÜ: AnchorGenerator'ı import ediyoruz ---
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import FeaturePyramidNetwork

try:
    import timm
except ImportError:
    raise ImportError("Lütfen 'timm' kütüphanesini kurun: pip install timm")

try:
    from pycocotools import mask as coco_mask
except Exception:
    coco_mask = None

# ----------------------------
# Configuration
# ----------------------------
COCO_ROOT = r"C:/Users/ali.donbaloglu/Desktop/Lens/coco_isaretsiz"
TRAIN_SPLIT = "train"
VAL_SPLIT_NAME = "valid"
TEST_SPLIT = "test"
OUTPUT_DIR = os.path.join("results", "MaskRCNN")
MODEL_DIR = os.path.join("Modeller")
BACKBONE_NAME = "efficientnet_b5" 
MODEL_PATH = os.path.join(MODEL_DIR, f"MaskRCNN_COCO_{BACKBONE_NAME}.pt")
BATCH_SIZE = 2
LEARNING_RATE = 0.0005
NUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 0.001
RANDOM_SEED = 42
NUM_WORKERS = 4
WEIGHT_DECAY = 5e-05
PRINT_FREQ = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

class COCOSegmentationDataset(Dataset):
    def __init__(self, json_path: str, images_root: str, category_ids: Optional[List[int]] = None, transforms=None) -> None:
        if coco_mask is None: raise ImportError("pycocotools gerekli.")
        self.images_root = images_root; self.transforms = transforms
        with open(json_path, "r", encoding="utf-8") as f: data = json.load(f)
        self.images: List[Dict] = data.get("images", [])
        annotations: List[Dict] = data.get("annotations", [])
        categories: List[Dict] = data.get("categories", [])
        if category_ids is None: category_ids = [c["id"] for c in categories]
        self.category_id_to_label: Dict[int, int] = {cid: i + 1 for i, cid in enumerate(sorted(category_ids))}
        self.num_classes: int = len(self.category_id_to_label) + 1
        self.image_id_to_anns: Dict[int, List[Dict]] = {}
        for ann in annotations:
            if ann["category_id"] not in self.category_id_to_label: continue
            self.image_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    def __len__(self) -> int: return len(self.images)
    def __getitem__(self, idx: int):
        img_info = self.images[idx]
        img_path = os.path.join(self.images_root, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        image = F.pil_to_tensor(img).float() / 255.0
        h, w = img_info.get("height", image.shape[-2]), img_info.get("width", image.shape[-1])
        anns = self.image_id_to_anns.get(img_info["id"], [])
        masks, boxes, labels = [], [], []
        for ann in anns:
            x, y, bw, bh = ann.get("bbox", [0,0,0,0]);
            if bw <= 0 or bh <= 0: continue
            seg = ann.get("segmentation")
            if isinstance(seg, list): rle = coco_mask.merge(coco_mask.frPyObjects(seg, h, w))
            elif isinstance(seg, dict): rle = seg
            else: continue
            m = (coco_mask.decode(rle) > 0).astype(np.uint8)
            masks.append(torch.from_numpy(m)); boxes.append([x, y, x+bw, y+bh]); labels.append(self.category_id_to_label[ann["category_id"]])
        if not masks: target = {"boxes": torch.zeros((0,4), dtype=torch.float32), "labels": torch.zeros(0, dtype=torch.int64), "masks": torch.zeros((0,h,w), dtype=torch.uint8)}
        else: target = {"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(labels, dtype=torch.int64), "masks": torch.stack(masks)}
        if self.transforms: image = self.transforms(image)
        return image, target

def collate_fn(batch): return tuple(zip(*batch))
def _find_split_paths(split_root: str) -> Tuple[str, str]:
    for name in ["_annotations.coco.json", "annotations.coco.json"]:
        p = os.path.join(split_root, name)
        if os.path.isfile(p): return split_root, p
    raise FileNotFoundError(f"No COCO JSON found in {split_root}")

class TimmBackboneWithFPN(nn.Module):
    def __init__(self, model_name, pretrained=True, out_channels=256):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        self.fpn = FeaturePyramidNetwork(self.model.feature_info.channels(), out_channels)
        self.out_channels = out_channels
    def forward(self, x):
        features = self.model(x)
        return self.fpn({str(i): f for i, f in enumerate(features)})

# --- HATA DÜZELTMESİ BURADA UYGULANDI ---
def get_model(num_classes: int, backbone_name: str) -> MaskRCNN:
    print("Özelleştirilmiş omurga ve AnchorGenerator kullanılıyor...")
    backbone = TimmBackboneWithFPN(model_name=backbone_name, pretrained=True, out_channels=256)

    # EfficientNet backbone 4 seviyeli özellik haritası üretir. Her seviye için ayrı tuple tanımlıyoruz.
    anchor_sizes = ((32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0))

    # RPN (Region Proposal Network) için çapa üreticiyi oluşturuyoruz.
    rpn_anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # Modeli oluştururken özel çapa üreticimizi (rpn_anchor_generator) parametre olarak veriyoruz.
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator
    )

    # Tahmin başlıklarını (predictor heads) sınıf sayımıza göre ayarlıyoruz.
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, num_classes)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, num_classes)
    
    return model

@torch.no_grad()
def evaluate_pixel_f1(model, data_loader, device) -> float:
    model.eval()
    f1_scores = []
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            scores = out.get("scores", torch.tensor([])); masks = out.get("masks", torch.tensor([]))
            if len(scores) > 0 and len(masks) > 0:
                pred_binary = (torch.any(masks[scores > 0.5] > 0.5, dim=0).squeeze(0)).byte().cpu().numpy()
            else:
                pred_binary = np.zeros((images[0].shape[-2], images[0].shape[-1]), dtype=np.uint8)
            
            if "masks" in tgt and tgt["masks"].numel() > 0:
                gt_binary = (torch.any(tgt["masks"] > 0, dim=0)).byte().cpu().numpy()
            else:
                gt_binary = np.zeros_like(pred_binary)

            tp = np.logical_and(pred_binary, gt_binary).sum()
            fp = np.logical_and(pred_binary, 1 - gt_binary).sum()
            fn = np.logical_and(1 - pred_binary, gt_binary).sum()
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
            f1_scores.append(f1)
    return np.mean(f1_scores) if f1_scores else 0.0

def train_one_epoch(model, optimizer, data_loader, device, epoch: int) -> Dict[str, float]:
    model.train()
    epoch_losses = {}
    for step, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        for k, v in loss_dict.items(): epoch_losses[k] = epoch_losses.get(k, 0) + v.item()
        if (step + 1) % PRINT_FREQ == 0: print(f"Epoch {epoch+1} | Adım {step+1}/{len(data_loader)} | Loss: {losses.item():.4f}")
    avg_losses = {k: v / len(data_loader) for k, v in epoch_losses.items()}
    avg_losses['total_loss'] = sum(avg_losses.values())
    return avg_losses

def save_training_results(training_info: Dict, output_dir: str) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"training_results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f: json.dump(training_info, f, indent=2, ensure_ascii=False)
    txt_path = os.path.join(output_dir, f"training_summary_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("="*50 + "\nEĞİTİM ÖZETİ\n" + "="*50 + "\n")
        f.write(f"Omurga (Backbone): {training_info['config']['backbone']}\n")
        f.write(f"Tamamlanan Epoch: {training_info['training_progress']['completed_epochs']}\n")
        f.write(f"Toplam Süre: {training_info['training_time']['total_duration']:.2f} dakika\n")
        f.write(f"En İyi Val F1 Skoru: {training_info['training_progress']['best_val_f1']:.4f} (Epoch: {training_info['training_progress']['best_epoch']})\n")
        f.write(f"Model Kayıt Yolu: {training_info['model_info']['model_path']}\n\n")
        f.write("--- Final Metrikler (Validation) ---\n")
        f.write(f"  F1-Skor: {training_info['final_metrics']['validation'].get('f1', 0.0):.4f}\n\n")
        f.write("--- Hiperparametreler ---\n")
        for key, val in training_info['config'].items(): f.write(f"  {key}: {val}\n")
    print(f"\nEğitim sonuçları kaydedildi:\n  JSON: {json_path}\n  TXT: {txt_path}")

def main() -> None:
    set_seed(RANDOM_SEED)
    ensure_dir(OUTPUT_DIR); ensure_dir(MODEL_DIR)
    start_time = time.time(); training_start_dt = datetime.now()
    
    train_root, train_json = _find_split_paths(os.path.join(COCO_ROOT, TRAIN_SPLIT))
    val_root, val_json = _find_split_paths(os.path.join(COCO_ROOT, VAL_SPLIT_NAME))
    
    train_dataset = COCOSegmentationDataset(train_json, train_root)
    val_dataset = COCOSegmentationDataset(val_json, val_root, category_ids=list(train_dataset.category_id_to_label.keys()))
    num_classes = train_dataset.num_classes

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    model = get_model(num_classes=num_classes, backbone_name=BACKBONE_NAME)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    print(f"\nEğitim başlıyor... Omurga: {BACKBONE_NAME}, Cihaz: {DEVICE}")
    
    best_f1, best_epoch, epochs_no_improve, epoch = -1.0, 0, 0, 0
    loss_history, val_f1_history = [], []

    for epoch in range(NUM_EPOCHS):
        epoch_losses = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
        loss_history.append(epoch_losses)
        val_f1 = evaluate_pixel_f1(model, val_loader, DEVICE)
        val_f1_history.append(val_f1)
        lr_sched.step(val_f1)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Val Pixel-F1: {val_f1:.4f}")
        if val_f1 > best_f1 + EARLY_STOP_MIN_DELTA:
            best_f1, best_epoch, epochs_no_improve = val_f1, epoch + 1, 0
            torch.save({"model_state": model.state_dict(), "best_val_pixel_f1": best_f1, "epoch": best_epoch, "num_classes": num_classes}, MODEL_PATH)
            print(f"-> En iyi model kaydedildi: {MODEL_PATH}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping tetiklendi."); break
    
    print("\nEğitim tamamlandı!")
    total_time = (time.time() - start_time) / 60.0
    
    final_metrics = {}
    if os.path.isfile(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        val_f1_final = evaluate_pixel_f1(model, val_loader, DEVICE)
        final_metrics['validation'] = {'f1': val_f1_final}

    training_info = {
        "config": {"dataset_path": COCO_ROOT, "backbone": BACKBONE_NAME, "num_epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE},
        "training_time": {"start_time": training_start_dt.strftime("%Y-%m-%d %H:%M:%S"), "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "total_duration": total_time},
        "training_progress": {"completed_epochs": epoch + 1, "best_val_f1": best_f1, "best_epoch": best_epoch, "early_stopped": epochs_no_improve >= EARLY_STOP_PATIENCE, "val_f1_history": val_f1_history},
        "loss_history": loss_history, "final_metrics": final_metrics,
        "model_info": {"model_path": MODEL_PATH, "file_size_mb": os.path.getsize(MODEL_PATH)/(1024*1024) if os.path.isfile(MODEL_PATH) else 0}
    }
    save_training_results(training_info, OUTPUT_DIR)

if __name__ == "__main__":
    main()