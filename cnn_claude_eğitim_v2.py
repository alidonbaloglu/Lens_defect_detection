import os
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.roi_heads import fastrcnn_loss as tv_fastrcnn_loss
from torchvision.ops import boxes as box_ops
import torch.nn.functional as F
import numpy as np
from PIL import Image
import json
import cv2
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time

# ============================================
# 1. EN İYİ PARAMETRELERLE KONFİGÜRASYON
# ============================================
# Veri Seti ve Model Yolları
COCO_ROOT = r"C:/Users/ali.donbaloglu/Desktop/Lens/datasetler/merged_coco_dataset"
TRAIN_SPLIT = "train"
VAL_SPLIT = "valid"
OUTPUT_DIR = "results/best_params_run"
MODEL_PATH = os.path.join(OUTPUT_DIR, "maskrcnn_resnet101_best.pt")
LOG_PATH = os.path.join(OUTPUT_DIR, "training_log_best.json")

NUM_CLASSES = 7
BACKBONE_NAME = "resnet101"
NUM_EPOCHS = 80
BATCH_SIZE = 4
LEARNING_RATE = 1e-4      # ← Başlangıçta 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# 2. YARDIMCI SINIFLAR VE FONKSİYONLAR
# ============================================

class MultiModalPreprocessor:
    """Görüntüden RGB, Kenar (Edge) ve Gradyan kanallarını çıkarır."""
    def __call__(self, image_np: np.ndarray):
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Kenar tespiti (Canny)
        edges = cv2.Canny(gray, 60, 150)
        
        # Gradyan tespiti (Sobel)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        # Normalizasyon
        if gradient.max() > 0:
            gradient = (gradient / gradient.max() * 255).astype(np.uint8)
        
        return {
            'image': image_np, # Albumentations için orijinal RGB gerekli
            'edge': np.expand_dims(edges, axis=-1),
            'gradient': np.expand_dims(gradient, axis=-1)
        }

class RefinedMaskRCNNPredictor(torch.nn.Module):
    """Kontur (sınır) bilgisiyle zenginleştirilmiş maske tahmincisi."""
    def __init__(self, in_channels, dim_reduced, num_classes):
        super().__init__()
        self.conv5_mask = torch.nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)
        self.relu = torch.nn.ReLU(inplace=True)
        self.mask_fcn_logits = torch.nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        
        # Kontur iyileştirme katmanları
        self.contour_conv1 = torch.nn.Conv2d(dim_reduced, 128, 3, padding=1)
        self.contour_conv2 = torch.nn.Conv2d(128, 64, 3, padding=1)
        self.contour_conv3 = torch.nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x):
        x = self.conv5_mask(x)
        x = self.relu(x)
        
        mask_logits = self.mask_fcn_logits(x)
        
        # Kontur tahmini ve ana maskeye eklenmesi
        contour = self.relu(self.contour_conv1(x))
        contour = self.relu(self.contour_conv2(contour))
        contour_logits = self.contour_conv3(contour)
        
        # Ana maske ve kontur bilgisini birleştir (0.3 ağırlığı ayarlanabilir bir hiperparametredir)
        return mask_logits + 0.3 * contour_logits

class BoundaryAwareMaskLoss:
    """Maske sınırlarına daha fazla ağırlık veren özel kayıp fonksiyonu."""
    def __init__(self, boundary_weight=1.8):
        self.boundary_weight = boundary_weight

    def __call__(self, mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
        # PyTorch'un orijinal mask_rcnn_loss fonksiyonundan alınmıştır
        mask_targets = [
            torch.cat([gt_masks[i][j] for j in js])
            for i, js in enumerate(mask_matched_idxs)
        ]
        
        # Sınıf etiketlerine göre doğru kanalları seç
        labels = [torch.cat(list(gt_labels), dim=0)]
        proposals = torch.cat(proposals, dim=0)
        pos_matched_idxs = torch.cat([torch.as_tensor(list(range(len(p)))) for p in proposals], dim=0)

        # Doğru sınıf etiketlerini al
        labels = labels[0]
        
        # Maske hedeflerini oluştur
        mask_targets_cat = torch.cat(mask_targets, dim=0)

        # Logitlerden ilgili sınıf tahminlerini seç
        selected_logits = mask_logits[pos_matched_idxs, labels]

        # BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(
            selected_logits, mask_targets_cat, reduction='none'
        )

        # Sınır ağırlıklarını hesapla
        target_np = mask_targets_cat.cpu().detach().numpy()
        boundary_weights = torch.ones_like(mask_targets_cat)
        for i in range(target_np.shape[0]):
            mask = (target_np[i] > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boundary_mask = np.zeros_like(mask, dtype=np.float32)
            cv2.drawContours(boundary_mask, contours, -1, 1, thickness=3)
            # Sınır piksellerine `boundary_weight` kadar ek ağırlık ver
            boundary_weights[i] = (torch.from_numpy(boundary_mask).to(mask_targets_cat.device) * self.boundary_weight) + 1.0
        
        # Ağırlıklı kaybı hesapla
        weighted_loss = bce_loss * boundary_weights
        return weighted_loss.mean()

class BoundaryAwareRoIHeads(RoIHeads):
    """Boundary-Aware Loss'u kullanmak için özelleştirilmiş RoIHeads."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_loss = BoundaryAwareMaskLoss()

    def forward(self, features, proposals, image_shapes, targets=None):
        # Delegate to the base implementation. The main reason to keep this
        # subclass is to provide a custom mask predictor and mask loss object,
        # but the torchvision RoIHeads.forward implementation already handles
        # proposal sampling, mask processing and loss computation robustly.
        #
        # By calling super().forward we avoid reimplementing internal logic
        # (and the risk of missing private attributes like `mask_sampler`).
        return super().forward(features, proposals, image_shapes, targets)


# ============================================
# 3. VERİ SETİ SINIFI (GELİŞMİŞ AUGMENTATION İLE)
# ============================================
class MultiModalCOCODataset(Dataset):
    def __init__(self, json_path, img_root, use_augmentation=False):
        self.coco = COCO(json_path)
        self.img_root = img_root
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.preprocessor = MultiModalPreprocessor()
        self.use_augmentation = use_augmentation
        
        
        # For the non-aug path also ensure additional targets are handled
        self.transform = A.Compose([
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']),
        additional_targets={'edge': 'image', 'gradient': 'image'})

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_root, img_info['file_name'])
        img_np = np.array(Image.open(img_path).convert('RGB'))
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes, labels, masks = [], [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, w, h])
            labels.append(ann['category_id'])
            masks.append(self.coco.annToMask(ann))
        
        modalities = self.preprocessor(img_np)
        
        if self.transform and len(boxes) > 0:
            try:
                transformed = self.transform(
                    image=modalities['image'],
                    bboxes=boxes,
                    labels=labels,
                    masks=masks,
                    edge=modalities['edge'],
                    gradient=modalities['gradient']
                )
            except ValueError as e:
                # Albumentations can sometimes produce out-of-bounds bbox values
                # after geometric transforms; catch and fallback to a safe
                # non-augmented conversion for this sample to avoid crashing
                # the DataLoader worker.
                import warnings
                warnings.warn(f"Augmentation produced invalid bboxes, skipping augment for this sample: {e}")
                # Build tensors from original modalities (no augmentation)
                rgb_tensor = A.pytorch.ToTensorV2()(image=modalities['image'])['image'] / 255.0
                edge_tensor = torch.from_numpy(modalities['edge']).permute(2,0,1).float() / 255.0
                gradient_tensor = torch.from_numpy(modalities['gradient']).permute(2,0,1).float() / 255.0
                multi_modal_tensor = torch.cat([rgb_tensor, edge_tensor, gradient_tensor], dim=0)

                target = {}
                transformed_boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in boxes]
                target['boxes'] = torch.as_tensor(transformed_boxes, dtype=torch.float32)
                target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
                target['masks'] = torch.as_tensor(np.array(masks), dtype=torch.uint8)
                return multi_modal_tensor, target
            # RGB, Edge, Gradient kanallarını birleştir
            # If augmentation succeeded, Albumentations with additional_targets
            # and ToTensorV2 returns tensors for image/edge/gradient
            rgb_tensor = transformed['image'] / 255.0
            # transformed['edge'] and ['gradient'] are tensors or arrays; normalize to (C,H,W)
            edge_t = transformed['edge']
            gradient_t = transformed['gradient']
            if isinstance(edge_t, np.ndarray):
                edge_tensor = torch.from_numpy(edge_t).permute(2,0,1).float() / 255.0
            else:
                edge_tensor = edge_t.float().unsqueeze(0) / 255.0 if edge_t.dim() == 2 else edge_t.float() / 255.0

            if isinstance(gradient_t, np.ndarray):
                gradient_tensor = torch.from_numpy(gradient_t).permute(2,0,1).float() / 255.0
            else:
                gradient_tensor = gradient_t.float().unsqueeze(0) / 255.0 if gradient_t.dim() == 2 else gradient_t.float() / 255.0

            multi_modal_tensor = torch.cat([rgb_tensor, edge_tensor, gradient_tensor], dim=0)

            target = {}
            # Bbox formatını [x_min, y_min, x_max, y_max] yap
            transformed_boxes = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in transformed['bboxes']]
            # Ensure boxes is a (N,4) tensor even when empty. torch.as_tensor([])
            # yields a 1-D empty tensor which breaks torchvision checks.
            if len(transformed_boxes) == 0:
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            else:
                target['boxes'] = torch.as_tensor(transformed_boxes, dtype=torch.float32)

            if len(transformed.get('labels', [])) == 0:
                target['labels'] = torch.zeros((0,), dtype=torch.int64)
            else:
                target['labels'] = torch.as_tensor(transformed['labels'], dtype=torch.int64)
            # transformed['masks'] can be a list of numpy arrays or tensors;
            # convert robustly to a (N,H,W) numpy array first, then to tensor.
            masks_out = transformed['masks']
            if isinstance(masks_out, (list, tuple)):
                processed = []
                for mm in masks_out:
                    if isinstance(mm, np.ndarray):
                        processed.append(mm)
                    elif torch.is_tensor(mm):
                        processed.append(mm.cpu().numpy())
                    else:
                        processed.append(np.array(mm))
                masks_np = np.stack(processed, axis=0) if len(processed) > 0 else np.zeros((0, img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
            else:
                if torch.is_tensor(masks_out):
                    masks_np = masks_out.cpu().numpy()
                else:
                    masks_np = np.array(masks_out)

            target['masks'] = torch.as_tensor(masks_np, dtype=torch.uint8)
        else: # Eğer resimde nesne yoksa veya augmentation yoksa
             multi_modal_tensor = torch.cat([
                 A.pytorch.ToTensorV2()(image=img_np)['image'] / 255.0,
                 torch.from_numpy(modalities['edge']).permute(2,0,1).float() / 255.0,
                 torch.from_numpy(modalities['gradient']).permute(2,0,1).float() / 255.0,
             ], dim=0)
             target = {
                 'boxes': torch.zeros((0, 4), dtype=torch.float32),
                 'labels': torch.zeros(0, dtype=torch.int64),
                 'masks': torch.zeros((0, img_np.shape[0], img_np.shape[1]), dtype=torch.uint8)
             }
        
        return multi_modal_tensor, target

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    return tuple(zip(*batch))

# ============================================
# 4. MODEL OLUŞTURMA
# ============================================
def create_enhanced_maskrcnn(num_classes):
    """Tüm iyileştirmeleri içeren modeli oluşturur."""
    # Try to use the convenience constructor if available (some torchvision
    # versions don't provide maskrcnn_resnet101_fpn). If it's missing, build a
    # ResNet-101+FPN backbone manually and construct MaskRCNN.
    if hasattr(torchvision.models.detection, 'maskrcnn_resnet101_fpn'):
        model = torchvision.models.detection.maskrcnn_resnet101_fpn(weights='DEFAULT')
    else:
        # Build ResNet101+FPN backbone and pass to MaskRCNN
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        # BACKBONE_NAME is defined at module level (e.g. 'resnet101')
        backbone = resnet_fpn_backbone(BACKBONE_NAME, pretrained=True)

        # Adjust first conv to accept 5-channel multimodal input (3 RGB + edge + gradient)
        original_conv1 = backbone.body.conv1
        new_conv1 = torch.nn.Conv2d(
            5, original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=(original_conv1.bias is not None)
        )
        with torch.no_grad():
            # Copy RGB weights
            new_conv1.weight[:, :3, :, :] = original_conv1.weight
            # Initialize extra channels as mean of RGB weights
            new_conv1.weight[:, 3:, :, :] = original_conv1.weight.mean(dim=1, keepdim=True)

        backbone.body.conv1 = new_conv1

        # Create MaskRCNN with this backbone
        model = MaskRCNN(backbone, num_classes=num_classes)

    # Ensure model has been created; now replace box/mask predictors and RoIHeads
    # 3. Kutu tahmincisini (box predictor) bizim sınıf sayımıza göre ayarla
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)

    # 4. Maske tahmincisini (mask predictor) bizim özel `RefinedMaskRCNNPredictor` ile değiştir
    # Some torchvision variants name internal modules slightly differently, so guard access
    try:
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = RefinedMaskRCNNPredictor(in_features_mask, 256, num_classes)
    except Exception:
        # If mask_predictor layout differs, try to infer channels from mask_head output
        try:
            # mask_head often ends with a conv to reduce channels
            in_features_mask = model.roi_heads.mask_head[-1].in_channels
            model.roi_heads.mask_predictor = RefinedMaskRCNNPredictor(in_features_mask, 256, num_classes)
        except Exception:
            # Fallback: skip replacing mask predictor (still works with default)
            pass

    # 5. RoIHeads'i, özel Boundary-Aware Loss'u kullanan versiyonla değiştir
    box_roi_pool = model.roi_heads.box_roi_pool
    mask_roi_pool = model.roi_heads.mask_roi_pool
    box_head = model.roi_heads.box_head
    mask_head = model.roi_heads.mask_head

    model.roi_heads = BoundaryAwareRoIHeads(
        box_roi_pool=box_roi_pool,
        box_head=box_head,
        box_predictor=model.roi_heads.box_predictor,
        fg_iou_thresh=0.5, bg_iou_thresh=0.5,
        batch_size_per_image=512, positive_fraction=0.25,
        bbox_reg_weights=None,
        score_thresh=0.05, nms_thresh=0.5, detections_per_img=100,
        mask_roi_pool=mask_roi_pool,
        mask_head=mask_head,
        mask_predictor=model.roi_heads.mask_predictor
    )

    # Adjust the model's internal transform to accept 5-channel images.
    # The default transform expects 3-channel RGB; since we feed 5 channels
    # (RGB + edge + gradient) we must provide matching mean/std vectors.
    try:
        from torchvision.models.detection.transform import GeneralizedRCNNTransform

        # Try to reuse original min/max size settings
        try:
            orig_t = model.transform
            min_size = orig_t.min_size
            max_size = orig_t.max_size
            # orig image_mean/std may be tensors or lists
            img_mean = list(orig_t.image_mean) if hasattr(orig_t, 'image_mean') else [0.485, 0.456, 0.406]
            img_std = list(orig_t.image_std) if hasattr(orig_t, 'image_std') else [0.229, 0.224, 0.225]
        except Exception:
            min_size = 800
            max_size = 1333
            img_mean = [0.485, 0.456, 0.406]
            img_std = [0.229, 0.224, 0.225]

        # Extend mean/std to 5 channels for edge and gradient. Use 0.5/0.5 as a
        # reasonable default normalization for those extra single-channel maps.
        if len(img_mean) == 3:
            img_mean = img_mean + [0.5, 0.5]
            img_std = img_std + [0.5, 0.5]
        else:
            # Ensure length is at least 5
            while len(img_mean) < 5:
                img_mean.append(0.5)
                img_std.append(0.5)

        model.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean=img_mean, image_std=img_std)
    except Exception:
        # If anything goes wrong here, fall back to leaving model.transform as-is.
        pass

    return model

# ============================================
# 5. EĞİTİM VE DEĞERLENDİRME DÖNGÜSÜ
# ============================================

def evaluate_pixel_f1(model, dataloader, device):
    """Basit F1 (IoU) değerlendirmesi yapar."""
    model.eval()
    total_f1 = 0
    count = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = list(img.to(device) for img in images)
            predictions = model(images)
            for pred, target in zip(predictions, targets):
                if len(pred['masks']) > 0 and len(target['masks']) > 0:
                    pred_mask = (pred['masks'][0, 0] > 0.5).cpu().numpy().astype(np.uint8)
                    true_mask = target['masks'][0].cpu().numpy().astype(np.uint8)
                    
                    intersection = np.logical_and(pred_mask, true_mask).sum()
                    union = np.logical_or(pred_mask, true_mask).sum()
                    
                    if union > 0:
                        iou = intersection / union
                        total_f1 += iou
                        count += 1
    return total_f1 / count if count > 0 else 0.0

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Veri setlerini hazırla
    train_images_root = os.path.join(COCO_ROOT, TRAIN_SPLIT)
    train_json = os.path.join(COCO_ROOT, TRAIN_SPLIT, "_annotations.coco.json")
    val_images_root = os.path.join(COCO_ROOT, VAL_SPLIT)
    val_json = os.path.join(COCO_ROOT, VAL_SPLIT, "_annotations.coco.json")
    
    train_dataset = MultiModalCOCODataset(train_json, train_images_root)
    val_dataset = MultiModalCOCODataset(val_json, val_images_root)

    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Modeli oluştur ve cihaza taşı
    model = create_enhanced_maskrcnn(NUM_CLASSES)
    model.to(DEVICE)

    # Optimizatör ve Öğrenme Hızı Zamanlayıcısı
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Öğrenme hızı planlayıcı (10. epoch’tan sonra 5e-5’e düş)
    def adjust_learning_rate(optimizer, epoch):
        if epoch == 10:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 5e-5
            print("\n📉 Öğrenme hızı 1e-4 → 5e-5 olarak düşürüldü.\n")

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=10, min_lr=1e-7
    )

    best_f1 = 0.0
    patience_counter = 0
    training_log = {"epochs": []}

    print("Eğitim başlıyor...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        for i, (images, targets) in enumerate(train_loader):
            images = list(img.to(DEVICE) for img in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_loader)}, Loss: {losses.item():.4f}", end='\r')

        avg_loss = epoch_loss / len(train_loader)
        val_f1 = evaluate_pixel_f1(model, val_loader, DEVICE)
        
        # Öğrenme hızını güncelle
        lr_scheduler.step(val_f1)
         # 🔹 Buraya ekliyorsun
        adjust_learning_rate(optimizer, epoch + 1)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {avg_loss:.4f}, Val F1: {val_f1:.4f}, LR: {current_lr:.7f}")
        
        training_log["epochs"].append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_f1": val_f1,
            "lr": current_lr
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> Yeni en iyi model kaydedildi! F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"  -> F1 skoru artmadı. Sabır: {patience_counter}/{EARLY_STOP_PATIENCE}")
        
        with open(LOG_PATH, 'w') as f:
            json.dump(training_log, f, indent=4)
        
        if patience_counter >= EARLY_STOP_PATIENCE:
            print("Erken durdurma tetiklendi. Eğitim sonlandırılıyor.")
            break
            
    print(f"Eğitim tamamlandı. En iyi F1 skoru: {best_f1:.4f}")

if __name__ == "__main__":
    main()