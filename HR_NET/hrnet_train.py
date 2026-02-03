"""
HRNet ile Faster R-CNN - COCO Dataset için Komple Kod
4 sınıf (background dahil) için hazır training kodu
Validation ve Test değerlendirmesi ile birlikte
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from collections import OrderedDict
import os
from PIL import Image
import json
import numpy as np
from collections import defaultdict

# ============================================
# 1. HRNet Backbone
# ============================================

class HRNetBackbone(nn.Module):
    """HRNet backbone for Faster R-CNN"""
    
    def __init__(self, pretrained=True):
        super(HRNetBackbone, self).__init__()
        
        import timm
        self.hrnet = timm.create_model('hrnet_w32', pretrained=pretrained, features_only=True)
        self.out_channels = 256
        self.fusion = nn.Conv2d(1984, 256, kernel_size=1)
        
    def forward(self, x):
        features = self.hrnet(x)
        target_size = features[-1].shape[2:]
        
        upsampled_features = []
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = nn.functional.interpolate(
                    feat, size=target_size, mode='bilinear', align_corners=False
                )
            upsampled_features.append(feat)
        
        fused = torch.cat(upsampled_features, dim=1)
        output = self.fusion(fused)
        
        return OrderedDict([('0', output)])


# ============================================
# 2. COCO Dataset Loader
# ============================================

class COCODataset(torch.utils.data.Dataset):
    """COCO formatındaki dataset için custom dataset class"""
    
    def __init__(self, root_dir, annotation_file, transforms=None):
        """
        Args:
            root_dir: Görüntülerin bulunduğu klasör
            annotation_file: COCO format JSON annotation dosyası
            transforms: Augmentation transforms
        """
        self.root_dir = root_dir
        self.transforms = transforms
        
        # COCO annotations yükle
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        
        # Image ID'ye göre annotations'ları grupla
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
            
        # Sadece annotation'ı olan resimleri tut (Training için genellikle gereklidir)
        self.images = [img for img in self.images if img['id'] in self.img_to_anns]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Görüntü bilgisini al
        img_info = self.images[idx]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        
        # Annotations'ları al
        img_id = img_info['id']
        anns = self.img_to_anns.get(img_id, [])
        
        # Boxes ve labels hazırla
        boxes = []
        labels = []
        
        for ann in anns:
            # COCO format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # Faster R-CNN format: [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        # Tensor'a çevir
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # Eğer kutu yoksa boş ama doğru boyutta tensor oluştur (N, 4)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        # Target dictionary oluştur
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])
        
        # Transforms uygula
        if self.transforms is not None:
            img = self.transforms(img)
        else:
            # Default transform
            img = torchvision.transforms.ToTensor()(img)
        
        return img, target


# ============================================
# 3. Collate Function
# ============================================

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return tuple(zip(*batch))


# ============================================
# 4. Model Oluşturma
# ============================================

def create_hrnet_faster_rcnn(num_classes, pretrained_backbone=True):
    """HRNet backbone ile Faster R-CNN modeli oluştur"""
    
    backbone = HRNetBackbone(pretrained=pretrained_backbone)
    
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


# ============================================
# 5. Training Fonksiyonu
# ============================================

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """Bir epoch training"""
    model.train()
    total_loss = 0
    
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
        # Progress
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch}], Step [{i+1}/{len(data_loader)}], Loss: {losses.item():.4f}")
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss


# ============================================
# 6. IoU ve Metrik Hesaplama Fonksiyonları
# ============================================

def calculate_iou(box1, box2):
    """İki bounding box arasındaki IoU hesapla"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def calculate_ap(precisions, recalls):
    """11-point interpolation ile AP hesapla"""
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        precisions_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
        if precisions_at_recall:
            ap += max(precisions_at_recall)
    return ap / 11


def evaluate_model(model, data_loader, device, iou_threshold=0.5, conf_threshold=0.5):
    """
    Model değerlendirmesi - mAP, Precision, Recall, F1 hesapla
    """
    model.eval()
    
    all_predictions = []  # (image_id, class_id, confidence, box)
    all_ground_truths = []  # (image_id, class_id, box)
    
    # Debug için istatistikler
    total_raw_preds = 0
    max_score_seen = 0.0
    score_distribution = {'0.0-0.1': 0, '0.1-0.25': 0, '0.25-0.5': 0, '0.5-0.75': 0, '0.75-1.0': 0}
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            predictions = model(images)
            
            for idx, (pred, target) in enumerate(zip(predictions, targets)):
                image_id = target['image_id'].item()
                
                # Ground truth boxes
                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy()
                
                for box, label in zip(gt_boxes, gt_labels):
                    all_ground_truths.append((image_id, label, box))
                
                # Predictions
                pred_boxes = pred['boxes'].cpu().numpy()
                pred_labels = pred['labels'].cpu().numpy()
                pred_scores = pred['scores'].cpu().numpy()
                
                # Debug: ham tahmin sayısı
                total_raw_preds += len(pred_scores)
                
                for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                    # Score dağılımı
                    if score > max_score_seen:
                        max_score_seen = score
                    if score < 0.1:
                        score_distribution['0.0-0.1'] += 1
                    elif score < 0.25:
                        score_distribution['0.1-0.25'] += 1
                    elif score < 0.5:
                        score_distribution['0.25-0.5'] += 1
                    elif score < 0.75:
                        score_distribution['0.5-0.75'] += 1
                    else:
                        score_distribution['0.75-1.0'] += 1
                    
                    if score >= conf_threshold:
                        all_predictions.append((image_id, label, score, box))
    
    # Debug çıktısı
    print(f"\n[DEBUG] Toplam ham tahmin sayısı: {total_raw_preds}")
    print(f"[DEBUG] En yüksek confidence score: {max_score_seen:.4f}")
    print(f"[DEBUG] Score dağılımı: {score_distribution}")
    print(f"[DEBUG] Threshold ({conf_threshold}) üstü tahmin: {len(all_predictions)}")
    print(f"[DEBUG] Toplam ground truth: {len(all_ground_truths)}\n")
    
    # Sınıf bazlı metrikler
    class_ids = set([gt[1] for gt in all_ground_truths])
    class_metrics = {}
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    aps = []
    
    for class_id in class_ids:
        # Bu sınıf için GT ve predictions
        class_gt = [(gt[0], gt[2]) for gt in all_ground_truths if gt[1] == class_id]
        class_preds = [(p[0], p[2], p[3]) for p in all_predictions if p[1] == class_id]
        
        # Confidence'a göre sırala
        class_preds.sort(key=lambda x: x[1], reverse=True)
        
        # GT'leri işaretlemek için
        gt_matched = {i: False for i in range(len(class_gt))}
        
        tp = 0
        fp = 0
        
        precisions = []
        recalls = []
        
        for pred_img_id, pred_conf, pred_box in class_preds:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, (gt_img_id, gt_box) in enumerate(class_gt):
                if gt_img_id == pred_img_id and not gt_matched[gt_idx]:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp += 1
                gt_matched[best_gt_idx] = True
            else:
                fp += 1
            
            # Precision ve Recall hesapla
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / len(class_gt) if len(class_gt) > 0 else 0
            precisions.append(precision)
            recalls.append(recall)
        
        # FN: eşleşmeyen GT sayısı
        fn = sum(1 for matched in gt_matched.values() if not matched)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # AP hesapla
        if precisions and recalls:
            ap = calculate_ap(precisions, recalls)
        else:
            ap = 0
        aps.append(ap)
        
        # Sınıf metrikleri
        class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
        
        class_metrics[class_id] = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision': class_precision,
            'Recall': class_recall,
            'F1': class_f1,
            'AP': ap
        }
    
    # Genel metrikler
    mAP = np.mean(aps) if aps else 0
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    metrics = {
        'mAP': mAP,
        'Precision': overall_precision,
        'Recall': overall_recall,
        'F1': overall_f1,
        'Total_TP': total_tp,
        'Total_FP': total_fp,
        'Total_FN': total_fn,
        'class_metrics': class_metrics
    }
    
    return metrics


def print_metrics(metrics, title="Evaluation Results"):
    """Metrikleri güzel formatta yazdır"""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"mAP@0.5:     {metrics['mAP']:.4f}")
    print(f"Precision:   {metrics['Precision']:.4f}")
    print(f"Recall:      {metrics['Recall']:.4f}")
    print(f"F1-Score:    {metrics['F1']:.4f}")
    print(f"Total TP:    {metrics['Total_TP']}")
    print(f"Total FP:    {metrics['Total_FP']}")
    print(f"Total FN:    {metrics['Total_FN']}")
    
    if metrics.get('class_metrics'):
        print(f"\n{'─'*60}")
        print("Sınıf Bazlı Metrikler:")
        print(f"{'─'*60}")
        for class_id, cm in metrics['class_metrics'].items():
            print(f"  Class {class_id}: AP={cm['AP']:.4f}, P={cm['Precision']:.4f}, R={cm['Recall']:.4f}, F1={cm['F1']:.4f}")
    print(f"{'='*60}\n")


# ============================================
# 7. Ana Training Loop
# ============================================

def main():
    # ===== AYARLAR - BUNLARI KENDİ DATASET'İNE GÖRE DEĞİŞTİR =====
    TRAIN_IMG_DIR = "C:/Users/ali.donbaloglu/Desktop/Lens/datasetler/Lens_Kabin_Aralik_CNN/train"  # Train görüntüleri klasörü
    TRAIN_ANN_FILE = "C:/Users/ali.donbaloglu/Desktop/Lens/datasetler/Lens_Kabin_Aralik_CNN/train/_annotations.coco.json"  # Train annotations JSON
    VAL_IMG_DIR = "C:/Users/ali.donbaloglu/Desktop/Lens/datasetler/Lens_Kabin_Aralik_CNN/valid"  # Validation görüntüleri
    VAL_ANN_FILE = "C:/Users/ali.donbaloglu/Desktop/Lens/datasetler/Lens_Kabin_Aralik_CNN/valid/_annotations.coco.json"  # Validation annotations
    TEST_IMG_DIR = "C:/Users/ali.donbaloglu/Desktop/Lens/datasetler/Lens_Kabin_Aralik_CNN/test"  # Test görüntüleri
    TEST_ANN_FILE = "C:/Users/ali.donbaloglu/Desktop/Lens/datasetler/Lens_Kabin_Aralik_CNN/test/_annotations.coco.json"  # Test annotations
    
    NUM_CLASSES = 4  # Background dahil 4 sınıf
    NUM_EPOCHS = 2
    BATCH_SIZE = 2
    LEARNING_RATE = 0.005
    SAVE_DIR = "checkpoints"  # Model kaydetme klasörü
    IOU_THRESHOLD = 0.25  # IoU eşik değeri (düşük başla, model olgunlaşınca 0.5 yapılabilir)
    CONF_THRESHOLD = 0.1  # Confidence eşik değeri (eğitim başında düşük, sonra 0.5 yapılabilir)
    # ===========================================================
    
    # Checkpoint klasörü oluştur
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    # ===== DATASETS =====
    print("\nLoading datasets...")
    
    # Train dataset
    train_dataset = COCODataset(TRAIN_IMG_DIR, TRAIN_ANN_FILE)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    print(f"Train dataset size: {len(train_dataset)}")
    
    # Validation dataset
    val_dataset = COCODataset(VAL_IMG_DIR, VAL_ANN_FILE)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Test dataset
    test_dataset = COCODataset(TEST_IMG_DIR, TEST_ANN_FILE)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Model oluştur
    print("\nCreating model...")
    model = create_hrnet_faster_rcnn(num_classes=NUM_CLASSES, pretrained_backbone=True)
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1
    )
    
    # Training loop
    print("\nStarting training...")
    best_mAP = 0.0
    best_model_path = os.path.join(SAVE_DIR, 'best_model.pt')  # Önceden tanımla
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch+1)
        lr_scheduler.step()
        
        print(f"\nEpoch {epoch+1} - Average Loss: {train_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Validation değerlendirmesi
        print("\nRunning validation...")
        val_metrics = evaluate_model(model, val_loader, device, IOU_THRESHOLD, CONF_THRESHOLD)
        print_metrics(val_metrics, f"Validation Results - Epoch {epoch+1}")
        
        # Model kaydet - Her epoch
        checkpoint_path = os.path.join(SAVE_DIR, f'hrnet_fasterrcnn_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'val_mAP': val_metrics['mAP'],
            'val_precision': val_metrics['Precision'],
            'val_recall': val_metrics['Recall'],
            'val_f1': val_metrics['F1'],
        }, checkpoint_path)
        
        # En iyi model (mAP'e göre)
        if val_metrics['mAP'] > best_mAP:
            best_mAP = val_metrics['mAP']
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ Best model saved! mAP: {best_mAP:.4f}")
        
        # İlk epoch'ta da kaydet (en azından bir model olsun)
        if epoch == 0:
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ Initial model saved (epoch 1)")
    
    # Son model kaydet
    last_model_path = os.path.join(SAVE_DIR, 'last_model.pt')
    torch.save(model.state_dict(), last_model_path)
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation mAP: {best_mAP:.4f}")
    print("="*60)
    
    # ===== TEST SET DEĞERLENDİRMESİ =====
    print("\n" + "="*60)
    print("FINAL TEST SET EVALUATION")
    print("="*60)
    
    # Best model ile test
    print("\n--- Best Model Test Results ---")
    model.load_state_dict(torch.load(best_model_path, weights_only=False))
    best_test_metrics = evaluate_model(model, test_loader, device, IOU_THRESHOLD, CONF_THRESHOLD)
    print_metrics(best_test_metrics, "Test Results - Best Model")
    
    # Last model ile test
    print("\n--- Last Model Test Results ---")
    model.load_state_dict(torch.load(last_model_path, weights_only=False))
    last_test_metrics = evaluate_model(model, test_loader, device, IOU_THRESHOLD, CONF_THRESHOLD)
    print_metrics(last_test_metrics, "Test Results - Last Model")
    
    # Sonuçları JSON olarak kaydet
    results = {
        'training': {
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'best_val_mAP': best_mAP,
        },
        'test_results_best_model': {
            'mAP': best_test_metrics['mAP'],
            'Precision': best_test_metrics['Precision'],
            'Recall': best_test_metrics['Recall'],
            'F1': best_test_metrics['F1'],
            'TP': best_test_metrics['Total_TP'],
            'FP': best_test_metrics['Total_FP'],
            'FN': best_test_metrics['Total_FN'],
        },
        'test_results_last_model': {
            'mAP': last_test_metrics['mAP'],
            'Precision': last_test_metrics['Precision'],
            'Recall': last_test_metrics['Recall'],
            'F1': last_test_metrics['F1'],
            'TP': last_test_metrics['Total_TP'],
            'FP': last_test_metrics['Total_FP'],
            'FN': last_test_metrics['Total_FN'],
        }
    }
    
    results_path = os.path.join(SAVE_DIR, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {results_path}")


# ============================================
# 8. Inference Fonksiyonu
# ============================================

def run_inference(model_path, image_path, num_classes=4, threshold=0.5):
    """Eğitilmiş model ile inference yap"""
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Model yükle
    model = create_hrnet_faster_rcnn(num_classes=num_classes, pretrained_backbone=False)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()
    
    # Görüntü yükle
    image = Image.open(image_path).convert("RGB")
    image_tensor = torchvision.transforms.ToTensor()(image).to(device)
    
    # Inference
    with torch.no_grad():
        prediction = model([image_tensor])
    
    # Sonuçları filtrele
    pred = prediction[0]
    boxes = pred['boxes'][pred['scores'] > threshold].cpu().numpy()
    labels = pred['labels'][pred['scores'] > threshold].cpu().numpy()
    scores = pred['scores'][pred['scores'] > threshold].cpu().numpy()
    
    print(f"Detected {len(boxes)} objects:")
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        print(f"  Object {i+1}: Class {label}, Score: {score:.3f}, Box: {box}")
    
    return boxes, labels, scores


# ============================================
# 9. KULLANIM
# ============================================

if __name__ == "__main__":
    # Training başlat
    main()
    
    # Inference örneği (training bittikten sonra)
    # run_inference(
    #     model_path="checkpoints/best_model.pt",
    #     image_path="C:/Users/omerd/Desktop/omer/HRNet/test/sample.jpg",
    #     num_classes=4,
    #     threshold=0.5
    # )