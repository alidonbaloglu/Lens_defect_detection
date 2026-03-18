import torch
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.ops import box_iou
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import sys
from pathlib import Path as _Path
# Ensure project root is on sys.path to import training module
_current_file = _Path(__file__).resolve()
_project_root = _current_file.parents[2]  # C:\Users\...\Desktop\Lens
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from mask_rcnn_coco_object_detection import COCOSegmentationDataset, get_model

# Kategorileri JSON'dan yükle
def load_categories(coco_json_path):
    with open(coco_json_path, 'r') as f:
        data = json.load(f)
    
    categories = {}
    for cat in data['categories']:
        categories[cat['id']] = cat['name']
    
    return categories, data

# IoU hesaplama ve eşleştirme
def match_predictions_to_gt(pred_boxes, pred_labels, pred_scores, 
                           gt_boxes, gt_labels, iou_threshold=0.5, 
                           conf_threshold=0.5):
    """
    Tahminleri ground truth ile eşleştir
    """
    matches = []
    
    # Confidence threshold'dan düşük tahminleri filtrele
    valid_preds = pred_scores > conf_threshold
    pred_boxes = pred_boxes[valid_preds]
    pred_labels = pred_labels[valid_preds]
    pred_scores = pred_scores[valid_preds]

    # Eğitimde kullanılan dataset 1-based sınıf etiketleri (0=background) kullanır.
    # Background (0) etiketli pred/gt'yi filtrele ve indeksleri 0-based'e çevir.
    if len(pred_labels) > 0:
        keep_fg = pred_labels > 0
        pred_boxes = pred_boxes[keep_fg]
        pred_labels = pred_labels[keep_fg] - 1
        pred_scores = pred_scores[keep_fg]
    
    if len(pred_boxes) == 0:
        # Hiç tahmin yoksa, tüm GT'ler "tespit edilemedi" olarak işaretle
        for gt_label in gt_labels:
            matches.append({
                'gt_label': gt_label.item(),
                'pred_label': -1,  # -1: No detection
                'iou': 0.0
            })
        return matches
    
    if len(gt_boxes) == 0:
        # GT yoksa ama tahmin varsa (False Positive durumu)
        return matches
    
    # IoU matrisini hesapla
    iou_matrix = box_iou(gt_boxes, pred_boxes)
    
    # Her GT için en iyi eşleşmeyi bul
    matched_pred_indices = set()
    
    for gt_idx in range(len(gt_boxes)):
        gt_label = (gt_labels[gt_idx].item() - 1) if gt_labels[gt_idx].item() > 0 else -1
        
        # Bu GT için tüm tahminlerin IoU'larını al
        ious = iou_matrix[gt_idx]
        
        # En yüksek IoU'yu ve indeksini bul
        best_iou = 0.0
        best_pred_idx = -1
        
        for pred_idx in range(len(pred_boxes)):
            # Bu tahmin daha önce kullanılmışsa atla
            if pred_idx in matched_pred_indices:
                continue
                
            if ious[pred_idx] > best_iou and ious[pred_idx] >= iou_threshold:
                best_iou = ious[pred_idx].item()
                best_pred_idx = pred_idx
        
        if best_pred_idx >= 0 and gt_label >= 0:
            # Eşleşme bulundu
            matched_pred_indices.add(best_pred_idx)
            matches.append({
                'gt_label': gt_label,
                'pred_label': pred_labels[best_pred_idx].item(),
                'iou': best_iou
            })
        else:
            # Eşleşme bulunamadı (False Negative)
            matches.append({
                'gt_label': gt_label,
                'pred_label': -1,  # No detection
                'iou': 0.0
            })
    
    return matches

# Test setinde model çalıştırma
def evaluate_model(model, test_loader, device, iou_threshold=0.5, conf_threshold=0.5):
    """
    Model tahminlerini topla ve eşleştir
    """
    model.eval()
    all_matches = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            
            # Model tahmini
            predictions = model(images)
            
            # Her görüntü için eşleştirme yap
            for pred, target in zip(predictions, targets):
                pred_boxes = pred['boxes'].cpu()
                pred_labels = pred['labels'].cpu()
                pred_scores = pred['scores'].cpu()
                
                gt_boxes = target['boxes'].cpu()
                gt_labels = target['labels'].cpu()
                
                # Eşleştirme yap
                matches = match_predictions_to_gt(
                    pred_boxes, pred_labels, pred_scores,
                    gt_boxes, gt_labels,
                    iou_threshold, conf_threshold
                )
                
                all_matches.extend(matches)
    
    return all_matches


# COCO tarzı veri kümesi
class CocoDetectionDataset(Dataset):
    def __init__(self, images_dir, coco_json_path, transform=None):
        self.images_dir = Path(images_dir)
        self.coco_json_path = Path(coco_json_path)
        self.transform = transform if transform is not None else T.ToTensor()

        with open(self.coco_json_path, 'r') as f:
            data = json.load(f)

        # Görselleri ve anotasyonları hazırla
        self.images = data.get('images', [])
        annotations = data.get('annotations', [])
        categories = data.get('categories', [])

        # Kategori id -> index (0-based, contiguous) eşlemesi
        cat_ids = [c['id'] for c in categories]
        cat_ids_sorted = sorted(cat_ids)
        self.cat_id_to_idx = {cid: i for i, cid in enumerate(cat_ids_sorted)}
        self.idx_to_cat_id = {i: cid for cid, i in self.cat_id_to_idx.items()}
        self.categories_indexed = {i: next((c['name'] for c in categories if c['id'] == cid), f'Class_{i}')
                                   for i, cid in self.idx_to_cat_id.items()}

        # image_id -> anotasyon listesi
        self.image_to_anns = defaultdict(list)
        for ann in annotations:
            self.image_to_anns[ann['image_id']].append(ann)

        # image_id -> file_name
        self.id_to_file = {img['id']: img['file_name'] for img in self.images}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        image_id = img_info['id']
        file_name = img_info['file_name']

        img_path = self.images_dir / file_name
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        anns = self.image_to_anns.get(image_id, [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, w, h = ann['bbox']  # COCO formatı: [x, y, w, h]
            x2 = x + w
            y2 = y + h
            boxes.append([x, y, x2, y2])
            cat_id = ann['category_id']
            labels.append(self.cat_id_to_idx.get(cat_id, 0))
            areas.append(w * h)
            iscrowd.append(ann.get('iscrowd', 0))

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            areas = torch.tensor(areas, dtype=torch.float32)
            iscrowd = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id], dtype=torch.int64),
            'area': areas,
            'iscrowd': iscrowd,
        }

        return image, target


def collate_fn(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets


def create_test_dataloader(images_dir, coco_json_path, batch_size=2, num_workers=0):
    dataset = CocoDetectionDataset(images_dir, coco_json_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return loader

# Confusion Matrix oluşturma
def create_confusion_matrix(matches, num_classes, include_no_detection=True):
    """
    Eşleşmelerden confusion matrix oluştur
    """
    # No detection kategorisini dahil et
    if include_no_detection:
        matrix_size = num_classes + 1
    else:
        matrix_size = num_classes
    
    confusion_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
    
    for match in matches:
        gt_label = match['gt_label']
        pred_label = match['pred_label']
        
        # Etiketleri ayarla (0-indexed)
        if pred_label == -1:  # No detection
            if include_no_detection:
                pred_idx = num_classes  # Son indeks
            else:
                continue  # No detection'ı dahil etme
        else:
            pred_idx = pred_label
        
        gt_idx = gt_label
        confusion_matrix[gt_idx, pred_idx] += 1
    
    return confusion_matrix

# Confusion Matrix görselleştirme
def plot_confusion_matrix(cm, categories, save_path='confusion_matrix.png'):
    """
    Confusion matrix'i görselleştir
    """
    # Kategori isimlerini hazırla
    class_names = [categories.get(i, f'Class_{i}') for i in range(len(categories))]
    class_names.append('No Detection')
    
    plt.figure(figsize=(12, 10))
    
    # Normalize edilmiş confusion matrix (opsiyonel)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # NaN'ları 0 yap
    
    # Heatmap çiz
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Object Detection', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix kaydedildi: {save_path}")
    
    # Normalize edilmiş versiyonu da kaydet
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Ratio'}, vmin=0, vmax=1)
    
    plt.title('Normalized Confusion Matrix - Object Detection', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_normalized.png'), dpi=300, bbox_inches='tight')
    print(f"Normalized confusion matrix kaydedildi: {save_path.replace('.png', '_normalized.png')}")
    
    return cm_normalized

# Sınıf bazlı metrikleri hesapla
def calculate_metrics(cm, categories):
    """
    Precision, Recall, F1-Score hesapla
    """
    num_classes = len(categories)
    metrics = {}
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_name = categories.get(i, f'Class_{i}')
        metrics[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'support': int(cm[i, :].sum())
        }
    
    return metrics

# Ana fonksiyon
def main():
    # Ayarlar
    MODEL_PATH = 'MaskRCNN_COCO_kabin_v3.pt'
    COCO_JSON_PATH = 'C:/Users/ali.donbaloglu/Desktop/Lens/datasetler/Kabin_object_detection_v1/test/_annotations.coco.json'
    TEST_IMAGES_PATH = 'C:/Users/ali.donbaloglu/Desktop/Lens/datasetler/Kabin_object_detection_v1/test'
    IOU_THRESHOLD = 0.5
    CONF_THRESHOLD = 0.5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("1. Kategoriler yükleniyor...")
    categories, coco_data = load_categories(COCO_JSON_PATH)
    print(f"   {len(categories)} kategori bulundu: {categories}")
    
    print("\n2. Model yükleniyor...")
    # Test dataset ve num_classes'i eğitimdeki sınıf haritasıyla uyumlu olacak şekilde kullan
    test_dataset = COCOSegmentationDataset(COCO_JSON_PATH, TEST_IMAGES_PATH, transforms=None)
    num_classes = test_dataset.num_classes  # 0=background, 1..N sınıflar
    model = get_model(num_classes)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            res = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif isinstance(checkpoint, dict):
            res = model.load_state_dict(checkpoint, strict=False)
        else:
            # Tam model nesnesi kaydedildiyse (nadir), onu kullanmayıp mimariye yüklemeye devam ederiz
            print("   Uyarı: Beklenmeyen checkpoint formatı, state_dict bekleniyor.")
            res = None
        model.to(DEVICE)
        model.eval()
        if res is not None:
            unexpected = getattr(res, 'unexpected_keys', [])
            missing = getattr(res, 'missing_keys', [])
            if unexpected or missing:
                print(f"   Uyarı: {len(unexpected)} beklenmeyen anahtar ve {len(missing)} eksik anahtar yoksayıldı.")
                if unexpected:
                    print("   Beklenmeyen anahtar(lar) örnek:", unexpected[:5])
                if missing:
                    print("   Eksik anahtar(lar) örnek:", missing[:5])
        print("   Model yüklendi!")
    except Exception as e:
        raise RuntimeError(f"Model yükleme başarısız: {e}")
    
    print("\n3. Test dataloader hazırlanıyor...")
    # Eğitimdeki dataloader ile uyumlu
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: (list(x for x, _ in batch), list(y for _, y in batch)),
    )
    
    
    # Model hazır; tahminleri topla

    print("\n4. Model tahminleri toplanıyor...")
    print(f"   IoU Threshold: {IOU_THRESHOLD}")
    print(f"   Confidence Threshold: {CONF_THRESHOLD}")
    matches = evaluate_model(model, test_loader, DEVICE, IOU_THRESHOLD, CONF_THRESHOLD)
    print(f"   Toplam {len(matches)} eşleştirme yapıldı")
    
    print("\n5. Confusion matrix oluşturuluyor...")
    cm = create_confusion_matrix(matches, len(categories), include_no_detection=True)
    
    print("\n6. Confusion matrix görselleştiriliyor...")
    cm_normalized = plot_confusion_matrix(cm, categories, 'confusion_matrix_od.png')
    
    print("\n7. Sınıf bazlı metrikler hesaplanıyor...")
    metrics = calculate_metrics(cm, categories)
    
    print("\n" + "="*50)
    print("SINIF BAZLI METRİKLER")
    print("="*50)
    for class_name, metric in metrics.items():
        print(f"\n{class_name}:")
        print(f"  Precision: {metric['precision']:.4f}")
        print(f"  Recall:    {metric['recall']:.4f}")
        print(f"  F1-Score:  {metric['f1_score']:.4f}")
        print(f"  Support:   {metric['support']}")
    
    # Sonuçları JSON'a kaydet
    results = {
        'confusion_matrix': cm.tolist(),
        'metrics': metrics,
        'config': {
            'iou_threshold': IOU_THRESHOLD,
            'conf_threshold': CONF_THRESHOLD
        }
    }
    with open('confusion_matrix_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n8. Sonuçlar 'confusion_matrix_results.json' dosyasına kaydedildi!")

if __name__ == '__main__':
    main()