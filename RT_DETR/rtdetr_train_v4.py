"""
RT-DETR Eğitim Scripti - Improved Version with Epoch Logging
=============================================================
Kabin yüzey hatası tespiti için RT-DETR modeli eğitimi.
Mask R-CNN kodundaki parametreler ve sonuç kaydetme yapısı ile uyumlu.
Epoch bazlı metrik kaydetme ve mAP@0.25 desteği eklenmiştir.

Kullanım:
    python rtdetr_train.py

Gereksinimler:
    pip install ultralytics>=8.0.0
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import random
import numpy as np
import time

# Ultralytics import
try:
    from ultralytics import RTDETR
    from ultralytics.utils.callbacks.base import add_integration_callbacks
    import torch
except ImportError:
    print("Ultralytics yüklü değil. Yüklemek için:")
    print("pip install ultralytics")
    exit(1)

# Shapely import (polygon IoU için)
try:
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.geometry import box as ShapelyBox
except ImportError:
    print("Shapely yüklü değil. Yüklemek için:")
    print("pip install shapely")
    ShapelyPolygon = None
    ShapelyBox = None

# ----------------------------
# Konfigürasyon - Mask R-CNN ile Uyumlu
# ----------------------------
# Dataset yolları
YOLO_ROOT = r"C:/Users/ali.donbaloglu/Desktop/Lens/datasetler/Kabin_merged_yolo"
DATASET_YAML = r"C:/Users/ali.donbaloglu/Desktop/Lens/RT_DETR/kabin_dataset_yolo.yaml"

# Çıktı klasörleri
OUTPUT_PROJECT = r"C:/Users/ali.donbaloglu/Desktop/Lens/results/RTDETR"
OUTPUT_NAME = f"deneme_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
MODEL_DIR = r"C:/Users/ali.donbaloglu/Desktop/Lens/Modeller"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "RTDETR_deneme_BEST.pt")
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "RTDETR_deneme_LAST.pt")

# MODEL SEÇİMİ
MODEL_NAME = "rtdetr-x.pt"
#MODEL_NAME = "RT-DETR-R101"
# ✅ İYİLEŞTİRİLMİŞ HİPERPARAMETRELER
RANDOM_SEED = 42
NUM_EPOCHS = 1
BATCH_SIZE = 2  # 4 → 8 (daha stabil gradients)
IMAGE_SIZE = 1000
PATIENCE = 30  # 50 → 20 (erken dur, crash'i önle)
LEARNING_RATE = 0.00005  # 0.0001 → 0.00005 (daha düşük, daha güvenli)
LR_FINAL = 0.01
WEIGHT_DECAY = 1e-4
IOU_THRESHOLD = 0.25
CONF_THRESHOLD = 0.001

# ✅ YENİ EKLEMELER
GRADIENT_CLIP = 1.0  # Gradient explosion'ı önle
USE_AMP = True  # Mixed precision training (stabilite için)
WARMUP_EPOCHS = 5  # İlk 5 epoch yavaş başla
WARMUP_LR = 0.00001  # Warmup başlangıç LR
CLS_WEIGHT = 1.5  # Focal loss için class weight (recall artışı)

# ✅ İYİLEŞTİRİLMİŞ AUGMENTATION
USE_AUGMENTATION = False
AUG_HSVO = 0.015
AUG_HSVS = 0.2
AUG_HSVV = 0.2
AUG_DEGREES = 5.0  # 0.0 → 5.0 (hafif rotasyon ekle)
AUG_TRANSLATE = 0.05
AUG_SCALE = 0.1
AUG_SHEAR = 0.0
AUG_FLIPUD = 0.0
AUG_FLIPLR = 0.5
AUG_MOSAIC = 0.3  # 0.0 → 0.3 (ilk 50 epoch için)
AUG_MIXUP = 0.1  # 0.0 → 0.1 (veri çeşitliliği)
AUG_COPY_PASTE = 0.1  # Yeni: copy-paste augmentation

# Global değişkenler - epoch logging için
epoch_history = []
epoch_logs_dir = None
epoch_start_time = None


def set_seed(seed: int = 42):
    """Tekrarlanabilirlik için seed ayarla"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except:
        pass


def ensure_dir(path: str):
    """Klasör yoksa oluştur"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ----------------------------
# ✅ EPOCH LOGGING CALLBACKS
# ----------------------------
def on_train_epoch_start(trainer):
    """Her epoch başında çağrılır"""
    global epoch_start_time
    epoch_start_time = time.time()


def on_fit_epoch_end(trainer):
    """Her epoch sonunda çağrılır - metrikleri kaydet"""
    global epoch_history, epoch_logs_dir, epoch_start_time
    
    epoch = trainer.epoch + 1
    epoch_time = time.time() - epoch_start_time if epoch_start_time else 0
    
    # Metrikleri topla
    metrics = trainer.metrics if hasattr(trainer, 'metrics') else {}
    
    # Loss değerlerini al
    loss_items = trainer.loss_items if hasattr(trainer, 'loss_items') else None
    train_loss = float(loss_items.mean()) if loss_items is not None else 0.0
    
    # Learning rate
    lr = trainer.optimizer.param_groups[0]['lr'] if trainer.optimizer else LEARNING_RATE
    
    # Epoch bilgisi
    epoch_info = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_map50': float(metrics.get('metrics/mAP50(B)', 0)),
        'val_map50_95': float(metrics.get('metrics/mAP50-95(B)', 0)),
        'val_precision': float(metrics.get('metrics/precision(B)', 0)),
        'val_recall': float(metrics.get('metrics/recall(B)', 0)),
        'learning_rate': lr,
        'epoch_time': epoch_time,
    }
    
    epoch_history.append(epoch_info)
    
    # Per-epoch JSON dosyası kaydet
    if epoch_logs_dir:
        epoch_json_path = os.path.join(epoch_logs_dir, f"epoch_{epoch:03d}.json")
        try:
            with open(epoch_json_path, "w", encoding="utf-8") as f:
                json.dump(epoch_info, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Epoch {epoch} JSON kaydı başarısız: {e}")
    
    # Konsola özet yazdır
    print(f"\n📊 Epoch {epoch} Özet:")
    print(f"   📉 Train Loss: {train_loss:.4f}")
    print(f"   🎯 Val mAP50: {epoch_info['val_map50']:.4f}")
    print(f"   📈 Val mAP50-95: {epoch_info['val_map50_95']:.4f}")
    print(f"   📈 Val Precision: {epoch_info['val_precision']:.4f}")
    print(f"   📈 Val Recall: {epoch_info['val_recall']:.4f}")
    print(f"   🔧 LR: {lr:.2e}")
    print(f"   ⏱️  Süre: {epoch_time:.2f}s")


def save_training_results(training_info: dict, output_dir: str):
    """Eğitim sonuçlarını JSON ve TXT olarak kaydet"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON kaydet
    json_path = os.path.join(output_dir, f"training_results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False, default=str)
    
    # TXT kaydet
    txt_path = os.path.join(output_dir, f"training_summary_{timestamp}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("RT-DETR EĞİTİM SONUÇLARI (EPOCH LOGGING ENABLED)\n")
        f.write("=" * 100 + "\n\n")
        
        # Konfigürasyon
        f.write("EĞİTİM KONFİGÜRASYONU:\n")
        f.write("-" * 50 + "\n")
        config = training_info.get('config', {})
        f.write(f"Model: {config.get('model', '-')}\n")
        f.write(f"Veri Seti: {config.get('dataset_path', '-')}\n")
        f.write(f"Epoch Sayısı: {config.get('num_epochs', '-')}\n")
        f.write(f"Batch Size: {config.get('batch_size', '-')}\n")
        f.write(f"Image Size: {config.get('image_size', '-')}\n")
        f.write(f"Learning Rate: {config.get('learning_rate', '-')}\n")
        f.write(f"LR Final (lrf): {config.get('lr_final', '-')}\n")
        f.write(f"Weight Decay: {config.get('weight_decay', '-')}\n")
        f.write(f"Patience: {config.get('patience', '-')}\n")
        f.write(f"IoU Threshold: {config.get('iou_threshold', '-')}\n")
        f.write(f"Conf Threshold: {config.get('conf_threshold', '-')}\n")
        f.write(f"Sınıf Sayısı: {config.get('num_classes', '-')}\n")
        f.write(f"Augmentation: {'Aktif' if config.get('augmentation', False) else 'Kapalı'}\n")
        if config.get('device'):
            f.write(f"Cihaz: {config.get('device')}\n")
        if config.get('gpu_name'):
            f.write(f"GPU: {config.get('gpu_name')}\n")
        f.write("\n")
        
        # Eğitim süresi
        f.write("EĞİTİM SÜRESİ:\n")
        f.write("-" * 50 + "\n")
        time_info = training_info.get('training_time', {})
        f.write(f"Başlangıç: {time_info.get('start_time', '-')}\n")
        f.write(f"Bitiş: {time_info.get('end_time', '-')}\n")
        f.write(f"Toplam Süre: {time_info.get('total_duration', '-')}\n")
        f.write("\n")
        
        # ✅ EPOCH GEÇMİŞİ TABLOSU
        f.write("EPOCH GEÇMİŞİ:\n")
        f.write("=" * 100 + "\n")
        epoch_hist = training_info.get('epoch_history', [])
        if epoch_hist:
            # Tablo başlığı
            f.write(f"{'Epoch':>5}  {'TrainLoss':>10}  {'mAP50':>8}  {'mAP50-95':>10}  {'Precision':>10}  {'Recall':>8}  {'LR':>12}  {'Time(s)':>8}\n")
            f.write("-" * 100 + "\n")
            for e in epoch_hist:
                ep = e.get('epoch', '-')
                tl = e.get('train_loss', 0.0)
                m50 = e.get('val_map50', 0.0)
                m50_95 = e.get('val_map50_95', 0.0)
                prec = e.get('val_precision', 0.0)
                rec = e.get('val_recall', 0.0)
                lr = e.get('learning_rate', 0.0)
                et = e.get('epoch_time', 0.0)
                f.write(f"{ep:>5}  {tl:10.4f}  {m50:8.4f}  {m50_95:10.4f}  {prec:10.4f}  {rec:8.4f}  {lr:12.2e}  {et:8.1f}\n")
        else:
            f.write("(Epoch history not available)\n")
        f.write("\n")
        
        # En iyi model sonuçları
        f.write("EN İYİ MODEL SONUÇLARI:\n")
        f.write("=" * 100 + "\n")
        best = training_info.get('best_results', {})
        f.write(f"En İyi Epoch: {best.get('epoch', '-')}\n")
        f.write(f"mAP50: {best.get('map50', 0):.4f}\n")
        f.write(f"mAP50-95: {best.get('map50_95', 0):.4f}\n")
        f.write(f"Precision: {best.get('precision', 0):.4f}\n")
        f.write(f"Recall: {best.get('recall', 0):.4f}\n")
        f.write("\n")
        
        # ✅ mAP@0.25 SONUÇLARI (Özel bölüm)
        map_025_results = training_info.get('map_025_results', {})
        if map_025_results and map_025_results.get('best'):
            f.write("\n" + "=" * 100 + "\n")
            f.write(f"mAP@{IOU_THRESHOLD} SONUÇLARI (Özel IoU Threshold):\n")
            f.write("=" * 100 + "\n")
            
            best_map025 = map_025_results.get('best', {})
            last_map025 = map_025_results.get('last', {})
            
            f.write("EN İYİ MODEL (BEST):\n")
            f.write(f"  mAP@{IOU_THRESHOLD}: {best_map025.get(f'map@{IOU_THRESHOLD}', 0):.4f}\n")
            f.write(f"  mAP50: {best_map025.get('map50', 0):.4f}\n")
            f.write(f"  mAP50-95: {best_map025.get('map50_95', 0):.4f}\n")
            f.write(f"  Precision: {best_map025.get('precision', 0):.4f}\n")
            f.write(f"  Recall: {best_map025.get('recall', 0):.4f}\n")
            
            if last_map025 and last_map025.get('map50', 0) > 0:
                f.write("\nSON MODEL (LAST):\n")
                f.write(f"  mAP@{IOU_THRESHOLD}: {last_map025.get(f'map@{IOU_THRESHOLD}', 0):.4f}\n")
                f.write(f"  mAP50: {last_map025.get('map50', 0):.4f}\n")
                f.write(f"  mAP50-95: {last_map025.get('map50_95', 0):.4f}\n")
                f.write(f"  Precision: {last_map025.get('precision', 0):.4f}\n")
                f.write(f"  Recall: {last_map025.get('recall', 0):.4f}\n")
            
            f.write("\n")
        
        # Sınıf bazlı sonuçlar
        per_class = best.get('per_class', {})
        if per_class:
            f.write("SINIF BAZLI METRİKLER:\n")
            f.write("-" * 100 + "\n")
            for class_name, metrics in per_class.items():
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {metrics.get('precision', 0):.4f}\n")
                f.write(f"  Recall: {metrics.get('recall', 0):.4f}\n")
                f.write(f"  mAP50: {metrics.get('map50', 0):.4f}\n")
        
        # ✅ EN İYİ MODEL TEST SONUÇLARI
        best_test = training_info.get('best_model_test', {})
        if best_test:
            f.write("\n" + "=" * 100 + "\n")
            f.write("EN İYİ MODEL (BEST) TEST SONUÇLARI:\n")
            f.write("=" * 100 + "\n")
            f.write(f"mAP50: {best_test.get('map50', 0):.4f}\n")
            f.write(f"mAP50-95: {best_test.get('map50_95', 0):.4f}\n")
            f.write(f"Precision: {best_test.get('precision', 0):.4f}\n")
            f.write(f"Recall: {best_test.get('recall', 0):.4f}\n")
            if 'map_025' in best_test:
                f.write(f"mAP@{IOU_THRESHOLD}: {best_test.get('map_025', 0):.4f}\n")
        
        # ✅ SON MODEL TEST SONUÇLARI
        last_test = training_info.get('last_model_test', {})
        if last_test:
            f.write("\n" + "=" * 100 + "\n")
            f.write("SON MODEL (LAST) TEST SONUÇLARI:\n")
            f.write("=" * 100 + "\n")
            f.write(f"mAP50: {last_test.get('map50', 0):.4f}\n")
            f.write(f"mAP50-95: {last_test.get('map50_95', 0):.4f}\n")
            f.write(f"Precision: {last_test.get('precision', 0):.4f}\n")
            f.write(f"Recall: {last_test.get('recall', 0):.4f}\n")
            if 'map_025' in last_test:
                f.write(f"mAP@{IOU_THRESHOLD}: {last_test.get('map_025', 0):.4f}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("NOT: Bu sonuçlar otomatik olarak oluşturulmuştur.\n")
        f.write(f"Epoch logları: {training_info.get('model_paths', {}).get('epoch_logs', '-')}\n")
        f.write("=" * 100 + "\n")
    
    print(f"✅ Eğitim sonuçları kaydedildi:")
    print(f"   JSON: {json_path}")
    print(f"   TXT:  {txt_path}")
    
    return json_path, txt_path


def compute_iou(box1, box2):
    """İki bounding box arasındaki IoU değerini hesapla
    
    Args:
        box1: [x1, y1, x2, y2] formatında kutu
        box2: [x1, y1, x2, y2] formatında kutu
    
    Returns:
        IoU değeri (0-1 arası)
    """
    # Kesişim alanını hesapla
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Her kutunun alanı
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Birleşim alanı
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def calculate_ap(precisions, recalls):
    """Precision-Recall eğrisinden AP hesapla (11-point interpolation)
    
    Args:
        precisions: Precision değerleri listesi
        recalls: Recall değerleri listesi
    
    Returns:
        AP değeri
    """
    if len(precisions) == 0 or len(recalls) == 0:
        return 0.0
    
    # Recall'a göre sırala
    sorted_indices = np.argsort(recalls)
    recalls = np.array(recalls)[sorted_indices]
    precisions = np.array(precisions)[sorted_indices]
    
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap


def compute_polygon_iou(pred_box, gt_polygon_points):
    """Tahmin edilen bbox ile polygon ground truth arasında IoU hesapla
    
    Args:
        pred_box: [x1, y1, x2, y2] tahmin edilen kutu (normalized)
        gt_polygon_points: [[x1,y1], [x2,y2], ...] polygon noktaları (normalized)
    
    Returns:
        IoU değeri
    """
    if ShapelyPolygon is None or ShapelyBox is None:
        # Shapely yüklü değilse basit bbox IoU hesapla
        # Polygon'un bounding box'ını hesapla
        xs = [p[0] for p in gt_polygon_points]
        ys = [p[1] for p in gt_polygon_points]
        gt_bbox = [min(xs), min(ys), max(xs), max(ys)]
        return compute_iou(pred_box, gt_bbox)
    
    try:
        # Tahmin kutusunu Shapely polygon'a çevir
        pred_poly = ShapelyBox(pred_box[0], pred_box[1], pred_box[2], pred_box[3])
        
        # Ground truth polygon'u oluştur
        gt_poly = ShapelyPolygon(gt_polygon_points)
        
        # Geçerlilik kontrolü
        if not pred_poly.is_valid:
            pred_poly = pred_poly.buffer(0)
        if not gt_poly.is_valid:
            gt_poly = gt_poly.buffer(0)
        
        # Kesişim ve birleşim
        intersection = pred_poly.intersection(gt_poly).area
        union = pred_poly.union(gt_poly).area
        
        if union <= 0:
            return 0.0
        
        return intersection / union
        
    except Exception as e:
        # Hata durumunda bounding box IoU hesapla
        xs = [p[0] for p in gt_polygon_points]
        ys = [p[1] for p in gt_polygon_points]
        gt_bbox = [min(xs), min(ys), max(xs), max(ys)]
        return compute_iou(pred_box, gt_bbox)


def read_polygon_labels(label_path):
    """Polygon etiketleri oku
    
    YOLO Segmentation formatı:
    class_id x1 y1 x2 y2 x3 y3 ... (normalized koordinatlar)
    
    Returns:
        List of (class_id, polygon_points)
    """
    polygons = []
    
    if not os.path.exists(label_path):
        return polygons
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:  # En az class + 4 koordinat (2 nokta minimum)
                    continue
                
                cls_id = int(parts[0])
                
                # Koordinatları çiftler halinde al
                coords = [float(x) for x in parts[1:]]
                
                # Eğer sadece 4 değer varsa, bu YOLO bbox formatı (cx, cy, w, h)
                if len(coords) == 4:
                    cx, cy, w, h = coords
                    # Bbox'ı polygon noktalarına çevir (4 köşe)
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    polygon_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                else:
                    # Polygon formatı: koordinatları çiftler halinde al
                    polygon_points = [(coords[i], coords[i+1]) for i in range(0, len(coords)-1, 2)]
                
                if len(polygon_points) >= 3:  # Geçerli polygon için en az 3 nokta
                    polygons.append((cls_id, polygon_points))
        
    except Exception as e:
        pass  # Hata durumunda boş liste döndür
    
    return polygons


def evaluate_with_custom_iou(model, data_yaml: str, iou_threshold: float = 0.25):
    """Polygon ground truth kullanarak mAP hesapla
    
    ✅ POLYGON ETİKETLER İÇİN DOĞRU HESAPLAMA
    """
    print(f"\n🔍 mAP@{iou_threshold} hesaplanıyor (Polygon GT)...")
    
    try:
        import yaml
        import glob
        
        # Dataset YAML'dan bilgileri oku
        with open(data_yaml, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        dataset_root = data_config.get('path', '')
        if not dataset_root:
            dataset_root = os.path.dirname(data_yaml)
        
        val_path = data_config.get('val', '')
        if not os.path.isabs(val_path):
            val_path = os.path.join(dataset_root, val_path)
        
        images_dir = val_path
        labels_dir = val_path.replace('/images', '/labels').replace('\\images', '\\labels')
        
        if not os.path.exists(images_dir):
            print(f"⚠️ Validation images klasörü bulunamadı: {images_dir}")
            return _fallback_evaluation(model, data_yaml, iou_threshold)
        
        # Görüntüleri bul
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        
        if len(image_files) == 0:
            print(f"⚠️ Validation görüntüsü bulunamadı")
            return _fallback_evaluation(model, data_yaml, iou_threshold)
        
        print(f"   📊 {len(image_files)} validation görüntüsü bulundu")
        
        num_classes = len(data_config.get('names', []))
        if num_classes == 0:
            num_classes = 2
        
        # Metrikler
        all_predictions = {cls_id: [] for cls_id in range(num_classes)}
        total_gt = {cls_id: 0 for cls_id in range(num_classes)}
        
        total_tp = 0
        total_fp = 0
        total_gt_count = 0
        
        debug_count = 0
        max_debug = 3
        
        # Her görüntü için
        for img_idx, img_path in enumerate(image_files):
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(labels_dir, img_name + '.txt')
            
            # ✅ POLYGON etiketleri oku
            gt_polygons = read_polygon_labels(label_path)
            
            if len(gt_polygons) == 0:
                continue
            
            for cls_id, _ in gt_polygons:
                total_gt[cls_id] = total_gt.get(cls_id, 0) + 1
                total_gt_count += 1
            
            # Model tahmini
            results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)
            
            if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
                continue
            
            boxes = results[0].boxes
            
            # Tahminleri al
            pred_boxes = []
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                box = boxes.xyxyn[i].cpu().numpy()
                
                x1 = max(0.0, min(1.0, float(box[0])))
                y1 = max(0.0, min(1.0, float(box[1])))
                x2 = max(0.0, min(1.0, float(box[2])))
                y2 = max(0.0, min(1.0, float(box[3])))
                
                pred_boxes.append((cls_id, conf, x1, y1, x2, y2))
            
            pred_boxes.sort(key=lambda x: x[1], reverse=True)
            
            # Debug
            if debug_count < max_debug and len(gt_polygons) > 0 and len(pred_boxes) > 0:
                print(f"\n   [DEBUG] Görüntü {img_idx + 1}: {img_name}")
                print(f"      GT polygons: {len(gt_polygons)}, Pred boxes: {len(pred_boxes)}")
                print(f"      İlk GT: cls={gt_polygons[0][0]}, {len(gt_polygons[0][1])} nokta")
                debug_count += 1
            
            # Her GT için eşleşme durumu
            gt_matched = [False] * len(gt_polygons)
            
            # Her tahmin için TP/FP belirle
            for pred in pred_boxes:
                pred_cls, pred_conf, px1, py1, px2, py2 = pred
                
                best_iou = 0.0
                best_gt_idx = -1
                
                # ✅ POLYGON GT ile karşılaştır
                for gt_idx, (gt_cls, gt_poly_points) in enumerate(gt_polygons):
                    if gt_cls != pred_cls:
                        continue
                    
                    if gt_matched[gt_idx]:
                        continue
                    
                    # Polygon IoU hesapla
                    iou = compute_polygon_iou([px1, py1, px2, py2], gt_poly_points)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # Debug
                if debug_count <= max_debug and img_idx < max_debug:
                    if best_iou > 0:
                        print(f"         Pred cls={pred_cls}, IoU={best_iou:.3f} (vs polygon)")
                
                # Threshold kontrolü
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    gt_matched[best_gt_idx] = True
                    all_predictions[pred_cls].append((pred_conf, True))
                    total_tp += 1
                else:
                    all_predictions[pred_cls].append((pred_conf, False))
                    total_fp += 1
        
        print(f"\n   📈 Toplam: TP={total_tp}, FP={total_fp}, GT={total_gt_count}")
        
        # AP hesapla
        class_aps = []
        for cls_id in range(num_classes):
            predictions = all_predictions[cls_id]
            gt_count = total_gt.get(cls_id, 0)
            
            if gt_count == 0:
                continue
            
            predictions.sort(key=lambda x: x[0], reverse=True)
            
            tp_cumsum = 0
            fp_cumsum = 0
            precisions = []
            recalls = []
            
            for conf, is_tp in predictions:
                if is_tp:
                    tp_cumsum += 1
                else:
                    fp_cumsum += 1
                
                precision = tp_cumsum / (tp_cumsum + fp_cumsum) if (tp_cumsum + fp_cumsum) > 0 else 0
                recall = tp_cumsum / gt_count if gt_count > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
            
            # AP hesapla
            ap = calculate_ap(precisions, recalls)
            class_aps.append(ap)
            
            class_name = data_config.get('names', {}).get(cls_id, f"class_{cls_id}")
            print(f"      Sınıf {cls_id} ({class_name}): AP@{iou_threshold}={ap:.4f} (GT={gt_count})")
        
        map_custom = np.mean(class_aps) if len(class_aps) > 0 else 0.0
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / total_gt_count if total_gt_count > 0 else 0.0
        
        # Standart metrikler
        std_results = model.val(data=data_yaml, conf=CONF_THRESHOLD, verbose=False, plots=False)
        map50 = float(getattr(std_results.box, 'map50', 0.0))
        map50_95 = float(getattr(std_results.box, 'map', 0.0))
        
        metrics = {
            f'map@{iou_threshold}': float(map_custom),
            'map50': map50,
            'map50_95': map50_95,
            'precision': overall_precision,
            'recall': overall_recall,
            'per_class_ap': class_aps,
            'method': 'polygon_gt'
        }
        
        print(f"\n   ✅ mAP@{iou_threshold}: {map_custom:.4f} (POLYGON GT)")
        print(f"      mAP50 (standart bbox): {map50:.4f}")
        print(f"      mAP50-95: {map50_95:.4f}")
        print(f"      Precision: {overall_precision:.4f}")
        print(f"      Recall: {overall_recall:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"⚠️ Hata: {e}")
        import traceback
        traceback.print_exc()
        return _fallback_evaluation(model, data_yaml, iou_threshold)


def _fallback_evaluation(model, data_yaml: str, iou_threshold: float):
    """Hata durumunda standart validation ile fallback"""
    print(f"   ⚠️ Fallback: Standart validation kullanılıyor...")
    try:
        results = model.val(data=data_yaml, conf=CONF_THRESHOLD, verbose=False, plots=False)
        map50 = float(getattr(results.box, 'map50', 0.0))
        map50_95 = float(getattr(results.box, 'map', 0.0))
        precision = float(getattr(results.box, 'mp', 0.0))
        recall = float(getattr(results.box, 'mr', 0.0))
        
        return {
            f'map@{iou_threshold}': map50,  # Fallback: map50 kullan
            'map50': map50,
            'map50_95': map50_95,
            'precision': precision,
            'recall': recall,
            'per_class_ap': None,
            'note': 'Fallback to standard mAP50'
        }
    except:
        return {
            f'map@{iou_threshold}': 0.0,
            'map50': 0.0,
            'map50_95': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'per_class_ap': None
        }


def train():
    """RT-DETR eğitimini başlat"""
    global epoch_history, epoch_logs_dir
    
    set_seed(RANDOM_SEED)
    ensure_dir(OUTPUT_PROJECT)
    ensure_dir(MODEL_DIR)
    
    # Epoch history'yi sıfırla
    epoch_history = []
    
    print("=" * 80)
    print("RT-DETR EĞİTİM (EPOCH LOGGING ENABLED)")
    print("=" * 80)
    
    # Cihaz bilgisi
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n🖥️  Cihaz: {device}")
        print(f"🎮 GPU: {gpu_name}")
        print(f"💾 GPU Bellek: {gpu_memory:.2f} GB")
    else:
        print(f"\n🖥️  Cihaz: {device}")
    
    print(f"\n📁 Dataset: {YOLO_ROOT}")
    print(f"🔧 Model: {MODEL_NAME}")
    print(f"📊 Epochs: {NUM_EPOCHS}")
    print(f"📦 Batch Size: {BATCH_SIZE}")
    print(f"🖼️  Image Size: {IMAGE_SIZE}")
    print(f"📈 Learning Rate: {LEARNING_RATE} → {LEARNING_RATE * LR_FINAL} (cosine)")
    print(f"⏱️  Patience: {PATIENCE}")
    print(f"🎯 IoU Threshold: {IOU_THRESHOLD}")
    print(f"🎲 Augmentation: {'Aktif' if USE_AUGMENTATION else 'Kapalı'}")
    print("=" * 80)
    
    # Çıktı klasörünü hazırla
    run_dir = os.path.join(OUTPUT_PROJECT, OUTPUT_NAME)
    epoch_logs_dir = os.path.join(run_dir, "epoch_logs")
    ensure_dir(run_dir)
    ensure_dir(epoch_logs_dir)
    print(f"\n📂 Epoch logları: {epoch_logs_dir}")
    
    # Model yükle
    print(f"\n📥 Model yükleniyor: {MODEL_NAME}")
    model = RTDETR(MODEL_NAME)
    
    # ✅ Callback'leri ekle
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    
    # Eğitimi başlat
    print("\n🚀 Eğitim başlıyor...")
    start_time = datetime.now()
    start_timestamp = time.time()
    
    # ✅ İyileştirilmiş Augmentation parametreleri
    aug_params = {}
    if USE_AUGMENTATION:
        aug_params = {
            "hsv_h": AUG_HSVO,
            "hsv_s": AUG_HSVS,
            "hsv_v": AUG_HSVV,
            "degrees": AUG_DEGREES,
            "translate": AUG_TRANSLATE,
            "scale": AUG_SCALE,
            "shear": AUG_SHEAR,
            "flipud": AUG_FLIPUD,
            "fliplr": AUG_FLIPLR,
            "mosaic": AUG_MOSAIC,
            "mixup": AUG_MIXUP,
            "copy_paste": AUG_COPY_PASTE,
        }
    
    results = model.train(
        data=DATASET_YAML,
        epochs=NUM_EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        lr0=LEARNING_RATE,
        lrf=LR_FINAL,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,  # ✅ Yeni
        warmup_momentum=0.8,  # ✅ Yeni
        warmup_bias_lr=WARMUP_LR,  # ✅ Yeni
        box=7.5,  # Box loss gain
        cls=0.5,  # Class loss gain
        dfl=1.5,  # DFL loss gain
        label_smoothing=0.0,  # Label smoothing epsilon
        nbs=64,  # Nominal batch size
        overlap_mask=True,  # Masks should overlap during training
        mask_ratio=4,  # Mask downsample ratio
        dropout=0.0,  # Use dropout regularization
        val=True,
        device=0 if device == "cuda" else "cpu",
        project=OUTPUT_PROJECT,
        name=OUTPUT_NAME,
        exist_ok=True,
        seed=RANDOM_SEED,
        verbose=True,
        plots=True,
        save=True,
        iou=IOU_THRESHOLD,
        conf=CONF_THRESHOLD,
        amp=USE_AMP,  # ✅ Mixed precision
        **aug_params
    )
    
    end_time = datetime.now()
    total_duration = (time.time() - start_timestamp) / 60
    
    print("\n" + "=" * 80)
    print("EĞİTİM TAMAMLANDI!")
    print("=" * 80)
    print(f"⏱️  Toplam süre: {total_duration:.2f} dakika")
    print(f"📂 Sonuçlar: {run_dir}")
    print(f"📂 Epoch logları: {epoch_logs_dir}")
    
    # Sonuçları topla
    best_model_src = os.path.join(run_dir, "weights", "best.pt")
    last_model_src = os.path.join(run_dir, "weights", "last.pt")
    
    # En iyi modeli kopyala
    if os.path.exists(best_model_src):
        shutil.copy2(best_model_src, BEST_MODEL_PATH)
        print(f"✅ En iyi model: {BEST_MODEL_PATH}")
    
    # Son epoch modelini kopyala
    if os.path.exists(last_model_src):
        shutil.copy2(last_model_src, LAST_MODEL_PATH)
        print(f"✅ Son model: {LAST_MODEL_PATH}")
    
    # ✅ EN İYİ MODEL İLE TEST DEĞERLENDİRME
    best_model_test = {}
    map_025_best = {}
    if os.path.exists(BEST_MODEL_PATH):
        print("\n" + "=" * 80)
        print("EN İYİ MODEL (BEST) DEĞERLENDİRME")
        print("=" * 80)
        best_model = RTDETR(BEST_MODEL_PATH)
        
        # Standart validation
        print("🔍 Standart validation...")
        best_val = best_model.val(data=DATASET_YAML, verbose=False)
        best_model_test = {
            'map50': float(best_val.box.map50),
            'map50_95': float(best_val.box.map),
            'precision': float(best_val.box.mp),
            'recall': float(best_val.box.mr),
        }
        print(f"   mAP50: {best_model_test['map50']:.4f}")
        print(f"   mAP50-95: {best_model_test['map50_95']:.4f}")
        print(f"   Precision: {best_model_test['precision']:.4f}")
        print(f"   Recall: {best_model_test['recall']:.4f}")
        
        # mAP@0.25 ile özel değerlendirme
        map_025_best = evaluate_with_custom_iou(best_model, DATASET_YAML, IOU_THRESHOLD)
        if map_025_best:
            best_model_test['map_025'] = map_025_best.get(f'map@{IOU_THRESHOLD}', 0)
            print(f"   mAP@{IOU_THRESHOLD}: {best_model_test['map_025']:.4f}")
    
    # ✅ SON MODEL İLE TEST DEĞERLENDİRME
    last_model_test = {}
    map_025_last = {}
    if os.path.exists(LAST_MODEL_PATH):
        print("\n" + "=" * 80)
        print("SON MODEL (LAST) DEĞERLENDİRME")
        print("=" * 80)
        last_model = RTDETR(LAST_MODEL_PATH)
        
        # Standart validation
        print("🔍 Standart validation...")
        last_val = last_model.val(data=DATASET_YAML, verbose=False)
        last_model_test = {
            'map50': float(last_val.box.map50),
            'map50_95': float(last_val.box.map),
            'precision': float(last_val.box.mp),
            'recall': float(last_val.box.mr),
        }
        print(f"   mAP50: {last_model_test['map50']:.4f}")
        print(f"   mAP50-95: {last_model_test['map50_95']:.4f}")
        print(f"   Precision: {last_model_test['precision']:.4f}")
        print(f"   Recall: {last_model_test['recall']:.4f}")
        
        # mAP@0.25 ile özel değerlendirme
        map_025_last = evaluate_with_custom_iou(last_model, DATASET_YAML, IOU_THRESHOLD)
        if map_025_last:
            last_model_test['map_025'] = map_025_last.get(f'map@{IOU_THRESHOLD}', 0)
            print(f"   mAP@{IOU_THRESHOLD}: {last_model_test['map_025']:.4f}")
    
    # En iyi epoch'u bul
    best_epoch = 0
    best_map50 = 0
    for e in epoch_history:
        if e.get('val_map50', 0) > best_map50:
            best_map50 = e.get('val_map50', 0)
            best_epoch = e.get('epoch', 0)
    
    # Training info hazırla
    training_info = {
        "config": {
            "model": MODEL_NAME,
            "dataset_path": YOLO_ROOT,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "learning_rate": LEARNING_RATE,
            "lr_final": LR_FINAL,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "iou_threshold": IOU_THRESHOLD,
            "conf_threshold": CONF_THRESHOLD,
            "gradient_clip": GRADIENT_CLIP,  # ✅ Yeni
            "use_amp": USE_AMP,  # ✅ Yeni
            "warmup_epochs": WARMUP_EPOCHS,  # ✅ Yeni
            "cls_weight": CLS_WEIGHT,  # ✅ Yeni
            "num_classes": 2,
            "augmentation": USE_AUGMENTATION,
            "device": device,
            "gpu_name": gpu_name,
            "seed": RANDOM_SEED,
        },
        "training_time": {
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration": f"{total_duration:.2f} dakika",
        },
        "epoch_history": epoch_history,
        "best_results": {
            "epoch": best_epoch,
            "map50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "map50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
        },
        "best_model_test": best_model_test,
        "last_model_test": last_model_test,
        
        # ✅ mAP@0.25 sonuçlarını ayrı bir bölüm olarak ekle
        "map_025_results": {
            "best": map_025_best,
            "last": map_025_last,
            "iou_threshold": IOU_THRESHOLD,
            "note": f"Custom IoU threshold evaluation at {IOU_THRESHOLD}"
        },
        
        "model_paths": {
            "best": BEST_MODEL_PATH,
            "last": LAST_MODEL_PATH,
            "run_dir": run_dir,
            "epoch_logs": epoch_logs_dir,
        }
    }
    
    # Sonuçları kaydet
    save_training_results(training_info, run_dir)
    
    return results


def validate(model_path: str = None):
    """Eğitilmiş modeli validation set üzerinde değerlendir"""
    if model_path is None:
        model_path = BEST_MODEL_PATH
    
    if not os.path.exists(model_path):
        print(f"❌ Model bulunamadı: {model_path}")
        return None
    
    print(f"📥 Model yükleniyor: {model_path}")
    model = RTDETR(model_path)
    
    print("\n🔍 Validation başlıyor...")
    
    # Standart validation
    results = model.val(data=DATASET_YAML)
    
    print("\n" + "=" * 80)
    print("VALIDATION SONUÇLARI:")
    print("=" * 80)
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")
    
    # mAP@0.25 ile özel değerlendirme
    print(f"\n🎯 mAP@{IOU_THRESHOLD} ile değerlendirme...")
    map_025 = evaluate_with_custom_iou(model, DATASET_YAML, IOU_THRESHOLD)
    if map_025:
        print(f"\nmAP@{IOU_THRESHOLD} SONUÇLARI:")
        print("-" * 40)
        print(f"mAP50: {map_025.get('map50', 0):.4f}")
        print(f"Precision: {map_025.get('precision', 0):.4f}")
        print(f"Recall: {map_025.get('recall', 0):.4f}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--val":
        validate()
    else:
        train()
