"""
RT-DETR Ensemble Model Test Script
----------------------------------
İki farklı RT-DETR modelini (çizik ve siyah nokta) ensemble olarak kullanarak
test verisini değerlendiren script.

Ensemble Stratejisi:
- Her iki modelden gelen tahminler birleştirilir
- Çizik modeli: scratch (class 0) tahminlerinde daha iyi
- Siyah nokta modeli: black_dot (class 1) tahminlerinde daha iyi
- NMS uygulanarak çakışan tahminler filtrelenir

Kullanım:
    python ensemble_test.py --source <test_klasoru>
    python ensemble_test.py --source ./test --conf 0.25 --iou 0.25
"""

import os
import cv2
import numpy as np
import pandas as pd
import argparse
import torch
from ultralytics import RTDETR
from datetime import datetime

# --- AYARLAR ---
# Model Yolları
SCRATCH_MODEL_PATH = r"C:/Users/ali.donbaloglu/Desktop/Lens/RT_Detr_Ensemble/model/best_cizik.pt"
BLACKDOT_MODEL_PATH = r"C:/Users/ali.donbaloglu/Desktop/Lens/RT_Detr_Ensemble/model/best_siyahnokta.pt"

# Sınıf İsimleri
CLASS_NAMES = ["scratch", "black_dot"]
# Class ID mapping: scratch=0, black_dot=1

# Varsayılan Eşik Değerleri
DEFAULT_CONF_THRESHOLD = 0.35
DEFAULT_IOU_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.40  # Ensemble NMS için

# Ensemble Ağırlık Ayarları
# Her model kendi güçlü olduğu sınıfta STRONG_WEIGHT, zayıf olduğu sınıfta WEAK_WEIGHT alır
SCRATCH_MODEL_STRONG_WEIGHT = 1.0   # Çizik modeli scratch tahminlerinde
SCRATCH_MODEL_WEAK_WEIGHT = 0.5     # Çizik modeli black_dot tahminlerinde
BLACKDOT_MODEL_STRONG_WEIGHT = 1.0  # Siyah nokta modeli black_dot tahminlerinde
BLACKDOT_MODEL_WEAK_WEIGHT = 0.5    # Siyah nokta modeli scratch tahminlerinde

# Çıktı Ayarları - Tüm sonuçlar tek klasörde toplanacak
OUTPUT_BASE_DIR = 'ensemble_results'  # Ana çıktı klasörü

# Görselleştirme Ayarları
LINE_THICKNESS = 1
FONT_SCALE = 0.6

# Renk Haritası (BGR)
COLORS = {
    'gt': (0, 0, 255),       # Kırmızı - Ground Truth
    'scratch': (0, 255, 0),   # Yeşil - Çizik tahminleri
    'black_dot': (255, 0, 0), # Mavi - Siyah nokta tahminleri
    'ensemble': (0, 255, 255) # Sarı - Ensemble sonucu
}


# --- YARDIMCI FONKSİYONLAR ---

def get_ground_truth(label_path):
    """
    YOLO formatındaki polygon etiketlerini oku ve bounding box'a çevir.
    Format: class_id x1 y1 x2 y2 x3 y3 x4 y4 ... (normalized)
    """
    gt_list = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = [float(p) for p in line.strip().split()]
                if len(parts) < 5:
                    continue
                    
                class_id = int(parts[0])
                points = parts[1:]

                # Polygon koordinatlarını bounding box'a çevir
                x_coords = points[0::2]
                y_coords = points[1::2]

                x_min, y_min = min(x_coords), min(y_coords)
                x_max, y_max = max(x_coords), max(y_coords)

                # YOLO format: center_x, center_y, width, height (normalized)
                gt_box = [
                    (x_min + x_max) / 2,
                    (y_min + y_max) / 2,
                    x_max - x_min,
                    y_max - y_min
                ]
                gt_list.append((class_id, gt_box))
        return gt_list
    except Exception as e:
        return []


def calculate_iou(boxA, boxB):
    """
    İki bounding box arasındaki IoU değerini hesapla.
    Box format: [center_x, center_y, width, height] (normalized)
    """
    def to_corners(box):
        x, y, w, h = box
        return [x - w/2, y - h/2, x + w/2, y + h/2]

    A = to_corners(boxA)
    B = to_corners(boxB)

    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]

    denominator = areaA + areaB - inter
    return inter / denominator if denominator > 0 else 0


def apply_nms(predictions, iou_threshold=0.5):
    """
    Non-Maximum Suppression uygula.
    predictions: [(class_id, box, confidence, model_source), ...]
    """
    if len(predictions) == 0:
        return []
    
    # Confidence'a göre sırala
    predictions = sorted(predictions, key=lambda x: x[2], reverse=True)
    
    kept = []
    while predictions:
        best = predictions.pop(0)
        kept.append(best)
        
        # Aynı sınıftan çakışan tahminleri filtrele
        predictions = [
            p for p in predictions
            if p[0] != best[0] or calculate_iou(best[1], p[1]) < iou_threshold
        ]
    
    return kept


def ensemble_predictions(scratch_results, blackdot_results, img_shape, nms_iou=0.5, 
                         scratch_weight=1.0, blackdot_weight=1.0):
    """
    İki modelin tahminlerini GERÇEK ENSEMBLE olarak birleştir.
    
    Strateji:
    - İki model aynı yerde (IoU >= nms_iou) aynı sınıfı tespit ederse:
      Skorları ağırlıklı olarak birleştir: (score1 * w1 + score2 * w2)
    - Sadece bir model tespit ettiyse: O modelin skorunu kullan
    
    Args:
        scratch_results: Çizik modelinden gelen tahminler
        blackdot_results: Siyah nokta modelinden gelen tahminler
        img_shape: Görüntü boyutu (h, w)
        nms_iou: Eşleştirme için IoU eşiği
    
    Returns:
        [(class_id, box, ensemble_confidence, model_source, details), ...]
    """
    h, w = img_shape[:2]
    
    # Her modelden tahminleri ayrı ayrı topla
    scratch_preds = []
    blackdot_preds = []
    
    # Çizik modelinden tahminleri al
    if scratch_results and scratch_results[0].boxes:
        for box in scratch_results[0].boxes:
            cls_id = int(box.cls.item())
            conf = box.conf.item()
            
            if hasattr(box, 'xywhn'):
                pred_box = box.xywhn[0].tolist()
            else:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                pred_box = [
                    ((x1 + x2) / 2) / w,
                    ((y1 + y2) / 2) / h,
                    (x2 - x1) / w,
                    (y2 - y1) / h
                ]
            
            scratch_preds.append({
                'cls_id': cls_id,
                'box': pred_box,
                'conf': conf,
                'matched': False
            })
    
    # Siyah nokta modelinden tahminleri al
    if blackdot_results and blackdot_results[0].boxes:
        for box in blackdot_results[0].boxes:
            cls_id = int(box.cls.item())
            conf = box.conf.item()
            
            if hasattr(box, 'xywhn'):
                pred_box = box.xywhn[0].tolist()
            else:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                pred_box = [
                    ((x1 + x2) / 2) / w,
                    ((y1 + y2) / 2) / h,
                    (x2 - x1) / w,
                    (y2 - y1) / h
                ]
            
            blackdot_preds.append({
                'cls_id': cls_id,
                'box': pred_box,
                'conf': conf,
                'matched': False
            })
    
    ensemble_results = []
    
    # Her çizik modeli tahmini için siyah nokta modelinde eşleşme ara
    for sp in scratch_preds:
        best_match = None
        best_iou = 0
        
        for bp in blackdot_preds:
            if bp['matched']:
                continue
            
            # Aynı sınıf mı kontrol et
            if sp['cls_id'] != bp['cls_id']:
                continue
            
            iou = calculate_iou(sp['box'], bp['box'])
            if iou >= nms_iou and iou > best_iou:
                best_iou = iou
                best_match = bp
        
        if best_match:
            # İKİ MODEL DE TESPİT ETTİ - ENSEMBLE SKORU HESAPLA
            best_match['matched'] = True
            sp['matched'] = True
            
            cls_id = sp['cls_id']
            
            # Sınıfa göre ağırlıkları belirle
            if cls_id == 0:  # scratch
                w1 = SCRATCH_MODEL_STRONG_WEIGHT
                w2 = BLACKDOT_MODEL_WEAK_WEIGHT
            else:  # black_dot
                w1 = SCRATCH_MODEL_WEAK_WEIGHT
                w2 = BLACKDOT_MODEL_STRONG_WEIGHT
            
            # Ağırlıklı ortalama skor
            ensemble_conf = (sp['conf'] * w1 + best_match['conf'] * w2) / (w1 + w2)
            
            # Box'ları da birleştir (ortalama al)
            ensemble_box = [
                (sp['box'][0] + best_match['box'][0]) / 2,
                (sp['box'][1] + best_match['box'][1]) / 2,
                (sp['box'][2] + best_match['box'][2]) / 2,
                (sp['box'][3] + best_match['box'][3]) / 2
            ]
            
            ensemble_results.append((
                cls_id,
                ensemble_box,
                ensemble_conf,
                'ensemble',  # Her iki model de tespit etti
                f"S:{sp['conf']:.3f}*{w1}+B:{best_match['conf']:.3f}*{w2}"
            ))
        else:
            # SADECE ÇİZİK MODELİ TESPİT ETTİ
            sp['matched'] = True
            cls_id = sp['cls_id']
            
            # Tek model skoru, ağırlıkla çarp
            if cls_id == 0:
                weight = SCRATCH_MODEL_STRONG_WEIGHT
            else:
                weight = SCRATCH_MODEL_WEAK_WEIGHT
            
            weighted_conf = sp['conf'] * weight
            
            ensemble_results.append((
                cls_id,
                sp['box'],
                weighted_conf,
                'scratch_only',
                f"S:{sp['conf']:.3f}*{weight}"
            ))
    
    # Eşleşmeyen siyah nokta modeli tahminleri
    for bp in blackdot_preds:
        if not bp['matched']:
            cls_id = bp['cls_id']
            
            # Tek model skoru, ağırlıkla çarp
            if cls_id == 1:
                weight = BLACKDOT_MODEL_STRONG_WEIGHT
            else:
                weight = BLACKDOT_MODEL_WEAK_WEIGHT
            
            weighted_conf = bp['conf'] * weight
            
            ensemble_results.append((
                cls_id,
                bp['box'],
                weighted_conf,
                'blackdot_only',
                f"B:{bp['conf']:.3f}*{weight}"
            ))
    
    return ensemble_results


def draw_boxes_ensemble(image, gt_items, ensemble_preds, class_names):
    """
    Ground truth ve ensemble tahminlerini görüntü üzerine çiz.
    """
    h, w = image.shape[:2]
    
    # 🔴 Ground Truth
    for cls_id, box in gt_items:
        x, y, bw, bh = box
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)

        # 5 pixel offset (Genişletme)
        OFFSET = 5
        x1 = max(0, x1 - OFFSET)
        y1 = max(0, y1 - OFFSET)
        x2 = min(w, x2 + OFFSET)
        y2 = min(h, y2 + OFFSET)

        label_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), COLORS['gt'], LINE_THICKNESS)
        cv2.putText(image, f"GT: {label_name}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLORS['gt'], LINE_THICKNESS)

    # 🟢 Ensemble Tahminleri
    for pred in ensemble_preds:
        # 5 elemanlı tuple: (cls_id, box, conf, model_source, details)
        if len(pred) == 5:
            cls_id, box, conf, model_source, details = pred
        else:
            cls_id, box, conf, model_source = pred
            details = ""
        
        x, y, bw, bh = box
        
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)

        # 5 pixel offset
        OFFSET = 5
        x1 = max(0, x1 - OFFSET)
        y1 = max(0, y1 - OFFSET)
        x2 = min(w, x2 + OFFSET)
        y2 = min(h, y2 + OFFSET)

        label_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
        
        # Model kaynağına göre renk ve etiket
        if model_source == 'ensemble':
            color = (0, 255, 255)  # Sarı - Her iki model de tespit etti
            tag = "E"  # Ensemble
        elif model_source == 'scratch_only':
            color = COLORS.get(label_name, (0, 255, 0))
            tag = "S"  # Sadece çizik modeli
        elif model_source == 'blackdot_only':
            color = COLORS.get(label_name, (255, 0, 0))
            tag = "B"  # Sadece siyah nokta modeli
        else:
            color = COLORS.get(label_name, COLORS['ensemble'])
            tag = "?"
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, LINE_THICKNESS)
        
        cv2.putText(image, f"[{tag}] {label_name} {conf:.2f}",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, LINE_THICKNESS)

    return image


def evaluate_predictions(gt_items, ensemble_preds, iou_threshold=0.25):
    """
    Ensemble tahminlerini ground truth ile karşılaştırarak değerlendir.
    Returns: [(gt_label, pred_label, iou, result_type, model_source, confidence, details), ...]
    """
    results = []
    matched_preds = set()
    
    # Her ground truth için en iyi eşleşen tahmini bul
    for gt_class_id, gt_box in gt_items:
        gt_label = CLASS_NAMES[gt_class_id] if gt_class_id < len(CLASS_NAMES) else str(gt_class_id)
        
        best_iou = 0
        best_pred_label = "Tespit Edilemedi"
        best_pred_idx = -1
        best_pred_class_id = -1
        best_model_source = ""
        best_confidence = 0.0
        best_details = ""
        
        for pred_idx, pred in enumerate(ensemble_preds):
            # 5 elemanlı tuple: (cls_id, pred_box, conf, model_source, details)
            if len(pred) == 5:
                cls_id, pred_box, conf, model_source, details = pred
            else:
                cls_id, pred_box, conf, model_source = pred
                details = ""
            
            iou = calculate_iou(gt_box, pred_box)
            
            if iou > best_iou and pred_idx not in matched_preds:
                best_iou = iou
                best_pred_label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                best_pred_idx = pred_idx
                best_pred_class_id = cls_id
                best_model_source = model_source
                best_confidence = conf
                best_details = details
        
        # Sonuç tipini belirle
        if best_iou >= iou_threshold:
            if best_pred_class_id == gt_class_id:
                result_type = "TP"  # True Positive
            else:
                result_type = "FP"  # False Positive (yanlış sınıf)
            matched_preds.add(best_pred_idx)
        else:
            result_type = "FN"  # False Negative
        
        results.append({
            "Gerçek Etiket": gt_label,
            "Ensemble Tahmin": best_pred_label,
            "Ensemble Confidence": f"{best_confidence:.4f}" if best_confidence > 0 else "-",
            "Skor Detayı": best_details if best_details else "-",
            "IoU": f"{best_iou:.2f}",
            "Sonuç": result_type,
            "Model Kaynağı": best_model_source if best_model_source else "-"
        })
    
    # Eşleşmeyen tahminler (False Positive)
    for pred_idx, pred in enumerate(ensemble_preds):
        if pred_idx not in matched_preds:
            if len(pred) == 5:
                cls_id, pred_box, conf, model_source, details = pred
            else:
                cls_id, pred_box, conf, model_source = pred
                details = ""
            pred_label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            results.append({
                "Gerçek Etiket": "YOK",
                "Ensemble Tahmin": pred_label,
                "Ensemble Confidence": f"{conf:.4f}",
                "Skor Detayı": details if details else "-",
                "IoU": "0.00",
                "Sonuç": "FP",
                "Model Kaynağı": model_source
            })
    
    return results


def calculate_metrics(all_results):
    """
    TP, FP, FN değerlerinden metrik hesapla.
    """
    tp = sum(1 for r in all_results if r["Sonuç"] == "TP")
    fp = sum(1 for r in all_results if r["Sonuç"] == "FP")
    fn = sum(1 for r in all_results if r["Sonuç"] == "FN")
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


def calculate_class_metrics(all_results, class_names):
    """
    Her sınıf için doğru TP, FP, FN hesapla.
    
    Doğru tanımlar:
    - TP (sınıf X): GT=X ve Tahmin=X ve eşleşme var (IoU >= threshold)
    - FP (sınıf X): Tahmin=X ama ya GT yok ya da GT != X
    - FN (sınıf X): GT=X ama ya tahmin yok ya da tahmin != X veya eşleşme yok
    
    Returns: {class_name: {TP, FP, FN, Precision, Recall, F1}, ...}
    """
    class_metrics = {}
    
    for class_name in class_names:
        # TP: GT ve tahmin aynı sınıf ve sonuç TP
        tp = sum(1 for r in all_results 
                 if r["Sonuç"] == "TP" and r["Gerçek Etiket"] == class_name)
        
        # FP: Tahmin bu sınıf ama ya GT yok ("YOK") ya da GT farklı sınıf
        # FP durumları:
        # 1. GT yok (Gerçek Etiket = "YOK") ve Tahmin = class_name
        # 2. GT var ama farklı sınıf, tahmin = class_name (yanlış sınıf tahmini)
        fp = sum(1 for r in all_results 
                 if r["Sonuç"] == "FP" and r["Ensemble Tahmin"] == class_name)
        
        # FN: GT bu sınıf ama tespit edilemedi (sonuç FN)
        fn = sum(1 for r in all_results 
                 if r["Sonuç"] == "FN" and r["Gerçek Etiket"] == class_name)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[class_name] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }
    
    return class_metrics


# --- ANA SCRIPT ---

def save_summary_txt(output_dir, metrics, class_metrics_dict, model_stats, config):
    """
    Özet metrikleri txt dosyasına kaydet.
    """
    txt_path = os.path.join(output_dir, 'summary.txt')
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("RT-DETR ENSEMBLE MODEL TEST - ÖZET RAPOR\n")
        f.write("=" * 60 + "\n\n")
        
        # Test Konfigürasyonu
        f.write("📋 TEST KONFİGÜRASYONU\n")
        f.write("-" * 40 + "\n")
        f.write(f"Tarih/Saat: {config['timestamp']}\n")
        f.write(f"Test Verisi: {config['dataset_dir']}\n")
        f.write(f"Çizik Modeli: {config['scratch_model']}\n")
        f.write(f"Siyah Nokta Modeli: {config['blackdot_model']}\n")
        f.write(f"Confidence Eşiği: {config['conf_thresh']}\n")
        f.write(f"IoU Eşiği: {config['iou_thresh']}\n")
        f.write(f"NMS IoU Eşiği: {config['nms_iou_thresh']}\n")
        f.write(f"Toplam Görsel: {config['total_images']}\n")
        f.write("\n")
        
        # Genel Metrikler
        f.write("📊 GENEL METRİKLER\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Metrik':<25} {'Değer':>10}\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'True Positive (TP)':<25} {metrics['TP']:>10}\n")
        f.write(f"{'False Positive (FP)':<25} {metrics['FP']:>10}\n")
        f.write(f"{'False Negative (FN)':<25} {metrics['FN']:>10}\n")
        f.write(f"{'Precision':<25} {metrics['Precision']:>10.4f}\n")
        f.write(f"{'Recall':<25} {metrics['Recall']:>10.4f}\n")
        f.write(f"{'F1 Score':<25} {metrics['F1']:>10.4f}\n")
        f.write("\n")
        
        # Sınıf Bazlı Metrikler
        f.write("🎯 SINIF BAZLI SONUÇLAR\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Sınıf':<15} {'TP':>6} {'FP':>6} {'FN':>6} {'Precision':>10} {'Recall':>10} {'F1':>10}\n")
        f.write("-" * 60 + "\n")
        
        for class_name, m in class_metrics_dict.items():
            f.write(f"{class_name:<15} {m['TP']:>6} {m['FP']:>6} {m['FN']:>6} {m['Precision']:>10.4f} {m['Recall']:>10.4f} {m['F1']:>10.4f}\n")
        f.write("\n")
        
        # Model Kaynak Analizi
        f.write("🔍 MODEL KAYNAK ANALİZİ\n")
        f.write("-" * 40 + "\n")
        f.write(f"Ensemble (her iki model) TP: {model_stats['ensemble_tp']}\n")
        f.write(f"Sadece Çizik Modeli TP: {model_stats['scratch_only_tp']}\n")
        f.write(f"Sadece Siyah Nokta Modeli TP: {model_stats['blackdot_only_tp']}\n")
        f.write("\n")
        
        f.write("=" * 60 + "\n")
        f.write("Rapor otomatik olarak oluşturulmuştur.\n")
    
    return txt_path


def main_ensemble(dataset_dir, conf_thresh, iou_thresh, nms_iou_thresh):
    """
    Ana ensemble test fonksiyonu.
    """
    # Timestamp ile çıktı klasörü oluştur
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_BASE_DIR, f"test_{timestamp}")
    images_output_dir = os.path.join(output_dir, "images")
    
    print("=" * 60)
    print("RT-DETR ENSEMBLE MODEL TEST")
    print("=" * 60)
    print(f"Çizik modeli: {SCRATCH_MODEL_PATH}")
    print(f"Siyah nokta modeli: {BLACKDOT_MODEL_PATH}")
    print(f"Test verisi: {dataset_dir}")
    print(f"Conf Threshold: {conf_thresh}")
    print(f"IoU Threshold: {iou_thresh}")
    print(f"NMS IoU Threshold: {nms_iou_thresh}")
    print(f"Çıktı klasörü: {output_dir}")
    print("=" * 60)
    
    # Model kontrolleri
    if not os.path.exists(SCRATCH_MODEL_PATH):
        print(f"HATA: Çizik modeli bulunamadı -> {SCRATCH_MODEL_PATH}")
        return
    
    if not os.path.exists(BLACKDOT_MODEL_PATH):
        print(f"HATA: Siyah nokta modeli bulunamadı -> {BLACKDOT_MODEL_PATH}")
        return
    
    # Cihaz belirle (GPU varsa kullan, yoksa CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"  🚀 GPU (CUDA) algılandı, modeller GPU'da çalışacak.")
    else:
        print(f"  ℹ️ GPU bulunamadı, modeller CPU'da çalışacak.")

    # Modelleri yükle
    print("\n📥 Modeller yükleniyor...")
    try:
        scratch_model = RTDETR(SCRATCH_MODEL_PATH).to(device)
        print("  ✓ Çizik modeli yüklendi")
    except Exception as e:
        print(f"  ✗ Çizik modeli yüklenirken hata: {e}")
        return
    
    try:
        blackdot_model = RTDETR(BLACKDOT_MODEL_PATH).to(device)
        print("  ✓ Siyah nokta modeli yüklendi")
    except Exception as e:
        print(f"  ✗ Siyah nokta modeli yüklenirken hata: {e}")
        return
    
    # Veri klasörlerini bul (Gelişmiş rekürsif arama)
    images_dir = dataset_dir
    labels_dir = dataset_dir
    
    # En az bir görsel var mı diye rekürsif kontrol et
    has_images = False
    for root, dirs, files in os.walk(dataset_dir):
        if any(f.lower().endswith(('.jpg', '.png', '.jpeg')) for f in files):
            has_images = True
            break
            
    if not has_images:
        print(f"\nHATA: Klasörde veya alt klasörlerinde hiç görsel bulunamadı: {dataset_dir}")
        return
    
    # Çıktı klasörlerini oluştur
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_output_dir, exist_ok=True)
    
    # Görüntü dosyalarını özyinelemeli (recursive) olarak listele
    image_files_with_rel_path = []
    for root, dirs, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, images_dir)
                image_files_with_rel_path.append(rel_path)
    
    if not image_files_with_rel_path:
        print("\nKlasörde işlenecek görsel bulunamadı.")
        return
    
    print(f"\n📷 Toplam {len(image_files_with_rel_path)} görsel işlenecek...\n")
    
    all_results = []
    
    try:
        for idx, rel_image_path in enumerate(image_files_with_rel_path, 1):
            image_path = os.path.join(images_dir, rel_image_path)
            image_name = os.path.basename(rel_image_path)
            folder_name = os.path.dirname(rel_image_path) or "root"
            
            # Label yolunu belirle (Akıllı eşleştirme)
            # Önce aynı klasörde ara, sonra /images/ yerine /labels/ koyarak dene
            label_filename = os.path.splitext(image_name)[0] + '.txt'
            label_path = os.path.join(os.path.dirname(image_path), label_filename)
            
            if not os.path.exists(label_path):
                # images -> labels klasör değişikliğini dene
                possible_label_path = image_path.replace('/images/', '/labels/').replace('\\images\\', '\\labels\\')
                possible_label_path = os.path.splitext(possible_label_path)[0] + '.txt'
                if os.path.exists(possible_label_path):
                    label_path = possible_label_path
            
            # Ground truth'u oku
            gt_items = get_ground_truth(label_path)
            
            # Görüntüyü oku
            img = cv2.imread(image_path)
            if img is None:
                print(f"  [{idx}/{len(image_files_with_rel_path)}] ⚠ Görsel okunamadı: {image_name}")
                continue
            
            # Her iki modelden tahmin al
            scratch_results = scratch_model(image_path, conf=conf_thresh, verbose=False)
            blackdot_results = blackdot_model(image_path, conf=conf_thresh, verbose=False)
            
            # Tahminleri birleştir (ensemble)
            ensemble_preds = ensemble_predictions(
                scratch_results, 
                blackdot_results, 
                img.shape,
                nms_iou=nms_iou_thresh
            )
            
            # Değerlendirme yap
            image_results = evaluate_predictions(gt_items, ensemble_preds, iou_thresh)
            
            # Her sonuca görsel adını ve klasör adını ekle
            for r in image_results:
                r["Fotoğraf Adı"] = image_name
                r["Klasör"] = folder_name
            
            all_results.extend(image_results)
            
            # Görselleştir ve kaydet (klasör yapısını çıktıda da koru)
            drawn_img = draw_boxes_ensemble(img, gt_items, ensemble_preds, CLASS_NAMES)
            
            # Çıktı görüntüsü için klasör oluştur (ensemble_results içinde)
            if folder_name == "root":
                image_output_subdir = images_output_dir
            else:
                image_output_subdir = os.path.join(images_output_dir, folder_name)
                
            os.makedirs(image_output_subdir, exist_ok=True)
            
            output_path = os.path.join(image_output_subdir, image_name)
            cv2.imwrite(output_path, drawn_img)
            
            # İlerleme göster
            print(f"  [{idx}/{len(image_files_with_rel_path)}] ✓ {rel_image_path} - GT: {len(gt_items)}, Pred: {len(ensemble_preds)}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ İşlem kullanıcı tarafından durduruldu. O ana kadarki sonuçlar hazırlanıyor...")
    
    # Metrikleri hesapla
    print("\n" + "=" * 60)
    print("📊 SONUÇLAR")
    print("=" * 60)
    
    metrics = calculate_metrics(all_results)
    print(f"\n  True Positive (TP):  {metrics['TP']}")
    print(f"  False Positive (FP): {metrics['FP']}")
    print(f"  False Negative (FN): {metrics['FN']}")
    print(f"\n  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1 Score:  {metrics['F1']:.4f}")
    
    # Sınıf bazlı metrikler (doğru hesaplama ile)
    print("\n  --- Sınıf Bazlı Sonuçlar ---")
    class_metrics_dict = calculate_class_metrics(all_results, CLASS_NAMES)
    for class_name, class_metrics in class_metrics_dict.items():
        print(f"\n  {class_name.upper()}:")
        print(f"    TP: {class_metrics['TP']}, FP: {class_metrics['FP']}, FN: {class_metrics['FN']}")
        print(f"    Precision: {class_metrics['Precision']:.4f}, Recall: {class_metrics['Recall']:.4f}, F1: {class_metrics['F1']:.4f}")
    
    # Model kaynak analizi
    print("\n  --- Model Kaynak Analizi ---")
    ensemble_preds_tp = [r for r in all_results if r["Model Kaynağı"] == "ensemble" and r["Sonuç"] == "TP"]
    scratch_only_preds = [r for r in all_results if r["Model Kaynağı"] == "scratch_only" and r["Sonuç"] == "TP"]
    blackdot_only_preds = [r for r in all_results if r["Model Kaynağı"] == "blackdot_only" and r["Sonuç"] == "TP"]
    print(f"  Ensemble (her iki model) TP: {len(ensemble_preds_tp)}")
    print(f"  Sadece çizik modeli TP: {len(scratch_only_preds)}")
    print(f"  Sadece siyah nokta modeli TP: {len(blackdot_only_preds)}")
    
    model_stats = {
        'ensemble_tp': len(ensemble_preds_tp),
        'scratch_only_tp': len(scratch_only_preds),
        'blackdot_only_tp': len(blackdot_only_preds)
    }
    
    # Konfigürasyon bilgisi
    config = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset_dir': dataset_dir,
        'scratch_model': SCRATCH_MODEL_PATH,
        'blackdot_model': BLACKDOT_MODEL_PATH,
        'conf_thresh': conf_thresh,
        'iou_thresh': iou_thresh,
        'nms_iou_thresh': nms_iou_thresh,
        'total_images': len(image_files_with_rel_path)
    }
    
    # Sonuçları kaydet
    if all_results:
        # Sütun sırasını düzenle
        df = pd.DataFrame(all_results)
        column_order = ["Klasör", "Fotoğraf Adı", "Gerçek Etiket", "Ensemble Tahmin", "Ensemble Confidence", "Skor Detayı", "IoU", "Sonuç", "Model Kaynağı"]
        df = df[column_order]
        
        try:
            # Excel dosyası
            excel_path = os.path.join(output_dir, 'results.xlsx')
            
            # Klasöre göre sırala (tek sayfada gruplandırma için)
            df = df.sort_values(by=["Klasör", "Fotoğraf Adı"])
            
            # Birden fazla sayfa ile kaydet
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Tüm sonuçları tek bir sayfada kaydet
                df.to_excel(writer, sheet_name='Detaylı Sonuçlar', index=False)
                
                # Özet metrikleri ayrı bir sayfaya yaz
                summary_df = pd.DataFrame([{
                    "Metrik": "True Positive (TP)",
                    "Değer": metrics['TP']
                }, {
                    "Metrik": "False Positive (FP)",
                    "Değer": metrics['FP']
                }, {
                    "Metrik": "False Negative (FN)",
                    "Değer": metrics['FN']
                }, {
                    "Metrik": "Precision",
                    "Değer": f"{metrics['Precision']:.4f}"
                }, {
                    "Metrik": "Recall",
                    "Değer": f"{metrics['Recall']:.4f}"
                }, {
                    "Metrik": "F1 Score",
                    "Değer": f"{metrics['F1']:.4f}"
                }])
                summary_df.to_excel(writer, sheet_name='Özet Metrikler', index=False)
            
            print(f"\n✅ Excel dosyası kaydedildi: {excel_path}")
        except Exception as e:
            print(f"\n❌ Excel kaydetme hatası: {e}")
        
        # Özet TXT dosyası
        try:
            txt_path = save_summary_txt(output_dir, metrics, class_metrics_dict, model_stats, config)
            print(f"✅ Özet rapor kaydedildi: {txt_path}")
        except Exception as e:
            print(f"❌ TXT kaydetme hatası: {e}")
    
    print(f"\n✅ Tüm sonuçlar kaydedildi: {output_dir}")
    print(f"   ├── summary.txt (Özet rapor)")
    print(f"   └── results.xlsx (Detaylı sonuçlar)")
    print(f"✅ Analiz edilmiş fotoğraflar kaynak klasörlerdeki 'analyzed' dizinine kaydedildi.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RT-DETR Ensemble Model Test Script")
    parser.add_argument(
        "--source", 
        type=str, 
        default=r"C:/Users/ali.donbaloglu/Desktop/Lens/Model_test/Yeni/PARTLAR",
        help="Test verisi klasör yolu (images/ ve labels/ alt klasörleri içermeli)"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=DEFAULT_CONF_THRESHOLD,
        help="Confidence (güven) eşiği"
    )
    parser.add_argument(
        "--iou", 
        type=float, 
        default=DEFAULT_IOU_THRESHOLD,
        help="IoU eşiği (değerlendirme için)"
    )
    parser.add_argument(
        "--nms-iou", 
        type=float, 
        default=NMS_IOU_THRESHOLD,
        help="NMS IoU eşiği (ensemble birleştirme için)"
    )
    
    args = parser.parse_args()
    main_ensemble(args.source, args.conf, args.iou, args.nms_iou)

