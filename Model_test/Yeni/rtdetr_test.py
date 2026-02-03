import os
import cv2
import pandas as pd
import argparse
from ultralytics import RTDETR

# --- AYARLAR ---
# Varsayılan model yolu
DEFAULT_MODEL_PATH = r"C:/Users/ali.donbaloglu/Desktop/Lens/training_plots/Yeni_modeller/RT-DETR_X_V3/Kabin_aralik_20260108_114749/weights/best.pt"
CLASS_NAMES = ["scratch", "black_dot"]

OUTPUT_EXCEL_PATH = 'rtdetr_sinifli_iou_test_sonuclari_deneme.xlsx'
OUTPUT_IMAGE_DIR = 'rtdetr_gorsel_sonuclar_deneme'

# Varsayılan Eşik Değerleri
DEFAULT_CONF_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.25

# Görselleştirme Ayarları
LINE_THICKNESS = 1 
FONT_SCALE = 0.6

# --- YARDIMCI FONKSİYONLAR ---
def get_ground_truth(label_path):
    gt_list = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = [float(p) for p in line.split()]
                class_id = int(parts[0])
                points = parts[1:]

                # Polygon formatında olabilir, bounding box'a çevirelim
                x_coords = points[0::2]
                y_coords = points[1::2]

                x_min, y_min = min(x_coords), min(y_coords)
                x_max, y_max = max(x_coords), max(y_coords)

                gt_box = [
                    (x_min + x_max) / 2,
                    (y_min + y_max) / 2,
                    x_max - x_min,
                    y_max - y_min
                ]
                gt_list.append((class_id, gt_box))
        return gt_list
    except:
        return []

def calculate_iou(boxA, boxB):
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

    return inter / (areaA + areaB - inter) if (areaA + areaB - inter) > 0 else 0

def draw_boxes(image, gt_items, boxes, class_names):
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
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), LINE_THICKNESS)
        cv2.putText(image, f"GT: {label_name}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), LINE_THICKNESS)

    # 🟢 Tahminler
    for box in boxes:
        cls_id = int(box.cls.item())
        conf = box.conf.item()
        
        # RT-DETR/YOLO uyumluluğu için
        if hasattr(box, 'xywhn'):
            x, y, bw, bh = box.xywhn[0].tolist()
        else:
             # Fallback
            x1_raw, y1_raw, x2_raw, y2_raw = box.xyxy[0].tolist()
            x = ((x1_raw + x2_raw) / 2) / w
            y = ((y1_raw + y2_raw) / 2) / h
            bw = (x2_raw - x1_raw) / w
            bh = (y2_raw - y1_raw) / h

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

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), LINE_THICKNESS)
        cv2.putText(image, f"PRED: {label_name} {conf:.2f}",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 0), LINE_THICKNESS)

    return image

# --- ANA SCRIPT ---
def main_rtdetr(dataset_dir, model_path, conf_thresh, iou_thresh):
    if not os.path.exists(model_path):
        print(f"HATA: Model dosyası bulunamadı -> {model_path}")
        return

    print(f"Model yükleniyor: {model_path}")
    try:
        model = RTDETR(model_path) # RT-DETR yükleme
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        return

    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')

    if not os.path.exists(images_dir):
        if any(f.lower().endswith(('.jpg', '.png')) for f in os.listdir(dataset_dir)):
             images_dir = dataset_dir
             labels_dir = dataset_dir
             print(f"Uyarı: 'images' klasörü bulunamadı, görseller ana dizinde aranıyor: {dataset_dir}")
        else:
            print(f"HATA: Görsellerin bulunduğu klasör bulunamadı: {images_dir}")
            return

    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    
    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not image_files:
        print("Klasörde işlenecek görsel bulunamadı.")
        return

    print(f"Toplam {len(image_files)} görsel işlenecek...")
    
    results_list = []

    for image_name in image_files:
        image_path = os.path.join(images_dir, image_name)
        label_filename = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)

        gt_items = get_ground_truth(label_path)
        img = cv2.imread(image_path)
        if img is None: continue

        # Model Tahmini
        results = model(image_path, conf=conf_thresh, verbose=False)
        boxes = results[0].boxes if results[0].boxes else []

        # Değerlendirme Döngüsü - TP, FP, FN ayrımı
        matched_preds = set()  # Eşleşen tahminlerin indeksleri
        
        for gt_class_id, gt_box in gt_items:
            gt_label = CLASS_NAMES[gt_class_id] if gt_class_id < len(CLASS_NAMES) else str(gt_class_id)
            
            best_iou = 0
            best_label = "Tespit Edilemedi"
            best_pred_idx = -1
            best_pred_class_id = -1

            for pred_idx, box in enumerate(boxes):
                pred_id = int(box.cls.item())
                
                # Box Formatı
                if hasattr(box, 'xywhn'):
                    pred_box = box.xywhn[0].tolist()
                else:
                    h_img, w_img = img.shape[:2]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    pred_box = [((x1 + x2) / 2) / w_img, ((y1 + y2) / 2) / h_img, (x2 - x1) / w_img, (y2 - y1) / h_img]

                iou = calculate_iou(gt_box, pred_box)

                if iou > best_iou and pred_idx not in matched_preds:
                    best_iou = iou
                    best_label = CLASS_NAMES[pred_id] if pred_id < len(CLASS_NAMES) else str(pred_id)
                    best_pred_idx = pred_idx
                    best_pred_class_id = pred_id

            # Sonuç Kuralı - TP / FN ayrımı
            if best_iou >= iou_thresh:
                if best_pred_class_id == gt_class_id:
                    sonuc = "TP"  # True Positive: Doğru konum + doğru sınıf
                else:
                    sonuc = "FP"  # False Positive: Doğru konum ama yanlış sınıf (sınıf hatalı)
                matched_preds.add(best_pred_idx)
            else:
                sonuc = "FN"  # False Negative: Ground truth tespit edilemedi

            results_list.append({
                "Fotoğraf Adı": image_name,
                "Gerçek Etiket": gt_label,
                "Tahmin Edilen Etiket": best_label,
                "IoU": f"{best_iou:.2f}",
                "Sonuç": sonuc
            })
        
        # FP'ler: Eşleşmeyen tahminler (Ground truth ile eşleşmeyen tüm tahminler)
        for pred_idx, box in enumerate(boxes):
            if pred_idx not in matched_preds:
                pred_id = int(box.cls.item())
                pred_label = CLASS_NAMES[pred_id] if pred_id < len(CLASS_NAMES) else str(pred_id)
                conf = box.conf.item()
                
                results_list.append({
                    "Fotoğraf Adı": image_name,
                    "Gerçek Etiket": "YOK",
                    "Tahmin Edilen Etiket": pred_label,
                    "IoU": f"0.00",
                    "Sonuç": "FP"  # False Positive: Ground truth olmadan yapılan tahmin
                })

        # 📸 Görsel Kaydet
        drawn_img = draw_boxes(img, gt_items, boxes, CLASS_NAMES)
        cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, image_name), drawn_img)

    # Excel Kayıt
    if results_list:
        df = pd.DataFrame(results_list)
        try:
            df.to_excel(OUTPUT_EXCEL_PATH, index=False)
            print(f"✅ Excel başarıyla kaydedildi: {OUTPUT_EXCEL_PATH}")
        except Exception as e:
             print(f"❌ Excel kaydetme hatası: {e}")
    
    print(f"✅ Görseller '{OUTPUT_IMAGE_DIR}' klasörüne kaydedildi.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RT-DETR Evaluation Script")
    parser.add_argument("--source", type=str, default="C:/Users/ali.donbaloglu/Desktop/Lens/Model_test/Yeni/test", help="Test verisi klasör yolu")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Model dosya yolu (.pt)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF_THRESHOLD, help="Güven Eşiği")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_THRESHOLD, help="IoU Eşiği")
    
    args = parser.parse_args()
    main_rtdetr(args.source, args.model, args.conf, args.iou)
