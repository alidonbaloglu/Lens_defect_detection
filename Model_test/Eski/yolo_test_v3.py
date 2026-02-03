import os
import pandas as pd
from ultralytics import YOLO

# --- AYARLAR ---
MODEL_PATH = 'C:/Users/ali.donbaloglu/Desktop/Lens/Modeller/Kabin_yolo/weights/best.pt'
DATASET_DIR = 'C:/Users/ali.donbaloglu/Desktop/Lens/yolo_kabin_dataset/test'
CLASS_NAMES = ["0", "black_dot", "scratch", "watermark"]
OUTPUT_EXCEL_PATH = 'yolo_sinifli_iou_test_sonuclari.xlsx'

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.4

# --- YARDIMCI FONKSİYONLAR ---
def get_ground_truth(label_path):
    """Birden fazla ground truth döndürür: [(class_id, box), ...]"""
    gt_list = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [float(p) for p in line.split()]
                class_id = int(parts[0])
                points = parts[1:]
                x_coords = points[0::2]
                y_coords = points[1::2]
                x_min, y_min = min(x_coords), min(y_coords)
                x_max, y_max = max(x_coords), max(y_coords)
                gt_box = [(x_min + x_max)/2, (y_min + y_max)/2, x_max - x_min, y_max - y_min]
                gt_list.append((class_id, gt_box))
        return gt_list
    except (FileNotFoundError, IndexError, ValueError):
        return []

def calculate_iou(boxA, boxB):
    """İki kutu arasındaki IoU oranını hesaplar."""
    def to_corners(box):
        x_center, y_center, w, h = box
        return [x_center - w/2, y_center - h/2, x_center + w/2, y_center + h/2]

    boxA_corners, boxB_corners = to_corners(boxA), to_corners(boxB)
    xA = max(boxA_corners[0], boxB_corners[0])
    yA = max(boxA_corners[1], boxB_corners[1])
    xB = min(boxA_corners[2], boxB_corners[2])
    yB = min(boxA_corners[3], boxB_corners[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    denom = float(boxAArea + boxBArea - interArea)
    return interArea / denom if denom > 0 else 0

# --- ANA SCRIPT ---
def main_yolo():
    model = YOLO(MODEL_PATH)
    images_dir = os.path.join(DATASET_DIR, 'images')
    labels_dir = os.path.join(DATASET_DIR, 'labels')
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    results_list = []

    for image_name in image_files:
        image_path = os.path.join(images_dir, image_name)
        label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + '.txt')

        gt_items = get_ground_truth(label_path)

        if not gt_items:
            results_list.append({'Fotoğraf Adı': image_name, 'Gerçek Etiket': 'Etiket Dosyası Yok/Bozuk',
                                 'Tahmin Edilen Etiket': 'N/A', 'IoU': 0, 'Sonuç': 'Belirsiz'})
            continue

        # YOLO model tahminleri
        results = model(image_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
        boxes = results[0].boxes if results[0].boxes else []

        for gt_class_id, gt_box in gt_items:
            gt_label_str = CLASS_NAMES[gt_class_id]

            best_iou = 0
            best_match_label = "Tespit Edilemedi"
            found_correct = False

            for box in boxes:
                pred_class_id = int(box.cls.item())
                pred_box = box.xywhn[0].tolist()  # normalized [x_center, y_center, w, h]
                iou = calculate_iou(gt_box, pred_box)

                # Eğer sınıf ve konum doğruysa -> doğru sonuç
                if pred_class_id == gt_class_id and iou > IOU_THRESHOLD:
                    found_correct = True
                    best_iou = iou
                    best_match_label = CLASS_NAMES[pred_class_id]
                    break

                # IoU en yüksek olan tahmini sakla
                if iou > best_iou:
                    best_iou = iou
                    best_match_label = CLASS_NAMES[pred_class_id]

            sonuc = "Doğru" if found_correct else ("Yanlış" if best_match_label != "Tespit Edilemedi" else "Tespit Edilemedi")

            results_list.append({
                'Fotoğraf Adı': image_name,
                'Gerçek Etiket': gt_label_str,
                'Tahmin Edilen Etiket': best_match_label,
                'IoU': f"{best_iou:.2f}",
                'Sonuç': sonuc
            })

            print(f"{image_name} | Etiket='{gt_label_str}', Tahmin='{best_match_label}', IoU={best_iou:.2f} -> {sonuc}")

    df = pd.DataFrame(results_list)
    df.to_excel(OUTPUT_EXCEL_PATH, index=False)
    print(f"\n✅ İşlem tamamlandı! Sonuçlar '{OUTPUT_EXCEL_PATH}' dosyasına kaydedildi.")

if __name__ == '__main__':
    main_yolo()
