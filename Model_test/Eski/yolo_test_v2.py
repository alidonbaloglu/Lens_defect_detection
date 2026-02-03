import os
import pandas as pd
from ultralytics import YOLO

# --- AYARLAR ---
MODEL_PATH = 'C:/Users/ali.donbaloglu/Desktop/Lens/Modeller/Kabin_yolo/weights/best.pt'
DATASET_DIR = 'C:/Users/ali.donbaloglu/Desktop/Lens/yolo_kabin_dataset/test'
CLASS_NAMES = [
    "0",
    "black_dot",
    "scratch",
    "watermark",
    ]
OUTPUT_EXCEL_PATH = 'yolo_sinifli_iou_test_sonuclari.xlsx'

# YENİ AYARLAR
CONFIDENCE_THRESHOLD = 0.25 # Modelin bir tespiti geçerli sayması için minimum güven skoru
IOU_THRESHOLD = 0.4 # Bir tahminin konum olarak 'Doğru' kabul edilmesi için gereken minimum örtüşme oranı

# --- YARDIMCI FONKSİYONLAR (Değişiklik Gerekmez) ---
def get_ground_truth(label_path):
    """Poligon etiket dosyasını okur, sınıf ID'sini ve poligonu çevreleyen kutuyu döndürür."""
    try:
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            if not line:
                return None, None
            
            parts = [float(p) for p in line.split()]
            class_id = int(parts[0])
            
            # Poligon noktalarını (x, y) çiftleri olarak ayır
            points = parts[1:]
            x_coords = points[0::2]
            y_coords = points[1::2]
            
            # Poligonu çevreleyen sınırlayıcı kutuyu oluştur (normalized)
            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)
            
            # YOLO formatına çevir: [x_center, y_center, width, height]
            gt_box = [(x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min]
            
            return class_id, gt_box
    except (FileNotFoundError, IndexError, ValueError):
        return None, None

def calculate_iou(boxA, boxB):
    """İki kutu arasındaki IoU oranını hesaplar. Kutular [x_center, y_center, width, height] formatında olmalı."""
    # [x_center, y_center, width, height] -> [x1, y1, x2, y2]
    def to_corners(box):
        x_center, y_center, w, h = box
        return [x_center - w / 2, y_center - h / 2, x_center + w / 2, y_center + h / 2]

    boxA_corners, boxB_corners = to_corners(boxA), to_corners(boxB)
    
    xA = max(boxA_corners[0], boxB_corners[0])
    yA = max(boxA_corners[1], boxB_corners[1])
    xB = min(boxA_corners[2], boxB_corners[2])
    yB = min(boxA_corners[3], boxB_corners[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

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

        gt_class_id, gt_box = get_ground_truth(label_path)
        
        if gt_class_id is None:
            gt_label_str = "Etiket Dosyası Yok/Bozuk"
            results_list.append({'Fotoğraf Adı': image_name, 'Gerçek Etiket': gt_label_str, 'Tahmin Edilen Etiket': 'N/A', 'IoU': 0, 'Sonuç': 'Belirsiz'})
            continue
        
        gt_label_str = CLASS_NAMES[gt_class_id]
        
        results = model(image_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        best_iou = 0
        best_match_label = "Tespit Edilemedi"
        found_correct = False

        if results[0].boxes:
            for box in results[0].boxes:
                pred_class_id = int(box.cls.item())
                pred_box = box.xywhn[0].tolist() # Normalized [x_center, y_center, w, h]
                
                iou = calculate_iou(gt_box, pred_box)
                
                # Hem sınıf doğruysa hem de IoU eşiği geçiyorsa, bu 'Doğru' bir tespittir.
                if pred_class_id == gt_class_id and iou > IOU_THRESHOLD:
                    found_correct = True
                    best_iou = iou
                    best_match_label = CLASS_NAMES[pred_class_id]
                    break # Doğru eşleşme bulundu, diğerlerine bakmaya gerek yok.
                
                # Sınıf yanlış olsa bile en iyi örtüşen kutunun bilgisini sakla
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
        print(f"{image_name}: Gerçek='{gt_label_str}', Tahmin='{best_match_label}', IoU={best_iou:.2f} -> {sonuc}")

    df = pd.DataFrame(results_list)
    df.to_excel(OUTPUT_EXCEL_PATH, index=False)
    print(f"\nİşlem tamamlandı! Sonuçlar '{OUTPUT_EXCEL_PATH}' dosyasına kaydedildi.")

if __name__ == '__main__':
    # Hangi script'i çalıştırmak istediğinize göre ilgili fonksiyonu çağırın.
    main_yolo()