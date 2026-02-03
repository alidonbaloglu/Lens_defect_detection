import os
import cv2
import pandas as pd
from ultralytics import YOLO

# --- AYARLAR ---
MODEL_PATH = 'C:/Users/ali.donbaloglu/Desktop/Lens/training_plots/Yeni_modeller/Yolo11_Large/weights/best.pt'
DATASET_DIR = 'C:/Users/ali.donbaloglu/Desktop/Lens/Model_test/Yeni/test'
CLASS_NAMES = ["scratch", "black_dot"]

OUTPUT_EXCEL_PATH = 'yolo_Realtest_sonuclari.xlsx'
OUTPUT_IMAGE_DIR = 'yolo_Realtest_sonuclar'

# Görselleştirme Ayarları
LINE_THICKNESS = 0.6
FONT_SCALE = 0.6 

# YOLO Ayarları
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.25

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

# --- YARDIMCI FONKSİYONLAR ---
def get_ground_truth(label_path):
    gt_list = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = [float(p) for p in line.split()]
                class_id = int(parts[0])
                points = parts[1:]

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

def draw_boxes(image, gt_items, boxes):
    h, w = image.shape[:2]

    # 🔴 Ground Truth
    for cls_id, box in gt_items:
        x, y, bw, bh = box
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)

        OFFSET = 5
        x1 = max(0, x1 - OFFSET)
        y1 = max(0, y1 - OFFSET)
        x2 = min(w, x2 + OFFSET)
        y2 = min(h, y2 + OFFSET)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(image, f"GT: {CLASS_NAMES[cls_id]}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # 🟢 YOLO Tahminleri
    for box in boxes:
        cls_id = int(box.cls.item())
        conf = box.conf.item()
        x, y, bw, bh = box.xywhn[0].tolist()

        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)

        OFFSET = 5
        x1 = max(0, x1 - OFFSET)
        y1 = max(0, y1 - OFFSET)
        x2 = min(w, x2 + OFFSET)
        y2 = min(h, y2 + OFFSET)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(image, f"PRED: {CLASS_NAMES[cls_id]} {conf:.2f}",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    return image

# --- ANA SCRIPT ---
def main_yolo():
    model = YOLO(MODEL_PATH)

    images_dir = os.path.join(DATASET_DIR, 'images')
    labels_dir = os.path.join(DATASET_DIR, 'labels')

    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    results_list = []

    for image_name in image_files:
        image_path = os.path.join(images_dir, image_name)
        label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + '.txt')

        gt_items = get_ground_truth(label_path)
        img = cv2.imread(image_path)

        results = model(image_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
        boxes = results[0].boxes if results[0].boxes else []

        for gt_class_id, gt_box in gt_items:
            gt_label = CLASS_NAMES[gt_class_id]

            best_iou = 0
            best_label = "Tespit Edilemedi"

            for box in boxes:
                pred_id = int(box.cls.item())
                pred_box = box.xywhn[0].tolist()
                iou = calculate_iou(gt_box, pred_box)

                if iou > best_iou:
                    best_iou = iou
                    best_label = CLASS_NAMES[pred_id]

            sonuc = "Doğru" if best_iou >= IOU_THRESHOLD else "Yanlış"

            results_list.append({
                "Fotoğraf Adı": image_name,
                "Gerçek Etiket": gt_label,
                "Tahmin Edilen Etiket": best_label,
                "IoU": f"{best_iou:.2f}",
                "Sonuç": sonuc
            })

        # 📸 Görsel Kaydet
        if img is not None:
            drawn = draw_boxes(img, gt_items, boxes)
            cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, image_name), drawn)

    df = pd.DataFrame(results_list)
    df.to_excel(OUTPUT_EXCEL_PATH, index=False)

    print("✅ Excel ve görseller başarıyla kaydedildi.")

if __name__ == "__main__":
    main_yolo()
