import os
import pandas as pd
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import MaskRCNN
from PIL import Image

# --- AYARLAR ---
MASKRCNN_MODEL_PATH = "C:/Users/ali.donbaloglu/Desktop/Lens/training_plots/Merged_v3/MaskRCNN_COCO_merged_v3.pt"
MASKRCNN_BACKBONE = "resnet101"
DATASET_DIR = 'C:/Users/ali.donbaloglu/Desktop/Lens/Model_test/Eski/dataset/test'
CLASS_NAMES = [
    "background",
    "defect",
    "cizik",
    "siyah_nokta"
]
OUTPUT_EXCEL_PATH = 'maskrcnn_deneme.xlsx'

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.25


# --- YARDIMCI FONKSİYONLAR ---
def get_ground_truth(label_path):
    """Birden fazla ground truth etiketi döndürür: [(class_id, gt_box), ...]"""
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
                gt_box = [(x_min + x_max) / 2, (y_min + y_max) / 2,
                          x_max - x_min, y_max - y_min]
                gt_list.append((class_id, gt_box))
        return gt_list
    except (FileNotFoundError, ValueError):
        return []


def calculate_iou(boxA, boxB):
    """İki kutu arasındaki IoU oranını hesaplar. [x_center, y_center, width, height] formatında olmalı."""
    def to_corners(box):
        x_center, y_center, w, h = box
        return [x_center - w / 2, y_center - h / 2,
                x_center + w / 2, y_center + h / 2]

    boxA_corners, boxB_corners = to_corners(boxA), to_corners(boxB)
    xA = max(boxA_corners[0], boxB_corners[0])
    yA = max(boxA_corners[1], boxB_corners[1])
    xB = min(boxA_corners[2], boxB_corners[2])
    yB = min(boxA_corners[3], boxB_corners[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    if boxAArea + boxBArea - interArea == 0:
        return 0
    return interArea / float(boxAArea + boxBArea - interArea)


def load_maskrcnn_model(weights_path, backbone_name, num_classes, device):
    backbone = resnet_fpn_backbone(backbone_name, weights=None)
    model = MaskRCNN(backbone, num_classes=num_classes)
    try:
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        state_dict = ckpt.get('model_state', ckpt)
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        return model
    except Exception as e:
        print(f"HATA: Mask R-CNN modeli yüklenemedi. Hata: {e}")
        raise


# --- ANA SCRIPT ---
@torch.no_grad()
def main_maskrcnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_maskrcnn_model(MASKRCNN_MODEL_PATH, MASKRCNN_BACKBONE, len(CLASS_NAMES), device)

    images_dir = os.path.join(DATASET_DIR, 'images')
    labels_dir = os.path.join(DATASET_DIR, 'labels')
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    results_list = []

    for image_name in image_files:
        image_path = os.path.join(images_dir, image_name)
        label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + '.txt')

        gt_items = get_ground_truth(label_path)

        if not gt_items:
            results_list.append({
                'Fotoğraf Adı': image_name,
                'Gerçek Etiket': 'Etiket Dosyası Yok/Bozuk',
                'Tahmin Edilen Etiket': 'N/A',
                'IoU': 0,
                'Sonuç': 'Belirsiz'
            })
            continue

        img = Image.open(image_path).convert("RGB")
        img_tensor = F.to_tensor(img).to(device)
        outputs = model([img_tensor])

        # Mask R-CNN tahminlerini filtrele
        scores = outputs[0]['scores']
        keep_indices = scores > CONFIDENCE_THRESHOLD

        if keep_indices.any():
            boxes = outputs[0]['boxes'][keep_indices]
            labels = outputs[0]['labels'][keep_indices]
        else:
            boxes, labels = [], []

        img_w, img_h = img.size

        # Her ground truth objesi için kontrol yap
        for gt_class_id_yolo, gt_box in gt_items:
            gt_class_id_mrcnn = gt_class_id_yolo + 2  # 0 tabanlı -> 2 ofsetli
            gt_label_str = CLASS_NAMES[gt_class_id_mrcnn]

            best_iou = 0
            best_match_label = "Tespit Edilemedi"
            found_correct = False

            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box.tolist()
                pred_box = [((x1 + x2) / 2) / img_w, ((y1 + y2) / 2) / img_h,
                            (x2 - x1) / img_w, (y2 - y1) / img_h]
                pred_class_id = label.item()

                iou = calculate_iou(gt_box, pred_box)

                if pred_class_id == gt_class_id_mrcnn and iou > IOU_THRESHOLD:
                    found_correct = True
                    best_iou = iou
                    best_match_label = CLASS_NAMES[pred_class_id]
                    break

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
    main_maskrcnn()
