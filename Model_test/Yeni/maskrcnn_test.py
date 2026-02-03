import os
import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import MaskRCNN
from PIL import Image

# --- AYARLAR ---
MODEL_PATH = "C:/Users/ali.donbaloglu/Desktop/Lens/training_plots/Kabin_v4_improved/MaskRCNN_COCO_kabin_v5.pt"
BACKBONE = "resnet101"

DATASET_DIR = "C:/Users/ali.donbaloglu/Desktop/Lens/Model_test/Yeni/test"

CLASS_NAMES = [
    "background",
    "defect",
    "cizik",
    "siyah_nokta"
]

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.25

OUTPUT_EXCEL_PATH = "maskrcnn_iou_sonuclari.xlsx"
OUTPUT_IMAGE_DIR = "maskrcnn_gorsel_sonuclar"
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)


# --------------------------------------------------
# YARDIMCI FONKSİYONLAR
# --------------------------------------------------

def get_ground_truth(label_path):
    gt_list = []
    try:
        with open(label_path) as f:
            for line in f:
                parts = list(map(float, line.split()))
                cls = int(parts[0])
                pts = parts[1:]
                xs, ys = pts[0::2], pts[1::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                gt_box = [
                    (x_min + x_max) / 2,
                    (y_min + y_max) / 2,
                    x_max - x_min,
                    y_max - y_min
                ]
                gt_list.append((cls, gt_box))
    except:
        pass
    return gt_list


def calculate_iou(boxA, boxB):
    def to_xyxy(b):
        x, y, w, h = b
        return [x - w/2, y - h/2, x + w/2, y + h/2]

    A = to_xyxy(boxA)
    B = to_xyxy(boxB)

    xA, yA = max(A[0], B[0]), max(A[1], B[1])
    xB, yB = min(A[2], B[2]), min(A[3], B[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]

    denom = areaA + areaB - inter
    return inter / denom if denom > 0 else 0


def draw_boxes(img, gt_items, boxes, labels, scores):
    h, w = img.shape[:2]

    # Ground Truth (KIRMIZI)
    for cls_id_yolo, box in gt_items:
        cls_id = cls_id_yolo + 2
        label = CLASS_NAMES[cls_id]

        x, y, bw, bh = box
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(img, f"GT: {label}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # Predictions (YESIL)
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box.tolist())
        cls = label.item()
        conf = score.item()

        name = CLASS_NAMES[cls]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(img, f"PRED: {name} {conf:.2f}",
                    (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    return img


def load_model(path, device):
    backbone = resnet_fpn_backbone(BACKBONE, weights=None)
    model = MaskRCNN(backbone, num_classes=len(CLASS_NAMES))

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.to(device).eval()
    return model


# --------------------------------------------------
# ANA ÇALIŞMA
# --------------------------------------------------

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_PATH, device)

    images_dir = os.path.join(DATASET_DIR, "images")
    labels_dir = os.path.join(DATASET_DIR, "labels")

    results = []

    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, img_name.replace(".jpg", ".txt"))

        gt_items = get_ground_truth(label_path)

        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = F.to_tensor(img_pil).to(device)

        output = model([img_tensor])[0]

        mask = output["scores"] > CONFIDENCE_THRESHOLD
        boxes = output["boxes"][mask]
        labels = output["labels"][mask]
        scores = output["scores"][mask]

        w, h = img_pil.size

        for gt_cls_yolo, gt_box in gt_items:
            gt_cls = gt_cls_yolo + 2
            gt_name = CLASS_NAMES[gt_cls]

            best_iou = 0
            best_label = "Tespit Edilemedi"
            found = False

            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = box.tolist()
                pred_box = [
                    ((x1 + x2) / 2) / w,
                    ((y1 + y2) / 2) / h,
                    (x2 - x1) / w,
                    (y2 - y1) / h
                ]

                iou = calculate_iou(gt_box, pred_box)

                if label.item() == gt_cls and iou > IOU_THRESHOLD:
                    found = True
                    best_iou = iou
                    best_label = CLASS_NAMES[label.item()]
                    break

                if iou > best_iou:
                    best_iou = iou
                    best_label = CLASS_NAMES[label.item()]

            results.append({
                "Fotoğraf Adı": img_name,
                "Gerçek Etiket": gt_name,
                "Tahmin Edilen Etiket": best_label,
                "IoU": f"{best_iou:.2f}",
                "Sonuç": "Doğru" if found else "Yanlış"
            })

        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        drawn = draw_boxes(img_cv, gt_items, boxes, labels, scores)
        cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, img_name), drawn)

    pd.DataFrame(results).to_excel(OUTPUT_EXCEL_PATH, index=False)
    print("✅ Mask R-CNN değerlendirme tamamlandı.")


if __name__ == "__main__":
    main()
