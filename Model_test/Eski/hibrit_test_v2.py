import os
import pandas as pd
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import MaskRCNN
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ----------------------------------------
# AYARLAR (Lütfen bu bölümü dikkatlice düzenleyin)
# ----------------------------------------

# --- MASK R-CNN MODEL AYARLARI ---
MASKRCNN_MODEL_PATH = "C:/Users/ali.donbaloglu/Desktop/Lens/Modeller/MaskRCNN_COCO_v4.pt"
MASKRCNN_BACKBONE = "resnet101"
# Modelinizin hatasız yüklenmesi için eğitildiği 8 sınıflı tam liste
MASKRCNN_CLASS_NAMES = ['__background__', 'bilinmeyen_sinif','anomali'] 
# Mask R-CNN'in bulmasını istediğimiz genel sınıfın adı
TARGET_ANOMALI_CLASS_NAME = 'anomali' # veya 'anomali' olabilir, modelinize göre değiştirin
MASKRCNN_CONFIDENCE_THRESHOLD = 0.5

# --- YOLO MODEL AYARLARI ---
YOLO_MODEL_PATH = 'C:/Users/ali.donbaloglu/Desktop/Lens/Modeller/Hibrit_yolo/weights/best.pt'
# Modelin bildiği sınıflar (eğitimde kullanılan sırayla)
MODEL_YOLO_CLASS_NAMES = [
    'artik malzeme', 'cizik', 'enjeksiyon noktasi', 'siyah nokta', 'yirtik'
]
YOLO_CONFIDENCE_THRESHOLD = 0.25
YOLO_IMGSZ = 640

# --- VERİ SETİ ve EŞLEŞTİRME AYARLARI ---
DATASET_DIR = 'C:/Users/ali.donbaloglu/Desktop/Lens/Model_test/dataset/test'
# Test veri setindeki sınıflar (.txt dosyalarındaki ID sırasıyla)
DATASET_YOLO_CLASS_NAMES = [
    'cizik', 'enjeksiyon_noktasi', 'kirik', 'siyah_nokta', 'siyahlk', 'yabanci'
]
# Model tahmini ile veri seti etiketleri arasındaki özel eşleştirme kuralları
CLASS_MAPPING = {
    # Model Tahmini : [Kabul Edilecek Veri Seti Etiketleri]
    'artik malzeme'     : ['siyahlk'],
    'cizik'             : ['cizik'],
    'enjeksiyon noktasi': ['enjeksiyon_noktasi'],
    'siyah nokta'       : ['siyah_nokta', 'yabanci'],  # Bir-çok eşleşme
    'yirtik'            : ['kirik']
}

# --- DEĞERLENDİRME AYARLARI ---
# Mask R-CNN'in bulduğu kutunun, gerçek etiketin kutusuyla en az bu oranda
# örtüşmesi gerekir ki doğru bölgeyi bulduğu kabul edilsin.
STAGE_1_IOU_THRESHOLD = 0.4 
CROP_PADDING = 5
OUTPUT_EXCEL_PATH = 'final_hibrit_test_sonuclari.xlsx'

# ----------------------------------------------------
# SCRIPT BÖLÜMÜ (Değişiklik yapmanıza gerek yoktur)
# ----------------------------------------------------

# --- YARDIMCI FONKSİYONLAR ---
def get_ground_truth(label_path):
    """
    Poligon etiket dosyasını okur. Hata durumlarında 2 değer (None, None) döndürür.
    """
    try:
        with open(label_path, 'r') as f:
            line = f.readline().strip()
            # Hata durumunda 2 değer döndür
            if not line: return None, None
            
            parts = [float(p) for p in line.split()]
            class_id = int(parts[0])
            points = parts[1:]
            x_coords, y_coords = points[0::2], points[1::2]
            
            # Poligonun varlığını kontrol et
            if not x_coords or not y_coords: return None, None

            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)
            gt_box = [(x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min]
            
            # Başarılı durumda 2 değer döndür
            return class_id, gt_box
            
    except (FileNotFoundError, IndexError, ValueError):
        # Hata durumunda 2 değer döndür
        return None, None

def calculate_iou(boxA, boxB):
    def to_corners(box):
        x_center, y_center, w, h = box
        return [x_center - w / 2, y_center - h / 2, x_center + w / 2, y_center + h / 2]
    boxA_corners, boxB_corners = to_corners(boxA), to_corners(boxB)
    xA, yA = max(boxA_corners[0], boxB_corners[0]), max(boxA_corners[1], boxB_corners[1])
    xB, yB = min(boxA_corners[2], boxB_corners[2]), min(boxA_corners[3], boxB_corners[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def load_maskrcnn_model(weights_path, backbone_name, num_classes, device):
    backbone = resnet_fpn_backbone(backbone_name, weights=None)
    model = MaskRCNN(backbone, num_classes=num_classes)
    try:
        ckpt = torch.load(weights_path, map_location=device)
        state_dict = ckpt.get('model_state', ckpt)
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        return model
    except Exception as e: raise e

def load_yolo_model(weights_path):
    try: return YOLO(weights_path)
    except Exception as e: raise e

# --- ANA SCRIPT ---
@torch.no_grad()
def main_hibrit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")

    try:
        maskrcnn_model = load_maskrcnn_model(MASKRCNN_MODEL_PATH, MASKRCNN_BACKBONE, len(MASKRCNN_CLASS_NAMES), device)
        yolo_model = load_yolo_model(YOLO_MODEL_PATH)
        print("Tüm modeller başarıyla yüklendi.")
    except Exception as e:
        print(f"HATA: Modeller yüklenemedi. Hata: {e}")
        return

    images_dir, labels_dir = os.path.join(DATASET_DIR, 'images'), os.path.join(DATASET_DIR, 'labels')
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', 'jpeg'))]
    results_list = []

    try:
        target_class_id = MASKRCNN_CLASS_NAMES.index(TARGET_ANOMALI_CLASS_NAME)
    except ValueError:
        print(f"HATA: Hedef sınıf '{TARGET_ANOMALI_CLASS_NAME}' listede bulunamadı."); return

    for image_name in image_files:
        image_path = os.path.join(images_dir, image_name)
        label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + '.txt')

        gt_class_id_yolo, gt_box = get_ground_truth(label_path)
        
        if gt_class_id_yolo is None:
            results_list.append({'Fotoğraf': image_name, 'Gerçek Etiket': 'Etiket Yok/Bozuk', 'Aşama 1 Sonucu': 'N/A', 'Aşama 2 (YOLO) Sonucu': 'N/A', 'Nihai Sonuç': 'Belirsiz'})
            continue
        
        ground_truth_label = DATASET_YOLO_CLASS_NAMES[gt_class_id_yolo]
        original_pil_image = Image.open(image_path).convert("RGB")
        img_tensor = F.to_tensor(original_pil_image).to(device)

        outputs = maskrcnn_model([img_tensor])
        
        stage1_result = "Anomali Tespit Edilemedi"
        yolo_result = "N/A"
        is_correct = "Yanlış"
        
        scores = outputs[0]['scores']
        high_conf_indices = scores > MASKRCNN_CONFIDENCE_THRESHOLD

        best_roi_box = None
        max_iou_with_gt = 0

        if high_conf_indices.any():
            boxes = outputs[0]['boxes'][high_conf_indices]
            img_w, img_h = original_pil_image.size
            for box in boxes:
                x1, y1, x2, y2 = box.tolist()
                pred_box = [((x1+x2)/2)/img_w, ((y1+y2)/2)/img_h, (x2-x1)/img_w, (y2-y1)/img_h]
                iou = calculate_iou(gt_box, pred_box)
                if iou > max_iou_with_gt:
                    max_iou_with_gt = iou
                    best_roi_box = box.tolist()
        
        if max_iou_with_gt > STAGE_1_IOU_THRESHOLD:
            stage1_result = f"Doğru Bölge Bulundu (IoU: {max_iou_with_gt:.2f})"
            x1, y1, x2, y2 = map(int, best_roi_box)
            x1, y1 = max(0, x1 - CROP_PADDING), max(0, y1 - CROP_PADDING)
            x2, y2 = min(img_w, x2 + CROP_PADDING), min(img_h, y2 + CROP_PADDING)
            
            if x2 > x1 and y2 > y1:
                cropped_image = original_pil_image.crop((x1, y1, x2, y2))
                yolo_outputs = yolo_model.predict(source=cropped_image, conf=YOLO_CONFIDENCE_THRESHOLD, imgsz=YOLO_IMGSZ, augment=True, verbose=False)
                if yolo_outputs and yolo_outputs[0].boxes and len(yolo_outputs[0].boxes) > 0:
                    class_id = int(yolo_outputs[0].boxes.cls[torch.argmax(yolo_outputs[0].boxes.conf)].item())
                    yolo_result = MODEL_YOLO_CLASS_NAMES[class_id]
                else:
                    yolo_result = "Sınıflandırılamadı"
            else:
                 yolo_result = "Kırpma Alanı Geçersiz"

        elif max_iou_with_gt > 0:
            stage1_result = f"Yanlış Bölge Bulundu (IoU: {max_iou_with_gt:.2f})"
        
        if yolo_result in CLASS_MAPPING and ground_truth_label in CLASS_MAPPING[yolo_result]:
            is_correct = "Doğru"
        
        if stage1_result == "Anomali Tespit Edilemedi" or "N/A" in yolo_result or "Sınıflandırılamadı" in yolo_result or "Geçersiz" in yolo_result:
            is_correct = "Belirsiz"

        results_list.append({
            'Fotoğraf': image_name, 'Gerçek Etiket': ground_truth_label, 'Aşama 1 Sonucu': stage1_result,
            'Aşama 2 (YOLO) Sonucu': yolo_result, 'Nihai Sonuç': is_correct
        })
        print(f"{image_name}: Gerçek='{ground_truth_label}', Aşama1='{stage1_result}', YOLO='{yolo_result}' -> {is_correct}")

    df = pd.DataFrame(results_list)
    df.to_excel(OUTPUT_EXCEL_PATH, index=False)
    print(f"\nİşlem tamamlandı! Sonuçlar '{OUTPUT_EXCEL_PATH}' dosyasına kaydedildi.")

if __name__ == '__main__':
    main_hibrit()