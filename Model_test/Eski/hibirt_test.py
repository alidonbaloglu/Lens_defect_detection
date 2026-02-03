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
# Modelin hatasız yüklenmesi için 3 sınıflı olarak tanımlanması ŞART.
# Bilmediğiniz sınıfa geçici bir isim verin.
MASKRCNN_CLASS_NAMES = ['__background__', 'bilinmeyen_sinif','anomali'] 
MASKRCNN_CONFIDENCE_THRESHOLD = 0.50

# YENİ DEĞİŞKEN: Sadece bu sınıfla ilgileneceğiz.
TARGET_ANOMALI_CLASS_NAME = 'anomali' 

# --- YOLO MODEL AYARLARI ---
YOLO_MODEL_PATH = 'C:/Users/ali.donbaloglu/Desktop/Lens/Modeller/Hibrit_yolo/weights/best.pt'
# Modelin bildiği sınıflar (eğitimde kullanılan sırayla)
MODEL_YOLO_CLASS_NAMES = [
    'artik malzeme', 'cizik', 'enjeksiyon noktasi', 'siyah nokta', 'yirtik'
]
# Test veri setindeki sınıflar (.txt dosyalarındaki sırayla)
DATASET_YOLO_CLASS_NAMES = [
    'cizik', 'enjeksiyon_noktasi', 'kirik', 'siyah_nokta', 'siyahlk', 'yabanci'
]

# YENİ EŞLEŞTİRME SÖZLÜĞÜ (MAPPING DICTIONARY)
# Bu sözlük, hangi model tahmininin hangi veri seti etiketlerine karşılık geldiğini belirtir.
CLASS_MAPPING = {
    # Model Tahmini : [Kabul Edilecek Veri Seti Etiketleri]
    'artik malzeme'     : ['siyahlk'],
    'cizik'             : ['cizik'],
    'enjeksiyon noktasi': ['enjeksiyon_noktasi'],
    'siyah nokta'       : ['siyah_nokta', 'yabanci'],  # Bir-çok eşleşme
    'yirtik'            : ['kirik']
}

YOLO_CONFIDENCE_THRESHOLD = 0.25
YOLO_IMGSZ = 640

# --- VERİ SETİ VE ÇIKTI AYARLARI ---
DATASET_DIR = 'C:/Users/ali.donbaloglu/Desktop/Lens/Model_test/dataset/test'
OUTPUT_EXCEL_PATH = 'pipeline_test_sonuclari_filtrelenmis.xlsx'
CROP_PADDING = 5

# ----------------------------------------------------
# SCRIPT BÖLÜMÜ (Değişiklik yapmanıza gerek yoktur)
# ----------------------------------------------------

def load_maskrcnn_model(weights_path, backbone_name, num_classes, device):
    backbone = resnet_fpn_backbone(backbone_name, weights=None)
    model = MaskRCNN(backbone, num_classes=num_classes)
    try:
        ckpt = torch.load(weights_path, map_location=device)
        state_dict = ckpt.get('model_state', ckpt)
        model.load_state_dict(state_dict, strict=False)
        model.to(device).eval()
        print(f"Mask R-CNN modeli başarıyla yüklendi: {weights_path}")
        return model
    except Exception as e:
        print(f"HATA: Mask R-CNN modeli yüklenemedi. Hata: {e}")
        raise

def load_yolo_model(weights_path):
    try:
        model = YOLO(weights_path)
        print(f"YOLO modeli başarıyla yüklendi: {weights_path}")
        return model
    except Exception as e:
        print(f"HATA: YOLO modeli yüklenemedi. Hata: {e}")
        raise

def get_ground_truth_label(label_path: str, class_names: list) -> str:
    try:
        with open(label_path, 'r') as f:
            line = f.readline()
            if not line: return "Etiket Boş"
            class_id_yolo = int(line.split()[0])
            if class_id_yolo < len(class_names):
                return class_names[class_id_yolo]
            else:
                return "Geçersiz Sınıf ID"
    except FileNotFoundError:
        return "Etiket Dosyası Yok"
    except (IndexError, ValueError):
        return "Etiket Formatı Hatalı"

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")

    try:
        maskrcnn_model = load_maskrcnn_model(
            MASKRCNN_MODEL_PATH, MASKRCNN_BACKBONE, len(MASKRCNN_CLASS_NAMES), device
        )
        yolo_model = load_yolo_model(YOLO_MODEL_PATH)
    except Exception:
        return

    images_dir = os.path.join(DATASET_DIR, 'images')
    labels_dir = os.path.join(DATASET_DIR, 'labels')
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"{len(image_files)} adet fotoğraf test için bulundu.")
    
    debug_crops_dir = 'debug_crops'
    os.makedirs(debug_crops_dir, exist_ok=False)
    print(f"Debug crop'ları '{debug_crops_dir}' klasörüne kaydedilecek.")
    
    results_list = []

    try:
        target_class_id = MASKRCNN_CLASS_NAMES.index(TARGET_ANOMALI_CLASS_NAME)
    except ValueError:
        print(f"HATA: Hedef sınıf '{TARGET_ANOMALI_CLASS_NAME}' listede bulunamadı. Script durduruluyor.")
        return

    for image_name in image_files:
        image_path = os.path.join(images_dir, image_name)
        label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + '.txt')
        
        ground_truth_label = get_ground_truth_label(label_path, DATASET_YOLO_CLASS_NAMES)
        original_pil_image = Image.open(image_path).convert("RGB")
        img_tensor = F.to_tensor(original_pil_image).to(device)

        outputs = maskrcnn_model([img_tensor])
        
        maskrcnn_result = "Anomali Tespit Edilemedi"
        yolo_result = "N/A"

        scores = outputs[0].get("scores", torch.empty(0))
        labels = outputs[0].get("labels", torch.empty(0))
        boxes = outputs[0].get("boxes", torch.empty(0, 4))
        
        high_conf_indices = torch.where(scores >= MASKRCNN_CONFIDENCE_THRESHOLD)[0]

        if len(high_conf_indices) > 0:
            target_class_indices = torch.where(labels[high_conf_indices] == target_class_id)[0]

            if len(target_class_indices) > 0:
                maskrcnn_result = f"'{TARGET_ANOMALI_CLASS_NAME}' Tespit Edildi"
                
                anomali_scores = scores[high_conf_indices][target_class_indices]
                best_anomali_local_idx = torch.argmax(anomali_scores)
                
                best_anomali_global_idx = high_conf_indices[target_class_indices][best_anomali_local_idx]
                
                box = boxes[best_anomali_global_idx].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)

                img_w, img_h = original_pil_image.size
                x1 = max(0, x1 - CROP_PADDING)
                y1 = max(0, y1 - CROP_PADDING)
                x2 = min(img_w, x2 + CROP_PADDING)
                y2 = min(img_h, y2 + CROP_PADDING)
                
                cropped_image = original_pil_image.crop((x1, y1, x2, y2))
                cropped_image.save(os.path.join(debug_crops_dir, f"crop_{image_name}"))

                yolo_outputs = yolo_model.predict(source=cropped_image, conf=YOLO_CONFIDENCE_THRESHOLD, imgsz=YOLO_IMGSZ, augment=True, verbose=False)

                if yolo_outputs and yolo_outputs[0].boxes and len(yolo_outputs[0].boxes) > 0:
                    yolo_best_idx = torch.argmax(yolo_outputs[0].boxes.conf).item()
                    class_id = int(yolo_outputs[0].boxes.cls[yolo_best_idx].item())
                    yolo_result = MODEL_YOLO_CLASS_NAMES[class_id]
                else:
                    yolo_outputs_pass2 = yolo_model.predict(source=cropped_image, conf=max(0.05, YOLO_CONFIDENCE_THRESHOLD * 0.5), imgsz=max(1280, YOLO_IMGSZ), augment=True, verbose=False)
                    if yolo_outputs_pass2 and yolo_outputs_pass2[0].boxes and len(yolo_outputs_pass2[0].boxes) > 0:
                        yolo_best_idx = torch.argmax(yolo_outputs_pass2[0].boxes.conf).item()
                        class_id = int(yolo_outputs_pass2[0].boxes.cls[yolo_best_idx].item())
                        yolo_result = MODEL_YOLO_CLASS_NAMES[class_id]
                    else:
                        yolo_result = "Sınıflandırılamadı (Tespit Yok)"
            else:
                maskrcnn_result = "Diğer Sınıf Tespit Edildi (İşlenmedi)"
        
        # --- DEĞİŞİKLİK BURADA: Yeni Eşleştirme Mantığı ---
        # Varsayılan olarak sonucu 'Yanlış' kabul et
        is_correct = "Yanlış"
        
        # Eğer model bir sonuç ürettiyse ve bu sonuç mapping sözlüğünde varsa
        if yolo_result in CLASS_MAPPING:
            # Modelin tahminine izin verilen etiket listesini al
            allowed_ground_truths = CLASS_MAPPING[yolo_result]
            # Eğer gerçek etiket, bu izin verilen etiketlerden biriyse, sonucu 'Doğru' yap
            if ground_truth_label in allowed_ground_truths:
                is_correct = "Doğru"

        # Sonucu belirsiz yapacak istisnai durumları kontrol et
        if "Tespit Edilemedi" in maskrcnn_result or "Sınıflandırılamadı" in yolo_result or "N/A" in yolo_result or "İşlenmedi" in maskrcnn_result or "Etiket" in ground_truth_label:
            is_correct = "Belirsiz"

        results_list.append({
            'Fotoğraf Adı': image_name, 'Gerçek Etiket': ground_truth_label, 'MaskRCNN Sonucu': maskrcnn_result,
            'YOLO Sonucu': yolo_result, 'Nihai Sonuç': is_correct
        })
        print(f"{image_name}: Gerçek='{ground_truth_label}', M-RCNN='{maskrcnn_result}', YOLO='{yolo_result}' -> {is_correct}")

    df = pd.DataFrame(results_list)
    df.to_excel(OUTPUT_EXCEL_PATH, index=False, engine='openpyxl')
    print(f"\nİşlem tamamlandı! Sonuçlar '{OUTPUT_EXCEL_PATH}' dosyasına kaydedildi.")


if __name__ == '__main__':
    main()