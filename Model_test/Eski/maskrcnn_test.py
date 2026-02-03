import os
import pandas as pd
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import MaskRCNN
from PIL import Image
import numpy as np

# ----------------------------
# AYARLAR (Lütfen bu bölümü düzenleyin)
# ----------------------------

# 1. Test veri setinizin olduğu ana klasörün yolu
DATASET_DIR = 'C:/Users/ali.donbaloglu/Desktop/Lens/Model_test/dataset/test'  # Bu klasör içinde 'images' ve 'labels' olmalı

# 2. Modelinizin .pt veya .pth uzantılı ağırlık dosyasının tam yolu
#    Bu yol, video script'inizdeki get_weights_path() fonksiyonunun döndürdüğü yol ile aynı olmalıdır.
MODEL_WEIGHTS_PATH = "C:/Users/ali.donbaloglu/Desktop/Lens/Modeller/MaskRCNN_COCO_sinif_v1.pt"

# 3. Modelinizin omurga (backbone) mimarisi
#    "resnet50" veya "resnet101" olarak, eğittiğiniz modele uygun şekilde belirtin.
BACKBONE_NAME = "resnet101"

# 4. Sınıf isimleriniz
#    !!! ÖNEMLİ: Video script'inizdeki CLASS_NAMES listesi ile BİREBİR AYNI olmalıdır.
CLASS_NAMES = [
    "background",
    "defect",
    "cizik",
    "enjeksiyon_noktasi",
    "kirik",
    "siyah_nokta",
    "siyahlk",
    "yabanci",
    
]

# 5. Sonuçların kaydedileceği Excel dosyasının adı
OUTPUT_EXCEL_PATH = 'maskrcnn_test_sonuclari.xlsx'

# 6. Modelin tahmin yaparken kullanacağı güven skoru eşiği (confidence threshold)
CONFIDENCE_THRESHOLD = 0.25

# 7. Kullanılacak cihaz ("cuda" veya "cpu")
DEVICE = "cuda"

# ----------------------------------------------------
# SCRIPT BÖLÜMÜ (Değişiklik yapmanıza gerek yoktur)
# ----------------------------------------------------

# Video script'inizden alınan model oluşturma fonksiyonları
def build_model(num_classes: int, backbone_name: str) -> torchvision.models.detection.MaskRCNN:
    """FPN omurgasını oluşturur ve Mask R-CNN modelini kurar."""
    backbone = resnet_fpn_backbone(backbone_name, weights=None)
    model = MaskRCNN(backbone, num_classes=num_classes)
    return model

def load_model(weights_path: str, device: torch.device) -> torchvision.models.detection.MaskRCNN:
    """Verilen yoldan modeli ve eğitilmiş ağırlıkları yükler."""
    num_classes = len(CLASS_NAMES)
    model = build_model(num_classes, BACKBONE_NAME)
    try:
        ckpt = torch.load(weights_path, map_location=device)
        # Checkpoint'in 'model_state' anahtarı içerip içermediğini kontrol et
        if 'model_state' in ckpt:
            model.load_state_dict(ckpt["model_state"], strict=False)
        else:
            # Eğer 'model_state' yoksa, checkpoint'in kendisi state_dict'tir
            model.load_state_dict(ckpt, strict=False)
        model.to(device).eval()
        print(f"'{weights_path}' model ağırlıkları başarıyla yüklendi.")
        return model
    except FileNotFoundError:
        print(f"HATA: Model ağırlık dosyası bulunamadı: {weights_path}")
        raise
    except Exception as e:
        print(f"HATA: Model yüklenirken bir sorun oluştu. Hata: {e}")
        raise

def get_ground_truth_label(label_path: str, class_names: list) -> str:
    """Verilen yoldaki YOLO etiket dosyasını okur ve sınıf adını döndürür."""
    try:
        with open(label_path, 'r') as f:
            line = f.readline()
            if not line:
                return "Etiket Boş"
            
            class_id_yolo = int(line.split()[0])
            # YOLO class_id (0'dan başlar) -> Gerçek sınıf listesi (background hariç)
            # CLASS_NAMES listesi background içerdiği için index +1 olmalı
            # defect -> 0 (YOLO), 1 (CLASS_NAMES)
            # cizik -> 1 (YOLO), 2 (CLASS_NAMES) ...
            true_class_index = class_id_yolo + 2
            if true_class_index < len(class_names):
                return class_names[true_class_index]
            else:
                return "Geçersiz Sınıf ID"
                
    except FileNotFoundError:
        return "Etiket Dosyası Yok"
    except (IndexError, ValueError):
        return "Etiket Formatı Hatalı"

@torch.no_grad()
def main():
    """Ana test fonksiyonu."""
    # Test veri setinin yollarını belirle
    images_dir = os.path.join(DATASET_DIR, 'images')
    labels_dir = os.path.join(DATASET_DIR, 'labels')

    # Cihazı belirle
    device = torch.device(DEVICE if torch.cuda.is_available() and DEVICE == "cuda" else "cpu")
    print(f"Kullanılan cihaz: {device}")

    # Modeli yükle
    try:
        model = load_model(MODEL_WEIGHTS_PATH, device)
    except Exception:
        return  # Model yüklenemezse programdan çık

    # Test edilecek fotoğrafların listesini al
    try:
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"HATA: '{images_dir}' klasöründe hiç resim dosyası bulunamadı.")
            return
        print(f"{len(image_files)} adet fotoğraf test için bulundu.")
    except FileNotFoundError:
        print(f"HATA: Resim klasörü bulunamadı: '{images_dir}'")
        return

    results_list = []

    # Her bir fotoğraf için işlem yap
    for image_name in image_files:
        image_path = os.path.join(images_dir, image_name)
        
        # İlgili etiket dosyasının yolunu oluştur
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)

        # 1. Gerçek etiket değerini (ground truth) dosyadan oku
        ground_truth_label = get_ground_truth_label(label_path, CLASS_NAMES)

        # 2. Fotoğrafı aç, tensöre çevir ve model ile tahmin yap
        img = Image.open(image_path).convert("RGB")
        img_tensor = F.to_tensor(img).to(device)
        
        outputs = model([img_tensor])
        out = outputs[0]

        # 3. Tahmin sonucunu işle
        predicted_label = "Tespit Edilemedi"
        scores = out.get("scores", torch.empty(0)).cpu().numpy()
        labels = out.get("labels", torch.empty(0)).cpu().numpy()
        
        if len(scores) > 0:
            # En yüksek güven skoruna sahip tespiti bul
            best_score_index = np.argmax(scores)
            best_score = scores[best_score_index]

            if best_score >= CONFIDENCE_THRESHOLD:
                pred_class_id = labels[best_score_index]
                if pred_class_id < len(CLASS_NAMES):
                    predicted_label = CLASS_NAMES[pred_class_id]
                else:
                    predicted_label = f"Bilinmeyen Sınıf ID ({pred_class_id})"

        # 4. Sonuçları karşılaştır
        is_correct = "Doğru" if ground_truth_label == predicted_label else "Yanlış"
        
        if ground_truth_label in ["Etiket Dosyası Yok", "Etiket Boş", "Etiket Formatı Hatalı"] or predicted_label == "Tespit Edilemedi":
            is_correct = "Belirsiz"

        # 5. Sonuçları listeye ekle
        results_list.append({
            'Fotoğraf Adı': image_name,
            'Veri Setindeki Etiket Değeri': ground_truth_label,
            'Modelin Tahmin Ettiği Etiket Değeri': predicted_label,
            'Sonuç': is_correct
        })
        
        print(f"{image_name} işlendi: Gerçek Etiket='{ground_truth_label}', Tahmin='{predicted_label}' -> {is_correct}")

    # 6. Sonuçları Excel dosyasına kaydet
    if results_list:
        df = pd.DataFrame(results_list)
        df.to_excel(OUTPUT_EXCEL_PATH, index=False, engine='openpyxl')
        print(f"\nİşlem tamamlandı! Sonuçlar '{OUTPUT_EXCEL_PATH}' dosyasına kaydedildi.")
    else:
        print("\nHiçbir sonuç kaydedilmedi.")

if __name__ == '__main__':
    main()