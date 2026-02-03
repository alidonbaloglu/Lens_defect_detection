import os
import pandas as pd
from ultralytics import YOLO
import torch

# ------------------- AYARLAR BÖLÜMÜ -------------------
# Lütfen bu bölümdeki yolları ve ayarları kendi projenize göre güncelleyin.

# 1. Eğittiğiniz modelin .pt dosyasının yolu
MODEL_PATH = 'C:/Users/ali.donbaloglu/Desktop/Lens/Modeller/Lens_v2_1400/weights/best.pt'  # Örnek: 'runs/train/exp/weights/best.pt'

# 2. Test veri setinizin olduğu ana klasörün yolu
DATASET_DIR = 'C:/Users/ali.donbaloglu/Desktop/Lens/Model_test/dataset/test'  # Bu klasör içinde 'images' ve 'labels' olmalı

# 3. Test fotoğraflarının bulunduğu klasörün yolu
IMAGES_DIR = os.path.join(DATASET_DIR, 'images')

# 4. Test etiketlerinin (ground truth) bulunduğu klasörün yolu
LABELS_DIR = os.path.join(DATASET_DIR, 'labels')

# 5. Modelinizin tanıdığı sınıf isimleri (data.yaml dosyanızdaki sırayla)
CLASS_NAMES = ['cizik', 'enjeksiyon_noktasi', 'kirik', 'siyah_nokta', 'siyahlk', 'yabanci'] # Kendi sınıf isimlerinizle değiştirin

# 6. Sonuçların kaydedileceği Excel dosyasının adı
OUTPUT_EXCEL_PATH = 'test_sonuclari.xlsx'

# 7. Modelin tahmin yaparken kullanacağı güven skoru eşiği (confidence threshold)
CONFIDENCE_THRESHOLD = 0.25

# ------------------- SCRIPT BÖLÜMÜ -------------------
# Bu bölümü değiştirmenize gerek yoktur.

def get_ground_truth_label(label_path, class_names):
    """Verilen yoldaki YOLO etiket dosyasını okur ve sınıf adını döndürür."""
    try:
        with open(label_path, 'r') as f:
            line = f.readline()
            if line:
                class_id = int(line.split()[0])
                return class_names[class_id]
            return "Etiket Boş"
    except FileNotFoundError:
        return "Etiket Dosyası Yok"
    except (IndexError, ValueError):
        return "Etiket Formatı Hatalı"

def main():
    """Ana test fonksiyonu."""
    # Modeli yükle
    try:
        model = YOLO(MODEL_PATH)
        print(f"'{MODEL_PATH}' modeli başarıyla yüklendi.")
    except Exception as e:
        print(f"HATA: Model yüklenemedi! Yolun doğru olduğundan emin olun. Hata: {e}")
        return

    # Test edilecek fotoğrafların listesini al
    try:
        image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"HATA: '{IMAGES_DIR}' klasöründe hiç resim dosyası bulunamadı.")
            return
        print(f"{len(image_files)} adet fotoğraf test için bulundu.")
    except FileNotFoundError:
        print(f"HATA: Resim klasörü bulunamadı: '{IMAGES_DIR}'")
        return

    results_list = []

    # Her bir fotoğraf için işlem yap
    for image_name in image_files:
        image_path = os.path.join(IMAGES_DIR, image_name)
        
        # İlgili etiket dosyasının yolunu oluştur
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(LABELS_DIR, label_name)

        # 1. Gerçek etiket değerini dosyadan oku
        ground_truth_label = get_ground_truth_label(label_path, CLASS_NAMES)

        # 2. Model ile tahmin yap
        results = model(image_path, conf=CONFIDENCE_THRESHOLD, verbose=False) # verbose=False çıktıyı temizler
        
        predicted_label = "Tespit Edilemedi"
        
        # Tahmin sonucunu işle
        if results[0].boxes:
            # En yüksek güven skoruna sahip tespiti al
            best_prediction = results[0].boxes[torch.argmax(results[0].boxes.conf)]
            pred_class_id = int(best_prediction.cls.item())
            predicted_label = CLASS_NAMES[pred_class_id]
        
        # 3. Sonuçları karşılaştır
        is_correct = "True" if ground_truth_label == predicted_label else "False"
        
        # Eğer etiket dosyası yoksa veya tespit yoksa sonucu "Belirsiz" olarak ayarla
        if ground_truth_label in ["Etiket Dosyası Yok", "Etiket Boş", "Etiket Formatı Hatalı"] or predicted_label == "Tespit Edilemedi":
            is_correct = "Belirsiz"

        # Sonuçları listeye ekle
        results_list.append({
            'Fotoğraf Adı': image_name,
            'Veri Setindeki Etiket Değeri': ground_truth_label,
            'Modelin Tahmin Ettiği Etiket Değeri': predicted_label,
            'Sonuç': is_correct
        })
        
        print(f"{image_name} işlendi: Gerçek Etiket='{ground_truth_label}', Tahmin='{predicted_label}' -> {is_correct}")

    # 4. Sonuçları Excel dosyasına kaydet
    if results_list:
        df = pd.DataFrame(results_list)
        df.to_excel(OUTPUT_EXCEL_PATH, index=False)
        print(f"\nİşlem tamamlandı! Sonuçlar '{OUTPUT_EXCEL_PATH}' dosyasına kaydedildi.")
    else:
        print("\nHiçbir sonuç kaydedilmedi.")

if __name__ == '__main__':
    main()