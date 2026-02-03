# 🔍 Lens Hata Tespit Projesi

Lens üretiminde yüzey hatalarının (çizik, siyah nokta vb.) otomatik tespiti için geliştirilen derin öğrenme tabanlı görüntü işleme projesidir.

## 📋 Özellikler

- **RT-DETR Ensemble Model**: Çizik ve siyah nokta tespiti için özelleştirilmiş iki modelin ağırlıklı ensemble birleşimi
- **Çoklu Kamera Desteği**: 4 kamera ile eş zamanlı görüntü yakalama
- **Gerçek Zamanlı Analiz**: PyQt5 tabanlı kullanıcı arayüzü ile canlı tespit
- **Otomatik Veri Toplama**: Focus taraması ile optimum görüntü yakalama
- **YOLO & Mask R-CNN**: Alternatif model eğitimi ve değerlendirme

## 🛠️ Kurulum

### 1. Gereksinimler
- Windows 10/11 64-bit
- Python 3.10 veya 3.11
- NVIDIA GPU + CUDA (önerilen)

### 2. Sanal Ortam
```powershell
cd "C:\Users\ali.donbaloglu\Desktop\Lens"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. PyTorch Kurulumu
CUDA sürümünüze uygun PyTorch kurun:
```powershell
# CUDA 12.6 için
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# CPU için
pip install torch torchvision
```

### 4. Bağımlılıklar
```powershell
pip install -r requirements.txt
```

## 📁 Proje Yapısı

```
Lens/
├── kamera/                          # Kamera uygulamaları
│   ├── data_collector_qt_ensemble.py   # Ana veri toplama uygulaması (RT-DETR Ensemble)
│   ├── camera_ayar.py                  # Kamera ayarları
│   └── results/                        # Toplanan veriler
│
├── RT_Detr_Ensemble/               # RT-DETR Ensemble modeli
│   ├── model/
│   │   ├── best_cizik.pt              # Çizik tespit modeli
│   │   └── best_siyahnokta.pt         # Siyah nokta tespit modeli
│   └── ensemble_test.py               # Ensemble test scripti
│
├── Modeller/                       # Eğitilmiş model dosyaları
├── datasetler/                     # Eğitim veri setleri (COCO/YOLO format)
├── results/                        # Analiz sonuçları
├── training_plots/                 # Eğitim grafikleri
│
├── rtdetr_train_v3.py              # RT-DETR model eğitimi
├── yolo11_train.py                 # YOLO model eğitimi
├── mask_rcnn_coco.py               # Mask R-CNN eğitimi
├── mask_rcnn_infer.py              # Mask R-CNN çıkarım
│
├── requirements.txt                # Python bağımlılıkları
└── README.md                       # Bu dosya
```

## 🚀 Kullanım

### Veri Toplama Uygulaması (RT-DETR Ensemble)
```powershell
python kamera/data_collector_qt_ensemble.py
```

**Özellikler:**
- 4 kamera canlı görüntüsü (üst satır)
- Gerçek zamanlı hata tespiti (alt satır)
- Sol/Sağ parça seçimi ile kamera ayarları
- Focus taraması ile otomatik görüntü yakalama
- CSV formatında tespit sonuçları

**Kısayollar:**
- `Kaydet`: Tüm kameralardan görüntü yakala ve analiz et
- `Sonraki Part`: Yeni parça klasörü oluştur
- `Kayıtları Analiz Et`: Kaydedilmiş görüntüleri toplu analiz et

### Model Eğitimi

**RT-DETR Eğitimi:**
```powershell
python rtdetr_train_v3.py
```

**YOLO Eğitimi:**
```powershell
python yolo11_train.py
```

**Mask R-CNN Eğitimi:**
```powershell
cd Mask_RCNN
pip install -e .
cd ..
python mask_rcnn_coco.py
```

### Ensemble Ayarları

`kamera/data_collector_qt_ensemble.py` dosyasında:
```python
# Eşik Değerleri
DEFAULT_CONF_THRESHOLD = 0.40    # Güven eşiği
NMS_IOU_THRESHOLD = 0.40         # NMS IoU eşiği

# Model Ağırlıkları
SCRATCH_MODEL_STRONG_WEIGHT = 1.0   # Çizik modeli çizik için
SCRATCH_MODEL_WEAK_WEIGHT = 0.5     # Çizik modeli siyah nokta için
BLACKDOT_MODEL_STRONG_WEIGHT = 1.0  # Siyah nokta modeli siyah nokta için
BLACKDOT_MODEL_WEAK_WEIGHT = 0.5    # Siyah nokta modeli çizik için
```

## 🎯 Ensemble Stratejisi

1. Her iki modelden (çizik & siyah nokta) tahmin alınır
2. Aynı konumdaki (IoU ≥ 0.4) aynı sınıf tespitleri birleştirilir
3. Ağırlıklı ortalama ile ensemble güven skoru hesaplanır:
   - Her model kendi güçlü sınıfında daha yüksek ağırlık alır
4. Eşik altındaki tespitler filtrelenir

**Renk Kodları:**
- 🟡 Sarı: Her iki model tespit etti (Ensemble)
- 🟢 Yeşil: Sadece çizik modeli tespit etti
- 🔵 Mavi: Sadece siyah nokta modeli tespit etti

## ⚙️ Kamera Ayarları

Sol/Sağ parça için farklı kamera ayarları tanımlıdır:
- Parlaklık, kontrast, doygunluk
- Focus mesafesi
- Beyaz dengesi
- Keskinlik

## 📊 Çıktılar

- **Görüntüler**: `results/DataCollection/PartX/` klasöründe
- **Analiz Sonuçları**: CSV formatında tespit detayları
- **Eğitim Sonuçları**: `training_plots/` klasöründe

## ❗ Sık Karşılaşılan Sorunlar

| Sorun | Çözüm |
|-------|-------|
| CUDA uyuşmazlığı | PyTorch'u CUDA sürümünüze uygun kurun |
| Kamera açılmadı | DirectShow sürücülerini kontrol edin |
| Model bulunamadı | `RT_Detr_Ensemble/model/` içindeki .pt dosyalarını kontrol edin |
| `pycocotools` hatası | `pip install pycocotools-windows` |

## 📫 İletişim

Bu proje özel kullanım içindir.
