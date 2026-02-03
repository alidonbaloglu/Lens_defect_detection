# 🔍 Lens Hata Tespit Projesi

Lens üretiminde yüzey hatalarının (çizik, siyah nokta vb.) otomatik tespiti için geliştirilen derin öğrenme tabanlı görüntü işleme projesidir.

## 📋 Özellikler

- **RT-DETR Ensemble Model**: Çizik ve siyah nokta tespiti için özelleştirilmiş iki modelin ağırlıklı ensemble birleşimi
- **Çoklu Kamera Desteği**: 4 kamera ile eş zamanlı görüntü yakalama
- **Gerçek Zamanlı Analiz**: PyQt5 tabanlı kullanıcı arayüzü ile canlı tespit
- **Otomatik Veri Toplama**: Focus taraması ile optimum görüntü yakalama
- **YOLO & Mask R-CNN**: Alternatif model eğitimi ve değerlendirme

---

## 🚀 Hızlı Başlangıç (Yeni Kullanıcılar İçin)

### 1. Depoyu Klonlayın
```powershell
git clone https://github.com/alidonbaloglu/Lens_defect_detection.git
cd Lens_defect_detection
```

### 2. Sanal Ortam Oluşturun
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. PyTorch Kurulumu
CUDA sürümünüze uygun PyTorch kurun:
```powershell
# CUDA 12.6 için (GPU kullanımı için önerilir)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Sadece CPU için
pip install torch torchvision
```

### 4. Bağımlılıkları Yükleyin
```powershell
pip install -r requirements.txt
```

---

## 📦 Eksik Dosyalar ve Nasıl Elde Edilir

GitHub'da depo boyutu sınırlamaları nedeniyle aşağıdaki dosyalar paylaşılmamıştır:

### 🗂️ Veri Setleri (`datasetler/`)

Veri setleri Roboflow üzerinden indirilebilir veya kendi verilerinizi oluşturabilirsiniz:

**Roboflow'dan İndirme:**
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
dataset = project.version(1).download("yolov8")
```

**Manuel Veri Seti Yapısı (YOLO formatı):**
```
datasetler/
└── your_dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── data.yaml
```

### 🤖 Model Dosyaları

Model dosyaları Kaggle'dan indirilebilir:

**📥 Kaggle'dan İndirme:**
```powershell
# Kaggle CLI kurulumu
pip install kaggle

# Modelleri indir
kaggle datasets download -d alidonbalolu/lens-defect-detection-models
unzip lens-defect-detection-models.zip -d RT_Detr_Ensemble/model/
```

**Model Dataset Linki:** [kaggle.com/datasets/alidonbalolu/lens-defect-detection-models](https://www.kaggle.com/datasets/alidonbalolu/lens-defect-detection-models)

İndirilen dosyalar:
- `best_cizik.pt` - Çizik tespit modeli (RT-DETR)
- `best_siyahnokta.pt` - Siyah nokta tespit modeli (RT-DETR)

**Manuel İndirme:**
1. Yukarıdaki linke gidin
2. "Download" butonuna tıklayın
3. Dosyaları `RT_Detr_Ensemble/model/` klasörüne çıkarın

### 📁 Oluşturmanız Gereken Klasörler
```powershell
# Gerekli klasörleri oluşturun
mkdir -p datasetler
mkdir -p Modeller
mkdir -p RT_Detr_Ensemble/model
mkdir -p results
mkdir -p training_plots
```

---

## 📁 Proje Yapısı

```
Lens_defect_detection/
├── kamera/                          # Kamera uygulamaları
│   ├── data_collector_qt_ensemble.py   # Ana veri toplama uygulaması
│   └── camera_ayar.py                  # Kamera ayarları
│
├── RT_Detr_Ensemble/               # RT-DETR Ensemble (model/ klasörü eksik)
│   └── ensemble_test.py               # Ensemble test scripti
│
├── Model_test/                     # Test scriptleri
│   ├── Eski/                          # Eski test dosyaları
│   └── Yeni/                          # Yeni test dosyaları
│
├── Mask_RCNN/                      # Matterport Mask R-CNN
├── RT_DETR/                        # RT-DETR eğitim dosyaları
│
├── rtdetr_train_v3.py              # RT-DETR model eğitimi
├── yolo11_train.py                 # YOLO model eğitimi
├── mask_rcnn_coco.py               # Mask R-CNN eğitimi
│
├── requirements.txt                # Python bağımlılıkları
└── README.md                       # Bu dosya
```

---

## 🎯 Kullanım

### Veri Toplama Uygulaması (Kamera Gerektirir)
```powershell
python kamera/data_collector_qt_ensemble.py
```

> ⚠️ **Not**: Bu uygulama 4 USB kamera ve eğitilmiş model dosyalarını gerektirir.

### Model Eğitimi

**RT-DETR Eğitimi:**
```powershell
# Veri setinizi datasetler/ klasörüne koyun
# rtdetr_train_v3.py içindeki yolları güncelleyin
python rtdetr_train_v3.py
```

**YOLO Eğitimi:**
```powershell
python yolo11_train.py
```

### Test Scriptleri
```powershell
# Ensemble test
python Model_test/Yeni/ensemble_gerçek_test_.py

# YOLO test
python Model_test/Yeni/yolo_test.py
```

---

## ⚙️ Konfigürasyon

### Model Yollarını Güncelleme

`kamera/data_collector_qt_ensemble.py` dosyasında model yollarını kendi sisteminize göre güncelleyin:

```python
# Satır 25-26
SCRATCH_MODEL_PATH = r"SIZIN_YOL/RT_Detr_Ensemble/model/best_cizik.pt"
BLACKDOT_MODEL_PATH = r"SIZIN_YOL/RT_Detr_Ensemble/model/best_siyahnokta.pt"
```

### Eşik Değerleri
```python
DEFAULT_CONF_THRESHOLD = 0.40    # Güven eşiği (0.0-1.0)
NMS_IOU_THRESHOLD = 0.40         # NMS IoU eşiği
```

---

## 🎓 Ensemble Stratejisi

1. Her iki modelden (çizik & siyah nokta) tahmin alınır
2. Aynı konumdaki (IoU ≥ 0.4) aynı sınıf tespitleri birleştirilir
3. Ağırlıklı ortalama ile ensemble güven skoru hesaplanır
4. Eşik altındaki tespitler filtrelenir

**Renk Kodları:**
- 🟡 Sarı: Her iki model tespit etti (Ensemble)
- 🟢 Yeşil: Sadece çizik modeli tespit etti
- 🔵 Mavi: Sadece siyah nokta modeli tespit etti

---

## ❗ Sorun Giderme

| Sorun | Çözüm |
|-------|-------|
| `Model bulunamadı` | Model dosyalarını `RT_Detr_Ensemble/model/` içine koyun |
| `CUDA uyuşmazlığı` | PyTorch'u CUDA sürümünüze uygun kurun |
| `Kamera açılmadı` | USB kameraların bağlı olduğunu kontrol edin |
| `pycocotools hatası` | `pip install pycocotools-windows` |
| `mrcnn import hatası` | `cd Mask_RCNN && pip install -e .` |

---

## 📧 İletişim

Model dosyaları veya veri setleri için iletişime geçin.

---

## 📄 Lisans

Bu proje özel kullanım içindir.
