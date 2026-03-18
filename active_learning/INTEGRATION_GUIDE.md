# Active Learning Entegrasyon Rehberi
## RT-DETR Ensemble Sistemi için Belirsizlik Tabanlı Aktif Öğrenme

---

## 📋 İçindekiler

1. [Genel Bakış](#genel-bakış)
2. [Entegrasyon Adımları](#entegrasyon-adımları)
3. [Workflow](#workflow)
4. [Performans Beklentileri](#performans-beklentileri)
5. [En İyi Uygulamalar](#en-iyi-uygulamalar)
6. [Sorun Giderme](#sorun-giderme)

---

## 🎯 Genel Bakış

### Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────────┐
│                    Üretim Hattı                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ Kamera 0 │   │ Kamera 1 │   │ Kamera 2 │   │ Kamera 3 │ │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘ │
└───────┼──────────────┼──────────────┼──────────────┼────────┘
        │              │              │              │
        └──────────────┴──────────────┴──────────────┘
                           │
                    ┌──────▼───────┐
                    │  RT-DETR     │
                    │  Ensemble    │
                    │  İnference   │
                    └──────┬───────┘
                           │
            ┌──────────────┴──────────────┐
            │                             │
     ┌──────▼────────┐          ┌────────▼─────────┐
     │ Yüksek Güven  │          │ Düşük/Orta Güven │
     │ (>0.70)       │          │ (0.30-0.70)      │
     │               │          │                  │
     │ ✓ Doğrudan    │          │ ⚠ Etiketleme    │
     │   Kabul       │          │   Kuyruğuna      │
     └───────────────┘          └────────┬─────────┘
                                         │
                                  ┌──────▼──────┐
                                  │  Operatör   │
                                  │  Kontrolü   │
                                  └──────┬──────┘
                                         │
                                  ┌──────▼──────┐
                                  │ Doğrulanmış │
                                  │   Veriler   │
                                  └──────┬──────┘
                                         │
                        ┌────────────────┴────────────────┐
                        │                                 │
                 ┌──────▼──────┐                  ┌──────▼──────┐
                 │ Yeni Veriler│                  │ Eski Veriler│
                 │   (100%)    │                  │    (30%)    │
                 └──────┬──────┘                  └──────┬──────┘
                        │                                 │
                        └────────────────┬────────────────┘
                                         │
                                  ┌──────▼──────┐
                                  │  Artımlı    │
                                  │   Eğitim    │
                                  └──────┬──────┘
                                         │
                                  ┌──────▼──────┐
                                  │ Yeni Model  │
                                  │ (v2.0, v3.0)│
                                  └─────────────┘
```

### Neden Bu Yaklaşım?

**Problem:** 
- Statik modeller üretim ortamındaki değişikliklere adapte olamaz
- Data drift (ışık, toz, parça varyasyonu) performansı düşürür
- Yeni kusur tiplerini öğrenemez

**Çözüm: Active Learning**
1. **Belirsizlik Örneklemesi**: Model en çok zorlandığı örnekleri seçer
2. **İnsan Denetimi**: Operatör sadece kritik örnekleri kontrol eder
3. **Sürekli İyileştirme**: Model gerçek üretim verisiyle güçlenir
4. **Domain Adaptation**: Model kendi ortamına özelleşir

---

## 🔧 Entegrasyon Adımları

### Adım 1: Dosyaları Kopyalayın

```bash
# Active Learning modülünü projenize ekleyin
cp active_learning_module.py /path/to/your/project/

# Integration guide'ı referans olarak kullanın
cp integration_guide.py /path/to/your/project/
```

### Adım 2: Mevcut Kodu Güncelleyin

`data_collector_qt_ensemble.py` dosyasında aşağıdaki değişiklikleri yapın:

#### 2.1 Import'ları Ekleyin (Dosyanın başına)

```python
from active_learning_module import (
    UncertaintySampler,
    ActiveLearningDataManager,
    IncrementalTrainer
)
```

#### 2.2 Config'e Active Learning Ayarlarını Ekleyin

```python
# Active Learning Ayarları
ENABLE_ACTIVE_LEARNING = True
AL_LOW_CONF_THRESHOLD = 0.30
AL_HIGH_CONF_THRESHOLD = 0.70
AL_SAMPLE_HIGH_CONF = 0.05
AL_DATA_ROOT = "active_learning_data"
```

#### 2.3 DataCollectorWindow.__init__ Metodunu Güncelleyin

`__init__` metodunun sonuna ekleyin:

```python
# Active Learning bileşenlerini başlat
if ENABLE_ACTIVE_LEARNING:
    self.al_sampler = UncertaintySampler(
        low_conf_threshold=AL_LOW_CONF_THRESHOLD,
        high_conf_threshold=AL_HIGH_CONF_THRESHOLD,
        save_high_conf_ratio=AL_SAMPLE_HIGH_CONF
    )
    self.al_manager = ActiveLearningDataManager(root_dir=AL_DATA_ROOT)
    print("[AL] Active Learning sistemi başlatıldı")
else:
    self.al_sampler = None
    self.al_manager = None
```

#### 2.4 analyze_batch Metodunu Güncelleyin

`analyze_batch` metodunda, her görüntü analiz edildikten sonra (910. satırın civarında):

```python
# RT-DETR Ensemble analizi (mevcut kod)
vis_bgr, ensemble_preds, err_count, has_error = analyze_frame_ensemble(
    self._scratch_model, self._blackdot_model, img, DEFAULT_CONF_THRESHOLD
)
vis_bgr = draw_ok_nok_flag(vis_bgr, has_error)

# ========== BURAYA ACTIVE LEARNING KODUNU EKLEYİN ==========
uncertainty_score = 0.0
al_queued = False

if ENABLE_ACTIVE_LEARNING and self.al_sampler and self.al_manager:
    should_annotate, reason, uncertainty_score = self.al_sampler.should_annotate(ensemble_preds)
    
    if should_annotate:
        cam_idx = _extract_cam_idx(fname)
        metadata = {
            "camera_idx": cam_idx,
            "filename": fname,
            "part_dir": self.current_part_dir,
            "side": self.selected_side,
            "error_count": err_count,
            "has_error": has_error
        }
        
        file_id = self.al_manager.save_inference_result(
            image=img,
            predictions=ensemble_preds,
            metadata=metadata,
            uncertainty_info=(should_annotate, reason, uncertainty_score)
        )
        al_queued = True
        al_saved += 1
# ============================================================

# Devamında mevcut kod (out_name, cv2.imwrite vb.)
```

#### 2.5 UI'a Active Learning Butonları Ekleyin

`init_ui` veya benzeri bir metodda:

```python
if ENABLE_ACTIVE_LEARNING:
    # AL İstatistikleri
    btn_al_stats = QtWidgets.QPushButton("AL İstatistikleri")
    btn_al_stats.clicked.connect(self.show_al_statistics)
    your_layout.addWidget(btn_al_stats)
    
    # Etiketleme Arayüzü
    btn_annotate = QtWidgets.QPushButton("Etiketleme Arayüzü")
    btn_annotate.clicked.connect(self.open_annotation_interface)
    your_layout.addWidget(btn_annotate)
    
    # YOLO Dönüşümü
    btn_convert = QtWidgets.QPushButton("YOLO'ya Dönüştür")
    btn_convert.clicked.connect(self.convert_to_yolo)
    your_layout.addWidget(btn_convert)
    
    # Artımlı Eğitim
    btn_train = QtWidgets.QPushButton("Artımlı Eğitim")
    btn_train.clicked.connect(self.start_incremental_training)
    your_layout.addWidget(btn_train)
```

#### 2.6 Helper Fonksiyonları Ekleyin

`integration_guide.py` dosyasındaki fonksiyonları (`show_al_statistics`, 
`open_annotation_interface`, `convert_to_yolo`, vb.) sınıfınıza ekleyin.

### Adım 3: Gerekli Kütüphaneleri Kurun

```bash
pip install opencv-python numpy torch ultralytics
pip install PyQt5  # Eğer yoksa
```

### Adım 4: Test Edin

```python
# Sistemi başlatın
python data_collector_qt_ensemble.py

# 1. Part seçin ve görüntü toplayın
# 2. "Analiz Et" butonuna basın
# 3. AL İstatistiklerini kontrol edin
```

---

## 🔄 Workflow

### Günlük Kullanım

1. **Sabah (Üretim Başlangıcı)**
   ```
   - Sistemi başlat
   - Kameraları kalibre et
   - Part seçimi yap
   ```

2. **Gün İçi (Üretim Sırasında)**
   ```
   - Görüntü toplama devam ediyor
   - Her batch analizi sonrası AL kuyruğu dolacak
   - Belirsiz örnekler otomatik olarak etiketleme kuyruğuna eklenecek
   ```

3. **Öğle / Akşam (Etiketleme)**
   ```
   - "AL İstatistikleri" butonuna bas
   - Kuyrukta bekleyen örnek sayısını gör
   - "Etiketleme Arayüzü" butonuna bas
   - CVAT veya LabelImg ile etiketle
   - Etiketlenmiş dosyaları "annotated" klasörüne taşı
   ```

4. **Haftalık (Model Güncelleme)**
   ```
   - "YOLO'ya Dönüştür" butonuna bas
   - En az 50-100 etiketlenmiş örnek biriktiğinde
   - "Artımlı Eğitim" başlat
   - Yeni modeli test et
   - İyiyse üretim modelini değiştir
   ```

### Etiketleme İş Akışı

#### Seçenek 1: CVAT (Önerilen)

1. **CVAT Kurulumu** (Lokal veya Cloud)
   ```bash
   git clone https://github.com/opencv/cvat
   cd cvat
   docker-compose up -d
   ```

2. **Projeyi İçe Aktar**
   - CVAT arayüzüne gir
   - Yeni proje oluştur
   - `active_learning_data/annotation_queue/` klasöründeki görselleri yükle
   - Mevcut JSON'ları pre-annotation olarak kullan

3. **Etiketleme**
   - Yanlış kutuları düzelt
   - Eksikleri ekle
   - Fazlaları sil

4. **Dışa Aktar**
   - YOLO 1.1 formatında dışa aktar
   - Dosyaları `annotated` klasörüne taşı

#### Seçenek 2: LabelImg (Basit)

1. **Kurulum**
   ```bash
   pip install labelImg
   labelImg
   ```

2. **Klasör Seç**
   - Open Dir: `active_learning_data/annotation_queue/`
   - Change Save Dir: `active_learning_data/annotated/`

3. **Etiketle ve Kaydet**

### Model Güncelleme İş Akışı

```
[Etiketlenmiş Veriler] → [YOLO Dönüşümü] → [Veri Birleştirme] → [Eğitim] → [Değerlendirme] → [Deployment]
         ↓                      ↓                   ↓                ↓              ↓             ↓
    annotated/         training_ready/      merged_dataset/     models/      validation/    production/
   (JSON+PNG)           (txt+PNG)          (old+new mix)      (weights)     (metrics)      (best.pt)
```

---

## 📊 Performans Beklentileri

### Kısa Vadeli (1-2 Hafta)

| Metrik | Başlangıç | Beklenti | Açıklama |
|--------|-----------|----------|----------|
| **False Positive** | Baseline | ↓ %10-15 | Yanlış alarm azalması |
| **False Negative** | Baseline | ↓ %5-10 | Kaçan hata azalması |
| **Edge Cases** | Zor tespitler | ↑ %20-30 | Belirsiz durumlar iyileşmesi |

### Orta Vadeli (1-2 Ay)

| Metrik | Başlangıç | Beklenti | Açıklama |
|--------|-----------|----------|----------|
| **mAP@0.5** | Baseline | ↑ %5-10 | Genel doğruluk artışı |
| **Precision** | Baseline | ↑ %8-12 | Doğru tespit oranı |
| **Recall** | Baseline | ↑ %6-10 | Hatayı yakalama oranı |
| **Operatör Müdahalesi** | Yüksek | ↓ %30-40 | Manuel kontrol azalması |

### Uzun Vadeli (3-6 Ay)

| Metrik | Başlangıç | Beklenti | Açıklama |
|--------|-----------|----------|----------|
| **Domain Adaptation** | Generic | Özelleşmiş | Kendi ortamına tam uyum |
| **New Defect Types** | 2 sınıf | +N sınıf | Yeni kusur tipleri öğrenme |
| **Robustness** | Orta | Yüksek | Değişen koşullara dayanıklılık |
| **Production Time** | Baseline | ↓ %20-30 | İnceleme süresi azalması |

### Başarı Kriterleri

✅ **Haftalık etiketleme:** 50-100 görüntü (ortalama 30 dk/gün)
✅ **Model güncelleme:** 2-4 haftada bir
✅ **Performans artışı:** Her güncellemede +%2-5 mAP
✅ **Operatör geri bildirimi:** Pozitif (daha az yanlış alarm)

---

## 🎯 En İyi Uygulamalar

### 1. Eşik Değerleri Optimizasyonu

```python
# BAŞLANGIÇ: Muhafazakar yaklaşım (daha çok örnek topla)
AL_LOW_CONF_THRESHOLD = 0.35
AL_HIGH_CONF_THRESHOLD = 0.65

# 2. HAFTA: Modele güven arttıkça sıkılaştır
AL_LOW_CONF_THRESHOLD = 0.30
AL_HIGH_CONF_THRESHOLD = 0.70

# 1. AY: İyice optimize et
AL_LOW_CONF_THRESHOLD = 0.25
AL_HIGH_CONF_THRESHOLD = 0.75
```

### 2. Etiketleme Kalitesi

✅ **Yapılması Gerekenler:**
- Her kutu için 2-3 piksel hassasiyet
- Belirsiz durumlarda not ekle (CVAT'ta)
- Tutarlı sınıf ataması (scratch vs black_dot)
- Edge case'leri dikkatlice kontrol et

❌ **Kaçınılması Gerekenler:**
- Acele etiketleme (hata propagasyonu)
- Tek kişi bağımlılığı (farklı operatörler rotasyon yapmalı)
- Belirsiz örnekleri atlamak (bunlar en değerli örnekler!)

### 3. Veri Dengesi

```python
# Sınıf dengesini kontrol edin
import json
from collections import Counter

def check_class_balance(annotation_dir):
    class_counts = Counter()
    
    for json_file in Path(annotation_dir).glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            for pred in data.get('predictions', []):
                class_counts[pred['class']] += 1
    
    print("Sınıf Dağılımı:")
    for cls, count in class_counts.items():
        print(f"  Class {cls}: {count} ({count/sum(class_counts.values())*100:.1f}%)")
    
    # Eğer dengesizlik varsa (örn. %80-%20), 
    # az olan sınıftan daha fazla örnek toplayın
```

### 4. Catastrophic Forgetting Önleme

```python
# İyi Uygulamalar:
OLD_DATA_RATIO = 0.30  # Eski veriden %30 karıştır

# ❌ Yanlış:
OLD_DATA_RATIO = 0.0   # Sadece yeni veri → Eski bilgileri unutur

# ❌ Aşırı:
OLD_DATA_RATIO = 0.8   # Çok fazla eski veri → Yeni öğrenemez
```

### 5. Learning Rate Schedule

```python
# Fine-tuning için düşük LR kullanın
LEARNING_RATES = {
    "first_update": 0.0001,    # İlk güncelleme - temkinli
    "stable_updates": 0.00005,  # Sonraki güncellemeler - daha da temkinli
    "major_changes": 0.0002     # Büyük değişiklikler varsa (yeni kusur tipi)
}
```

### 6. Versiyonlama

```python
# Her model güncellemesini versiyonlayın
# Format: model_name_vX.Y_YYYYMMDD.pt

# Örnek:
# scratch_v1.0_20250101.pt  ← Başlangıç
# scratch_v1.1_20250115.pt  ← İlk AL güncelleme
# scratch_v1.2_20250201.pt  ← İkinci güncelleme
# scratch_v2.0_20250301.pt  ← Major güncelleme (yeni sınıf eklendi)

# Git benzeri yaklaşım
models/
  ├── scratch/
  │   ├── v1.0/
  │   │   ├── best.pt
  │   │   ├── metrics.csv
  │   │   └── training_log.txt
  │   ├── v1.1/
  │   └── v2.0/
  └── blackdot/
      └── ...
```

### 7. A/B Testing

```python
# Yeni modeli deployment'a almadan önce test edin

class ABTester:
    def __init__(self, model_a, model_b, test_set):
        self.model_a = model_a  # Mevcut üretim modeli
        self.model_b = model_b  # Yeni eğitilmiş model
        self.test_set = test_set
    
    def compare(self):
        results_a = self.evaluate(self.model_a)
        results_b = self.evaluate(self.model_b)
        
        # Eğer B, A'dan %5+ daha iyiyse → Deploy
        improvement = (results_b['mAP'] - results_a['mAP']) / results_a['mAP']
        
        if improvement > 0.05:
            print(f"✅ Model B daha iyi: +{improvement*100:.1f}%")
            return "deploy_b"
        elif improvement < -0.02:
            print(f"❌ Model B daha kötü: {improvement*100:.1f}%")
            return "keep_a"
        else:
            print(f"⚠ Belirsiz: {improvement*100:.1f}% fark")
            return "manual_review"
```

### 8. Monitoring

```python
# Her inference'ta metrikleri logla
import logging

logging.basicConfig(
    filename='active_learning_metrics.log',
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)

# Örnek log
logging.info(f"inference | cam={cam_idx} | conf={avg_conf:.3f} | uncertainty={unc:.3f} | queued={queued}")
logging.info(f"annotation | file_id={file_id} | corrected={n_corrections}")
logging.info(f"training | epoch={epoch} | mAP={map_score:.4f}")
```

---

## 🐛 Sorun Giderme

### Problem 1: Kuyruk Çok Hızlı Doluyur

**Belirtiler:**
- Etiketleme kuyruğunda saatte 100+ görüntü birikim
- Operatör yetişemiyor

**Çözüm:**
```python
# Eşik değerlerini sıkılaştırın
AL_LOW_CONF_THRESHOLD = 0.20  # Daha düşük
AL_HIGH_CONF_THRESHOLD = 0.80  # Daha yüksek
AL_SAMPLE_HIGH_CONF = 0.02    # Daha az sample

# Veya filtreleme ekleyin
def should_annotate_with_filter(predictions, ...):
    basic_result = sampler.should_annotate(predictions)
    
    # Ek filtre: Çok küçük tesitleri atla
    if basic_result[0]:
        for pred in predictions:
            x1, y1, x2, y2 = pred[:4]
            area = (x2 - x1) * (y2 - y1)
            if area < 100:  # Çok küçük
                return False, "too_small", 0.0
    
    return basic_result
```

### Problem 2: Model Performansı Düşüyor (Catastrophic Forgetting)

**Belirtiler:**
- Yeni model eskisinden kötü
- Eski kusur tiplerini tespit edemiyor

**Çözüm:**
```python
# Eski veri oranını artırın
OLD_DATA_RATIO = 0.5  # %30'dan %50'ye

# Learning rate'i düşürün
LEARNING_RATE = 0.00005  # Daha temkinli

# Daha uzun eğitim
EPOCHS = 100  # 50'den 100'e
```

### Problem 3: Etiketleme Kalitesi Düşük

**Belirtiler:**
- Model eğitim sonrası validation loss artıyor
- Tutarsız tahminler

**Çözüm:**
- İki operatör cross-check yapmalı
- Belirsiz örnekler için konsensus
- Etiketleme rehberi oluştur:
  ```
  # etiketleme_rehberi.md
  
  ## Scratch (Çizik)
  - Çizik en az 3 piksel genişliğinde olmalı
  - Kenarlar keskin ve düz
  - Renk: Genelde koyu gri veya beyaz
  
  ## Black Dot (Siyah Nokta)
  - Yuvarlak veya oval şekil
  - Çap en az 2 piksel
  - Renk: Siyah veya koyu kahverengi
  
  ## Belirsiz Durumlar
  - Toz vs black dot: Toz geçici, silmeyle gider
  - Yansıma vs scratch: Yansıma açıya göre değişir
  ```

### Problem 4: Eğitim Çok Yavaş

**Belirtiler:**
- GPU kullanımı düşük
- Eğitim saatlerce sürüyor

**Çözüm:**
```python
# GPU kontrolü
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Batch size artır (GPU memory yetiyorsa)
BATCH_SIZE = 32  # 16'dan 32'ye

# Workers artır
NUM_WORKERS = 8  # CPU core sayısına göre

# Mixed precision kullan
from ultralytics import RTDETR
model = RTDETR(model_path)
results = model.train(
    ...,
    amp=True  # Automatic Mixed Precision
)
```

### Problem 5: Deployment Sonrası Performans Farklı

**Belirtiler:**
- Eğitim sırasında iyi, üretimde kötü
- Test setinde yüksek mAP, gerçekte düşük

**Çözüm:**
```python
# Validation set'i üretim verisine benzer hale getir
# Test ve validation setlerini AL'den gelen verilerle oluştur

def create_production_like_test_set():
    """
    AL sisteminden toplanan ve etiketlenen verilerden
    test seti oluştur (production'a en yakın)
    """
    annotated_dir = Path("active_learning_data/annotated")
    test_dir = Path("test_set_production_like")
    
    # Rastgele %10 seç
    all_files = list(annotated_dir.glob("*.json"))
    test_files = np.random.choice(all_files, size=int(len(all_files)*0.1))
    
    for f in test_files:
        # Kopyala...
        pass
```

---

## 📈 Metrik Takibi

### Günlük İzleme

```python
# daily_metrics.py
import pandas as pd
from datetime import datetime

def log_daily_metrics():
    metrics = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'images_processed': 0,
        'al_queued': 0,
        'annotated': 0,
        'avg_confidence': 0.0,
        'avg_uncertainty': 0.0
    }
    
    # CSV'ye kaydet
    df = pd.DataFrame([metrics])
    df.to_csv('metrics_daily.csv', mode='a', header=False, index=False)
```

### Haftalık Rapor

```python
# weekly_report.py
def generate_weekly_report():
    df = pd.read_csv('metrics_daily.csv')
    
    report = f"""
    HAFTALIK ACTIVE LEARNING RAPORU
    ================================
    
    Tarih Aralığı: {df['date'].min()} - {df['date'].max()}
    
    📊 Veri Toplama:
      - Toplam İşlenen: {df['images_processed'].sum()}
      - AL Kuyruğuna Eklenen: {df['al_queued'].sum()}
      - Etiketlenen: {df['annotated'].sum()}
      - Etiketleme Oranı: {df['annotated'].sum()/df['al_queued'].sum()*100:.1f}%
    
    🎯 Belirsizlik:
      - Ortalama Confidence: {df['avg_confidence'].mean():.3f}
      - Ortalama Uncertainty: {df['avg_uncertainty'].mean():.3f}
    
    ✅ Öneriler:
      {generate_recommendations(df)}
    """
    
    print(report)
    return report
```

---

## 🎓 Literatür ve Referanslar

### Önemli Makaleler

1. **Active Learning for Deep Object Detection**
   - Haussmann et al., 2020
   - Uncertainty sampling strategies comparison

2. **Learning Loss for Active Learning**
   - Yoo & Kweon, CVPR 2019
   - Learning to predict which samples are informative

3. **The Power of Ensembles for Active Learning in Image Classification**
   - Beluch et al., CVPR 2018
   - Ensemble-based uncertainty metrics

4. **A Survey on Deep Semi-supervised Learning**
   - Yang et al., 2022
   - Pseudo-labeling and consistency regularization

### İlgili Kavramlar

- **Curriculum Learning**: Kolay → Zor örneklerle eğitim
- **Semi-Supervised Learning**: Etiketli + etiketsiz veri kullanımı
- **Domain Adaptation**: Source → Target domain transfer
- **Few-Shot Learning**: Az örnekle öğrenme

### Online Kaynaklar

- [Ultralytics YOLO Docs](https://docs.ultralytics.com)
- [CVAT Documentation](https://opencv.github.io/cvat/)
- [Active Learning Blog - Lilian Weng](https://lilianweng.github.io/posts/2022-02-20-active-learning/)

---

## 📝 Sonuç ve İleriki Adımlar

### Başlangıç Fazı (İlk 2 Hafta)

✅ Active Learning modülünü entegre et
✅ İlk 100 görüntüyü etiketle
✅ İlk model güncellemesini yap
✅ Performans karşılaştırması yap

### Büyüme Fazı (1-3 Ay)

✅ Operatörleri eğit
✅ Etiketleme prosedürü oluştur
✅ Haftalık model güncellemeleri
✅ Metrik takibi otomatikleştir

### Olgunluk Fazı (3-6 Ay)

✅ Tam otomatik pipeline
✅ Yeni kusur tipleri ekleme
✅ Multi-site deployment
✅ Makale/patent hazırlığı

---

## 🤝 Destek

Sorularınız veya sorunlarınız için:
- GitHub Issues
- Antropic Claude ile danışma
- Ultralytics Community Forum

---

**Son Güncelleme:** 2025-02-05
**Versiyon:** 1.0
**Yazar:** AI Assistant via Anthropic Claude
