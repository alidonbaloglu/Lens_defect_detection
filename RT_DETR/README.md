# RT-DETR Kabin Hata Tespiti

RT-DETR (Real-Time Detection Transformer) ile kabin yüzey hatası tespiti.

## Kurulum

```bash
pip install ultralytics>=8.0.0
```

## Eğitim

```bash
cd RT_DETR
python rtdetr_train.py
```

Eğitim otomatik olarak:
1. COCO formatındaki dataset'i YOLO formatına dönüştürür
2. RT-DETR-L modelini yükler
3. Eğitimi başlatır
4. En iyi modeli `Modeller/RTDETR_kabin_v1.pt` olarak kaydeder

## Inference

```bash
# Tek görsel
python rtdetr_infer.py --source test_image.jpg

# Klasör
python rtdetr_infer.py --source ../datasetler/kabin_dataset_v1/test/

# Video
python rtdetr_infer.py --source ../output.mp4

# Kamera
python rtdetr_infer.py --source 0
```

## Model Seçenekleri

| Model | mAP (COCO) | FPS (T4 GPU) | Parametre |
|-------|-----------|--------------|-----------|
| rtdetr-l | 53.0% | 114 | 32M |
| rtdetr-x | 54.8% | 74 | 65M |

## Sınıflar

- `cizik` (çizik)
- `siyah_nokta` (siyah nokta)

## Dosyalar

- `rtdetr_train.py` - Eğitim scripti
- `rtdetr_infer.py` - Inference scripti
- `kabin_dataset.yaml` - Dataset konfigürasyonu
