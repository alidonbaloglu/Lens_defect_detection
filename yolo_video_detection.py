from ultralytics import YOLO
import cv2
import time
import torch

# Modeli yükle ve uygun cihaza taşı
model = YOLO("Modeller/Lens_v2_1400/weights/best.pt")
use_cuda = torch.cuda.is_available()
device = 0 if use_cuda else "cpu"

# Video dosyasını aç
cap = cv2.VideoCapture("Videolar/İşaretli/isaretsiz2.mp4")

# Video özelliklerini al
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # veya 'XVID'
fps = cap.get(cv2.CAP_PROP_FPS)
target_width = 1408
target_height = 1408

# VideoWriter ile çıktı videosunu oluştur (1920x1080)
out = cv2.VideoWriter("output.mp4", fourcc, fps if fps > 0 else 25, (target_width, target_height))

# FPS hesaplama için değişkenler
prev_time = 0
curr_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # FPS hesaplama
    curr_time = time.time()
    fps_calc = 1 / (curr_time - prev_time) if prev_time > 0 else 0
    prev_time = curr_time

    # Girişi hedef boyuta ölçekle ve tek seferde tahmin yap
    frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
    results = model.predict(source=frame_resized, 
                            imgsz=target_width, 
                            conf=0.3, 
                            device=device, 
                            half=use_cuda, 
                            verbose=False
                            )

    # Sonuçları çiz (zaten 1920x1080)
    result_img = results[0].plot()

    # FPS değerini ekrana yaz
    cv2.putText(result_img, f"FPS: {fps_calc:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Sonucu göster
    cv2.imshow("YOLO Video", result_img)

    # Sonucu dosyaya yaz
    out.write(result_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()