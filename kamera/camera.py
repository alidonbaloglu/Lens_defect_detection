import cv2
import numpy as np

def main():
    # Kamerayı başlat (0 = varsayılan kamera, birden fazla kamera varsa 1, 2 deneyin)
    cap = cv2.VideoCapture(0)
    
    # Kamera açılamazsa hata mesajı
    if not cap.isOpened():
        print("Hata: Kamera açılamadı!")
        return
    
    # Kamera çözünürlüğünü ayarla (C920 Pro 1080p destekler)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # FPS ayarla
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Kamera başlatıldı. Çıkmak için 'q' tuşuna basın.")
    print(f"Çözünürlük: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    
    while True:
        # Kameradan frame oku
        ret, frame = cap.read()
        
        if not ret:
            print("Hata: Frame okunamadı!")
            break
        
        # Görüntü üzerine bilgi ekle
        cv2.putText(frame, 'Logitech C920 Pro', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Cikmak icin 'q' basin", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Frame'i göster
        cv2.imshow('Logitech C920 Pro - Canli Goruntu', frame)
        
        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Kaynakları serbest bırak
    cap.release()
    cv2.destroyAllWindows()
    print("Kamera kapatıldı.")

if __name__ == "__main__":
    main()
