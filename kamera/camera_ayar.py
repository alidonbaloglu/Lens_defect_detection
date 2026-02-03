import cv2
import numpy as np

# Global değişkenler
cap = None

# Trackbar callback fonksiyonları
def set_brightness(val):
    cap.set(cv2.CAP_PROP_BRIGHTNESS, val)

def set_contrast(val):
    cap.set(cv2.CAP_PROP_CONTRAST, val)

def set_saturation(val):
    cap.set(cv2.CAP_PROP_SATURATION, val)

def set_hue(val):
    cap.set(cv2.CAP_PROP_HUE, val)

def set_gain(val):
    cap.set(cv2.CAP_PROP_GAIN, val)

def set_exposure(val):
    cap.set(cv2.CAP_PROP_EXPOSURE, val)

def set_focus(val):
    cap.set(cv2.CAP_PROP_FOCUS, val)

def set_sharpness(val):
    cap.set(cv2.CAP_PROP_SHARPNESS, val)

def set_white_balance(val):
    cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, val)

def toggle_autofocus(val):
    # 0 = Manuel, 1 = Otomatik
    cap.set(cv2.CAP_PROP_AUTOFOCUS, val)

def toggle_auto_exposure(val):
    # 1 = Manuel mod, 3 = Otomatik mod
    if val == 0:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manuel
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # Otomatik

def main():
    global cap
    
    # Kamerayı başlat
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Hata: Kamera açılamadı!")
        return
    
    # Kamera çözünürlüğünü ayarla
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Pencereler oluştur
    cv2.namedWindow('Kamera Goruntüsu')
    cv2.namedWindow('Kamera Ayarlari')
    
    # Otomatik özellikleri kapat
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Autofocus kapat
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Auto exposure kapat (1=manuel)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # Auto white balance kapat
    
    # Trackbar'ları oluştur
    cv2.createTrackbar('Parlaklik', 'Kamera Ayarlari', 128, 255, set_brightness)
    cv2.createTrackbar('Kontrast', 'Kamera Ayarlari', 128, 255, set_contrast)
    cv2.createTrackbar('Doygunluk', 'Kamera Ayarlari', 128, 255, set_saturation)
    cv2.createTrackbar('Renk Tonu', 'Kamera Ayarlari', 0, 180, set_hue)
    cv2.createTrackbar('Kazanc (Gain)', 'Kamera Ayarlari', 50, 255, set_gain)
    cv2.createTrackbar('Pozlama', 'Kamera Ayarlari', 150, 2000, set_exposure)
    cv2.createTrackbar('Focus', 'Kamera Ayarlari', 50, 255, set_focus)
    cv2.createTrackbar('Keskinlik', 'Kamera Ayarlari', 128, 255, set_sharpness)
    cv2.createTrackbar('Beyaz Dengesi', 'Kamera Ayarlari', 4000, 7000, set_white_balance)
    cv2.createTrackbar('AutoFocus (0=K 1=A)', 'Kamera Ayarlari', 0, 1, toggle_autofocus)
    cv2.createTrackbar('AutoExposure (0=K 1=A)', 'Kamera Ayarlari', 0, 1, toggle_auto_exposure)
    
    print("=" * 60)
    print("LOGITECH C920 PRO - MANUEL KAMERA KONTROLLERI")
    print("=" * 60)
    print("Trackbar'ları kullanarak kamera ayarlarını değiştirebilirsiniz")
    print("\nKısayollar:")
    print("  'q' - Çıkış")
    print("  'r' - Varsayılan ayarlara dön")
    print("  's' - Anlık görüntü kaydet")
    print("  'i' - Mevcut ayarları göster")
    print("=" * 60)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Hata: Frame okunamadı!")
            break
        
        frame_count += 1
        
        # Bilgi metni ekle
        info_text = f"FPS: {cap.get(cv2.CAP_PROP_FPS):.0f} | Frame: {frame_count}"
        cv2.putText(frame, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Otomatik mod durumu
        autofocus_status = "ACIK" if cap.get(cv2.CAP_PROP_AUTOFOCUS) else "KAPALI"
        autoexp_status = "ACIK" if cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) == 3 else "KAPALI"
        cv2.putText(frame, f"AutoFocus: {autofocus_status} | AutoExposure: {autoexp_status}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow('Kamera Goruntüsu', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' tuşu - Çıkış
        if key == ord('q'):
            break
        
        # 'r' tuşu - Reset (varsayılan ayarlar)
        elif key == ord('r'):
            cv2.setTrackbarPos('Parlaklik', 'Kamera Ayarlari', 128)
            cv2.setTrackbarPos('Kontrast', 'Kamera Ayarlari', 128)
            cv2.setTrackbarPos('Doygunluk', 'Kamera Ayarlari', 128)
            cv2.setTrackbarPos('Renk Tonu', 'Kamera Ayarlari', 0)
            cv2.setTrackbarPos('Kazanc (Gain)', 'Kamera Ayarlari', 50)
            cv2.setTrackbarPos('Pozlama', 'Kamera Ayarlari', 150)
            cv2.setTrackbarPos('Focus', 'Kamera Ayarlari', 50)
            cv2.setTrackbarPos('Keskinlik', 'Kamera Ayarlari', 128)
            cv2.setTrackbarPos('Beyaz Dengesi', 'Kamera Ayarlari', 4000)
            print("Ayarlar varsayılan değerlere döndürüldü")
        
        # 's' tuşu - Screenshot
        elif key == ord('s'):
            filename = f"screenshot_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Görüntü kaydedildi: {filename}")
        
        # 'i' tuşu - Info
        elif key == ord('i'):
            print("\n" + "=" * 60)
            print("MEVCUT KAMERA AYARLARI:")
            print("=" * 60)
            print(f"Parlaklik:      {cap.get(cv2.CAP_PROP_BRIGHTNESS):.0f}")
            print(f"Kontrast:       {cap.get(cv2.CAP_PROP_CONTRAST):.0f}")
            print(f"Doygunluk:      {cap.get(cv2.CAP_PROP_SATURATION):.0f}")
            print(f"Renk Tonu:      {cap.get(cv2.CAP_PROP_HUE):.0f}")
            print(f"Kazanc:         {cap.get(cv2.CAP_PROP_GAIN):.0f}")
            print(f"Pozlama:        {cap.get(cv2.CAP_PROP_EXPOSURE):.0f}")
            print(f"Focus:          {cap.get(cv2.CAP_PROP_FOCUS):.0f}")
            print(f"Keskinlik:      {cap.get(cv2.CAP_PROP_SHARPNESS):.0f}")
            print(f"Beyaz Dengesi:  {cap.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U):.0f}")
            print(f"AutoFocus:      {autofocus_status}")
            print(f"AutoExposure:   {autoexp_status}")
            print("=" * 60)
    
    # Temizlik
    cap.release()
    cv2.destroyAllWindows()
    print("\nKamera kapatıldı.")

if __name__ == "__main__":
    main()