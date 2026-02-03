import cv2
import numpy as np

# ============================================================================
# KAMERA AYARLARI - Buradan istediğiniz değerleri sabit olarak ayarlayın
# ============================================================================
CAMERA_SETTINGS = {
    'width': 1920,              # Çözünürlük genişlik
    'height': 1080,             # Çözünürlük yükseklik
    'fps': 30,                  # Frame rate
    'brightness': 128,          # Parlaklık (0-255)
    'contrast': 128,            # Kontrast (0-255)
    'saturation': 128,          # Doygunluk (0-255)
    'hue': 0,                   # Renk tonu (0-180)
    'gain': 50,                 # Kazanç (0-255)
    'exposure': 150,            # Pozlama (-13 to -1 veya pozitif değerler)
    'focus': 30,                # Focus mesafesi (0-255, düşük=yakın, yüksek=uzak)
    'sharpness': 128,           # Keskinlik (0-255)
    'white_balance': 4500,      # Beyaz dengesi (2800-6500)
    'autofocus': False,         # True=Otomatik, False=Manuel
    'auto_exposure': False,     # True=Otomatik, False=Manuel
    'auto_white_balance': False # True=Otomatik, False=Manuel
}

# Trackbar kullanımı (True=Açık, False=Kapalı)
USE_TRACKBARS = True

# ============================================================================

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
    cap.set(cv2.CAP_PROP_AUTOFOCUS, val)

def toggle_auto_exposure(val):
    if val == 0:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manuel
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # Otomatik

def apply_camera_settings():
    """Sabit kamera ayarlarını uygula"""
    
    # Çözünürlük ve FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SETTINGS['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SETTINGS['height'])
    cap.set(cv2.CAP_PROP_FPS, CAMERA_SETTINGS['fps'])
    
    # Otomatik özellikleri ayarla
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if CAMERA_SETTINGS['autofocus'] else 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if CAMERA_SETTINGS['auto_exposure'] else 1)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1 if CAMERA_SETTINGS['auto_white_balance'] else 0)
    
    # Manuel ayarlar (otomatik modlar kapalıysa)
    if not CAMERA_SETTINGS['autofocus']:
        cap.set(cv2.CAP_PROP_FOCUS, CAMERA_SETTINGS['focus'])
    
    if not CAMERA_SETTINGS['auto_exposure']:
        cap.set(cv2.CAP_PROP_EXPOSURE, CAMERA_SETTINGS['exposure'])
    
    if not CAMERA_SETTINGS['auto_white_balance']:
        cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, CAMERA_SETTINGS['white_balance'])
    
    # Diğer ayarlar
    cap.set(cv2.CAP_PROP_BRIGHTNESS, CAMERA_SETTINGS['brightness'])
    cap.set(cv2.CAP_PROP_CONTRAST, CAMERA_SETTINGS['contrast'])
    cap.set(cv2.CAP_PROP_SATURATION, CAMERA_SETTINGS['saturation'])
    cap.set(cv2.CAP_PROP_HUE, CAMERA_SETTINGS['hue'])
    cap.set(cv2.CAP_PROP_GAIN, CAMERA_SETTINGS['gain'])
    cap.set(cv2.CAP_PROP_SHARPNESS, CAMERA_SETTINGS['sharpness'])
    
    print("\n✓ Kamera ayarları uygulandı:")
    print(f"  Çözünürlük: {CAMERA_SETTINGS['width']}x{CAMERA_SETTINGS['height']} @ {CAMERA_SETTINGS['fps']} FPS")
    print(f"  Parlaklık: {CAMERA_SETTINGS['brightness']}")
    print(f"  Kontrast: {CAMERA_SETTINGS['contrast']}")
    print(f"  Focus: {CAMERA_SETTINGS['focus']} (AutoFocus: {'AÇIK' if CAMERA_SETTINGS['autofocus'] else 'KAPALI'})")
    print(f"  Pozlama: {CAMERA_SETTINGS['exposure']} (AutoExposure: {'AÇIK' if CAMERA_SETTINGS['auto_exposure'] else 'KAPALI'})")

def main():
    global cap
    
    # Kamerayı başlat
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Hata: Kamera açılamadı!")
        return
    
    # Sabit ayarları uygula
    apply_camera_settings()
    
    # Pencereyi oluştur
    cv2.namedWindow('Kamera Goruntüsu')
    
    # Trackbar'ları oluştur (eğer aktifse)
    if USE_TRACKBARS:
        cv2.namedWindow('Ayarlar', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Ayarlar', 400, 400)
        cv2.createTrackbar('Parlak', 'Ayarlar', CAMERA_SETTINGS['brightness'], 255, set_brightness)
        cv2.createTrackbar('Kontrast', 'Ayarlar', CAMERA_SETTINGS['contrast'], 255, set_contrast)
        cv2.createTrackbar('Doygun', 'Ayarlar', CAMERA_SETTINGS['saturation'], 255, set_saturation)
        cv2.createTrackbar('Renk', 'Ayarlar', CAMERA_SETTINGS['hue'], 180, set_hue)
        cv2.createTrackbar('Kazanc', 'Ayarlar', CAMERA_SETTINGS['gain'], 255, set_gain)
        cv2.createTrackbar('Pozlama', 'Ayarlar', CAMERA_SETTINGS['exposure'], 2000, set_exposure)
        cv2.createTrackbar('Focus', 'Ayarlar', CAMERA_SETTINGS['focus'], 255, set_focus)
        cv2.createTrackbar('Keskin', 'Ayarlar', CAMERA_SETTINGS['sharpness'], 255, set_sharpness)
        cv2.createTrackbar('B.Denge', 'Ayarlar', CAMERA_SETTINGS['white_balance'], 7000, set_white_balance)
        cv2.createTrackbar('AutoF', 'Ayarlar', 1 if CAMERA_SETTINGS['autofocus'] else 0, 1, toggle_autofocus)
        cv2.createTrackbar('AutoE', 'Ayarlar', 1 if CAMERA_SETTINGS['auto_exposure'] else 0, 1, toggle_auto_exposure)
    
    print("\n" + "=" * 60)
    print("LOGITECH C920 PRO - MANUEL KAMERA KONTROLLERI")
    print("=" * 60)
    if USE_TRACKBARS:
        print("Trackbar'ları kullanarak kamera ayarlarını değiştirebilirsiniz")
    else:
        print("Trackbar'lar kapalı - Sadece kod ayarları kullanılıyor")
    print("\nKısayollar:")
    print("  'q' - Çıkış")
    print("  'r' - Kod ayarlarına dön")
    print("  's' - Anlık görüntü kaydet")
    print("  'i' - Mevcut ayarları göster")
    print("=" * 60 + "\n")
    
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
        
        # 'r' tuşu - Kod ayarlarına dön
        elif key == ord('r'):
            apply_camera_settings()
            if USE_TRACKBARS:
                cv2.setTrackbarPos('Parlak', 'Ayarlar', CAMERA_SETTINGS['brightness'])
                cv2.setTrackbarPos('Kontrast', 'Ayarlar', CAMERA_SETTINGS['contrast'])
                cv2.setTrackbarPos('Doygun', 'Ayarlar', CAMERA_SETTINGS['saturation'])
                cv2.setTrackbarPos('Renk', 'Ayarlar', CAMERA_SETTINGS['hue'])
                cv2.setTrackbarPos('Kazanc', 'Ayarlar', CAMERA_SETTINGS['gain'])
                cv2.setTrackbarPos('Pozlama', 'Ayarlar', CAMERA_SETTINGS['exposure'])
                cv2.setTrackbarPos('Focus', 'Ayarlar', CAMERA_SETTINGS['focus'])
                cv2.setTrackbarPos('Keskin', 'Ayarlar', CAMERA_SETTINGS['sharpness'])
                cv2.setTrackbarPos('B.Denge', 'Ayarlar', CAMERA_SETTINGS['white_balance'])
                cv2.setTrackbarPos('AutoF', 'Ayarlar', 1 if CAMERA_SETTINGS['autofocus'] else 0)
                cv2.setTrackbarPos('AutoE', 'Ayarlar', 1 if CAMERA_SETTINGS['auto_exposure'] else 0)
        
        # 's' tuşu - Screenshot
        elif key == ord('s'):
            filename = f"screenshot_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ Görüntü kaydedildi: {filename}")
        
        # 'i' tuşu - Info
        elif key == ord('i'):
            print("\n" + "=" * 60)
            print("MEVCUT KAMERA AYARLARI:")
            print("=" * 60)
            print(f"Çözünürlük:     {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print(f"FPS:            {cap.get(cv2.CAP_PROP_FPS):.0f}")
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
            print("=" * 60 + "\n")
    
    # Temizlik
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Kamera kapatıldı.")

if __name__ == "__main__":
    main()