import cv2
import numpy as np
import threading
import time

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

# Ekranda gösterilecek pencere çözünürlüğü (capture ayrı, sadece gösterim için)
DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080

# Açılacak kamera indeksleri (örn. [0, 1, 2])
CAMERA_INDICES = [0,1,2,3]

# Kamera bazlı başlangıç ayarları (yalnızca farklı olacak alanları belirtin)
# Örnek:
# CAMERA_SETTINGS_PER_CAMERA = {
#     0: { 'exposure': 120, 'focus': 40 },
#     1: { 'exposure': 200, 'autofocus': True },
# }
CAMERA_SETTINGS_PER_CAMERA = {
    2: {
        'width': 3240,              # Çözünürlük genişlik
        'height': 2160,             # Çözünürlük yükseklik
        'brightness': 128,
        'contrast': 128,
        'saturation': 128,
        'focus': 20,
        'sharpness': 255,
        'white_balance': 3082,
    },
    0: {
        'brightness': 128,
        'contrast': 155,
        'saturation': 124,
        'focus': 30,
        'sharpness': 255,
        'white_balance': 3025,
    },
    1: {
        'brightness': 129,
        'contrast': 154,
        'saturation': 127,
        'focus': 30,
        'sharpness': 255,
        'white_balance': 3025,
    },
    3: {
        'brightness': 101,
        'contrast': 128,
        'saturation': 158,
        'focus': 30,
        'sharpness': 255,
        'white_balance': 3053,
    }
}

# ============================================================================

# Global değişkenler
caps = {}
active_cam_idx = None
latest_frames = {}
frame_locks = {}

def get_active_cap():
    if active_cam_idx is None:
        return None
    return caps.get(active_cam_idx)

def get_settings_for(camera_idx):
    settings = dict(CAMERA_SETTINGS)
    overrides = CAMERA_SETTINGS_PER_CAMERA.get(camera_idx, {})
    settings.update(overrides)
    return settings

def _reader_thread(camera_idx, cap):
    lock = frame_locks[camera_idx]
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.005)
            continue
        with lock:
            latest_frames[camera_idx] = frame

# Trackbar callback fonksiyonları
def set_brightness(val):
    cap = get_active_cap()
    if cap is not None:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, val)

def set_contrast(val):
    cap = get_active_cap()
    if cap is not None:
        cap.set(cv2.CAP_PROP_CONTRAST, val)

def set_saturation(val):
    cap = get_active_cap()
    if cap is not None:
        cap.set(cv2.CAP_PROP_SATURATION, val)

def set_hue(val):
    cap = get_active_cap()
    if cap is not None:
        cap.set(cv2.CAP_PROP_HUE, val)

def set_gain(val):
    cap = get_active_cap()
    if cap is not None:
        cap.set(cv2.CAP_PROP_GAIN, val)

def set_exposure(val):
    cap = get_active_cap()
    if cap is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, val)

def set_focus(val):
    cap = get_active_cap()
    if cap is not None:
        cap.set(cv2.CAP_PROP_FOCUS, val)

def set_sharpness(val):
    cap = get_active_cap()
    if cap is not None:
        cap.set(cv2.CAP_PROP_SHARPNESS, val)

def set_white_balance(val):
    cap = get_active_cap()
    if cap is not None:
        cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, val)

def toggle_autofocus(val):
    cap = get_active_cap()
    if cap is not None:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, val)

def toggle_auto_exposure(val):
    cap = get_active_cap()
    if cap is None:
        return
    if val == 0:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manuel
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # Otomatik

def apply_camera_settings(cap, settings):
    """Sabit kamera ayarlarını bir capture'a uygula"""
    if cap is None:
        return
    # Çözünürlük ve FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['height'])
    cap.set(cv2.CAP_PROP_FPS, settings['fps'])
    # Otomatik özellikleri ayarla
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if settings['autofocus'] else 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if settings['auto_exposure'] else 1)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1 if settings['auto_white_balance'] else 0)
    # Manuel ayarlar (otomatik modlar kapalıysa)
    if not settings['autofocus']:
        cap.set(cv2.CAP_PROP_FOCUS, settings['focus'])
    if not settings['auto_exposure']:
        cap.set(cv2.CAP_PROP_EXPOSURE, settings['exposure'])
    if not settings['auto_white_balance']:
        cap.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, settings['white_balance'])
    # Diğer ayarlar
    cap.set(cv2.CAP_PROP_BRIGHTNESS, settings['brightness'])
    cap.set(cv2.CAP_PROP_CONTRAST, settings['contrast'])
    cap.set(cv2.CAP_PROP_SATURATION, settings['saturation'])
    cap.set(cv2.CAP_PROP_HUE, settings['hue'])
    cap.set(cv2.CAP_PROP_GAIN, settings['gain'])
    cap.set(cv2.CAP_PROP_SHARPNESS, settings['sharpness'])

def main():
    global caps, active_cam_idx
    
    # Kameraları başlat
    for idx in CAMERA_INDICES:
        # Windows'ta DirectShow backend ile açmayı dene
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        except Exception:
            cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            print(f"Hata: Kamera açılamadı! (index={idx})")
            continue
        # MJPG fourcc ve küçük buffer ile gecikmeyi azalt
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass
        if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        apply_camera_settings(cap, get_settings_for(idx))
        # Isınma: birkaç frame oku
        for _ in range(5):
            cap.read()
            time.sleep(0.005)
        caps[idx] = cap
        frame_locks[idx] = threading.Lock()
        latest_frames[idx] = None
    
    # Okuyucu thread'leri başlat
    for idx, cap in caps.items():
        t = threading.Thread(target=_reader_thread, args=(idx, cap), daemon=True)
        t.start()
    
    if not caps:
        print("Hata: Hiçbir kamera açılamadı!")
        return
    
    # Aktif kamerayı seç (ilk mevcut)
    active_cam_idx = sorted(caps.keys())[0]
    
    # Pencereleri oluştur (pencere boyutunu gösterim için küçült)
    for idx in caps.keys():
        cv2.namedWindow(f'Kamera {idx}', cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow(f'Kamera {idx}', DISPLAY_WIDTH, DISPLAY_HEIGHT)
        except Exception:
            pass
    
    # Trackbar'ları oluştur (eğer aktifse)
    if USE_TRACKBARS:
        cv2.namedWindow('Ayarlar', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Ayarlar', 400, 400)
        s = get_settings_for(active_cam_idx)
        cv2.createTrackbar('Parlak', 'Ayarlar', s['brightness'], 255, set_brightness)
        cv2.createTrackbar('Kontrast', 'Ayarlar', s['contrast'], 255, set_contrast)
        cv2.createTrackbar('Doygun', 'Ayarlar', s['saturation'], 255, set_saturation)
        cv2.createTrackbar('Renk', 'Ayarlar', s['hue'], 180, set_hue)
        cv2.createTrackbar('Kazanc', 'Ayarlar', s['gain'], 255, set_gain)
        cv2.createTrackbar('Pozlama', 'Ayarlar', s['exposure'], 2000, set_exposure)
        cv2.createTrackbar('Focus', 'Ayarlar', s['focus'], 255, set_focus)
        cv2.createTrackbar('Keskin', 'Ayarlar', s['sharpness'], 255, set_sharpness)
        cv2.createTrackbar('B.Denge', 'Ayarlar', s['white_balance'], 7000, set_white_balance)
        cv2.createTrackbar('AutoF', 'Ayarlar', 1 if s['autofocus'] else 0, 1, toggle_autofocus)
        cv2.createTrackbar('AutoE', 'Ayarlar', 1 if s['auto_exposure'] else 0, 1, toggle_auto_exposure)
    
    print("\n" + "=" * 60)
    print("LOGITECH C920 PRO - MANUEL KAMERA KONTROLLERI (ÇOKLU)")
    print("=" * 60)
    if USE_TRACKBARS:
        print("Trackbar'ları kullanarak kamera ayarlarını değiştirebilirsiniz")
    else:
        print("Trackbar'lar kapalı - Sadece kod ayarları kullanılıyor")
    print("\nKısayollar:")
    print("  'q' - Çıkış")
    print("  'r' - Aktif kamerada kod ayarlarına dön")
    print("  's' - Aktif kameradan anlık görüntü kaydet")
    print("  'TAB' - Aktif kamerayı değiştir")
    print("  '0-9' - Belirli kamerayı aktif yap")
    print("  'i' - Mevcut ayarları göster")
    print("=" * 60 + "\n")
    
    frame_count = {idx: 0 for idx in caps.keys()}
    
    while True:
        any_ok = False
        for idx, cap in list(caps.items()):
            with frame_locks[idx]:
                frame = latest_frames.get(idx)
            if frame is None:
                continue
            any_ok = True
            frame_count[idx] += 1
            # Bilgi metni ekle
            info_text = f"Cam {idx}"
            cv2.putText(frame, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Otomatik mod durumu
            autofocus_status = "ACIK" if cap.get(cv2.CAP_PROP_AUTOFOCUS) else "KAPALI"
            autoexp_status = "ACIK" if cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) == 3 else "KAPALI"
            status_text = f"AutoFocus: {autofocus_status} | AutoExposure: {autoexp_status}"
            if idx == active_cam_idx:
                status_text += " | ACTIVE"
            cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            # Orijinal çözünürlüğü koru for screenshots; gösterim için yeniden boyutlandır
            try:
                disp_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)
            except Exception:
                disp_frame = frame
            cv2.imshow(f'Kamera {idx}', disp_frame)
        if not any_ok:
            print("Hata: Kameralardan frame okunamadı!")
            break
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' tuşu - Çıkış
        if key == ord('q'):
            break
        
        # 'r' tuşu - Kod ayarlarına dön
        elif key == ord('r'):
            s = get_settings_for(active_cam_idx)
            apply_camera_settings(get_active_cap(), s)
            if USE_TRACKBARS:
                cv2.setTrackbarPos('Parlak', 'Ayarlar', s['brightness'])
                cv2.setTrackbarPos('Kontrast', 'Ayarlar', s['contrast'])
                cv2.setTrackbarPos('Doygun', 'Ayarlar', s['saturation'])
                cv2.setTrackbarPos('Renk', 'Ayarlar', s['hue'])
                cv2.setTrackbarPos('Kazanc', 'Ayarlar', s['gain'])
                cv2.setTrackbarPos('Pozlama', 'Ayarlar', s['exposure'])
                cv2.setTrackbarPos('Focus', 'Ayarlar', s['focus'])
                cv2.setTrackbarPos('Keskin', 'Ayarlar', s['sharpness'])
                cv2.setTrackbarPos('B.Denge', 'Ayarlar', s['white_balance'])
                cv2.setTrackbarPos('AutoF', 'Ayarlar', 1 if s['autofocus'] else 0)
                cv2.setTrackbarPos('AutoE', 'Ayarlar', 1 if s['auto_exposure'] else 0)
        # TAB ile aktif kamerayı değiştir
        elif key == 9:  # TAB
            keys_sorted = sorted(caps.keys())
            cur_idx = keys_sorted.index(active_cam_idx)
            active_cam_idx = keys_sorted[(cur_idx + 1) % len(keys_sorted)]
            print(f"Aktif kamera değişti: {active_cam_idx}")
            if USE_TRACKBARS:
                s = get_settings_for(active_cam_idx)
                cv2.setTrackbarPos('Parlak', 'Ayarlar', s['brightness'])
                cv2.setTrackbarPos('Kontrast', 'Ayarlar', s['contrast'])
                cv2.setTrackbarPos('Doygun', 'Ayarlar', s['saturation'])
                cv2.setTrackbarPos('Renk', 'Ayarlar', s['hue'])
                cv2.setTrackbarPos('Kazanc', 'Ayarlar', s['gain'])
                cv2.setTrackbarPos('Pozlama', 'Ayarlar', s['exposure'])
                cv2.setTrackbarPos('Focus', 'Ayarlar', s['focus'])
                cv2.setTrackbarPos('Keskin', 'Ayarlar', s['sharpness'])
                cv2.setTrackbarPos('B.Denge', 'Ayarlar', s['white_balance'])
                cv2.setTrackbarPos('AutoF', 'Ayarlar', 1 if s['autofocus'] else 0)
                cv2.setTrackbarPos('AutoE', 'Ayarlar', 1 if s['auto_exposure'] else 0)
        # 0-9 ile doğrudan seçim
        elif key in [ord(str(d)) for d in range(10)]:
            target = int(chr(key))
            if target in caps:
                active_cam_idx = target
                print(f"Aktif kamera değişti: {active_cam_idx}")
                if USE_TRACKBARS:
                    s = get_settings_for(active_cam_idx)
                    cv2.setTrackbarPos('Parlak', 'Ayarlar', s['brightness'])
                    cv2.setTrackbarPos('Kontrast', 'Ayarlar', s['contrast'])
                    cv2.setTrackbarPos('Doygun', 'Ayarlar', s['saturation'])
                    cv2.setTrackbarPos('Renk', 'Ayarlar', s['hue'])
                    cv2.setTrackbarPos('Kazanc', 'Ayarlar', s['gain'])
                    cv2.setTrackbarPos('Pozlama', 'Ayarlar', s['exposure'])
                    cv2.setTrackbarPos('Focus', 'Ayarlar', s['focus'])
                    cv2.setTrackbarPos('Keskin', 'Ayarlar', s['sharpness'])
                    cv2.setTrackbarPos('B.Denge', 'Ayarlar', s['white_balance'])
                    cv2.setTrackbarPos('AutoF', 'Ayarlar', 1 if s['autofocus'] else 0)
                    cv2.setTrackbarPos('AutoE', 'Ayarlar', 1 if s['auto_exposure'] else 0)
        
        # 's' tuşu - Screenshot
        elif key == ord('s'):
            # Kaydedilecek kare: en son okunan orijinal çözünürlüklü frame
            with frame_locks[active_cam_idx]:
                orig_frame = latest_frames.get(active_cam_idx)
            if orig_frame is not None:
                filename = f"screenshot_cam{active_cam_idx}_{frame_count[active_cam_idx]}.jpg"
                cv2.imwrite(filename, orig_frame)
                print(f"✓ Görüntü kaydedildi: {filename}")
        
        # 'i' tuşu - Info
        elif key == ord('i'):
            cap = get_active_cap()
            if cap is not None:
                autofocus_status = "ACIK" if cap.get(cv2.CAP_PROP_AUTOFOCUS) else "KAPALI"
                autoexp_status = "ACIK" if cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) == 3 else "KAPALI"
                print("\n" + "=" * 60)
                print(f"MEVCUT KAMERA AYARLARI (Aktif: {active_cam_idx}):")
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
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Kameralar kapatıldı.")

if __name__ == "__main__":
    main()