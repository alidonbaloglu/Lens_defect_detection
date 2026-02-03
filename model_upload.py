from roboflow import Roboflow

# 1. Giriş Yapın
# API Anahtarınızı Roboflow ayarlarından almalısınız.
rf = Roboflow(api_key="6S2m6JHTsTjsMB4oNnYe")

# 2. Çalışma Alanını (Workspace) Seçin
# Roboflow URL'nizdeki isimdir (örn: app.roboflow.com/kullanici-adi/...)
workspace = rf.workspace("ali-3cyhu")

# 3. Modeli Yükleyin (Versionless Deploy)
workspace.deploy_model(
    model_type="yolov11",           # Kullandığınız model tipi (yolov8 veya yolov11)
    model_path="C:/Users/ali.donbaloglu/Desktop/Montaj_proces/Modeller/Makale/Yolov11/M_model/runs/detect/train/weights", # 'best.pt' dosyasının bulunduğu KLASÖR yolu
    project_ids=["montaj_deneme-sppbx"],       # Modeli hangi projeye eklemek istiyorsunuz?
    model_name="montaj-model",   # Modele vereceğiniz isim
    filename="best.pt"              # Ağırlık dosyasının adı (varsayılan best.pt)
)