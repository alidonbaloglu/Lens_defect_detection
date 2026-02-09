"""
data_collector_qt_ensemble.py dosyasına eklenecek değişiklikler
Active Learning entegrasyonu için
"""

# ============================================================
# 1. IMPORT EKLEMELERI (dosyanın başına ekleyin)
# ============================================================

from active_learning_module import (
    UncertaintySampler,
    ActiveLearningDataManager,
    IncrementalTrainer
)


# ============================================================
# 2. GLOBAL CONFIG EKLEMELERI (Config bölümüne ekleyin)
# ============================================================

# Active Learning Ayarları
ENABLE_ACTIVE_LEARNING = True  # AL sistemi aktif/pasif
AL_LOW_CONF_THRESHOLD = 0.30    # Düşük güven eşiği
AL_HIGH_CONF_THRESHOLD = 0.70   # Yüksek güven eşiği
AL_SAMPLE_HIGH_CONF = 0.05      # Yüksek güvenli olanlardan %5 sakla
AL_DATA_ROOT = "active_learning_data"


# ============================================================
# 3. DataCollectorWindow SINIFINA EKLEMELER
# ============================================================

class DataCollectorWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # ... mevcut __init__ kodları ...
        
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
        
        # AL istatistikleri için label ekle (UI'a)
        if ENABLE_ACTIVE_LEARNING:
            self.label_al_stats = QtWidgets.QLabel("AL Kuyruğu: 0")
            self.label_al_stats.setStyleSheet("color: cyan; font-size: 14px;")
            # Bu label'ı uygun bir layouta ekleyin
        
        # ... geri kalan __init__ kodları ...
    
    
    def analyze_batch(self):
        """
        Mevcut analyze_batch fonksiyonunu Active Learning destekli hale getirin.
        Aşağıdaki değişiklikleri yapın:
        """
        if self.current_part_dir is None:
            self.show_banner("Önce Part seçilmeli!", ok=False)
            return

        raw_dir = os.path.join(self.current_part_dir, "raw")
        out_dir = os.path.join(self.current_part_dir, "analyzed")
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.isdir(raw_dir):
            self.show_banner("RAW dizini yok!", ok=False)
            return

        files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            self.show_banner("Analiz edilecek görüntü yok!", ok=False)
            return

        total = 0
        failed = 0
        al_saved = 0  # Active Learning'e kaydedilen sayısı

        # CSV metrics
        csv_path = os.path.join(out_dir, "detection_metrics.csv")
        try:
            csv_fh = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_fh)
            csv_writer.writerow(["image_id", "camera_idx", "error_count", "avg_score", "uncertainty_score", "al_queued", "timestamp"])
        except Exception:
            csv_fh = None
            csv_writer = None

        for fname in files:
            fpath = os.path.join(raw_dir, fname)
            try:
                img = cv2.imread(fpath)
                if img is None:
                    failed += 1
                    continue

                # RT-DETR Ensemble analizi
                vis_bgr, ensemble_preds, err_count, has_error = analyze_frame_ensemble(
                    self._scratch_model, self._blackdot_model, img, DEFAULT_CONF_THRESHOLD
                )
                vis_bgr = draw_ok_nok_flag(vis_bgr, has_error)
                
                # ========== ACTIVE LEARNING ENTEGRASYONU ==========
                uncertainty_score = 0.0
                al_queued = False
                
                if ENABLE_ACTIVE_LEARNING and self.al_sampler and self.al_manager:
                    # Belirsizlik analizi
                    should_annotate, reason, uncertainty_score = self.al_sampler.should_annotate(ensemble_preds)
                    
                    if should_annotate:
                        # Metadata hazırla
                        cam_idx = _extract_cam_idx(fname)
                        metadata = {
                            "camera_idx": cam_idx,
                            "filename": fname,
                            "part_dir": self.current_part_dir,
                            "side": self.selected_side,
                            "error_count": err_count,
                            "has_error": has_error
                        }
                        
                        # Active Learning sistemine kaydet
                        file_id = self.al_manager.save_inference_result(
                            image=img,
                            predictions=ensemble_preds,
                            metadata=metadata,
                            uncertainty_info=(should_annotate, reason, uncertainty_score)
                        )
                        al_queued = True
                        al_saved += 1
                        print(f"[AL] Etiketleme kuyruğuna eklendi: {fname} (reason: {reason}, score: {uncertainty_score:.3f})")
                # ================================================
                
                out_name = os.path.splitext(fname)[0] + "_analyzed.png"
                out_path = os.path.join(out_dir, out_name)
                
                if cv2.imwrite(out_path, vis_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
                    total += 1
                    print(f"[OK] Analiz kaydedildi -> {out_name}")
                    
                    cam_idx = _extract_cam_idx(fname)
                    avg_score = 0.0
                    if ensemble_preds:
                        scores_list = [p[2] for p in ensemble_preds if len(p) >= 3]
                        avg_score = float(np.mean(scores_list)) if scores_list else 0.0
                    
                    # CSV'ye yaz (AL bilgisiyle)
                    if csv_writer is not None:
                        try:
                            csv_writer.writerow([
                                os.path.basename(fpath), 
                                cam_idx if cam_idx is not None else "", 
                                err_count, 
                                f"{avg_score:.4f}",
                                f"{uncertainty_score:.4f}",
                                "yes" if al_queued else "no",
                                datetime.utcnow().isoformat()
                            ])
                        except Exception:
                            pass
                    
                    # ... mevcut best_analyzed_per_cam güncellemesi ...
                    if cam_idx is not None:
                        prev = self._best_analyzed_per_cam.get(cam_idx)
                        if prev is None or err_count > prev[1] or (err_count == prev[1] and avg_score > prev[2]):
                            self._best_analyzed_per_cam[cam_idx] = (out_path, err_count, avg_score)
                else:
                    failed += 1
                    print(f"[ERROR] Analiz kaydedilemedi -> {out_name}")

                # Update status
                if total % 5 == 0:
                    status_msg = f"Analiz: {total}/{len(files)}"
                    if ENABLE_ACTIVE_LEARNING:
                        status_msg += f" | AL Kuyruğu: +{al_saved}"
                    self.label_status.setText(status_msg)
                    QtWidgets.QApplication.processEvents()

            except Exception as e:
                failed += 1
                print(f"[ERROR] Analiz sırasında hata ({fname}): {e}")

        # Close CSV
        try:
            if csv_fh is not None:
                csv_fh.close()
        except Exception:
            pass

        # Final mesaj
        msg = f"Analiz: {total} başarılı, {failed} hatalı"
        if ENABLE_ACTIVE_LEARNING:
            msg += f" | AL'ye eklenen: {al_saved}"
        self.show_banner(msg, ok=(total > 0))
        self.label_status.setText(msg)
        
        # AL kuyruğu istatistiklerini güncelle
        if ENABLE_ACTIVE_LEARNING and self.al_manager:
            stats = self.al_manager.get_annotation_queue_stats()
            self.label_al_stats.setText(f"AL Kuyruğu: {stats['count']}")

        # ... geri kalan mevcut kod (best_analyzed gösterimi vb.) ...


# ============================================================
# 4. YENİ BUTONLAR VE MENÜ ÖĞELERİ EKLEYIN
# ============================================================

    def init_ui_with_al_buttons(self):
        """
        Mevcut init_ui fonksiyonuna Active Learning butonları ekleyin.
        Sol menüye veya uygun bir yere ekleyebilirsiniz.
        """
        
        if ENABLE_ACTIVE_LEARNING:
            # AL İstatistikleri butonu
            btn_al_stats = QtWidgets.QPushButton("AL İstatistikleri")
            btn_al_stats.clicked.connect(self.show_al_statistics)
            # Bu butonu uygun layouta ekleyin
            
            # Etiketleme arayüzü butonu
            btn_annotate = QtWidgets.QPushButton("Etiketleme Arayüzü")
            btn_annotate.clicked.connect(self.open_annotation_interface)
            # Bu butonu uygun layouta ekleyin
            
            # YOLO dönüşüm butonu
            btn_convert = QtWidgets.QPushButton("YOLO Formatına Dönüştür")
            btn_convert.clicked.connect(self.convert_to_yolo)
            # Bu butonu uygun layouta ekleyin
            
            # Artımlı eğitim butonu
            btn_train = QtWidgets.QPushButton("Artımlı Eğitim Başlat")
            btn_train.clicked.connect(self.start_incremental_training)
            # Bu butonu uygun layouta ekleyin
    
    
    def show_al_statistics(self):
        """Active Learning istatistiklerini göster"""
        if not self.al_manager:
            return
        
        stats = self.al_manager.get_annotation_queue_stats()
        
        msg = f"Active Learning İstatistikleri\n\n"
        msg += f"Etiketleme Kuyruğu: {stats['count']} görüntü\n\n"
        msg += "Belirsizlik Nedenleri:\n"
        for reason, count in stats['reasons'].items():
            msg += f"  • {reason}: {count}\n"
        
        QtWidgets.QMessageBox.information(self, "AL İstatistikleri", msg)
    
    
    def open_annotation_interface(self):
        """
        Basit etiketleme arayüzünü aç.
        Daha gelişmiş bir uygulama için CVAT veya LabelImg entegrasyonu yapılabilir.
        """
        msg = "Etiketleme arayüzü açılıyor...\n\n"
        msg += f"Dizin: {AL_DATA_ROOT}/annotation_queue/\n\n"
        msg += "Bu dizindeki görüntüleri CVAT, LabelImg veya başka bir araçla etiketleyebilirsiniz.\n"
        msg += "Etiketlenmiş dosyaları 'annotated' klasörüne taşıyın."
        
        QtWidgets.QMessageBox.information(self, "Etiketleme", msg)
        
        # Dizini aç
        queue_path = os.path.join(AL_DATA_ROOT, "annotation_queue")
        if os.path.exists(queue_path):
            os.startfile(queue_path)  # Windows için
            # Linux için: subprocess.call(['xdg-open', queue_path])
    
    
    def convert_to_yolo(self):
        """Etiketlenmiş verileri YOLO formatına dönüştür"""
        if not self.al_manager:
            return
        
        reply = QtWidgets.QMessageBox.question(
            self, 
            "YOLO Dönüşümü",
            "Etiketlenmiş veriler YOLO formatına dönüştürülecek. Devam edilsin mi?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            try:
                count = self.al_manager.convert_annotations_to_yolo(ENSEMBLE_CLASS_NAMES)
                msg = f"Başarılı: {count} dosya YOLO formatına dönüştürüldü.\n\n"
                msg += f"Dizin: {AL_DATA_ROOT}/training_ready/"
                QtWidgets.QMessageBox.information(self, "Başarılı", msg)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Hata", f"Dönüşüm hatası: {e}")
    
    
    def start_incremental_training(self):
        """Artımlı eğitim başlat"""
        msg = "Artımlı Eğitim Parametreleri\n\n"
        msg += "Bu işlem uzun sürebilir ve GPU gerektirir.\n"
        msg += "Devam etmek istediğinizden emin misiniz?"
        
        reply = QtWidgets.QMessageBox.question(
            self,
            "Artımlı Eğitim",
            msg,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            # Eğitim dialog'u açabilir veya doğrudan başlatabilirsiniz
            self.show_training_dialog()
    
    
    def show_training_dialog(self):
        """Eğitim parametrelerini almak için dialog"""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Artımlı Eğitim Ayarları")
        dialog.setModal(True)
        
        layout = QtWidgets.QFormLayout()
        
        # Parametreler
        epochs_spin = QtWidgets.QSpinBox()
        epochs_spin.setRange(10, 500)
        epochs_spin.setValue(50)
        layout.addRow("Epochs:", epochs_spin)
        
        batch_spin = QtWidgets.QSpinBox()
        batch_spin.setRange(1, 64)
        batch_spin.setValue(16)
        layout.addRow("Batch Size:", batch_spin)
        
        lr_spin = QtWidgets.QDoubleSpinBox()
        lr_spin.setRange(0.00001, 0.01)
        lr_spin.setValue(0.0001)
        lr_spin.setDecimals(5)
        lr_spin.setSingleStep(0.00001)
        layout.addRow("Learning Rate:", lr_spin)
        
        old_data_ratio = QtWidgets.QDoubleSpinBox()
        old_data_ratio.setRange(0.0, 1.0)
        old_data_ratio.setValue(0.3)
        old_data_ratio.setDecimals(2)
        old_data_ratio.setSingleStep(0.05)
        layout.addRow("Eski Veri Oranı:", old_data_ratio)
        
        # Butonlar
        btn_layout = QtWidgets.QHBoxLayout()
        btn_start = QtWidgets.QPushButton("Başlat")
        btn_cancel = QtWidgets.QPushButton("İptal")
        btn_layout.addWidget(btn_start)
        btn_layout.addWidget(btn_cancel)
        
        layout.addRow(btn_layout)
        dialog.setLayout(layout)
        
        def start_training():
            dialog.accept()
            # Eğitimi ayrı bir thread'de başlat
            self.run_training_async(
                epochs=epochs_spin.value(),
                batch=batch_spin.value(),
                lr=lr_spin.value(),
                old_ratio=old_data_ratio.value()
            )
        
        btn_start.clicked.connect(start_training)
        btn_cancel.clicked.connect(dialog.reject)
        
        dialog.exec_()
    
    
    def run_training_async(self, epochs: int, batch: int, lr: float, old_ratio: float):
        """
        Eğitimi asenkron olarak çalıştır.
        Gerçek uygulamada QThread kullanın.
        """
        try:
            # Eski veri seti yolu (mevcut eğitim setiniz)
            old_dataset = "path/to/your/old/dataset"  # BURAYA KENDİ YOLUNUZU YAZIN
            new_dataset = os.path.join(AL_DATA_ROOT, "training_ready")
            merged_output = os.path.join(AL_DATA_ROOT, "merged_dataset")
            
            # Trainer oluştur
            trainer = IncrementalTrainer(
                old_dataset_path=old_dataset,
                new_dataset_path=new_dataset,
                old_data_ratio=old_ratio
            )
            
            # Veri setlerini birleştir
            self.label_status.setText("Veri setleri birleştiriliyor...")
            merged_path = trainer.merge_datasets(merged_output)
            
            # Eğitimi başlat
            self.label_status.setText(f"Eğitim başlatılıyor... (epochs={epochs}, batch={batch}, lr={lr})")
            
            # Scratch model için
            trainer.train_model(
                base_model_path=SCRATCH_MODEL_PATH,
                merged_dataset_path=str(merged_path),
                output_dir=os.path.join(AL_DATA_ROOT, "models", "scratch"),
                epochs=epochs,
                batch=batch,
                lr0=lr
            )
            
            # BlackDot model için
            trainer.train_model(
                base_model_path=BLACKDOT_MODEL_PATH,
                merged_dataset_path=str(merged_path),
                output_dir=os.path.join(AL_DATA_ROOT, "models", "blackdot"),
                epochs=epochs,
                batch=batch,
                lr0=lr
            )
            
            self.label_status.setText("Eğitim tamamlandı!")
            QtWidgets.QMessageBox.information(
                self,
                "Başarılı",
                "Artımlı eğitim tamamlandı. Yeni modeller kaydedildi."
            )
            
        except Exception as e:
            self.label_status.setText(f"Eğitim hatası: {e}")
            QtWidgets.QMessageBox.critical(self, "Hata", f"Eğitim hatası: {e}")


# ============================================================
# 5. HELPER FUNCTION: KAMERA İNDEX ÇIKARTMA
# ============================================================

def _extract_cam_idx(filename: str) -> Optional[int]:
    """
    Dosya adından kamera index'ini çıkarır.
    Örnek: cam_2_focus_30_00123.png -> 2
    """
    try:
        if "cam_" in filename:
            parts = filename.split("cam_")[1].split("_")
            return int(parts[0])
    except:
        pass
    return None
