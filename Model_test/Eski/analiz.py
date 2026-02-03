import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- AYARLAR ---
# Excel dosyanızın adını buraya yazın.
EXCEL_DOSYA_ADI = 'hibrit_iou_test_sonuclari.xlsx' # Dosya adını kendinizinkiyle değiştirin
# -----------------

def final_analiz_raporu_olustur(dosya_adi):
    """
    Yeni sütun yapısına sahip Excel dosyasını analiz eder ve 3 adet performans grafiği oluşturur.
    """
    try:
        # Yeni ve tam sütun isimlerini manuel olarak tanımlıyoruz.
        sutun_isimleri = ['Gerçek Etiket', 'MaskRCNN Sonucu', 'YOLO Sonucu', 'Nihai Sonuç']
        
        df = pd.read_excel(
            dosya_adi, 
            header=None, 
            names=sutun_isimleri,
            skiprows=1
        )
        print(f"'{dosya_adi}' dosyası başarıyla yüklendi.")

    except FileNotFoundError:
        print(f"HATA: '{dosya_adi}' dosyası bulunamadı. Lütfen dosyanın script ile aynı klasörde olduğundan emin olun.")
        return
    except Exception as e:
        print(f"Dosya okunurken bir hata oluştu: {e}")
        return

    # --- Grafik 1: Sınıf Bazında Doğru Tahmin Sayısı ---
    dogru_tahminler = df[df['Nihai Sonuç'] == 'Doğru']
    if not dogru_tahminler.empty:
        dogru_tahmin_sayilari = dogru_tahminler['Gerçek Etiket'].value_counts()
        plt.figure(figsize=(12, 7))
        sns.barplot(x=dogru_tahmin_sayilari.index, y=dogru_tahmin_sayilari.values, palette='Greens_r')
        plt.title('Sınıf Bazında Doğru Tahmin Sayıları', fontsize=16)
        plt.xlabel('Gerçek Sınıf Etiketi', fontsize=12)
        plt.ylabel('Doğru Tahmin Sayısı', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        for i, v in enumerate(dogru_tahmin_sayilari.values):
            plt.text(i, v + 0.5, str(v), ha='center', color='black', fontweight='bold')
        plt.tight_layout()
        plt.savefig('dogru_tahmin_sayilari.png')
        print("'dogru_tahmin_sayilari.png' grafiği oluşturuldu.")
    else:
        print("Grafik 1 (Doğru Tahmin Sayıları) için yeterli 'Doğru' veri bulunamadı.")


    # --- Grafik 2: Karışıklık Matrisi ---
    # 'YOLO Sonucu' sütunu olduğu için bu grafik artık mümkün.
    clean_df = df.copy()
    clean_df.dropna(subset=['Gerçek Etiket', 'YOLO Sonucu'], inplace=True)
    unwanted_labels = ["Etiket Dosyası Yok", "Etiket Boş", "Etiket Formatı Hatalı"]
    clean_df = clean_df[~clean_df['Gerçek Etiket'].isin(unwanted_labels)]

    if not clean_df.empty:
        # Gerçek Etiket (x-ekseni) ve YOLO Sonucu (y-ekseni) karşılaştırması
        confusion_matrix = pd.crosstab(
            clean_df['YOLO Sonucu'],
            clean_df['Gerçek Etiket']
        )
        plt.figure(figsize=(14, 10))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)
        plt.title('Karışıklık Matrisi', fontsize=16)
        plt.xlabel('Gerçek Etiket', fontsize=12)
        plt.ylabel('Modelin Tahmini (YOLO Sonucu)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('karisiklik_matrisi.png')
        print("'karisiklik_matrisi.png' grafiği oluşturuldu.")
    else:
        print("Grafik 2 (Karışıklık Matrisi) için yeterli veri bulunamadı.")

    # --- Grafik 3: Her Sınıf İçin Sonuç Dağılımı (Sayı Etiketli) ---
    if 'Gerçek Etiket' in df.columns and not df['Gerçek Etiket'].dropna().empty:
        result_distribution = df.groupby('Gerçek Etiket')['Nihai Sonuç'].value_counts().unstack(fill_value=0)
        
        sutun_sirasi = ['Doğru', 'Yanlış', 'Belirsiz']
        for sutun in sutun_sirasi:
            if sutun not in result_distribution.columns:
                result_distribution[sutun] = 0
        result_distribution = result_distribution[sutun_sirasi]

        result_distribution['toplam'] = result_distribution.sum(axis=1)
        result_distribution = result_distribution.sort_values('toplam', ascending=False).drop('toplam', axis=1)
        
        ax = result_distribution.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='coolwarm')
        
        for container in ax.containers:
            labels = [int(v.get_height()) if v.get_height() > 0 else '' for v in container]
            ax.bar_label(container, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=10)

        plt.title('Her Sınıf İçin Sonuç Dağılımı (Doğru/Yanlış/Belirsiz)', fontsize=16)
        plt.xlabel('Gerçek Sınıf Etiketi', fontsize=12)
        plt.ylabel('Toplam Fotoğraf Sayısı', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Sonuç')
        plt.tight_layout()
        plt.savefig('sonuc_dagilimi.png')
        print("'sonuc_dagilimi.png' grafiği oluşturuldu.")
    else:
        print("Grafik 3 (Sonuç Dağılımı) için yeterli veri bulunamadı.")


if __name__ == '__main__':
    final_analiz_raporu_olustur(EXCEL_DOSYA_ADI)