import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_training_data(json_path):
    """Load training results from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def _parse_duration_to_hours(duration_val):
    """Convert duration (possibly string like HH:MM:SS.s) to hours as float."""
    if isinstance(duration_val, (int, float)):
        # assume seconds
        return float(duration_val) / 3600.0
    if isinstance(duration_val, str):
        try:
            # HH:MM:SS(.micro)
            parts = duration_val.split(':')
            if len(parts) == 3:
                h = float(parts[0])
                m = float(parts[1])
                s = float(parts[2])
                return h + m / 60.0 + s / 3600.0
        except Exception:
            pass
    return None


def extract_epoch_series(data):
    """Extract epoch-wise series from the JSON schema used in Kabin_v4_improved runs."""
    epochs = []
    losses = {
        'loss_classifier': [], 'loss_box_reg': [], 'loss_mask': [],
        'loss_objectness': [], 'loss_rpn_box_reg': [], 'total_loss': []
    }
    val_overall = {'precision': [], 'recall': [], 'f1': [], 'iou': []}
    per_class = {}  # {class_name: {'precision':[], 'recall':[], 'f1':[], 'iou':[]}}
    lr = []
    map_025 = []
    map_50_95 = []
    val_fps = []
    epoch_times = []

    history = data.get('epoch_history', [])
    for ep in history:
        epochs.append(ep.get('epoch'))
        t = ep.get('train_losses', {})
        for k in losses.keys():
            losses[k].append(t.get(k, np.nan))

        # validation overall metrics
        ov = ((ep.get('val_metrics') or {}).get('overall')) or {}
        for k in val_overall.keys():
            val_overall[k].append(ov.get(k, np.nan))

        # per-class metrics (dynamically collect classes)
        pc = ((ep.get('val_metrics') or {}).get('per_class')) or {}
        for cls_name, m in pc.items():
            if cls_name not in per_class:
                per_class[cls_name] = {mk: [] for mk in ['precision', 'recall', 'f1', 'iou']}
            for mk in ['precision', 'recall', 'f1', 'iou']:
                per_class[cls_name][mk].append(m.get(mk, np.nan))
        # ensure all classes align in length for epochs without entries
        for cls_name in per_class.keys():
            if cls_name not in pc:
                for mk in ['precision', 'recall', 'f1', 'iou']:
                    per_class[cls_name][mk].append(np.nan)

        # others
        map_025.append(ep.get('val_map_025', np.nan))
        map_50_95.append(ep.get('val_map_50_95', np.nan))
        val_fps.append(ep.get('val_fps', ((ep.get('val_metrics') or {}).get('inference_fps'))))
        lr.append(ep.get('learning_rate', np.nan))
        epoch_times.append(ep.get('epoch_time', np.nan))

    return {
        'epochs': epochs,
        'losses': losses,
        'val_overall': val_overall,
        'per_class': per_class,
        'lr': lr,
        'map_025': map_025,
        'map_50_95': map_50_95,
        'val_fps': val_fps,
        'epoch_times': epoch_times,
    }


def plot_losses(epochs, losses, save_path=None):
    plt.figure(figsize=(15, 10))

    keys = ['loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg', 'total_loss']
    titles = ['Sınıflandırma Kaybı', 'Kutu Regresyon Kaybı', 'Maske Kaybı', 'Nesne Varlık Kaybı', 'RPN Kutu Regresyon', 'Toplam Kayıp']
    colors = ['b', 'g', 'r', 'm', 'c', 'k']

    for i, (k, title, color) in enumerate(zip(keys, titles, colors), start=1):
        plt.subplot(2, 3, i)
        plt.plot(epochs, losses[k], color+'-', linewidth=2)
        plt.title(title, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Kayıp')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_val_overall_metrics(epochs, val_overall, save_path=None):
    plt.figure(figsize=(14, 8))
    for k, color in zip(['precision', 'recall', 'f1', 'iou'], ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']):
        plt.plot(epochs, val_overall[k], label=k.upper(), linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Skor')
    plt.title('Doğrulama (Validation) Genel Metrikler')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_val_maps(epochs, map_025, map_50_95, save_path=None):
    plt.figure(figsize=(14, 8))
    plt.plot(epochs, map_025, label='mAP@0.25', linewidth=2)
    plt.plot(epochs, map_50_95, label='mAP@50:95', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('mAP Gelişimi')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_lr_and_fps(epochs, lr, fps, save_path=None):
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax1.plot(epochs, lr, 'tab:purple', label='Öğrenme Oranı (LR)', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('LR', color='tab:purple')
    ax1.tick_params(axis='y', labelcolor='tab:purple')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, fps, 'tab:gray', label='Val FPS', linewidth=2)
    ax2.set_ylabel('FPS', color='tab:gray')
    ax2.tick_params(axis='y', labelcolor='tab:gray')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_per_class_metrics(epochs, per_class, save_path=None):
    # Create one figure per metric to avoid overcrowding when many classes
    metrics = ['precision', 'recall', 'f1', 'iou']
    titles = {
        'precision': 'Sınıf Bazlı Doğruluk (Precision)',
        'recall': 'Sınıf Bazlı Duyarlılık (Recall)',
        'f1': 'Sınıf Bazlı F1',
        'iou': 'Sınıf Bazlı IoU',
    }
    for metric in metrics:
        plt.figure(figsize=(14, 8))
        for cls_name, md in per_class.items():
            plt.plot(epochs, md[metric], label=cls_name, linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel(metric.upper())
        plt.title(titles[metric])
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        if save_path:
            base, ext = os.path.splitext(save_path)
            plt.savefig(f"{base}_{metric}{ext}", dpi=300, bbox_inches='tight')
        plt.show()


def plot_coco_stats_over_epochs(epochs, epoch_history, save_path=None):
    # Plot AP, AP50, AP75 from stats_50_95, and AP from stats_025 across epochs
    ap_50_95 = []
    ap50 = []
    ap75 = []
    ap_025 = []
    ar100_50_95 = []
    ar100_025 = []
    for ep in epoch_history:
        s5095 = ((ep.get('val_coco_stats') or {}).get('stats_50_95')) or {}
        s025 = ((ep.get('val_coco_stats') or {}).get('stats_025')) or {}
        ap_50_95.append(s5095.get('AP', np.nan))
        ap50.append(s5095.get('AP50', np.nan))
        ap75.append(s5095.get('AP75', np.nan))
        ap_025.append(s025.get('AP', np.nan))
        ar100_50_95.append(s5095.get('AR100', np.nan))
        ar100_025.append(s025.get('AR100', np.nan))

    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, ap_50_95, label='mAP (50:95)', linewidth=2)
    plt.plot(epochs, ap50, label='AP50', linewidth=2)
    plt.plot(epochs, ap75, label='AP75', linewidth=2)
    plt.plot(epochs, ap_025, label='mAP (0.25)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('AP')
    plt.title('COCO AP İstatistikleri (Val)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, ar100_50_95, label='AR100 (50:95)', linewidth=2)
    plt.plot(epochs, ar100_025, label='AR100 (0.25)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('AR')
    plt.title('COCO AR İstatistikleri (Val)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_best_vs_test_overall(data, save_path=None):
    best = (data.get('best_validation') or {}).get('overall_metrics') or {}
    test = (data.get('test_metrics') or {}).get('overall') or {}
    metrics = ['precision', 'recall', 'f1', 'iou']
    val_values = [best.get(m, np.nan) for m in metrics]
    test_values = [test.get(m, np.nan) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    plt.figure(figsize=(12, 8))
    bars1 = plt.bar(x - width/2, val_values, width, label='En İyi Val', alpha=0.85)
    bars2 = plt.bar(x + width/2, test_values, width, label='Test', alpha=0.85)

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                plt.text(bar.get_x() + bar.get_width()/2., h + 0.01, f'{h:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.xlabel('Metrikler')
    plt.ylabel('Skor')
    plt.title('Genel Metrikler Karşılaştırması (En İyi Val vs Test)')
    plt.xticks(x, [m.upper() for m in metrics])
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_best_vs_test_per_class(data, save_path=None):
    best_pc = (data.get('best_validation') or {}).get('per_class_metrics') or {}
    test_pc = (data.get('test_metrics') or {}).get('per_class') or {}
    classes = sorted(set(list(best_pc.keys()) + list(test_pc.keys())))
    if not classes:
        return

    metrics = ['precision', 'recall', 'f1', 'iou']
    for m in metrics:
        plt.figure(figsize=(12, 8))
        x = np.arange(len(classes))
        width = 0.35
        val_vals = [((best_pc.get(c) or {}).get(m, np.nan)) for c in classes]
        test_vals = [((test_pc.get(c) or {}).get(m, np.nan)) for c in classes]
        plt.bar(x - width/2, val_vals, width, label='En İyi Val', alpha=0.85)
        plt.bar(x + width/2, test_vals, width, label='Test', alpha=0.85)
        plt.xlabel('Sınıf')
        plt.ylabel(m.upper())
        plt.title(f'Sınıf Bazlı {m.upper()} (En İyi Val vs Test)')
        plt.xticks(x, classes, rotation=0)
        plt.grid(True, alpha=0.3, axis='y')
        plt.ylim(0, 1.0)
        plt.legend()
        plt.tight_layout()
        if save_path:
            base, ext = os.path.splitext(save_path)
            plt.savefig(f"{base}_{m}{ext}", dpi=300, bbox_inches='tight')
        plt.show()


def training_summary_text(data, epochs_len):
    # derive values safely
    best_ep = (data.get('best_validation') or {}).get('epoch')
    best_f1 = (data.get('best_validation') or {}).get('f1_score')
    train_time_hours = _parse_duration_to_hours((data.get('training_time') or {}).get('total_duration'))
    cfg = data.get('config') or {}
    txt = [
        "Eğitim Özeti:",
        f"• Toplam Eğitim Süresi: {train_time_hours:.2f} saat" if train_time_hours is not None else "• Toplam Eğitim Süresi: -",
        f"• En İyi Epoch: {best_ep}",
        f"• En İyi F1: {best_f1:.4f}" if best_f1 is not None else "• En İyi F1: -",
        f"• Tamamlanan Epoch: {epochs_len}",
        "",
        "Model:",
        f"• Backbone: {cfg.get('backbone')}",
        f"• Optimizasyon: {((cfg.get('optimizer') or {}).get('name'))} (lr={((cfg.get('optimizer') or {}).get('lr'))})",
        f"• Scheduler: {((cfg.get('scheduler') or {}).get('name'))}",
        "",
        "Veriseti:",
        f"• Train: {cfg.get('train_samples')}",
        f"• Val: {cfg.get('val_samples')}",
        f"• Test: {cfg.get('test_samples')}",
        f"• Sınıf Sayısı: {cfg.get('num_classes')}",
    ]
    return "\n".join(txt)


def plot_training_summary(data, series, save_path=None):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # F1 progression (validation overall f1)
    f1_scores = series['val_overall']['f1']
    epochs = series['epochs']
    ax1.plot(epochs, f1_scores, 'b-', linewidth=2, marker='o', markersize=3)
    try:
        best_epoch = (data.get('best_validation') or {}).get('epoch')
        best_f1 = (data.get('best_validation') or {}).get('f1_score')
        if best_epoch is not None and best_f1 is not None:
            ax1.axhline(y=float(best_f1), color='r', linestyle='--', alpha=0.7)
            ax1.axvline(x=int(best_epoch), color='g', linestyle='--', alpha=0.7)
    except Exception:
        pass
    ax1.set_title('F1 Skoru Gelişimi (Val)', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('F1')
    ax1.grid(True, alpha=0.3)

    # Total loss
    ax2.plot(epochs, series['losses']['total_loss'], 'r-', linewidth=2, marker='o', markersize=3)
    ax2.set_title('Toplam Kayıp (Train)', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Kayıp')
    ax2.grid(True, alpha=0.3)

    # Final metrics comparison
    best = (data.get('best_validation') or {}).get('overall_metrics') or {}
    test = (data.get('test_metrics') or {}).get('overall') or {}
    metrics = ['precision', 'recall', 'f1', 'iou']
    val_vals = [best.get(m, np.nan) for m in metrics]
    test_vals = [test.get(m, np.nan) for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35
    ax3.bar(x - width/2, val_vals, width, label='En İyi Val', alpha=0.8)
    ax3.bar(x + width/2, test_vals, width, label='Test', alpha=0.8)
    ax3.set_title('Genel Metrikler Karşılaştırma', fontweight='bold')
    ax3.set_xlabel('Metrikler')
    ax3.set_ylabel('Skor')
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.upper() for m in metrics])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, 1.0)

    # Text info
    ax4.axis('off')
    info_text = training_summary_text(data, len(epochs))
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=10, va='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Generate comprehensive visualizations for the specified training results JSON."""
    json_path = r"C:/Users/ali.donbaloglu/Desktop/Lens/training_plots/Kabin_object_detection_sinifli/training_results_OD_Sinif.json"

    # Save plots next to the JSON under 'plots'
    output_dir = os.path.join(os.path.dirname(json_path), 'plots')
    os.makedirs(output_dir, exist_ok=True)

    print('Veri yükleniyor...')
    data = load_training_data(json_path)
    series = extract_epoch_series(data)
    epochs = series['epochs']

    print('Grafikler üretiliyor...')
    # 1) Train losses
    print('1. Kayıp bileşenleri...')
    plot_losses(epochs, series['losses'], save_path=os.path.join(output_dir, 'loss_components.png'))

    # 2) Validation overall metrics
    print('2. Doğrulama genel metrikler...')
    plot_val_overall_metrics(epochs, series['val_overall'], save_path=os.path.join(output_dir, 'val_overall_metrics.png'))

    # 3) mAP series
    print('3. mAP gelişimi...')
    plot_val_maps(epochs, series['map_025'], series['map_50_95'], save_path=os.path.join(output_dir, 'val_map_series.png'))

    # 4) LR and FPS
    print('4. Öğrenme oranı ve FPS...')
    plot_lr_and_fps(epochs, series['lr'], series['val_fps'], save_path=os.path.join(output_dir, 'lr_fps.png'))

    # 5) Per-class metrics
    if series['per_class']:
        print('5. Sınıf bazlı metrikler...')
        plot_per_class_metrics(epochs, series['per_class'], save_path=os.path.join(output_dir, 'per_class_metrics.png'))

    # 6) COCO stats across epochs
    if data.get('epoch_history'):
        print('6. COCO istatistikleri (epoch bazlı)...')
        plot_coco_stats_over_epochs(epochs, data['epoch_history'], save_path=os.path.join(output_dir, 'coco_stats.png'))

    # 7) Best vs Test comparisons (overall and per-class)
    print('7. En iyi Val vs Test (genel)...')
    plot_best_vs_test_overall(data, save_path=os.path.join(output_dir, 'best_vs_test_overall.png'))

    print('8. En iyi Val vs Test (sınıf bazlı)...')
    plot_best_vs_test_per_class(data, save_path=os.path.join(output_dir, 'best_vs_test_per_class.png'))

    # 9) Summary
    print('9. Özet...')
    plot_training_summary(data, series, save_path=os.path.join(output_dir, 'training_summary.png'))

    print(f"\nTüm grafikler '{output_dir}' klasörüne kaydedildi.")


if __name__ == '__main__':
    main()
