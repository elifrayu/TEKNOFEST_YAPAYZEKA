"""
HELIXAI — Akademik Görselleştirme Modülü (helixai_visuals.py)
==============================================================
Tüm paneller (MASTER, KANSER, PAH, CFTR) için ortak kullanılır.

Üretilen grafikler:
  01_fold_distributions.png   — fold bazlı F1/MCC/Specificity dağılımı
  02_shap_top10.png           — SHAP top-10 bar chart
  03_pipeline_schema.png      — pipeline akış şeması
  04_real_roc_curve.png       — OOF ROC eğrisi (gerçek)
  05_pr_curve.png             — OOF PR eğrisi (gerçek)
  06_confusion_matrix.png     — OOF confusion matrix (gerçek)
  07_calibration_curve.png    — OOF kalibrasyon eğrisi (gerçek)
  08_threshold_distribution.png — threshold vs F1/Specificity (gerçek)

Kullanım:
  from helixai_visuals import generate_all_visuals
  img_paths = generate_all_visuals(report, fold_results, output_dir)
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score, f1_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# ── Renk paleti ──────────────────────────────
PALETTE = {
    'primary':   '#2E86AB',
    'secondary': '#A23B72',
    'accent':    '#F18F01',
    'positive':  '#C73E1D',
    'negative':  '#3B1F2B',
    'grid':      '#E8E8E8',
    'text':      '#2D2D2D',
    'bg':        '#FAFAFA',
}

def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=PALETTE['bg'])
    plt.close(fig)
    print(f"  Kaydedildi: {path}")
    return path

def _get_oof(report):
    """Rapor JSON'undan OOF y_true ve y_prob çeker."""
    y_true = np.array(report.get('oof_y_true', []))
    y_prob = np.array(report.get('oof_y_prob', []))
    return y_true, y_prob

def _has_oof(report):
    y_true, y_prob = _get_oof(report)
    return len(y_true) > 0 and len(y_prob) > 0

# ══════════════════════════════════════════════
# 1. FOLD DAĞILIMI
# ══════════════════════════════════════════════
def plot_fold_distributions(report, output_dir):
    panel = report.get('panel', '?')
    f1s   = report.get('fold_f1_values',   [])
    mccs  = report.get('fold_mcc_values',  [])
    specs = report.get('fold_spec_values', [])

    if not f1s:
        # cv_summary'den al
        cv = report.get('cv_summary', {})
        f1s   = [cv.get('f1',  {}).get('mean', 0)]
        mccs  = [cv.get('mcc', {}).get('mean', 0)]
        specs = [cv.get('specificity', {}).get('mean', 0)]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor(PALETTE['bg'])
    fig.suptitle(f'[{panel}] Fold Dağılımı — Cross-Validation', fontsize=14,
                 color=PALETTE['text'], fontweight='bold', y=1.02)

    data_list  = [f1s, mccs, specs]
    labels     = ['F1 Skoru', 'MCC', 'Specificity']
    colors     = [PALETTE['primary'], PALETTE['secondary'], PALETTE['accent']]

    for ax, data, label, color in zip(axes, data_list, labels, colors):
        ax.set_facecolor(PALETTE['bg'])
        if len(data) > 1:
            ax.boxplot(data, patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=0.4),
                       medianprops=dict(color=color, linewidth=2),
                       whiskerprops=dict(color=PALETTE['text']),
                       capprops=dict(color=PALETTE['text']),
                       flierprops=dict(marker='o', color=color, alpha=0.5))
            for i, v in enumerate(data):
                ax.scatter([1 + (np.random.rand()-0.5)*0.2], [v],
                           color=color, alpha=0.6, s=30, zorder=3)
        else:
            ax.bar([1], data, color=color, alpha=0.7, width=0.4)

        mean_val = float(np.mean(data))
        std_val  = float(np.std(data))
        ax.axhline(mean_val, color=color, linestyle='--', linewidth=1.5, alpha=0.8)
        ax.set_title(f'{label}\n{mean_val:.4f} ± {std_val:.4f}',
                     color=PALETTE['text'], fontsize=11)
        ax.set_xticks([])
        ax.set_ylim(max(0, mean_val - 4*std_val - 0.1), min(1.0, mean_val + 4*std_val + 0.1))
        ax.grid(axis='y', color=PALETTE['grid'], linewidth=0.8)
        ax.spines[['top','right','bottom']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, '01_fold_distributions.png')
    return _save(fig, path)


# ══════════════════════════════════════════════
# 2. SHAP TOP-10
# ══════════════════════════════════════════════
def plot_shap_top10(report, output_dir):
    panel    = report.get('panel', '?')
    shap_d   = report.get('shap_top10', {})

    if not shap_d:
        print("  [SHAP] shap_top10 verisi bulunamadı, atlandı.")
        return None

    feats  = list(shap_d.keys())[:10]
    values = [float(shap_d[f]) for f in feats]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['bg'])

    colors = [PALETTE['primary'] if i < 3 else
              PALETTE['secondary'] if i < 6 else
              PALETTE['accent'] for i in range(len(feats))]

    bars = ax.barh(range(len(feats)), values, color=colors, alpha=0.85, height=0.65)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=10, color=PALETTE['text'])
    ax.invert_yaxis()
    ax.set_xlabel('Ortalama |SHAP Değeri|', color=PALETTE['text'], fontsize=11)
    ax.set_title(f'[{panel}] SHAP Özellik Önemi — Top 10',
                 color=PALETTE['text'], fontsize=13, fontweight='bold')

    for bar, val in zip(bars, values):
        ax.text(val + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9, color=PALETTE['text'])

    ax.grid(axis='x', color=PALETTE['grid'], linewidth=0.8)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(colors=PALETTE['text'])

    plt.tight_layout()
    path = os.path.join(output_dir, '02_shap_top10.png')
    return _save(fig, path)


# ══════════════════════════════════════════════
# 3. PİPELINE ŞEMASI
# ══════════════════════════════════════════════
def plot_pipeline_schema(report, output_dir):
    panel  = report.get('panel', '?')
    config = report.get('config', {})
    n_fold = config.get('n_splits', 5)
    n_rep  = config.get('n_repeats', 5)
    models = list(config.get('weights', {'XGBoost': 0.35, 'LightGBM': 0.30, 'CatBoost': 0.35}).keys())

    steps = [
        ('1. Veri Yükleme',      f'{panel} CSV\nLabel: 0/1'),
        ('2. Global Filtre',     'Eksik >%90\nkaldır'),
        ('3. Biyolojik QC',      'Stop kodon\ngeçersiz AA'),
        ('4. AA Mühendislik',    'BLOSUM62\nGrantham\n+4 fark'),
        ('5. AF Log Dönüşüm',    '-log10(AF+ε)\nAL_ kolonlar'),
        ('6. Fold Eğitimi',      f'{n_fold}-Fold × {n_rep}\nRepeated CV'),
        ('7. Kalibrasyon',       'Platt Scaling\nfold içi'),
        ('8. Prior Correction',  'Bayes düzeltme\ntest priorı'),
        ('9. Ensemble',          '+'.join([m[:3] for m in models[:3]])+'\nweighted avg'),
        ('10. SHAP',             'Açıklanabilirlik\nTop-10 özellik'),
    ]

    fig, ax = plt.subplots(figsize=(16, 4))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['bg'])
    ax.set_xlim(0, len(steps))
    ax.set_ylim(-0.5, 1.5)
    ax.axis('off')

    step_colors = [PALETTE['primary'], PALETTE['primary'], PALETTE['primary'],
                   PALETTE['secondary'], PALETTE['secondary'],
                   PALETTE['accent'], PALETTE['accent'],
                   PALETTE['positive'], PALETTE['positive'],
                   PALETTE['primary']]

    box_w, box_h = 0.85, 0.7
    for i, ((title, desc), color) in enumerate(zip(steps, step_colors)):
        x = i + 0.5
        rect = mpatches.FancyBboxPatch((x - box_w/2, 0.15), box_w, box_h,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, alpha=0.15,
                                        edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, 0.5 + box_h*0.18, title, ha='center', va='center',
                fontsize=7.5, fontweight='bold', color=color)
        ax.text(x, 0.5 - box_h*0.18, desc, ha='center', va='center',
                fontsize=6.5, color=PALETTE['text'], alpha=0.8)
        if i < len(steps) - 1:
            ax.annotate('', xy=(x + box_w/2 + 0.02, 0.5),
                        xytext=(x + box_w/2 + 0.02, 0.5),
                        arrowprops=dict(arrowstyle='->', color=PALETTE['text'], lw=1.2))
            ax.plot([x + box_w/2 + 0.01, x + 1 - box_w/2 - 0.01],
                    [0.5, 0.5], color=PALETTE['text'], lw=1.2, alpha=0.5)

    ax.set_title(f'[{panel}] HELIXAI Pipeline Akış Şeması',
                 color=PALETTE['text'], fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    path = os.path.join(output_dir, '03_pipeline_schema.png')
    return _save(fig, path)


# ══════════════════════════════════════════════
# 4. GERÇEK ROC EĞRİSİ (OOF)
# ══════════════════════════════════════════════
def plot_real_roc(report, output_dir):
    panel = report.get('panel', '?')
    y_true, y_prob = _get_oof(report)

    if not _has_oof(report):
        print("  [ROC] OOF verisi yok, atlandı.")
        return None

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['bg'])

    ax.plot(fpr, tpr, color=PALETTE['primary'], lw=2.5,
            label=f'ROC Eğrisi (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Rastgele Tahmin')
    ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE['primary'])

    # Optimal threshold noktası
    opt_idx = np.argmax(tpr - fpr)
    ax.scatter(fpr[opt_idx], tpr[opt_idx], marker='o', color=PALETTE['positive'],
               s=80, zorder=5, label=f'Optimal Eşik = {thresholds[opt_idx]:.3f}')

    ax.set_xlabel('Yanlış Pozitif Oranı (FPR)', color=PALETTE['text'], fontsize=11)
    ax.set_ylabel('Doğru Pozitif Oranı (TPR)', color=PALETTE['text'], fontsize=11)
    ax.set_title(f'[{panel}] ROC Eğrisi — OOF Tahminleri',
                 color=PALETTE['text'], fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(color=PALETTE['grid'], linewidth=0.8)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(colors=PALETTE['text'])

    plt.tight_layout()
    path = os.path.join(output_dir, '04_real_roc_curve.png')
    return _save(fig, path)


# ══════════════════════════════════════════════
# 5. GERÇEK PR EĞRİSİ (OOF)
# ══════════════════════════════════════════════
def plot_pr_curve(report, output_dir):
    panel = report.get('panel', '?')
    y_true, y_prob = _get_oof(report)

    if not _has_oof(report):
        print("  [PR] OOF verisi yok, atlandı.")
        return None

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    baseline = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['bg'])

    ax.plot(recall, precision, color=PALETTE['secondary'], lw=2.5,
            label=f'PR Eğrisi (AP = {pr_auc:.4f})')
    ax.axhline(baseline, color='k', linestyle='--', lw=1.2, alpha=0.5,
               label=f'Baseline (prevalans = {baseline:.3f})')
    ax.fill_between(recall, precision, alpha=0.08, color=PALETTE['secondary'])

    # F1 max noktası
    if len(thresholds) > 0:
        f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
        best_idx = np.argmax(f1_scores)
        ax.scatter(recall[best_idx], precision[best_idx], marker='*',
                   color=PALETTE['positive'], s=150, zorder=5,
                   label=f'F1 Max = {f1_scores[best_idx]:.4f} @ T={thresholds[best_idx]:.3f}')

    ax.set_xlabel('Recall (Duyarlılık)', color=PALETTE['text'], fontsize=11)
    ax.set_ylabel('Precision (Kesinlik)', color=PALETTE['text'], fontsize=11)
    ax.set_title(f'[{panel}] Precision-Recall Eğrisi — OOF Tahminleri',
                 color=PALETTE['text'], fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(color=PALETTE['grid'], linewidth=0.8)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(colors=PALETTE['text'])

    plt.tight_layout()
    path = os.path.join(output_dir, '05_pr_curve.png')
    return _save(fig, path)


# ══════════════════════════════════════════════
# 6. GERÇEK CONFUSION MATRIX (OOF)
# ══════════════════════════════════════════════
def plot_confusion_matrix(report, output_dir):
    panel = report.get('panel', '?')
    y_true, y_prob = _get_oof(report)

    if not _has_oof(report):
        print("  [CM] OOF verisi yok, atlandı.")
        return None

    # OOF threshold
    config    = report.get('config', {})
    threshold = config.get('final_threshold_oof', 0.5)
    y_pred    = (y_prob >= threshold).astype(int)
    cm        = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['bg'])

    # Heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8)

    labels_axis = ['Benign (0)', 'Patojenik (1)']
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels_axis, color=PALETTE['text'], fontsize=11)
    ax.set_yticklabels(labels_axis, color=PALETTE['text'], fontsize=11)

    thresh_cm = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]}',
                    ha='center', va='center', fontsize=18, fontweight='bold',
                    color='white' if cm[i, j] > thresh_cm else PALETTE['text'])

    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1v  = f1_score(y_true, y_pred, zero_division=0)

    ax.set_xlabel('Tahmin Edilen Sınıf', color=PALETTE['text'], fontsize=11)
    ax.set_ylabel('Gerçek Sınıf', color=PALETTE['text'], fontsize=11)
    ax.set_title(
        f'[{panel}] Confusion Matrix — OOF (Eşik={threshold:.3f})\n'
        f'Sensitivity={sens:.4f} | Specificity={spec:.4f} | F1={f1v:.4f}',
        color=PALETTE['text'], fontsize=12, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, '06_confusion_matrix.png')
    return _save(fig, path)


# ══════════════════════════════════════════════
# 7. GERÇEK KALİBRASYON EĞRİSİ (OOF)
# ══════════════════════════════════════════════
def plot_calibration_curve(report, output_dir):
    panel = report.get('panel', '?')
    y_true, y_prob = _get_oof(report)

    if not _has_oof(report):
        print("  [Kalibrasyon] OOF verisi yok, atlandı.")
        return None

    n_bins = min(10, max(5, len(y_true) // 20))
    try:
        fraction_of_positives, mean_predicted = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform')
    except Exception as e:
        print(f"  [Kalibrasyon] Hata: {e}, atlandı.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(PALETTE['bg'])

    # Sol: Kalibrasyon eğrisi
    ax = axes[0]
    ax.set_facecolor(PALETTE['bg'])
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.6, label='Mükemmel Kalibrasyon')
    ax.plot(mean_predicted, fraction_of_positives, 's-',
            color=PALETTE['accent'], lw=2, markersize=8,
            label=f'Model (n_bins={n_bins})')
    ax.fill_between(mean_predicted, fraction_of_positives, mean_predicted,
                    alpha=0.1, color=PALETTE['accent'])
    ax.set_xlabel('Ortalama Tahmin Olasılığı', color=PALETTE['text'], fontsize=11)
    ax.set_ylabel('Gözlemlenen Frekans', color=PALETTE['text'], fontsize=11)
    ax.set_title(f'[{panel}] Kalibrasyon Eğrisi (OOF)',
                 color=PALETTE['text'], fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(color=PALETTE['grid'], linewidth=0.8)
    ax.spines[['top','right']].set_visible(False)

    # Sağ: Olasılık dağılımı
    ax2 = axes[1]
    ax2.set_facecolor(PALETTE['bg'])
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    ax2.hist(neg_probs, bins=30, alpha=0.6, color=PALETTE['primary'],
             label=f'Benign (n={len(neg_probs)})', density=True)
    ax2.hist(pos_probs, bins=30, alpha=0.6, color=PALETTE['positive'],
             label=f'Patojenik (n={len(pos_probs)})', density=True)
    ax2.set_xlabel('Tahmin Edilen Olasılık', color=PALETTE['text'], fontsize=11)
    ax2.set_ylabel('Yoğunluk', color=PALETTE['text'], fontsize=11)
    ax2.set_title('Olasılık Dağılımı (Sınıf Bazlı)',
                  color=PALETTE['text'], fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(color=PALETTE['grid'], linewidth=0.8)
    ax2.spines[['top','right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, '07_calibration_curve.png')
    return _save(fig, path)


# ══════════════════════════════════════════════
# 8. GERÇEK THRESHOLD DAĞILIMI (OOF)
# ══════════════════════════════════════════════
def plot_real_threshold_distribution(report, output_dir):
    panel = report.get('panel', '?')
    y_true, y_prob = _get_oof(report)

    if not _has_oof(report):
        print("  [Threshold] OOF verisi yok, atlandı.")
        return None

    thresholds = np.arange(0.01, 0.99, 0.005)
    f1_vals, spec_vals, sens_vals = [], [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1_vals.append(f1_score(y_true, y_pred, zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec_vals.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        sens_vals.append(tp / (tp + fn) if (tp + fn) > 0 else 0)

    best_f1_idx = np.argmax(f1_vals)
    opt_t       = thresholds[best_f1_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(PALETTE['bg'])
    ax.set_facecolor(PALETTE['bg'])

    ax.plot(thresholds, f1_vals,   color=PALETTE['primary'],   lw=2, label='F1 Skoru')
    ax.plot(thresholds, spec_vals, color=PALETTE['secondary'], lw=2, label='Specificity')
    ax.plot(thresholds, sens_vals, color=PALETTE['accent'],    lw=2, label='Sensitivity (Recall)')

    ax.axvline(opt_t, color=PALETTE['positive'], linestyle='--', lw=1.8,
               label=f'Optimal Threshold = {opt_t:.3f}\n(F1 max = {f1_vals[best_f1_idx]:.4f})')
    ax.scatter([opt_t], [f1_vals[best_f1_idx]], color=PALETTE['positive'],
               s=100, zorder=5)

    ax.set_xlabel('Threshold (Sınıflandırma Eşiği)', color=PALETTE['text'], fontsize=11)
    ax.set_ylabel('Skor', color=PALETTE['text'], fontsize=11)
    ax.set_title(f'[{panel}] Threshold vs Metrikler — OOF Tahminleri',
                 color=PALETTE['text'], fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(color=PALETTE['grid'], linewidth=0.8)
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.tick_params(colors=PALETTE['text'])

    plt.tight_layout()
    path = os.path.join(output_dir, '08_threshold_distribution.png')
    return _save(fig, path)


# ══════════════════════════════════════════════
# ANA FONKSİYON
# ══════════════════════════════════════════════
def generate_all_visuals(report, fold_results=None, output_dir=None):
    """
    Tüm akademik görselleri üretir.

    Args:
        report      : pipeline'dan dönen JSON raporu (dict)
        fold_results: fold_results listesi (fold bazlı F1/MCC/Spec için)
        output_dir  : kayıt klasörü (None ise report'dan alınır)

    Returns:
        img_paths: {isim: dosya_yolu} dict
    """
    panel = report.get('panel', 'panel')
    if output_dir is None:
        output_dir = f'{panel.lower()}_results'
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*60)
    print(f"HELIXAI Akademik Görselleştirme — {panel}")
    print("="*60)

    # Fold bazlı değerleri rapordan veya fold_results'tan al
    if fold_results and 'fold_f1_values' not in report:
        try:
            report['fold_f1_values']  = [r['metrics']['f1']         for r in fold_results]
            report['fold_mcc_values'] = [r['metrics']['mcc']         for r in fold_results]
            report['fold_spec_values']= [r['metrics']['specificity'] for r in fold_results]
        except Exception:
            pass

    img_paths = {}

    img_paths['fold_dist']  = plot_fold_distributions(report, output_dir)
    img_paths['shap']       = plot_shap_top10(report, output_dir)
    img_paths['pipeline']   = plot_pipeline_schema(report, output_dir)
    img_paths['roc']        = plot_real_roc(report, output_dir)
    img_paths['pr']         = plot_pr_curve(report, output_dir)
    img_paths['cm']         = plot_confusion_matrix(report, output_dir)
    img_paths['calibration']= plot_calibration_curve(report, output_dir)
    img_paths['threshold']  = plot_real_threshold_distribution(report, output_dir)

    produced = sum(1 for v in img_paths.values() if v is not None)
    print(f"\n  Tamamlandı — {produced}/{len(img_paths)} grafik üretildi")
    print(f"  Klasör: {output_dir}")

    return img_paths