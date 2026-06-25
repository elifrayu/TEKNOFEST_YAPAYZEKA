"""
HELIXAI — Ortak Akademik Görselleştirme Modülü (helixai_visuals.py)
=====================================================================
Master / Kanser (ve ileride CFTR / PAH) panellerinin TÜMÜNÜN ortak
kullanacağı, run_repeated_cv() / run_loocv() çıktısı 'fold_results'
(bkz. helixai_common.py modül-başı sözleşmesi) ile çalışan, MODEL-
SAYISI-AGNOSTİK grafik üretim katmanı.

[NEDEN AYRI BİR DOSYA] helixai_common.py istatistiksel/ML mantığını
(metrik, eşik, kalibrasyon, prior correction) barındırıyor ve
matplotlib'e BAĞIMLI DEĞİL. Görselleştirme tamamen ayrı bir kaygı
(concern) olduğundan ve jüri ortamında matplotlib import'unun olası
backend sorunlarını (display yokluğu vb.) helixai_common.py'nin
çekirdek istatistik fonksiyonlarından İZOLE etmek için bu modül
ayrılmıştır. Panel pipeline'ları (kanser_pipeline_v1.py,
master_pipeline_v1.py, ...) hem helixai_common'ı hem bu modülü import eder.

[KRİTİK METODOLOJİK NOT — PRIOR CORRECTION & KALİBRASYON]
Bayesian Prior Correction (Saerens 2002), olasılıkları KASITLI olarak
eğitim-önselinden test-önseline doğru kaydırır. Bu nedenle:
  - ROC eğrisi: prior-BAĞIMSIZDIR (yalnızca sıralamaya/rank'e bakar);
    düzeltilmiş havuzda hesaplanması güvenlidir (AUC ham/düzeltilmiş
    havuzda birebir AYNIDIR, çünkü prior_correction() p'ye göre
    MONOTON bir dönüşümdür).
  - PR eğrisi: prior'a HASSASTIR. Bu yüzden BURADA iki eğri çizilir:
    (a) ham OOF havuzu (eğitim-önseli) — referans/karşılaştırma amaçlı,
    (b) Klinik Stres Testi ile yeniden örneklenmiş havuz (test-önseli
        simülasyonu) — gerçek yarışma senaryosuna en yakın olan BUDUR.
  - Calibration (reliability) eğrisi: HAM (düzeltme ÖNCESİ) olasılıklar
    üzerinden hesaplanır. Düzeltilmiş olasılıkları, hâlâ eğitim-önselini
    taşıyan OOF etiketlerine göre "kalibre" göstermek YANILTICI olurdu —
    çünkü düzeltme zaten FARKLI bir popülasyon önseli (test_prior) için
    tasarlanmıştır. Bu, raporun "dürüstlük" ilkesiyle birebir uyumludur.
  - Confusion Matrix: analyze_report.py ile BİREBİR TUTARLI olması için
    report['stress_test']['before'/'after'] içindeki tp/fp/fn/tn DOĞRUDAN
    kullanılır — burada YENİDEN HESAPLAMA yapılmaz (iki araç arasında
    rakam çatışması riskini sıfırlar).
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')          # [Jüri-güvenli] display olmayan ortamlarda da çalışır
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve

import helixai_common as hc

# ══════════════════════════════════════════════
# RENK PALETİ (renk-körlüğüne dayanıklı, tüm grafiklerde SABİT)
# ══════════════════════════════════════════════
COLOR_PATHO   = '#D62728'   # Patojenik — kırmızı
COLOR_BENIGN  = '#1F77B4'   # Benign    — mavi
COLOR_BEFORE  = '#7F7F7F'   # Stres testi ÖNCESİ — gri (referans)
COLOR_AFTER   = '#2CA02C'   # Stres testi SONRASI — yeşil (güvenilen)
COLOR_ACCENT  = '#9467BD'   # vurgu (eşik çizgisi vb.)
FIGSIZE_WIDE  = (10, 5)
FIGSIZE_SQ    = (6, 5)
DPI           = 140

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})


# ══════════════════════════════════════════════
# 1. OOF HAVUZLAMA YARDIMCISI
# ══════════════════════════════════════════════
def get_oof_pools(fold_results):
    """
    fold_results -> (labels, probs_raw, probs_corrected) — TÜM fold/LOO
    iterasyonlarının val_* dizileri TEK havuzda birleştirilir (concatenate).
    Bu, helixai_common.compute_oof_pooled_threshold /
    run_stress_test_comparison ile AYNI havuzlama mantığıdır (tek yerden
    DEĞİL ama BİREBİR aynı formülle — kasıtlı, döngüsel import'tan kaçınmak
    için burada yeniden uygulanmıştır).
    """
    labels      = np.concatenate([np.asarray(r['val_labels'])          for r in fold_results])
    probs_raw   = np.concatenate([np.asarray(r['val_probs_raw'])       for r in fold_results])
    probs_corr  = np.concatenate([np.asarray(r['val_probs_corrected']) for r in fold_results])
    return labels, probs_raw, probs_corr


def _fold_is_loocv(fold_results):
    """LOOCV'de her fold'un val seti n=1'dir -> fold-bazlı F1 anlamsızdır."""
    sizes = [len(np.asarray(r['val_labels'])) for r in fold_results]
    return len(sizes) > 0 and max(sizes) <= 1


def _save(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Görsel kaydedildi] {path}")
    return path


# ══════════════════════════════════════════════
# 2. 01 — FOLD-BAZLI METRİK DAĞILIMLARI
# ══════════════════════════════════════════════
def plot_fold_distributions(report, fold_results, output_dir, panel_name="?", prefix="panel"):
    """
    Repeated-CV fold'ları arasındaki F1 / MCC / Specificity dağılımını
    box+strip plot olarak gösterir. CV'nin TEK bir sayıya (ortalamaya)
    indirgenmesinin gizleyebileceği varyansı görünür kılar.

    [LOOCV UYARISI] Her fold n=1 ise (CFTR), fold-bazlı F1/MCC anlamsızdır
    (0 ya da 1 olur, dağılım değil nokta-tahmindir) — bu durumda fonksiyon
    grafiği ÜRETMEZ, açıklayıcı bir placeholder döner (PDR'da bu sınırlama
    zaten belirtiliyor — bkz. helixai_common.run_loocv docstring).
    """
    if _fold_is_loocv(fold_results):
        fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
        ax.text(0.5, 0.5,
                "LOOCV: her fold n=1 örnek içerir.\n"
                "Fold-bazlı F1/MCC dağılımı tanımsızdır.\n"
                "Bkz. OOF-havuzlanmış (pooled) sonuçlar.",
                ha='center', va='center', fontsize=11, color=COLOR_BEFORE)
        ax.set_title(f"{panel_name} — Fold Dağılımları (N/A: LOOCV)")
        ax.axis('off')
        return _save(fig, output_dir, '01_fold_distributions.png')

    f1s   = report.get('fold_f1_values')   or [r['metrics']['f1']          for r in fold_results]
    mccs  = report.get('fold_mcc_values')  or [r['metrics']['mcc']         for r in fold_results]
    specs = report.get('fold_spec_values') or [r['metrics']['specificity'] for r in fold_results]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    data_labels = [('F1 (Patojenik)', f1s, COLOR_PATHO),
                   ('MCC',            mccs, COLOR_ACCENT),
                   ('Specificity (Benign)', specs, COLOR_BENIGN)]
    for ax, (label, vals, color) in zip(axes, data_labels):
        vals = np.asarray(vals, dtype=float)
        bp = ax.boxplot(vals, vert=True, widths=0.45, patch_artist=True,
                         showmeans=True, meanline=True)
        for box in bp['boxes']:
            box.set(facecolor=color, alpha=0.25, edgecolor=color, linewidth=1.5)
        for med in bp['medians']:
            med.set(color=color, linewidth=2)
        rng = np.random.RandomState(hc.SEED)
        jitter = rng.normal(1.0, 0.04, size=len(vals))
        ax.scatter(jitter, vals, color=color, alpha=0.55, s=22, zorder=3,
                   edgecolor='white', linewidth=0.4)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_ylabel('Değer')
        ax.text(0.02, 0.02, f"μ={vals.mean():.3f}\nσ={vals.std():.3f}\nn={len(vals)} fold",
                transform=ax.transAxes, fontsize=8.5, va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, alpha=0.85))

    fig.suptitle(f"{panel_name} — Repeated-CV Fold Metrik Dağılımları "
                 f"({report.get('config', {}).get('n_splits', '?')}-Fold × "
                 f"{report.get('config', {}).get('n_repeats', '?')} Tekrar)",
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save(fig, output_dir, '01_fold_distributions.png')


# ══════════════════════════════════════════════
# 3. 02 — SHAP TOP-10
# ══════════════════════════════════════════════
def plot_shap_top10(report, output_dir, panel_name="?", prefix="panel"):
    """report['shap_top10'] (importance.head(10).to_dict() çıktısı) -> yatay bar grafiği."""
    shap_dict = report.get('shap_top10') or {}
    if not shap_dict:
        fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
        ax.text(0.5, 0.5, "SHAP top-10 verisi raporda bulunamadı.",
                ha='center', va='center', color=COLOR_BEFORE)
        ax.axis('off')
        return _save(fig, output_dir, '02_shap_top10.png')

    feats = list(shap_dict.keys())[::-1]
    vals  = [float(v) for v in shap_dict.values()][::-1]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.barh(feats, vals, color=COLOR_ACCENT, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va='center', fontsize=8.5)

    shap_method = report.get('config', {}).get('shap_method', '?')
    ax.set_xlabel('Ortalama |SHAP değeri| (özellik önem derecesi)')
    ax.set_title(f"{panel_name} — SHAP Top-10 Özellik\n(yöntem: {shap_method})", fontsize=11)
    ax.set_xlim(0, max(vals) * 1.18)
    fig.tight_layout()
    return _save(fig, output_dir, '02_shap_top10.png')


# ══════════════════════════════════════════════
# 4. 03 — PIPELINE ŞEMASI (statik, veri-bağımsız metodoloji diyagramı)
# ══════════════════════════════════════════════
def plot_pipeline_schema(report, output_dir, panel_name="?", prefix="panel"):
    """
    StratifiedKFold -> Class Weighting -> Ensemble -> Calibration ->
    Bayesian Prior Correction -> OOF Threshold Search -> Stres Testi ->
    F1(birincil)+MCC(ikincil) Değerlendirme akışını gösteren statik kutu-ok
    diyagramı. Panel-spesifik değerler report['config']'den okunur.
    """
    cfg = report.get('config', {})
    weights = cfg.get('weights', {})
    w_str = " / ".join(f"{k.upper()}={v:.2f}" for k, v in weights.items()) if weights else "?"
    calib = cfg.get('calibration_method', 'sigmoid')
    calib_display = {'sigmoid': 'Platt / sigmoid', 'isotonic': 'Isotonic Regression'}.get(calib, calib)
    spw   = cfg.get('chosen_scale_pos_weight', '?')
    n_splits = cfg.get('n_splits', '?'); n_repeats = cfg.get('n_repeats', '?')

    steps = [
        f"1) Repeated Stratified\n{n_splits}-Fold × {n_repeats} tekrar",
        f"2) Sınıf Ağırlıklandırma\nscale_pos_weight={spw}",
        f"3) Ensemble Eğitimi\n{w_str}",
        f"4) Olasılık Kalibrasyonu\nyöntem: {calib_display}",
        "5) Bayesian Prior Correction\n(Saerens 2002 EM)",
        "6) OOF-Havuzlanmış\nEşik Araması",
        "7) Klinik Stres Testi\n(test-önseli simülasyonu)",
        "8) Değerlendirme\nF1 (birincil) + MCC (ikincil)",
    ]

    fig, ax = plt.subplots(figsize=(6.5, 11))
    n = len(steps)
    box_h = 0.085
    gap = (1.0 - n * box_h) / (n + 1)
    y = 1.0 - gap

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, n))
    for i, (text, color) in enumerate(zip(steps, colors)):
        y_top = y - i * (box_h + gap)
        box = FancyBboxPatch((0.08, y_top - box_h), 0.84, box_h,
                              boxstyle="round,pad=0.012,rounding_size=0.02",
                              linewidth=1.4, edgecolor=color, facecolor=color, alpha=0.18)
        ax.add_patch(box)
        ax.text(0.5, y_top - box_h / 2, text, ha='center', va='center',
                fontsize=9.3, color='black', fontweight='medium')
        if i < n - 1:
            y_arrow_top = y_top - box_h
            y_arrow_bot = y_top - box_h - gap + 0.006
            arrow = FancyArrowPatch((0.5, y_arrow_top), (0.5, y_arrow_bot),
                                     arrowstyle='-|>', mutation_scale=14,
                                     color='#444444', linewidth=1.3)
            ax.add_patch(arrow)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f"{panel_name} — Metodoloji Akışı (Şartname v2.0 uyumlu)", fontsize=12)
    fig.tight_layout()
    return _save(fig, output_dir, '03_pipeline_schema.png')


# ══════════════════════════════════════════════
# 5. 04 — GERÇEK ROC EĞRİSİ
# ══════════════════════════════════════════════
def plot_real_roc(oof_labels, oof_probs_corrected, output_dir, panel_name="?", prefix="panel"):
    """
    [Not] ROC eğrisi prior-bağımsızdır (TPR/FPR sınıf dengesinden etkilenmez
    ve prior_correction() p'ye göre monoton bir dönüşüm olduğundan AUC, ham
    ve düzeltilmiş havuzda BİREBİR aynıdır) — bu yüzden tek eğri yeterlidir
    ve yeniden örnekleme (stres testi) GEREKTİRMEZ.
    """
    fpr, tpr, _ = roc_curve(oof_labels, oof_probs_corrected)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    ax.plot(fpr, tpr, color=COLOR_PATHO, linewidth=2.4, label=f"OOF ROC (AUC={roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle='--', color=COLOR_BEFORE, linewidth=1.2, label='Şans seviyesi (AUC=0.50)')
    ax.set_xlabel('Yanlış Pozitif Oranı (1 − Specificity)')
    ax.set_ylabel('Doğru Pozitif Oranı (Sensitivity / Recall)')
    ax.set_title(f"{panel_name} — Gerçek ROC Eğrisi (OOF-Havuzlanmış)")
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    return _save(fig, output_dir, '04_real_roc_curve.png')


# ══════════════════════════════════════════════
# 6. 05 — GERÇEK PR EĞRİSİ (ham OOF vs stres-testi-simülasyonu)
# ══════════════════════════════════════════════
def plot_pr_curve(oof_labels, oof_probs_corrected, test_prior, output_dir,
                   panel_name="?", prefix="panel", seed=None):
    """
    PR eğrisi prior'a HASSASTIR; bu yüzden İKİ eğri çizilir:
      (a) Ham OOF havuzu (eğitim-önseli)         — gri, referans
      (b) Klinik Stres Testi simülasyonu (test-önseli) — yeşil, GÜVENİLEN
    (b), helixai_common.simulate_clinical_stress_test ile AYNI yeniden
    örnekleme mantığını (aynı seed) kullanır — böylece bu grafikteki eğri,
    report['stress_test']['after'] sayılarıyla AYNI popülasyonu temsil eder.
    """
    seed = seed if seed is not None else hc.SEED
    prec_raw, rec_raw, _ = precision_recall_curve(oof_labels, oof_probs_corrected)

    p_sim, y_sim, bootstrap_used = hc.simulate_clinical_stress_test(
        oof_probs_corrected, oof_labels, test_prior, seed=seed)
    prec_sim, rec_sim, _ = precision_recall_curve(y_sim, p_sim)

    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    ax.plot(rec_raw, prec_raw, color=COLOR_BEFORE, linewidth=1.8, linestyle='--',
            label=f"OOF Havuzu (eğitim-önseli, π≈{oof_labels.mean():.3f})")
    ax.plot(rec_sim, prec_sim, color=COLOR_AFTER, linewidth=2.4,
            label=f"Klinik Stres Testi (test-önseli, π={test_prior:.3f})")
    ax.axhline(test_prior, color=COLOR_AFTER, linewidth=0.9, linestyle=':',
               label=f"Şans seviyesi (π_test={test_prior:.3f})")
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
    title_suffix = " [bootstrap kullanıldı]" if bootstrap_used else ""
    ax.set_title(f"{panel_name} — Gerçek PR Eğrisi{title_suffix}")
    ax.legend(loc='lower left', fontsize=8.3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    return _save(fig, output_dir, '05_pr_curve.png')


# ══════════════════════════════════════════════
# 7. 06 — GERÇEK CONFUSION MATRIX (analyze_report.py ile BİREBİR tutarlı)
# ══════════════════════════════════════════════
def _draw_cm(ax, m, title, color):
    if m is None:
        ax.text(0.5, 0.5, "Bu blok mevcut değil (None)", ha='center', va='center')
        ax.axis('off'); ax.set_title(title)
        return
    tp, fp, fn, tn = m['tp'], m['fp'], m['fn'], m['tn']
    cm = np.array([[tp, fn], [fp, tn]])  # satır=gerçek(Patho,Benign), kolon=tahmin(Patho,Benign)
    im = ax.imshow(cm, cmap=plt.cm.Blues if color == COLOR_AFTER else plt.cm.Greys, alpha=0.85)
    labels = [['TP', 'FN'], ['FP', 'TN']]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]}", ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color='white' if cm[i, j] > cm.max() * 0.5 else 'black')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Tahmin:\nPatojenik', 'Tahmin:\nBenign'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Gerçek:\nPatojenik', 'Gerçek:\nBenign'])
    f1 = m.get('f1', float('nan')); mcc = m.get('mcc', float('nan'))
    ax.set_title(f"{title}\nF1={f1:.3f} | MCC={mcc:.3f}", fontsize=10.5)


def plot_confusion_matrix(report, output_dir, panel_name="?", prefix="panel"):
    """
    [Tutarlılık garantisi] tp/fp/fn/tn değerleri report['stress_test']
    içinden DOĞRUDAN okunur, burada YENİDEN HESAPLANMAZ — bu grafik ile
    `python3 analyze_report.py <rapor.json>` çıktısı HER ZAMAN birebir
    aynı sayıları gösterir.
    """
    st = report.get('stress_test', {})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    _draw_cm(axes[0], st.get('before'), "Stres Testi ÖNCESİ\n(ham olasılık)", COLOR_BEFORE)
    _draw_cm(axes[1], st.get('after'),  "Stres Testi SONRASI\n(düzeltilmiş, test-önseli simülasyonu)", COLOR_AFTER)
    fig.suptitle(f"{panel_name} — Gerçek Confusion Matrix (PDR'da güvenilecek: SAĞ panel)",
                 fontsize=11.5, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return _save(fig, output_dir, '06_confusion_matrix.png')


# ══════════════════════════════════════════════
# 8. 07 — GERÇEK CALIBRATION (RELIABILITY) EĞRİSİ
# ══════════════════════════════════════════════
def plot_calibration_curve(oof_labels, oof_probs_raw, output_dir, panel_name="?", prefix="panel", n_bins=10):
    """
    [KRİTİK] HAM (düzeltme ÖNCESİ) olasılıklar kullanılır — bkz. modül başı
    not. Düzeltilmiş olasılıklar, eğitim-önselini taşıyan OOF etiketlerine
    göre kasıtlı olarak "yanlış kalibre" görünür (çünkü farklı bir prior
    için tasarlanmıştır); bu eğri SADECE ham model çıktısının kalitesini
    ölçer.
    """
    frac_pos, mean_pred = calibration_curve(oof_labels, oof_probs_raw, n_bins=n_bins, strategy='quantile')
    brier = hc.brier_score_loss(oof_labels, oof_probs_raw)

    fig = plt.figure(figsize=(6.5, 6.5))
    gs = fig.add_gridspec(4, 1, hspace=0.05)
    ax1 = fig.add_subplot(gs[:3, 0])
    ax2 = fig.add_subplot(gs[3, 0], sharex=ax1)

    ax1.plot([0, 1], [0, 1], linestyle='--', color=COLOR_BEFORE, linewidth=1.2, label='Mükemmel kalibrasyon')
    ax1.plot(mean_pred, frac_pos, marker='o', color=COLOR_PATHO, linewidth=2, markersize=6,
              label=f'Ham OOF (Brier={brier:.4f})')
    ax1.set_ylabel('Gözlenen Patojenik Oranı')
    ax1.set_title(f"{panel_name} — Reliability Eğrisi (HAM olasılıklar, düzeltme öncesi)", fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_xlim(-0.02, 1.02); ax1.set_ylim(-0.02, 1.02)
    ax1.tick_params(labelbottom=False)

    ax2.hist(oof_probs_raw, bins=30, range=(0, 1), color=COLOR_ACCENT, alpha=0.75)
    ax2.set_xlabel('Tahmin Edilen Olasılık (ham, P(Patojenik))')
    ax2.set_ylabel('Sayım')
    ax2.set_xlim(-0.02, 1.02)

    fig.text(0.5, 0.005,
             "Not: Bayes prior correction sonrası olasılıklar KASITLI olarak farklı bir önsele "
             "(test_prior) kaydırılır; bu nedenle kalibrasyon kontrolü HAM olasılıklarla yapılır.",
             ha='center', fontsize=7.3, color='#555555', wrap=True)
    return _save(fig, output_dir, '07_calibration_curve.png')


# ══════════════════════════════════════════════
# 9. 08 — GERÇEK THRESHOLD (EŞİK) DAĞILIMI
# ══════════════════════════════════════════════
def plot_real_threshold_distribution(oof_labels, oof_probs_corrected, final_threshold,
                                      output_dir, panel_name="?", prefix="panel"):
    """
    Düzeltilmiş (corrected) olasılıkların gerçek etikete göre dağılımı +
    üretimde kullanılan final_threshold çizgisi. Sınıfların ne kadar
    ayrıştığını ve seçilen eşiğin bu ayrışmaya göre nereye düştüğünü
    gösterir (final_threshold, düzeltilmiş olasılıklar üzerinde aranmıştır
    — bkz. helixai_common.compute_oof_pooled_threshold).
    """
    p_patho  = oof_probs_corrected[oof_labels == 1]
    p_benign = oof_probs_corrected[oof_labels == 0]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    bins = np.linspace(0, 1, 41)
    ax.hist(p_benign, bins=bins, alpha=0.55, color=COLOR_BENIGN, label=f'Gerçek: Benign (n={len(p_benign)})', density=True)
    ax.hist(p_patho,  bins=bins, alpha=0.55, color=COLOR_PATHO,  label=f'Gerçek: Patojenik (n={len(p_patho)})', density=True)
    ax.axvline(final_threshold, color='black', linewidth=2.2, linestyle='-',
               label=f'final_threshold = {final_threshold:.4f}')
    ax.set_xlabel('Düzeltilmiş Olasılık P_corr(Patojenik)')
    ax.set_ylabel('Yoğunluk (density)')
    ax.set_title(f"{panel_name} — Gerçek Eşik (Threshold) Dağılımı")
    ax.legend(loc='upper center', fontsize=9)
    ax.set_xlim(-0.01, 1.01)
    fig.tight_layout()
    return _save(fig, output_dir, '08_threshold_distribution.png')


# ══════════════════════════════════════════════
# 10. HTML DASHBOARD
# ══════════════════════════════════════════════
def generate_html_dashboard(report, output_dir, img_paths, panel_name="?", prefix=None):
    cv = report.get('cv_summary', {})
    st = report.get('stress_test', {})

    def fmt(m, k):
        v = (m or {}).get(k)
        return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

    rows_cv = "".join(
        f"<tr><td>{k}</td><td>{v.get('mean', float('nan')):.4f}</td><td>±{v.get('std', float('nan')):.4f}</td></tr>"
        for k, v in cv.items()
    )
    rows_st = "".join(
        f"<tr><td>{k}</td><td>{fmt(st.get('before'), k)}</td><td>{fmt(st.get('after'), k)}</td></tr>"
        for k in ['f1', 'mcc', 'sensitivity', 'specificity', 'pr_auc']
    )
    img_blocks = "".join(
        f'<div class="card"><h3>{os.path.basename(p)}</h3>'
        f'<img src="{os.path.basename(p)}" alt="{name}"></div>'
        for name, p in img_paths.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8">
<title>HELIXAI — {panel_name} Akademik Görselleştirme Panosu</title>
<style>
  body {{ font-family: -apple-system, Arial, sans-serif; margin: 0; padding: 24px;
          background: #f5f6f8; color: #222; }}
  h1 {{ font-size: 22px; }}
  .meta {{ color: #666; margin-bottom: 18px; }}
  table {{ border-collapse: collapse; margin-bottom: 24px; background: white; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 14px; text-align: right; font-size: 13px; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ background: #2c3e50; color: white; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }}
  .card {{ background: white; border-radius: 8px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.12); }}
  .card h3 {{ font-size: 11px; color: #888; margin: 0 0 8px; font-weight: normal; }}
  .card img {{ width: 100%; border-radius: 4px; }}
  .note {{ font-size: 12px; color: #666; background: #fff8e1; border-left: 3px solid #f0ad4e;
           padding: 8px 12px; margin-bottom: 18px; }}
</style></head>
<body>
  <h1>HELIXAI — {panel_name} Paneli | Akademik Görselleştirme Panosu</h1>
  <div class="meta">Versiyon: {report.get('version','?')} | Bu sayfa yalnızca araştırma/PDR
   sunumu amaçlıdır; klinik karar amaçlı değildir.</div>

  <div class="note">PDR'da raporlanacak/güvenilecek sayılar genellikle <b>"Stres Testi SONRASI"</b>
   sütunudur — test setinin gerçek (benign-ağırlıklı) dağılımını simüle eder.</div>

  <h2>Repeated-CV Özet (Fold Ortalaması)</h2>
  <table><tr><th>Metrik</th><th>Ortalama</th><th>Std</th></tr>{rows_cv}</table>

  <h2>Klinik Stres Testi: Öncesi vs Sonrası</h2>
  <table><tr><th>Metrik</th><th>Öncesi (ham)</th><th>Sonrası (düzeltilmiş)</th></tr>{rows_st}</table>

  <h2>Grafikler</h2>
  <div class="grid">{img_blocks}</div>
</body></html>"""

    prefix = prefix or panel_name.lower()
    path = os.path.join(output_dir, f'{prefix}_dashboard.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  [Dashboard kaydedildi] {path}")
    return path


# ══════════════════════════════════════════════
# 11. ORKESTRATÖR — generate_all_visuals
# ══════════════════════════════════════════════
def generate_all_visuals(report, fold_results, output_dir, test_prior,
                          panel_name=None, prefix=None):
    """
    [Genel, model-sayısı-agnostik] Tüm panellerde AYNI ŞEKİLDE çağrılır.

    Yan etki: `report` sözlüğüne şu alanları MUTASYONLA ekler (in-place):
      - fold_f1_values / fold_mcc_values / fold_spec_values (LOOCV değilse)
      - oof_y_true / oof_y_prob  (pooled OOF, JSON-serileştirilebilir liste)
    Ayrıca aynı OOF dizilerini output_dir/{prefix}_oof_predictions.npz
    olarak da kaydeder (büyük panellerde JSON'ı şişirmeden yeniden-analiz
    imkânı için).

    Kaldırılanlar (ESKİ, simülasyon/sunum amaçlı): radar_stress_test,
    prior_piechart, threshold_illustration, prior_correction_roc.
    Bunların yerini GERÇEK (OOF-tabanlı) ROC/PR/CM/Calibration/Threshold
    grafikleri alır (04–08).
    """
    panel_name = panel_name or report.get('panel', '?')
    prefix     = prefix or panel_name.lower()

    print("\n" + "=" * 60)
    print(f"[{panel_name}] HELIXAI Akademik Görselleştirme")
    print("=" * 60)
    os.makedirs(output_dir, exist_ok=True)

    # ---- fold-bazlı metrik listelerini rapora ekle (yoksa) ----
    if not _fold_is_loocv(fold_results) and 'fold_f1_values' not in report:
        try:
            report['fold_f1_values']   = [r['metrics']['f1']          for r in fold_results]
            report['fold_mcc_values']  = [r['metrics']['mcc']         for r in fold_results]
            report['fold_spec_values'] = [r['metrics']['specificity'] for r in fold_results]
        except Exception as e:
            print(f"  [UYARI] fold_*_values eklenemedi: {type(e).__name__}: {e}")

    # ---- OOF havuzunu hesapla + rapora yaz (1. madde: 'Pipeline'a EKLE') ----
    oof_labels, oof_probs_raw, oof_probs_corr = get_oof_pools(fold_results)
    report['oof_y_true'] = oof_labels.tolist()
    report['oof_y_prob'] = oof_probs_corr.tolist()   # final pipeline'da KULLANILAN olasılık

    npz_path = os.path.join(output_dir, f'{prefix}_oof_predictions.npz')
    np.savez(npz_path, y_true=oof_labels, y_prob_corrected=oof_probs_corr, y_prob_raw=oof_probs_raw)
    print(f"  [OOF tahminleri kaydedildi] {npz_path} (n={len(oof_labels)})")

    final_threshold = report.get('config', {}).get('final_threshold_oof', 0.5)

    img_paths = {}
    img_paths['fold_dist']    = plot_fold_distributions(report, fold_results, output_dir, panel_name, prefix)
    img_paths['shap']         = plot_shap_top10(report, output_dir, panel_name, prefix)
    img_paths['pipeline']     = plot_pipeline_schema(report, output_dir, panel_name, prefix)
    img_paths['roc']          = plot_real_roc(oof_labels, oof_probs_corr, output_dir, panel_name, prefix)
    img_paths['pr']           = plot_pr_curve(oof_labels, oof_probs_corr, test_prior, output_dir, panel_name, prefix)
    img_paths['cm']           = plot_confusion_matrix(report, output_dir, panel_name, prefix)
    img_paths['calibration']  = plot_calibration_curve(oof_labels, oof_probs_raw, output_dir, panel_name, prefix)
    img_paths['threshold']    = plot_real_threshold_distribution(
        oof_labels, oof_probs_corr, final_threshold, output_dir, panel_name, prefix)

    generate_html_dashboard(report, output_dir, img_paths, panel_name, prefix)

    print(f"\n[{panel_name}] Tamamlandı ({len(img_paths)} grafik + 1 dashboard + 1 OOF .npz)")
    return img_paths
