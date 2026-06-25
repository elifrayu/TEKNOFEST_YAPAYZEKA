"""
HELIXAI — PAH Transfer Öğrenme Deneyi (Master -> PAH)
=========================================================
helixai_common.py'ye bağımlıdır (AYNI klasörde olmalı).

AMAÇ: "PAH'ta benign veri kıtlığı (n=62) duvarını, Master panelinin daha
büyük benign havuzunu (n=782) kullanarak transfer öğrenmeyle aşabilir miyiz?"
hipotezini AMPİRİK olarak test etmek.

YÖNTEM (gradient boosting'e özgü "continued training" / fine-tuning):
  1. XGBoost/LightGBM/CatBoost'un HER BİRİ, önceden eğitilmiş bir modelden
     devam ederek eğitime devam edebilir (xgb_model= / init_model=
     parametreleri — bu üçü de doğrulanmış, gerçek API'ler, simülasyon
     değil).
  2. PRETRAIN: Master'ın TÜM verisiyle (2931 satır, 782 benign) N_PRETRAIN_
     TREES ağaçlık bir "temel" model eğitilir — genel, gen-bağımsız
     patojenite örüntülerini öğrenir.
  3. FINE-TUNE: Her PAH CV fold'unda, bu temel modelin ÜZERİNE, SADECE
     PAH'ın o fold'undaki eğitim verisiyle N_FINETUNE_TREES KADAR EK ağaç
     eklenir — PAH'a (PAH genine) özgü ince ayar yapılır.
  4. KARŞILAŞTIRMA: Transfer öğrenmeli model ile v5'teki "sıfırdan eğit"
     (baseline) modeli, AYNI CV fold'ları üzerinde yan yana çalıştırılıp
     F1/MCC/specificity karşılaştırılır — hipotez teoriden kabul edilmez,
     ampirik olarak doğrulanır veya reddedilir.

ÖNEMLİ MİMARİ KARARLAR (dürüstlük notları):
  - Master ve PAH AYRI ayrı preprocessing'den geçirilir (her ikisi de kendi
    global_col_filter eksiklik eşiğine göre farklı kolonlar düşürebilir);
    bu nedenle iki veri setinin SADECE ORTAK kolonları (intersection)
    kullanılır — aksi halde pretrain/fine-tune arasında özellik uyumsuzluğu
    (kolon sayısı/sırası farkı) modelleri bozar.
  - Özellik SEÇİMİ (k_max ile sınırlı SHAP/gain sıralaması) SADECE PAH
    verisiyle, TEK SEFER yapılır (CFTR'deki "global feature selection"
    mantığıyla aynı) — hedef alan (target domain) PAH olduğu için özellik
    seçimi PAH'ın kendi sinyaline göre yapılmalıdır, fold-içi dinamik
    seçim YOKTUR (pretrain/fine-tune arasında kolon tutarlılığı şarttır).
  - Imputer/Scaler, Master VE PAH'ın BİRLEŞİK havuzu üzerinde TEK SEFER fit
    edilir — pretrain ve fine-tune AYNI ölçeklendirmeyi kullanmalıdır,
    aksi halde "Master'ın 3.2 değeri" ile "PAH'ın 3.2 değeri" farklı
    şeyler ifade eder ve transfer anlamsızlaşır.
  - Continued-training API'leri (xgb_model=/init_model=) sürüm farkına
    karşı try/except ile sarmalıdır; başarısız olursa o model için
    SESSİZCE "sıfırdan eğit" (fine-tune'suz) moduna düşülür — pipeline
    çökmez.

Kullanım:
  python3 pah_transfer_pipeline.py --pah YARISMA_TRAIN_PAH.csv --master YARISMA_TRAIN_MASTER.csv
"""

import argparse, os, json, time
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import helixai_common as hc

# ══════════════════════════════════════════════
# SABİTLER
# ══════════════════════════════════════════════
SEED              = hc.SEED
N_SPLITS          = 5
N_REPEATS         = 10
THRESHOLD_STEP    = 0.005
K_MAX_FEATURES    = 40

DEFAULT_TEST_PRIOR = 100 / (100 + 250)   # PAH şartname onaylı, ≈0.2857
DEFAULT_SPW         = 1.0

W_XGB, W_LGB, W_CAT = 0.35, 0.30, 0.35

N_PRETRAIN_TREES  = 250   # Master üzerinde "temel" model — genel örüntüler
N_FINETUNE_TREES  = 60    # PAH fold'u üzerinde EK ağaç — gene-özgü ince ayar

# ══════════════════════════════════════════════
# 1. VERİ HAZIRLAMA (her panel kendi pipeline'ından geçer)
# ══════════════════════════════════════════════
def load_and_prep(path, label_col='Label'):
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c != label_col]
    tr_feats, _ = hc.global_col_filter(df[feat_cols])
    tr_feats = hc.biological_qc(tr_feats)
    tr_feats = hc.engineer_aa_features(tr_feats)
    tr_feats = hc.log_transform_af(tr_feats)
    y = df[label_col].astype(int)
    X = hc.to_numeric_df(tr_feats, label_col=label_col)
    return X, y

def align_columns(X_master, X_pah):
    """Sadece İKİ panelde de ORTAK olan kolonları kullan (kolon uyuşmazlığını önler)."""
    common = sorted(set(X_master.columns) & set(X_pah.columns))
    print(f"  [Hizalama] Master kolonu={X_master.shape[1]} | PAH kolonu={X_pah.shape[1]} "
          f"| ORTAK={len(common)}")
    return X_master[common].copy(), X_pah[common].copy()

# ══════════════════════════════════════════════
# 2. SABİT ÖZELLİK SEÇİMİ (PAH'ın kendi sinyaline göre, TEK SEFER)
# ══════════════════════════════════════════════
def select_fixed_features(X_pah, y_pah, k_max=K_MAX_FEATURES, seed=SEED):
    print("\n  [Sabit özellik seçimi] PAH verisiyle, transfer-uyumlu TEK SEFER seçim...")
    num_cols = X_pah.select_dtypes(include=[np.number]).columns.tolist()
    imp, sc = SimpleImputer(strategy='median'), RobustScaler()
    X2 = X_pah.copy()
    X2[num_cols] = imp.fit_transform(X2[num_cols])
    X2[num_cols] = sc.fit_transform(X2[num_cols])
    X2_f, _ = hc.correlation_filter(X2)
    Xs_tr, Xs_val, ys_tr, ys_val = train_test_split(X2_f, y_pah, test_size=0.2,
                                                      stratify=y_pah, random_state=seed)
    sel = hc.shap_feature_selection(Xs_tr, ys_tr, Xs_val, ys_val, k_max=k_max)
    print(f"  [Sabit özellik seçimi] {len(sel)} özellik seçildi (k_max={k_max}).")
    return sel

# ══════════════════════════════════════════════
# 3. BİRLEŞİK SCALER/IMPUTER (Master + PAH havuzu, TEK SEFER)
# ══════════════════════════════════════════════
def fit_combined_scaler(X_master_sel, X_pah_sel):
    combined = pd.concat([X_master_sel, X_pah_sel], axis=0)
    imp = SimpleImputer(strategy='median')
    sc  = RobustScaler()
    imp.fit(combined)
    sc.fit(imp.transform(combined))
    return imp, sc

def apply_scaler(X_sel, imp, sc):
    X_out = X_sel.copy()
    X_out[:] = sc.transform(imp.transform(X_sel))
    return X_out

# ══════════════════════════════════════════════
# 4. MODEL YAPILARI
# ══════════════════════════════════════════════
def build_xgb(spw, n_estimators):
    return xgb.XGBClassifier(
        n_estimators=n_estimators, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric='aucpr',
        random_state=SEED, verbosity=0,
    )

def build_lgb(spw, n_estimators):
    return lgb.LGBMClassifier(
        n_estimators=n_estimators, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, metric='average_precision',
        random_state=SEED, verbose=-1,
    )

def build_cat(spw, n_estimators):
    return CatBoostClassifier(
        iterations=n_estimators, learning_rate=0.05, depth=4,
        scale_pos_weight=spw, eval_metric='PRAUC',
        random_seed=SEED, verbose=False,
    )

# ══════════════════════════════════════════════
# 5. PRETRAIN — Master'ın TÜM verisiyle, TEK SEFER
# ══════════════════════════════════════════════
def pretrain_on_master(X_master_scaled, y_master, spw=DEFAULT_SPW):
    print(f"\n  [PRETRAIN] Master verisiyle ({len(X_master_scaled)} satır, "
          f"{N_PRETRAIN_TREES} ağaç) temel model eğitiliyor...")
    xgb_m = build_xgb(spw, N_PRETRAIN_TREES); xgb_m.fit(X_master_scaled, y_master)
    lgb_m = build_lgb(spw, N_PRETRAIN_TREES); lgb_m.fit(X_master_scaled, y_master)
    cat_m = build_cat(spw, N_PRETRAIN_TREES); cat_m.fit(X_master_scaled, y_master)
    print("  [PRETRAIN] Tamamlandı.")
    return {'xgb': xgb_m, 'lgb': lgb_m, 'cat': cat_m}

# ══════════════════════════════════════════════
# 6. FINE-TUNE — PAH fold'u üzerinde EK ağaçlarla devam eğitim
# ══════════════════════════════════════════════
def finetune_models(pretrained, X_tr_scaled, y_tr, spw=DEFAULT_SPW,
                     n_extra=N_FINETUNE_TREES, verbose_fallback=False):
    """
    [DÜRÜSTLÜK NOTU] xgb_model=/init_model= API'leri sürüme göre davranış
    farkı gösterebilir. Her model için ayrı try/except: başarısız olursa
    o TEK model için sessizce 'sıfırdan eğit' moduna (transfer'siz) düşülür
    — diğer 2 model transfer'i sürdürür, pipeline çökmez.
    """
    out = {}

    # XGBoost — xgb_model= ile devam eğitim
    try:
        m = build_xgb(spw, n_extra)
        m.fit(X_tr_scaled, y_tr, xgb_model=pretrained['xgb'].get_booster())
        out['xgb'] = m
    except Exception as e:
        if verbose_fallback:
            print(f"    [UYARI] XGB fine-tune başarısız ({type(e).__name__}: {e}) — sıfırdan eğitiliyor.")
        m = build_xgb(spw, N_PRETRAIN_TREES); m.fit(X_tr_scaled, y_tr)
        out['xgb'] = m

    # LightGBM — init_model= ile devam eğitim
    try:
        m = build_lgb(spw, n_extra)
        m.fit(X_tr_scaled, y_tr, init_model=pretrained['lgb'].booster_)
        out['lgb'] = m
    except Exception as e:
        if verbose_fallback:
            print(f"    [UYARI] LGB fine-tune başarısız ({type(e).__name__}: {e}) — sıfırdan eğitiliyor.")
        m = build_lgb(spw, N_PRETRAIN_TREES); m.fit(X_tr_scaled, y_tr)
        out['lgb'] = m

    # CatBoost — init_model= ile devam eğitim
    try:
        m = build_cat(spw, n_extra)
        m.fit(X_tr_scaled, y_tr, init_model=pretrained['cat'])
        out['cat'] = m
    except Exception as e:
        if verbose_fallback:
            print(f"    [UYARI] CAT fine-tune başarısız ({type(e).__name__}: {e}) — sıfırdan eğitiliyor.")
        m = build_cat(spw, N_PRETRAIN_TREES); m.fit(X_tr_scaled, y_tr)
        out['cat'] = m

    return out

# ══════════════════════════════════════════════
# 7. TEK FOLD EĞİTİMİ — strategy='transfer' veya 'baseline'
# ══════════════════════════════════════════════
def train_fold(X_tr_scaled, y_tr, X_val_scaled, y_val, train_prior, test_prior,
               strategy='transfer', pretrained=None, spw=DEFAULT_SPW, fold_id="?"):
    if strategy == 'transfer':
        models_raw = finetune_models(pretrained, X_tr_scaled, y_tr, spw=spw)
    else:
        models_raw = {
            'xgb': build_xgb(spw, N_PRETRAIN_TREES + N_FINETUNE_TREES),
            'lgb': build_lgb(spw, N_PRETRAIN_TREES + N_FINETUNE_TREES),
            'cat': build_cat(spw, N_PRETRAIN_TREES + N_FINETUNE_TREES),
        }
        models_raw['xgb'].fit(X_tr_scaled, y_tr)
        models_raw['lgb'].fit(X_tr_scaled, y_tr)
        models_raw['cat'].fit(X_tr_scaled, y_tr)

    weights = {'xgb': W_XGB, 'lgb': W_LGB, 'cat': W_CAT}
    models_cal = {name: hc.calibrate_fitted_model(mdl, X_val_scaled, y_val, method='sigmoid')
                  for name, mdl in models_raw.items()}

    p_ens  = hc.ensemble_predict_proba(models_cal, weights, X_val_scaled)
    p_corr = hc.prior_correction(p_ens, train_prior, test_prior)

    best_t, _ = hc.find_best_threshold(y_val, p_corr, step=THRESHOLD_STEP)
    metrics = hc.compute_metrics(y_val, p_corr, threshold=best_t)
    metrics['fold'] = fold_id
    metrics['n_features'] = X_tr_scaled.shape[1]

    p_raw_ens = (W_XGB * models_raw['xgb'].predict_proba(X_val_scaled)[:, 1] +
                 W_LGB * models_raw['lgb'].predict_proba(X_val_scaled)[:, 1] +
                 W_CAT * models_raw['cat'].predict_proba(X_val_scaled)[:, 1])

    return {
        'metrics': metrics, 'models_cal': models_cal, 'weights': weights,
        'threshold': best_t, 'val_probs_corrected': p_corr, 'val_probs_raw': p_raw_ens,
        'val_labels': y_val.values if hasattr(y_val, 'values') else y_val,
    }

# ══════════════════════════════════════════════
# 8. KARŞILAŞTIRMALI CV — transfer vs baseline, AYNI fold'larda
# ══════════════════════════════════════════════
def run_comparison_cv(X_pah_scaled, y_pah, test_prior, pretrained, spw=DEFAULT_SPW):
    print("\n" + "="*60)
    print(f"KARŞILAŞTIRMALI CV (Transfer vs Baseline) — {N_SPLITS}-Fold x {N_REPEATS} tekrar, "
          f"AYNI fold bölünmeleriyle")
    print("="*60)

    train_prior = hc.compute_train_prior(y_pah)
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)

    results_transfer, results_baseline = [], []

    for fold_idx, (tr_idx, val_idx) in enumerate(rskf.split(X_pah_scaled, y_pah)):
        repeat, fold = fold_idx // N_SPLITS + 1, fold_idx % N_SPLITS + 1
        label = f"R{repeat}F{fold}"
        X_tr, X_val = X_pah_scaled.iloc[tr_idx], X_pah_scaled.iloc[val_idx]
        y_tr, y_val = y_pah.iloc[tr_idx], y_pah.iloc[val_idx]

        res_t = train_fold(X_tr, y_tr, X_val, y_val, train_prior, test_prior,
                            strategy='transfer', pretrained=pretrained, spw=spw, fold_id=label)
        res_b = train_fold(X_tr, y_tr, X_val, y_val, train_prior, test_prior,
                            strategy='baseline', spw=spw, fold_id=label)
        results_transfer.append(res_t)
        results_baseline.append(res_b)

        print(f"  {label} | TRANSFER F1={res_t['metrics']['f1']:.4f} MCC={res_t['metrics']['mcc']:.4f} "
              f"| BASELINE F1={res_b['metrics']['f1']:.4f} MCC={res_b['metrics']['mcc']:.4f}")

    return results_transfer, results_baseline, train_prior

def summarize(fold_results):
    keys = ['f1', 'mcc', 'pr_auc', 'roc_auc', 'sensitivity', 'specificity', 'balanced_acc', 'brier']
    return {k: {'mean': float(np.mean([r['metrics'][k] for r in fold_results])),
                'std': float(np.std([r['metrics'][k] for r in fold_results]))}
            for k in keys}

# ══════════════════════════════════════════════
# ANA PIPELINE
# ══════════════════════════════════════════════
def run_transfer_experiment(pah_path, master_path, output_dir='pah_transfer_results',
                             label_col='Label', test_prior=DEFAULT_TEST_PRIOR, spw=DEFAULT_SPW):
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    print("\n" + "#"*60)
    print("HELIXAI -- PAH TRANSFER ÖĞRENME DENEYİ (Master -> PAH)")
    print("#"*60)

    print("\nADIM 1: Veri Yükleme + Ön İşleme (her panel kendi pipeline'ından)")
    X_master, y_master = load_and_prep(master_path, label_col)
    X_pah, y_pah        = load_and_prep(pah_path, label_col)
    print(f"  Master: {X_master.shape} | Patojenik/Benign = {(y_master==1).sum()}/{(y_master==0).sum()}")
    print(f"  PAH   : {X_pah.shape} | Patojenik/Benign = {(y_pah==1).sum()}/{(y_pah==0).sum()}")

    print("\nADIM 2: Kolon Hizalama (Master ∩ PAH)")
    X_master, X_pah = align_columns(X_master, X_pah)

    print("\nADIM 3: Sabit Özellik Seçimi")
    sel_feats = select_fixed_features(X_pah, y_pah, k_max=K_MAX_FEATURES)
    X_master_sel, X_pah_sel = X_master[sel_feats], X_pah[sel_feats]

    print("\nADIM 4: Birleşik Ölçeklendirme (Master + PAH havuzu, TEK SEFER)")
    imp, sc = fit_combined_scaler(X_master_sel, X_pah_sel)
    X_master_scaled = apply_scaler(X_master_sel, imp, sc)
    X_pah_scaled    = apply_scaler(X_pah_sel, imp, sc)

    print("\nADIM 5: Master Üzerinde Pretrain")
    pretrained = pretrain_on_master(X_master_scaled, y_master, spw=spw)

    results_transfer, results_baseline, train_prior = run_comparison_cv(
        X_pah_scaled, y_pah, test_prior, pretrained, spw=spw
    )

    summary_transfer = summarize(results_transfer)
    summary_baseline = summarize(results_baseline)

    print("\n" + "─"*60)
    print("CV ÖZET KARŞILAŞTIRMASI (50 fold, AYNI bölünmeler):")
    print(f"  {'Metrik':<14}{'Baseline':>14}{'Transfer':>14}{'Fark':>12}")
    for k in summary_baseline:
        b, t = summary_baseline[k]['mean'], summary_transfer[k]['mean']
        print(f"  {k:<14}{b:>14.4f}{t:>14.4f}{t-b:>+12.4f}")

    print("\n" + "="*60)
    print("STRES TESTİ KARŞILAŞTIRMASI (gerçek test dağılımı simülasyonu)")
    print("="*60)
    print("\n  -- BASELINE --")
    stress_baseline = hc.run_stress_test_comparison(results_baseline, test_prior, panel_name="PAH-Baseline")
    print("\n  -- TRANSFER --")
    stress_transfer = hc.run_stress_test_comparison(results_transfer, test_prior, panel_name="PAH-Transfer")

    print(f"\n  {'Metrik':<14}{'Baseline (after)':>18}{'Transfer (after)':>18}{'Fark':>10}")
    for k in ['f1', 'mcc', 'sensitivity', 'specificity']:
        b, t = stress_baseline['after'][k], stress_transfer['after'][k]
        print(f"  {k:<14}{b:>18.4f}{t:>18.4f}{t-b:>+10.4f}")

    elapsed = time.time() - t0
    verdict = "TRANSFER ÖĞRENME FAYDALI" if stress_transfer['after']['mcc'] > stress_baseline['after']['mcc'] else \
              "TRANSFER ÖĞRENME ANLAMLI FAYDA SAĞLAMADI"
    print(f"\n[SONUÇ] {verdict} (MCC farkı: {stress_transfer['after']['mcc'] - stress_baseline['after']['mcc']:+.4f})")
    print(f"Süre: {elapsed/60:.1f} dk")

    report = {
        'experiment': 'Master_to_PAH_transfer_learning',
        'config': {'n_pretrain_trees': N_PRETRAIN_TREES, 'n_finetune_trees': N_FINETUNE_TREES,
                   'k_max_features': K_MAX_FEATURES, 'selected_features': sel_feats,
                   'spw': spw, 'test_prior': test_prior, 'train_prior': train_prior},
        'cv_summary_baseline': summary_baseline,
        'cv_summary_transfer': summary_transfer,
        'stress_test_baseline': {k: v for k, v in stress_baseline['after'].items() if isinstance(v, (int, float))},
        'stress_test_transfer': {k: v for k, v in stress_transfer['after'].items() if isinstance(v, (int, float))},
        'verdict': verdict,
        'elapsed_min': round(elapsed / 60, 2),
    }
    rep_path = os.path.join(output_dir, 'pah_transfer_report.json')
    with open(rep_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nRapor kaydedildi: {rep_path}")
    return report

# ══════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PAH Transfer Öğrenme Deneyi (Master -> PAH)')
    parser.add_argument('--pah', type=str, required=True)
    parser.add_argument('--master', type=str, required=True)
    parser.add_argument('--label', type=str, default='Label')
    parser.add_argument('--output', type=str, default='pah_transfer_results')
    parser.add_argument('--test-prior', type=float, default=DEFAULT_TEST_PRIOR)
    parser.add_argument('--spw', type=float, default=DEFAULT_SPW)
    args = parser.parse_args()

    run_transfer_experiment(
        pah_path=args.pah, master_path=args.master, output_dir=args.output,
        label_col=args.label, test_prior=args.test_prior, spw=args.spw,
    )
