"""
HELIXAI — PAH Sentetik Benign Üretimi (Generative Oversampling)
====================================================================
helixai_common.py VE pah_transfer_pipeline.py'ye bağımlıdır (AYNI klasörde
olmalı — bu dosyadan load_and_prep/align_columns/select_fixed_features/
fit_combined_scaler fonksiyonları yeniden kullanılıyor, kod tekrarı yok).

YÖNTEM: v5'teki basit "kopyala + küçük gürültü ekle" oversampling'in daha
GÜÇLÜ bir versiyonu. Rastgele kopyalama yerine, Master panelinin 782
benign örneğinin İSTATİSTİKSEL DAĞILIMINI bir Gaussian Mixture Model (GMM)
ile öğrenip, bu dağılımdan PAH için YENİ, gerçek bir örneğin birebir kopyası
OLMAYAN ama istatistiksel olarak tutarlı sentetik benign örnekler üretiyoruz.

NEDEN MASTER'DAN ÖĞRENİLİYOR (PAH'IN KENDİ 62 ÖRNEĞİNDEN DEĞİL):
PAH'ın kendi 62 benign örneğinden bir GMM fit etmek, zaten kıt olan bilgiyi
yeniden paketlemekten ibaret olurdu (yeni bilgi katmaz). Master'ın 782
örneği, PAH'tan 12 kat daha fazla VE aynı özellik şemasını paylaşıyor —
bu nedenle GERÇEKTEN BAĞIMSIZ bir istatistiksel bilgi kaynağıdır.

[DÜRÜSTLÜK NOTU — PDR'da belirtilmelidir] Bu yöntem, "Master'daki benign
varyantların istatistiksel profili, PAH genindeki benign varyantlara
genellenebilir" varsayımına dayanır. Bu varsayım kesin DOĞRU DEĞİLDİR —
genler arası biyolojik farklılık olabilir. Bu yüzden v5/transfer'le AYNI
CV fold'larında, AYNI şekilde ablation ile (sentetik örnek sayısı 0/30/62/124)
ampirik olarak test edilip, gerçekten katkı sağlayıp sağlamadığı ölçülür —
teoriden kabul edilmez.

[SIZINTI/LEAKAGE GÜVENCESİ]
  - GMM SADECE Master'ın TÜM verisiyle, PAH'a hiç bakmadan TEK SEFER fit edilir.
  - Sentetik örnekler SADECE her CV fold'unun EĞİTİM kısmına eklenir;
    validation/kalibrasyon kısmı HER ZAMAN sadece gerçek PAH örnekleridir.

Kullanım:
  python3 pah_synthetic_benign_pipeline.py --pah YARISMA_TRAIN_PAH.csv --master YARISMA_TRAIN_MASTER.csv
"""

import argparse, os, json, time
import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RepeatedStratifiedKFold

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import helixai_common as hc
import pah_transfer_pipeline as tp   # load_and_prep, align_columns, select_fixed_features, fit_combined_scaler, apply_scaler buradan yeniden kullanılıyor

# ══════════════════════════════════════════════
# SABİTLER
# ══════════════════════════════════════════════
SEED            = hc.SEED
N_SPLITS        = 5
N_REPEATS       = 10
THRESHOLD_STEP  = 0.005
K_MAX_FEATURES  = 40

DEFAULT_TEST_PRIOR = 100 / (100 + 250)
DEFAULT_SPW         = 1.0
W_XGB, W_LGB, W_CAT = 0.35, 0.30, 0.35

GMM_N_COMPONENTS    = 3      # Master benign dağılımı için karışım bileşeni sayısı
SYNTHETIC_CANDIDATES = (0, 30, 62, 124)   # ablation: 0=baseline, 124=2x gerçek benign sayısı

# ══════════════════════════════════════════════
# 1. GENERATİF BENIGN MODELİ (Master'dan, TEK SEFER)
# ══════════════════════════════════════════════
def fit_benign_generator(X_master_scaled, y_master, n_components=GMM_N_COMPONENTS, seed=SEED):
    print(f"\n  [Generatif Model] Master'ın {int((y_master==0).sum())} benign örneğiyle "
          f"GMM (n_components={n_components}) fit ediliyor...")
    X_benign = X_master_scaled[y_master.values == 0]
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag',
                           random_state=seed, reg_covar=1e-3)
    gmm.fit(X_benign)
    print(f"  [Generatif Model] Tamamlandı (log-likelihood/örnek: {gmm.score(X_benign):.3f}).")
    return gmm

def sample_synthetic_benign(gmm, n, columns, seed=SEED):
    if n <= 0:
        return pd.DataFrame(columns=columns)
    X_syn, _ = gmm.sample(n_samples=n)
    rng = np.random.RandomState(seed)
    # GMM örneklemesi deterministik değildir (sklearn'ün kendi RNG'si); ek
    # bir karıştırma YAPILMIYOR — gmm.sample zaten seed'e bağlı reprodüktif.
    return pd.DataFrame(X_syn, columns=columns)

# ══════════════════════════════════════════════
# 2. MODEL YAPILARI (v5 ile aynı, tutarlılık için)
# ══════════════════════════════════════════════
def build_xgb(spw):
    return xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                              subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                              eval_metric='aucpr', random_state=SEED, verbosity=0)

def build_lgb(spw):
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                               subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                               metric='average_precision', random_state=SEED, verbose=-1)

def build_cat(spw):
    return CatBoostClassifier(iterations=300, learning_rate=0.05, depth=4,
                               scale_pos_weight=spw, eval_metric='PRAUC',
                               random_seed=SEED, verbose=False)

# ══════════════════════════════════════════════
# 3. TEK FOLD EĞİTİMİ — n_synthetic ile parametrize
# ══════════════════════════════════════════════
def train_fold(X_tr_scaled, y_tr, X_val_scaled, y_val, train_prior, test_prior,
               gmm, n_synthetic, spw=DEFAULT_SPW, fold_id="?"):
    # [LEAKAGE GÜVENCESİ] sentetik örnekler SADECE X_tr'ye eklenir, X_val'e ASLA dokunulmaz.
    if n_synthetic > 0:
        X_syn = sample_synthetic_benign(gmm, n_synthetic, X_tr_scaled.columns,
                                         seed=SEED + hash(fold_id) % 10000)
        y_syn = pd.Series([0] * n_synthetic)
        X_tr_aug = pd.concat([X_tr_scaled.reset_index(drop=True), X_syn], axis=0, ignore_index=True)
        y_tr_aug = pd.concat([y_tr.reset_index(drop=True), y_syn], axis=0, ignore_index=True)
    else:
        X_tr_aug, y_tr_aug = X_tr_scaled, y_tr

    models_raw = {'xgb': build_xgb(spw), 'lgb': build_lgb(spw), 'cat': build_cat(spw)}
    models_raw['xgb'].fit(X_tr_aug, y_tr_aug)
    models_raw['lgb'].fit(X_tr_aug, y_tr_aug)
    models_raw['cat'].fit(X_tr_aug, y_tr_aug)

    weights = {'xgb': W_XGB, 'lgb': W_LGB, 'cat': W_CAT}
    models_cal = {name: hc.calibrate_fitted_model(mdl, X_val_scaled, y_val, method='sigmoid')
                  for name, mdl in models_raw.items()}

    p_ens  = hc.ensemble_predict_proba(models_cal, weights, X_val_scaled)
    p_corr = hc.prior_correction(p_ens, train_prior, test_prior)
    best_t, _ = hc.find_best_threshold(y_val, p_corr, step=THRESHOLD_STEP)
    metrics = hc.compute_metrics(y_val, p_corr, threshold=best_t)
    metrics['fold'] = fold_id
    metrics['n_synthetic'] = n_synthetic

    p_raw_ens = (W_XGB * models_raw['xgb'].predict_proba(X_val_scaled)[:, 1] +
                 W_LGB * models_raw['lgb'].predict_proba(X_val_scaled)[:, 1] +
                 W_CAT * models_raw['cat'].predict_proba(X_val_scaled)[:, 1])

    return {'metrics': metrics, 'models_cal': models_cal, 'weights': weights,
            'threshold': best_t, 'val_probs_corrected': p_corr, 'val_probs_raw': p_raw_ens,
            'val_labels': y_val.values if hasattr(y_val, 'values') else y_val}

# ══════════════════════════════════════════════
# 4. ABLATION: kaç sentetik örnek en iyisi?
# ══════════════════════════════════════════════
def ablate_synthetic_count(X_pah_scaled, y_pah, test_prior, gmm,
                            candidates=SYNTHETIC_CANDIDATES, spw=DEFAULT_SPW,
                            n_splits=N_SPLITS, n_repeats=2):
    print("\n" + "="*60)
    print(f"ABLATION: Sentetik benign sayısı ({n_repeats}x{n_splits}-Fold)")
    print(f"  Adaylar: {candidates}")
    print("="*60)
    train_prior = hc.compute_train_prior(y_pah)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    results = {}
    for n_syn in candidates:
        f1s = []
        for tr_idx, val_idx in rskf.split(X_pah_scaled, y_pah):
            X_tr, X_val = X_pah_scaled.iloc[tr_idx], X_pah_scaled.iloc[val_idx]
            y_tr, y_val = y_pah.iloc[tr_idx], y_pah.iloc[val_idx]
            res = train_fold(X_tr, y_tr, X_val, y_val, train_prior, test_prior,
                              gmm, n_syn, spw=spw, fold_id=f"ablation_n{n_syn}")
            f1s.append(res['metrics']['f1'])
        results[n_syn] = {'mean_f1': float(np.mean(f1s)), 'std_f1': float(np.std(f1s))}
        print(f"  n_synthetic={n_syn:<4} | F1={results[n_syn]['mean_f1']:.4f} ± {results[n_syn]['std_f1']:.4f}")
    best_n = max(results, key=lambda k: results[k]['mean_f1'])
    print(f"\n  [Ablation] En iyi n_synthetic: {best_n} (F1={results[best_n]['mean_f1']:.4f})")
    return best_n, results

# ══════════════════════════════════════════════
# 5. KARŞILAŞTIRMALI TAM CV — baseline (n=0) vs en iyi n_synthetic
# ══════════════════════════════════════════════
def run_comparison_cv(X_pah_scaled, y_pah, test_prior, gmm, best_n, spw=DEFAULT_SPW):
    print("\n" + "="*60)
    print(f"KARŞILAŞTIRMALI TAM CV — baseline (n=0) vs n_synthetic={best_n} "
          f"({N_SPLITS}-Fold x {N_REPEATS} tekrar, AYNI bölünmeler)")
    print("="*60)
    train_prior = hc.compute_train_prior(y_pah)
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    results_baseline, results_synthetic = [], []
    for fold_idx, (tr_idx, val_idx) in enumerate(rskf.split(X_pah_scaled, y_pah)):
        label = f"R{fold_idx//N_SPLITS+1}F{fold_idx%N_SPLITS+1}"
        X_tr, X_val = X_pah_scaled.iloc[tr_idx], X_pah_scaled.iloc[val_idx]
        y_tr, y_val = y_pah.iloc[tr_idx], y_pah.iloc[val_idx]
        res_b = train_fold(X_tr, y_tr, X_val, y_val, train_prior, test_prior, gmm, 0, spw=spw, fold_id=label)
        res_s = train_fold(X_tr, y_tr, X_val, y_val, train_prior, test_prior, gmm, best_n, spw=spw, fold_id=label)
        results_baseline.append(res_b); results_synthetic.append(res_s)
        print(f"  {label} | Baseline F1={res_b['metrics']['f1']:.4f} MCC={res_b['metrics']['mcc']:.4f} "
              f"| Sentetik(+{best_n}) F1={res_s['metrics']['f1']:.4f} MCC={res_s['metrics']['mcc']:.4f}")
    return results_baseline, results_synthetic, train_prior

def summarize(fold_results):
    keys = ['f1', 'mcc', 'pr_auc', 'roc_auc', 'sensitivity', 'specificity', 'balanced_acc', 'brier']
    return {k: {'mean': float(np.mean([r['metrics'][k] for r in fold_results])),
                'std': float(np.std([r['metrics'][k] for r in fold_results]))} for k in keys}

# ══════════════════════════════════════════════
# ANA PIPELINE
# ══════════════════════════════════════════════
def run_synthetic_experiment(pah_path, master_path, output_dir='pah_synthetic_results',
                              label_col='Label', test_prior=DEFAULT_TEST_PRIOR, spw=DEFAULT_SPW):
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    print("\n" + "#"*60)
    print("HELIXAI -- PAH SENTETİK BENIGN ÜRETİMİ DENEYİ (Master -> PAH, GMM)")
    print("#"*60)

    print("\nADIM 1: Veri Yükleme + Ön İşleme")
    X_master, y_master = tp.load_and_prep(master_path, label_col)
    X_pah, y_pah        = tp.load_and_prep(pah_path, label_col)
    print(f"  Master: {X_master.shape} | Patojenik/Benign = {(y_master==1).sum()}/{(y_master==0).sum()}")
    print(f"  PAH   : {X_pah.shape} | Patojenik/Benign = {(y_pah==1).sum()}/{(y_pah==0).sum()}")

    print("\nADIM 2: Kolon Hizalama")
    X_master, X_pah = tp.align_columns(X_master, X_pah)

    print("\nADIM 3: Sabit Özellik Seçimi")
    sel_feats = tp.select_fixed_features(X_pah, y_pah, k_max=K_MAX_FEATURES)
    X_master_sel, X_pah_sel = X_master[sel_feats], X_pah[sel_feats]

    print("\nADIM 4: Birleşik Ölçeklendirme")
    imp, sc = tp.fit_combined_scaler(X_master_sel, X_pah_sel)
    X_master_scaled = tp.apply_scaler(X_master_sel, imp, sc)
    X_pah_scaled    = tp.apply_scaler(X_pah_sel, imp, sc)

    print("\nADIM 5: Generatif Benign Modeli (Master'dan)")
    gmm = fit_benign_generator(X_master_scaled, y_master)

    best_n, ablation_results = ablate_synthetic_count(X_pah_scaled, y_pah, test_prior, gmm, spw=spw)

    results_baseline, results_synthetic, train_prior = run_comparison_cv(
        X_pah_scaled, y_pah, test_prior, gmm, best_n, spw=spw
    )
    summary_baseline = summarize(results_baseline)
    summary_synthetic = summarize(results_synthetic)

    print("\n" + "─"*60)
    print("CV ÖZET KARŞILAŞTIRMASI:")
    print(f"  {'Metrik':<14}{'Baseline':>14}{'Sentetik':>14}{'Fark':>12}")
    for k in summary_baseline:
        b, s = summary_baseline[k]['mean'], summary_synthetic[k]['mean']
        print(f"  {k:<14}{b:>14.4f}{s:>14.4f}{s-b:>+12.4f}")

    print("\n  -- STRES TESTİ: BASELINE --")
    stress_b = hc.run_stress_test_comparison(results_baseline, test_prior, panel_name="PAH-Baseline")
    print("\n  -- STRES TESTİ: SENTETİK --")
    stress_s = hc.run_stress_test_comparison(results_synthetic, test_prior, panel_name="PAH-Sentetik")

    print(f"\n  {'Metrik':<14}{'Baseline(after)':>18}{'Sentetik(after)':>18}{'Fark':>10}")
    for k in ['f1', 'mcc', 'sensitivity', 'specificity']:
        b, s = stress_b['after'][k], stress_s['after'][k]
        print(f"  {k:<14}{b:>18.4f}{s:>18.4f}{s-b:>+10.4f}")

    elapsed = time.time() - t0
    verdict = "SENTETİK BENIGN FAYDALI" if stress_s['after']['mcc'] > stress_b['after']['mcc'] else "ANLAMLI FAYDA SAĞLAMADI"
    print(f"\n[SONUÇ] {verdict} (MCC farkı: {stress_s['after']['mcc']-stress_b['after']['mcc']:+.4f}) | Süre: {elapsed/60:.1f} dk")

    report = {
        'experiment': 'Master_to_PAH_synthetic_benign_GMM',
        'config': {'gmm_n_components': GMM_N_COMPONENTS, 'best_n_synthetic': best_n,
                   'k_max_features': K_MAX_FEATURES, 'selected_features': sel_feats,
                   'spw': spw, 'test_prior': test_prior, 'train_prior': train_prior},
        'ablation_n_synthetic': ablation_results,
        'cv_summary_baseline': summary_baseline, 'cv_summary_synthetic': summary_synthetic,
        'stress_test_baseline': {k: v for k, v in stress_b['after'].items() if isinstance(v, (int, float))},
        'stress_test_synthetic': {k: v for k, v in stress_s['after'].items() if isinstance(v, (int, float))},
        'verdict': verdict, 'elapsed_min': round(elapsed/60, 2),
    }
    with open(os.path.join(output_dir, 'pah_synthetic_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nRapor kaydedildi: {output_dir}/pah_synthetic_report.json")
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PAH Sentetik Benign Üretimi Deneyi')
    parser.add_argument('--pah', type=str, required=True)
    parser.add_argument('--master', type=str, required=True)
    parser.add_argument('--label', type=str, default='Label')
    parser.add_argument('--output', type=str, default='pah_synthetic_results')
    parser.add_argument('--test-prior', type=float, default=DEFAULT_TEST_PRIOR)
    parser.add_argument('--spw', type=float, default=DEFAULT_SPW)
    args = parser.parse_args()
    run_synthetic_experiment(args.pah, args.master, args.output, args.label, args.test_prior, args.spw)
