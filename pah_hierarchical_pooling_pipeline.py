"""
HELIXAI — PAH Hiyerarşik / Panel-Havuzlu Model (Partial Pooling)
====================================================================
helixai_common.py'ye bağımlıdır (AYNI klasörde olmalı).

YÖNTEM: İstatistikte "hiyerarşik/multilevel modelleme" veya "partial
pooling" olarak bilinen yaklaşımın, gradient boosting'e uyarlanmış bir
versiyonu. Tam Bayesgil hiyerarşik model (örn. PyMC/Stan ile) KASITLI
OLARAK kullanılmadı — hem yeni, ağır bir bağımlılık eklerdi (jüri ortamında
kurulu olmayabilir, reproducibility riski) hem de mevcut XGBoost/LightGBM/
CatBoost altyapısından kopardı. Bunun yerine, AYNI istatistiksel mantığı
("küçük grup tahminini, büyük/havuzlanmış grubun tahminine doğru çek")
PREDICTION-LEVEL SHRINKAGE ENSEMBLE ile uyguluyoruz:

    p_final = λ * p_GLOBAL + (1-λ) * p_LOCAL

  - p_LOCAL: SADECE PAH verisiyle eğitilmiş model (v5'teki baseline'ın AYNISI).
  - p_GLOBAL: SADECE DİĞER 3 PANELİN (Master+Kanser+CFTR) BİRLEŞİK verisiyle
    eğitilmiş, panel-bağımsız "genel patojenite" modeli — PAH'IN HİÇBİR
    SATIRINI GÖRMEZ (ne train ne val) — bu, λ ablation'ında PAH validation
    fold'larının global modele SIZMASINI engellemek için KASITLIDIR.
  - λ (shrinkage/karışım ağırlığı): teoriden SABİTLENMEZ, CV ile ampirik
    olarak taranır (λ=0 -> tam LOCAL/v5 baseline, λ=1 -> tam GLOBAL).

Bu, istatistikte James-Stein tahmincisi / empirical Bayes shrinkage ile
AYNI prensibe dayanır: az verisi olan bir grubun (PAH, n=372) tahminini,
çok verisi olan bir üst-grubun (4 panel havuzu, n=3802) tahminine doğru
ÇEKEREK varyansı azaltmak.

[DÜRÜSTLÜK NOTU] λ'nın optimal değeri TEORİDEN bilinemez — PAH'ın
diğer panellerle ne kadar "benzer" olduğuna bağlıdır. Bu yüzden λ ∈
{0.0, 0.1, ..., 0.5} ızgarası CV ile taranır; λ=0 kazanırsa bu, "panel-
havuzlama bu durumda işe yaramadı" anlamına gelir ve dürüstçe raporlanır.

Kullanım:
  python3 pah_hierarchical_pooling_pipeline.py --pah YARISMA_TRAIN_PAH.csv \\
      --master YARISMA_TRAIN_MASTER.csv --kanser YARISMA_TRAIN_KANSER.csv \\
      --cftr YARISMA_TRAIN_CFTR.csv
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
SEED            = hc.SEED
N_SPLITS        = 5
N_REPEATS       = 10
THRESHOLD_STEP  = 0.005
K_MAX_FEATURES  = 40

DEFAULT_TEST_PRIOR = 100 / (100 + 250)
DEFAULT_SPW         = 1.0
W_XGB, W_LGB, W_CAT = 0.35, 0.30, 0.35

LAMBDA_CANDIDATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)   # 0=tam LOCAL, 0.5=yarı yarıya

# ══════════════════════════════════════════════
# 1. VERİ HAZIRLAMA + HİZALAMA (4 panel)
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

def align_all(panels):
    """panels: {ad: (X,y)} -> ortak kolonlara hizalanmış {ad: (X,y)}"""
    common = None
    for name, (X, y) in panels.items():
        cols = set(X.columns)
        common = cols if common is None else (common & cols)
    common = sorted(common)
    print(f"  [Hizalama] {len(panels)} panel arasında ORTAK kolon sayısı: {len(common)}")
    return {name: (X[common].copy(), y) for name, (X, y) in panels.items()}, common

# ══════════════════════════════════════════════
# 2. SABİT ÖZELLİK SEÇİMİ (PAH'ın kendi sinyaline göre)
# ══════════════════════════════════════════════
def select_fixed_features(X_pah, y_pah, k_max=K_MAX_FEATURES, seed=SEED):
    num_cols = X_pah.select_dtypes(include=[np.number]).columns.tolist()
    imp, sc = SimpleImputer(strategy='median'), RobustScaler()
    X2 = X_pah.copy()
    X2[num_cols] = imp.fit_transform(X2[num_cols])
    X2[num_cols] = sc.fit_transform(X2[num_cols])
    X2_f, _ = hc.correlation_filter(X2)
    Xs_tr, Xs_val, ys_tr, ys_val = train_test_split(X2_f, y_pah, test_size=0.2,
                                                      stratify=y_pah, random_state=seed)
    sel = hc.shap_feature_selection(Xs_tr, ys_tr, Xs_val, ys_val, k_max=k_max)
    print(f"  [Sabit özellik seçimi] {len(sel)} özellik (k_max={k_max}).")
    return sel

def fit_combined_scaler(*X_sel_list):
    combined = pd.concat(X_sel_list, axis=0)
    imp, sc = SimpleImputer(strategy='median'), RobustScaler()
    imp.fit(combined); sc.fit(imp.transform(combined))
    return imp, sc

def apply_scaler(X_sel, imp, sc):
    out = X_sel.copy()
    out[:] = sc.transform(imp.transform(X_sel))
    return out

# ══════════════════════════════════════════════
# 3. MODEL YAPILARI
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

def fit_ensemble(X, y, spw):
    models = {'xgb': build_xgb(spw), 'lgb': build_lgb(spw), 'cat': build_cat(spw)}
    for m in models.values():
        m.fit(X, y)
    return models

def predict_ensemble(models, X):
    return (W_XGB * models['xgb'].predict_proba(X)[:, 1] +
            W_LGB * models['lgb'].predict_proba(X)[:, 1] +
            W_CAT * models['cat'].predict_proba(X)[:, 1])

# ══════════════════════════════════════════════
# 4. GLOBAL MODEL — 4 panelin TÜMÜYLE, TEK SEFER eğitilir
# ══════════════════════════════════════════════
def fit_global_model(X_all_panels_scaled, y_all_panels, spw=DEFAULT_SPW):
    print(f"\n  [GLOBAL MODEL] {len(X_all_panels_scaled)} satır (4 panel havuzu) ile eğitiliyor...")
    models = fit_ensemble(X_all_panels_scaled, y_all_panels, spw)
    print("  [GLOBAL MODEL] Tamamlandı.")
    return models

# ══════════════════════════════════════════════
# 5. TEK FOLD — LOCAL model + GLOBAL model + λ karışımı
# ══════════════════════════════════════════════
def train_fold(X_tr_scaled, y_tr, X_val_scaled, y_val, train_prior, test_prior,
               global_models, spw=DEFAULT_SPW, fold_id="?"):
    """Her fold'da LOCAL model SIFIRDAN PAH fold'unun eğitim kısmıyla eğitilir
    (v5 baseline ile AYNI); GLOBAL model fold-dışı, sabit, ÖNCEDEN eğitilmiştir.
    Kalibrasyon her ikisi için de PAH'ın val fold'unda yapılır."""
    local_models = fit_ensemble(X_tr_scaled, y_tr, spw)

    weights = {'xgb': W_XGB, 'lgb': W_LGB, 'cat': W_CAT}
    local_cal  = {n: hc.calibrate_fitted_model(m, X_val_scaled, y_val, method='sigmoid') for n, m in local_models.items()}
    global_cal = {n: hc.calibrate_fitted_model(m, X_val_scaled, y_val, method='sigmoid') for n, m in global_models.items()}

    p_local  = hc.ensemble_predict_proba(local_cal, weights, X_val_scaled)
    p_global = hc.ensemble_predict_proba(global_cal, weights, X_val_scaled)

    return p_local, p_global, y_val.values if hasattr(y_val, 'values') else y_val

# ══════════════════════════════════════════════
# 6. λ ABLATION — CV üzerinde en iyi karışım ağırlığını bul
# ══════════════════════════════════════════════
def ablate_lambda(X_pah_scaled, y_pah, test_prior, global_models,
                   candidates=LAMBDA_CANDIDATES, spw=DEFAULT_SPW,
                   n_splits=N_SPLITS, n_repeats=N_REPEATS):
    print("\n" + "="*60)
    print(f"λ (shrinkage) ABLATION — {n_repeats}x{n_splits}-Fold")
    print(f"  Adaylar: {candidates}")
    print("="*60)

    train_prior = hc.compute_train_prior(y_pah)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)

    all_p_local, all_p_global, all_labels = [], [], []
    for fold_idx, (tr_idx, val_idx) in enumerate(rskf.split(X_pah_scaled, y_pah)):
        X_tr, X_val = X_pah_scaled.iloc[tr_idx], X_pah_scaled.iloc[val_idx]
        y_tr, y_val = y_pah.iloc[tr_idx], y_pah.iloc[val_idx]
        p_l, p_g, labels = train_fold(X_tr, y_tr, X_val, y_val, train_prior, test_prior,
                                       global_models, spw=spw, fold_id=f"F{fold_idx}")
        all_p_local.append(p_l); all_p_global.append(p_g); all_labels.append(labels)

    p_local_pool  = np.concatenate(all_p_local)
    p_global_pool = np.concatenate(all_p_global)
    labels_pool   = np.concatenate(all_labels)

    results = {}
    for lam in candidates:
        p_mix = lam * p_global_pool + (1 - lam) * p_local_pool
        p_mix_corr = hc.prior_correction(p_mix, train_prior, test_prior)
        best_t, best_f1 = hc.find_best_threshold(labels_pool, p_mix_corr, step=THRESHOLD_STEP)
        m = hc.compute_metrics(labels_pool, p_mix_corr, threshold=best_t)
        results[lam] = {'f1': m['f1'], 'mcc': m['mcc'], 'specificity': m['specificity'],
                         'sensitivity': m['sensitivity'], 'threshold': best_t}
        print(f"  λ={lam:<4} | F1={m['f1']:.4f} | MCC={m['mcc']:.4f} | "
              f"specificity={m['specificity']:.4f} | sensitivity={m['sensitivity']:.4f}")

    best_lambda = max(results, key=lambda k: results[k]['mcc'])
    print(f"\n  [Ablation] En iyi λ (MCC'ye göre): {best_lambda} (MCC={results[best_lambda]['mcc']:.4f})")
    if best_lambda == 0.0:
        print("  Not: λ=0 kazandı — bu durumda panel-havuzlama FAYDA SAĞLAMADI, "
              "PAH'a özgü (local) model tek başına yeterli/daha iyi.")
    return best_lambda, results, (p_local_pool, p_global_pool, labels_pool, train_prior)

# ══════════════════════════════════════════════
# ANA PIPELINE
# ══════════════════════════════════════════════
def run_hierarchical_experiment(pah_path, master_path, kanser_path, cftr_path,
                                 output_dir='pah_hierarchical_results', label_col='Label',
                                 test_prior=DEFAULT_TEST_PRIOR, spw=DEFAULT_SPW):
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    print("\n" + "#"*60)
    print("HELIXAI -- PAH HİYERARŞİK / PANEL-HAVUZLU MODEL DENEYİ")
    print("#"*60)

    print("\nADIM 1: 4 Panelin Yüklenmesi + Ön İşlenmesi")
    panels_raw = {
        'pah':    load_and_prep(pah_path, label_col),
        'master': load_and_prep(master_path, label_col),
        'kanser': load_and_prep(kanser_path, label_col),
        'cftr':   load_and_prep(cftr_path, label_col),
    }
    for name, (X, y) in panels_raw.items():
        print(f"  {name:<8}: {X.shape} | Patojenik/Benign={int((y==1).sum())}/{int((y==0).sum())}")

    print("\nADIM 2: 4 Panel Arası Kolon Hizalama")
    panels_aligned, common_cols = align_all(panels_raw)

    print("\nADIM 3: Sabit Özellik Seçimi (PAH'ın kendi verisiyle)")
    X_pah, y_pah = panels_aligned['pah']
    sel_feats = select_fixed_features(X_pah, y_pah, k_max=K_MAX_FEATURES)

    print("\nADIM 4: Birleşik Ölçeklendirme (4 panel havuzu)")
    X_sel = {name: X[sel_feats] for name, (X, y) in panels_aligned.items()}
    imp, sc = fit_combined_scaler(*X_sel.values())
    X_scaled = {name: apply_scaler(X_sel[name], imp, sc) for name in X_sel}

    print("\nADIM 5: Global Model (SADECE diğer 3 panel — Master+Kanser+CFTR — TEK SEFER)")
    # [v1.1-KRİTİK DÜZELTME] ÖNCEKİ HATA: global model PAH'ın TÜM verisini de
    # (train+val) içeriyordu — bu, λ ablation'ında PAH validation fold'larını
    # global modelin ZATEN GÖRMÜŞ OLMASI anlamına geliyordu (sızıntı/leakage).
    # Sonuç: yapay olarak şişirilmiş (MCC +0.26 gibi gerçek olmayan) bir kazanç.
    # DÜZELTME: global model artık SADECE PAH-HARİCİ 3 panelle eğitiliyor —
    # PAH'ın hiçbir satırını (train da, val da) hiç görmüyor. Artık λ ablation'ı
    # sızıntısız.
    other_panels = [n for n in X_scaled if n != 'pah']
    print(f"  [DÜZELTME] Global model SADECE şu panellerle eğitiliyor: {other_panels} "
          f"(PAH TAMAMEN HARİÇ)")
    X_all = pd.concat([X_scaled[n] for n in other_panels], axis=0, ignore_index=True)
    y_all = pd.concat([panels_aligned[n][1].reset_index(drop=True) for n in other_panels], axis=0, ignore_index=True)
    global_models = fit_global_model(X_all, y_all, spw=spw)

    best_lambda, lambda_results, (p_local_pool, p_global_pool, labels_pool, train_prior) = ablate_lambda(
        X_scaled['pah'], y_pah, test_prior, global_models, spw=spw
    )

    # λ=0 (tam local / v5-baseline benzeri) ile en iyi λ karşılaştırması — stres testi seviyesinde
    print("\n" + "="*60)
    print("STRES TESTİ KARŞILAŞTIRMASI (λ=0 / tam-local vs en iyi λ)")
    print("="*60)
    p_mix_best = best_lambda * p_global_pool + (1 - best_lambda) * p_local_pool
    p_mix_best_corr = hc.prior_correction(p_mix_best, train_prior, test_prior)
    p_local_corr = hc.prior_correction(p_local_pool, train_prior, test_prior)

    stress_local = hc.run_stress_test_comparison(
        [{'val_probs_raw': p_local_pool, 'val_probs_corrected': p_local_corr, 'val_labels': labels_pool}],
        test_prior, panel_name="PAH-LocalOnly(lambda=0)"
    )
    stress_best = hc.run_stress_test_comparison(
        [{'val_probs_raw': p_mix_best, 'val_probs_corrected': p_mix_best_corr, 'val_labels': labels_pool}],
        test_prior, panel_name=f"PAH-HierarchicalPool(lambda={best_lambda})"
    )

    print(f"\n  {'Metrik':<14}{'Local(after)':>16}{'Hiyerarşik(after)':>20}{'Fark':>10}")
    for k in ['f1', 'mcc', 'sensitivity', 'specificity']:
        l, h = stress_local['after'][k], stress_best['after'][k]
        print(f"  {k:<14}{l:>16.4f}{h:>20.4f}{h-l:>+10.4f}")

    elapsed = time.time() - t0
    verdict = (f"PANEL-HAVUZLAMA FAYDALI (en iyi λ={best_lambda})" if best_lambda > 0
               else "PANEL-HAVUZLAMA FAYDA SAĞLAMADI (λ=0 kazandı, PAH-özel model yeterli)")
    print(f"\n[SONUÇ] {verdict} | Süre: {elapsed/60:.1f} dk")

    report = {
        'experiment': 'PAH_hierarchical_panel_pooling',
        'config': {'lambda_candidates': LAMBDA_CANDIDATES, 'best_lambda': best_lambda,
                   'k_max_features': K_MAX_FEATURES, 'selected_features': sel_feats,
                   'spw': spw, 'test_prior': test_prior, 'train_prior': train_prior,
                   'common_columns_across_4_panels': len(common_cols)},
        'lambda_ablation': lambda_results,
        'stress_test_local_only': {k: v for k, v in stress_local['after'].items() if isinstance(v, (int, float))},
        'stress_test_hierarchical': {k: v for k, v in stress_best['after'].items() if isinstance(v, (int, float))},
        'verdict': verdict, 'elapsed_min': round(elapsed/60, 2),
    }
    with open(os.path.join(output_dir, 'pah_hierarchical_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nRapor kaydedildi: {output_dir}/pah_hierarchical_report.json")
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PAH Hiyerarşik/Panel-Havuzlu Model Deneyi')
    parser.add_argument('--pah', type=str, required=True)
    parser.add_argument('--master', type=str, required=True)
    parser.add_argument('--kanser', type=str, required=True)
    parser.add_argument('--cftr', type=str, required=True)
    parser.add_argument('--label', type=str, default='Label')
    parser.add_argument('--output', type=str, default='pah_hierarchical_results')
    parser.add_argument('--test-prior', type=float, default=DEFAULT_TEST_PRIOR)
    parser.add_argument('--spw', type=float, default=DEFAULT_SPW)
    args = parser.parse_args()

    run_hierarchical_experiment(args.pah, args.master, args.kanser, args.cftr,
                                 args.output, args.label, args.test_prior, args.spw)
