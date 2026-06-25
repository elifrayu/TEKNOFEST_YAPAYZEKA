"""
HELIXAI — MASTER (Genel) Pipeline v1
======================================
helixai_common.py'ye bağımlıdır (AYNI klasörde olmalı).

EL YAZISI NOTU YORUMU (panel: Master):
  "Master → threshold ve kalibrasyona dikkat."

  Bu notun bu pipeline'a YANSIMASI:
  1. [YENİ] KALİBRASYON YÖNTEMİ ABLATION'I: Master panelinde toplam benign
     örnek sayısı 782 (gerçek veri, /mnt/user-data/uploads/YARISMA_TRAIN_MASTER.csv) —
     PAH (62) ve CFTR (21)'den ÇOK daha büyük. Isotonic Regression'ın
     ihtiyaç duyduğu "çok veri" koşulu burada karşılanıyor olabilir; PAH/
     CFTR'de Platt'ı seçmemizin gerekçesi (Isotonic'in veri açlığı çekmesi)
     Master'da GEÇERLİ OLMAYABİLİR. Bu yüzden yöntem TEORİDEN seçilmiyor:
     ana CV'den ÖNCE küçük bir karşılaştırma (Platt vs Isotonic, Brier
     skoruna göre) yapılıp KAZANAN yöntem tüm fold'larda kullanılıyor.
  2. [YENİ] Brier skoru artık her fold ve final raporda EXPLICIT loglanıyor
     (helixai_common.compute_metrics içine eklendi — kalibrasyon kalitesinin
     doğrudan, sayısal göstergesi; tüm 4 panelde ortak).
  3. [AYARLAMA] THRESHOLD_STEP, diğer panellerdeki 0.005 yerine 0.002'ye
     düşürüldü: Master'da hem çok daha fazla veri (2931 satır) hem de çok
     daha büyük bir test seti (≈3500 satır) olduğundan ince eşik araması
     hesaplama açısından karşılanabilir VE F1 üzerinde anlamlı fark yaratabilir.
  4. [AYARLAMA] N_REPEATS, PAH/Kanser'deki 10 yerine 5'e düşürüldü: büyük
     örneklem (N=2931) sayesinde 5-fold'un fold'ları zaten daha düşük
     varyanslı; 10 tekrar yerine 5 tekrar benzer kararlılığı daha az
     hesaplama maliyetiyle sağlar.

Diğer her şey (BLOSUM62/Grantham, AA feature engineering, AF log dönüşümü,
prior correction, EM sağlık kontrolü, SHAP güvenli düşüş, stres testi) ortak
helixai_common.py modülünden gelir — dört panel arasında metodolojik
tutarlılık sağlanır.

RESMİ PARAMETRELER (Şartname v2.0, Bölüm 3.2):
  Eğitim (yaklaşık) : 2000 Patojenik / 800 Benign
  Test   (yaklaşık) : 500 Patojenik / 3000 Benign
  Not: Gerçek teslim edilen eğitim dosyası 2149 Patojenik / 782 Benign (n=2931).

Kullanım:
  python master_pipeline_v1.py --demo
  python master_pipeline_v1.py --train YARISMA_TRAIN_MASTER.csv --test master_test.csv
"""

import argparse, os, json, time
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import helixai_common as hc

# ══════════════════════════════════════════════
# SABİTLER
# ══════════════════════════════════════════════
SEED            = hc.SEED
N_SPLITS        = 5
N_REPEATS       = 5          # [v1 AYARLAMA] büyük N -> daha az tekrar yeterli
THRESHOLD_STEP  = 0.002      # [v1 AYARLAMA] büyük veri -> ince eşik araması karşılanabilir

DEFAULT_TEST_PRIOR = 500 / (500 + 3000)   # ≈ 0.1429
DEFAULT_SPW         = 1.0

W_XGB, W_LGB, W_CAT = 0.35, 0.30, 0.35

SPW_CANDIDATES     = (0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
ABLATION_N_REPEATS = 2

# [v1-YENİ, PAH'taki gözlemden sonra eklendi] Master'da N=2931 PAH/Kanser'den
# çok daha büyük olduğundan daha fazla özellik aşırı öğrenme riskini o kadar
# artırmaz, ANCAK tavansız tarama hâlâ çok yavaş olur (her fold'da ~74 aday
# model fit). Daha yüksek ama hâlâ sınırlı bir tavan veriliyor.
K_MAX_FEATURES = 80

# ══════════════════════════════════════════════
# MODEL YAPILARI
# ══════════════════════════════════════════════
def build_xgb(spw=DEFAULT_SPW):
    return xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.04, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric='aucpr',
        early_stopping_rounds=40, random_state=SEED, verbosity=0,
    )

def build_lgb(spw=DEFAULT_SPW):
    return lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.04, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, metric='average_precision',
        random_state=SEED, verbose=-1,
    )

def build_cat(spw=DEFAULT_SPW):
    return CatBoostClassifier(
        iterations=400, learning_rate=0.04, depth=5,
        scale_pos_weight=spw, eval_metric='PRAUC',
        early_stopping_rounds=40, random_seed=SEED, verbose=False,
    )

# ══════════════════════════════════════════════
# [v1-YENİ] KALİBRASYON YÖNTEMİ SEÇİMİ (Platt vs Isotonic, Brier'e göre)
# ══════════════════════════════════════════════
def choose_calibration_method(X, y, seed=SEED):
    """
    [Hakem notuna yanıt: 'Master -> kalibrasyona dikkat']
    Tek bir temsili 70/30 eğitim/val bölmesiyle, basit bir XGB modeli
    üzerinden Platt (sigmoid) vs Isotonic kalibrasyonunu Brier skoruna göre
    karşılaştırır. Bu, ana 5x5 CV'den ÖNCE TEK SEFER yapılır; kazanan yöntem
    tüm fold'larda (3 model için de) kullanılır. Master'da yeterli benign
    örnek (782) bulunduğundan Isotonic burada (PAH/CFTR'nin aksine) gerçek
    bir aday haline gelir.
    """
    print("\n" + "="*60)
    print("ADIM 6: Kalibrasyon Yöntemi Seçimi (Platt vs Isotonic, Brier skoru)")
    print("="*60)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

    imp, sc = SimpleImputer(strategy='median'), RobustScaler()
    X_tr2, X_val2 = X_tr.copy(), X_val.copy()
    X_tr2[num_cols]  = imp.fit_transform(X_tr[num_cols])
    X_val2[num_cols] = imp.transform(X_val[num_cols])
    X_tr2[num_cols]  = sc.fit_transform(X_tr2[num_cols])
    X_val2[num_cols] = sc.transform(X_val2[num_cols])
    X_tr_f, dropped = hc.correlation_filter(X_tr2)
    X_val_f = X_val2.drop(columns=dropped, errors='ignore')

    m = build_xgb(DEFAULT_SPW)
    m.fit(X_tr_f, y_tr, eval_set=[(X_val_f, y_val)], verbose=False)

    results = {}
    for method in ['sigmoid', 'isotonic']:
        cal = hc.calibrate_fitted_model(m, X_val_f, y_val, method=method)
        p = cal.predict_proba(X_val_f)[:, 1]
        brier = hc.brier_score_loss(y_val, p)
        results[method] = brier
        print(f"  method={method:<9} | Brier={brier:.5f} (düşük daha iyi)")

    best_method = min(results, key=results.get)
    print(f"\n  [Seçim] Kazanan kalibrasyon yöntemi: '{best_method}' (Brier={results[best_method]:.5f})")
    return best_method, results

# ══════════════════════════════════════════════
# TEK FOLD EĞİTİMİ
# ══════════════════════════════════════════════
def train_fold(X_tr_raw, y_tr, X_val_raw, y_val, train_prior, test_prior,
               spw=DEFAULT_SPW, calib_method='sigmoid', fold_id="?"):
    num_cols = X_tr_raw.select_dtypes(include=[np.number]).columns.tolist()

    imp = SimpleImputer(strategy='median')
    X_tr, X_val = X_tr_raw.copy(), X_val_raw.copy()
    X_tr[num_cols]  = imp.fit_transform(X_tr_raw[num_cols])
    X_val[num_cols] = imp.transform(X_val_raw[num_cols])

    scaler = RobustScaler()
    X_tr[num_cols]  = scaler.fit_transform(X_tr[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])

    X_tr_f, dropped_corr = hc.correlation_filter(X_tr)
    X_val_f = X_val.drop(columns=dropped_corr, errors='ignore')

    sel = hc.shap_feature_selection(X_tr_f, y_tr, X_val_f, y_val, k_max=K_MAX_FEATURES)
    X_tr_s, X_val_s = X_tr_f[sel], X_val_f[sel]

    xgb_m = build_xgb(spw)
    xgb_m.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
    lgb_m = build_lgb(spw)
    lgb_m.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)],
              callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)])
    cat_m = build_cat(spw)
    cat_m.fit(X_tr_s, y_tr, eval_set=(X_val_s, y_val))

    n_ben_val, n_pat_val = int((y_val == 0).sum()), int((y_val == 1).sum())
    print(f"    [Kalibrasyon | {fold_id} | method={calib_method}] val seti: "
          f"{len(y_val)} örnek ({n_pat_val} patojenik / {n_ben_val} benign)")

    models_raw = {'xgb': xgb_m, 'lgb': lgb_m, 'cat': cat_m}
    weights = {'xgb': W_XGB, 'lgb': W_LGB, 'cat': W_CAT}
    models_cal = {name: hc.calibrate_fitted_model(mdl, X_val_s, y_val, method=calib_method)
                  for name, mdl in models_raw.items()}

    p_ens  = hc.ensemble_predict_proba(models_cal, weights, X_val_s)
    p_corr = hc.prior_correction(p_ens, train_prior, test_prior)

    best_t, _ = hc.find_best_threshold(y_val, p_corr, step=THRESHOLD_STEP)
    metrics = hc.compute_metrics(y_val, p_corr, threshold=best_t)
    metrics['fold'] = fold_id
    metrics['n_features'] = len(sel)

    p_raw_ens = (W_XGB * xgb_m.predict_proba(X_val_s)[:, 1] +
                 W_LGB * lgb_m.predict_proba(X_val_s)[:, 1] +
                 W_CAT * cat_m.predict_proba(X_val_s)[:, 1])

    return {
        'metrics': metrics, 'models_cal': models_cal, 'weights': weights,
        'imputer': imp, 'scaler': scaler, 'sel_feats': sel, 'dropped_corr': dropped_corr,
        'threshold': best_t, 'val_probs_corrected': p_corr, 'val_probs_raw': p_raw_ens,
        'val_labels': y_val.values,
    }

# ══════════════════════════════════════════════
# scale_pos_weight ABLATION (panel-spesifik: Master)
# ══════════════════════════════════════════════
def ablate_scale_pos_weight(X, y, test_prior, calib_method,
                             candidates=SPW_CANDIDATES, repeats=ABLATION_N_REPEATS):
    print("\n" + "="*60)
    print(f"ADIM 7: scale_pos_weight Ablation (panel: MASTER, {repeats}x{N_SPLITS}-Fold)")
    print(f"  Adaylar: {candidates}")
    print("="*60)
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=repeats, random_state=SEED)
    results = {}
    for spw in candidates:
        f1s = []
        for tr_idx, val_idx in rskf.split(X, y):
            X_tr, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
            y_tr, y_val = y.iloc[tr_idx].copy(), y.iloc[val_idx].copy()
            tr_prior = hc.compute_train_prior(y_tr)
            res = train_fold(X_tr, y_tr, X_val, y_val, tr_prior, test_prior,
                              spw=spw, calib_method=calib_method, fold_id=f"ablation_spw{spw}")
            f1s.append(res['metrics']['f1'])
        results[spw] = {'mean_f1': float(np.mean(f1s)), 'std_f1': float(np.std(f1s))}
        print(f"  spw={spw:<5} | F1={results[spw]['mean_f1']:.4f} ± {results[spw]['std_f1']:.4f}")
    best_spw = max(results, key=lambda k: results[k]['mean_f1'])
    print(f"\n  [Ablation] En iyi spw: {best_spw} (F1={results[best_spw]['mean_f1']:.4f})")
    return best_spw, results

# ══════════════════════════════════════════════
# ANA PIPELINE
# ══════════════════════════════════════════════
def run_master_pipeline(train_path, test_path=None, output_dir='master_results',
                         label_col='Label', test_prior=DEFAULT_TEST_PRIOR,
                         spw=None, run_ablation=True, ablation_repeats=ABLATION_N_REPEATS,
                         calib_method=None):
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    print("\n" + "#"*60)
    print("HELIXAI -- MASTER (Genel) Pipeline v1")
    print(f"Şartname v2.0 | Test Prior={test_prior:.4f} | {N_SPLITS}-Fold x {N_REPEATS} CV")
    print("#"*60)

    print("\nADIM 1: Veri Yükleme")
    train_df = pd.read_csv(train_path)
    print(f"  Eğitim: {train_df.shape} | {train_df[label_col].value_counts().to_dict()}")
    test_df = None
    if test_path and os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        print(f"  Test  : {test_df.shape}")

    print("\nADIM 2: Global Kolon Filtresi")
    feat_cols = [c for c in train_df.columns if c != label_col]
    tr_feats, dropped_global = hc.global_col_filter(train_df[feat_cols])
    te_feats = test_df[[c for c in feat_cols if c not in dropped_global]] if test_df is not None else None

    print("\nADIM 3: Biyolojik QC")
    tr_feats = hc.biological_qc(tr_feats)
    if te_feats is not None: te_feats = hc.biological_qc(te_feats)

    print("\nADIM 4: Amino Asit Özellik Mühendisliği")
    tr_feats = hc.engineer_aa_features(tr_feats)
    if te_feats is not None: te_feats = hc.engineer_aa_features(te_feats)

    print("\nADIM 5: Allel Frekans Log Dönüşümü")
    tr_feats = hc.log_transform_af(tr_feats)
    if te_feats is not None: te_feats = hc.log_transform_af(te_feats)

    y_train = train_df[label_col].astype(int)
    X_train = hc.to_numeric_df(tr_feats, label_col=label_col)
    X_test  = hc.to_numeric_df(te_feats, label_col=label_col) if te_feats is not None else None
    print(f"\n  Final özellik sayısı : {X_train.shape[1]}")
    print(f"  Patojenik / Benign   : {(y_train==1).sum()} / {(y_train==0).sum()}")

    if calib_method is None:
        calib_method, calib_compare = choose_calibration_method(X_train, y_train)
    else:
        calib_compare = None
        print(f"\nADIM 6: Kalibrasyon yöntemi manuel verildi: {calib_method}")

    ablation_results = None
    if spw is not None:
        chosen_spw = spw
        print(f"\nADIM 7: scale_pos_weight — manuel: {chosen_spw}")
    elif run_ablation:
        chosen_spw, ablation_results = ablate_scale_pos_weight(
            X_train, y_train, test_prior, calib_method, repeats=ablation_repeats
        )
    else:
        chosen_spw = DEFAULT_SPW
        print(f"\nADIM 7: scale_pos_weight — ablation atlandı, varsayılan: {chosen_spw}")

    fold_results, cv_summary, train_prior = hc.run_repeated_cv(
        X_train, y_train, test_prior, train_fold, N_SPLITS, N_REPEATS,
        fold_kwargs={'spw': chosen_spw, 'calib_method': calib_method}, panel_name="MASTER"
    )

    final_threshold = hc.compute_oof_pooled_threshold(fold_results)
    stress_test_results = hc.run_stress_test_comparison(fold_results, test_prior, panel_name="MASTER")

    test_metrics, em_check = None, None
    if X_test is not None:
        test_probs, _, raw_probs_mean = hc.predict_test(X_test, fold_results, train_prior, test_prior, final_threshold)
        est_prior = hc.estimate_test_prior_em(raw_probs_mean, train_prior)
        prior_diff = abs(est_prior - test_prior)
        em_check = {'sartname_prior': test_prior, 'em_estimated_prior': est_prior, 'abs_diff': prior_diff}
        print(f"\n  [Sağlık Kontrolü - EM] Şartname={test_prior:.4f} | EM-tahmini={est_prior:.4f} | Fark={prior_diff:.4f}")
        if prior_diff > 0.05:
            print("  [UYARI] Fark >0.05 — PDR'da not edilmeli.")

        if test_df is not None and label_col in test_df.columns:
            y_test = test_df[label_col].astype(int)
            test_metrics = hc.compute_metrics(y_test, test_probs, threshold=final_threshold)
            print("\n  TEST SETİ SONUÇLARI:")
            for k, v in test_metrics.items():
                if isinstance(v, float): print(f"    {k:<16}: {v:.4f}")
        else:
            pred_path = os.path.join(output_dir, 'master_test_predictions.csv')
            pd.DataFrame({'prob_pathogenic': test_probs,
                          'predicted_label': (test_probs >= final_threshold).astype(int)}
                         ).to_csv(pred_path, index=False)
            print(f"\n  Etiketsiz test tahminleri kaydedildi: {pred_path}")

    shap_imp, shap_method = hc.run_shap(fold_results, X_train, output_dir, panel_name="MASTER", prefix="master")

    elapsed = time.time() - t_start
    print("\n" + "#"*60)
    print("HELIXAI MASTER v1 -- FINAL RAPOR")
    print("#"*60)
    print(f"  Süre: {elapsed/60:.1f} dk | Kalibrasyon: {calib_method} | spw: {chosen_spw} | Eşik: {final_threshold:.4f}")
    for k, v in cv_summary.items():
        if isinstance(v, dict) and 'mean' in v:
            print(f"  {k:<16} {v['mean']:>8.4f} +/-{v['std']:>8.4f}")

    report = {
        'panel': 'MASTER', 'version': 'v1',
        'config': {'n_splits': N_SPLITS, 'n_repeats': N_REPEATS,
                   'weights': {'xgb': W_XGB, 'lgb': W_LGB, 'cat': W_CAT},
                   'train_prior_computed': train_prior, 'test_prior_param': test_prior,
                   'final_threshold_oof': final_threshold, 'chosen_scale_pos_weight': chosen_spw,
                   'calibration_method': calib_method, 'shap_method': shap_method},
        'calibration_method_comparison': calib_compare,
        'ablation_scale_pos_weight': ablation_results,
        'cv_summary': {k: {'mean': v['mean'], 'std': v['std']} for k, v in cv_summary.items() if isinstance(v, dict) and 'mean' in v},
        'oof_y_true': cv_summary.get('_oof_y_true', []),
        'oof_y_prob': cv_summary.get('_oof_y_prob', []),
        'stress_test': {'before': {k: v for k, v in stress_test_results['before'].items() if isinstance(v, (int, float))},
                        'after': {k: v for k, v in stress_test_results['after'].items() if isinstance(v, (int, float))},
                        'bootstrap_used': stress_test_results['bootstrap_used']},
        'em_test_prior_check': em_check, 'test': test_metrics,
        'shap_top10': shap_imp.head(10).to_dict(), 'elapsed_min': round(elapsed / 60, 2),
    }
    rep_path = os.path.join(output_dir, 'master_report_v1.json')
    with open(rep_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Rapor kaydedildi: {rep_path}")
    return report, fold_results

# ══════════════════════════════════════════════
# DEMO VERİSİ
# ══════════════════════════════════════════════
def generate_demo_data():
    np.random.seed(SEED)
    def make(n_path, n_ben):
        n = n_path + n_ben
        y = np.array([1]*n_path + [0]*n_ben)
        ix = np.random.permutation(n); y = y[ix]
        df = pd.DataFrame()
        df['AL_freq_global'] = np.where(y == 1, np.random.beta(1, 100, n), np.random.beta(5, 20, n))
        df['AL_freq_eur']    = np.where(y == 1, np.random.beta(1, 80, n),  np.random.beta(4, 15, n))
        df['AL_count']       = (df['AL_freq_global'] * 10000).astype(int)
        aa_list = list(hc.VALID_AA)
        df['AA_ref'] = np.random.choice(aa_list, n)
        df['AA_alt'] = np.random.choice(aa_list, n)
        df['EK_phylop'] = np.where(y == 1, np.random.normal(3.5, 1.0, n), np.random.normal(0.5, 1.5, n))
        df['EK_gerp']   = np.where(y == 1, np.random.normal(4.0, 1.2, n), np.random.normal(0.2, 2.0, n))
        df['CAT_pop']     = np.random.choice(['EUR', 'AFR', 'EAS', 'SAS', 'AMR'], n)
        df['CAT_quality'] = np.random.choice(['PASS', 'LOW_QUAL'], n, p=[0.9, 0.1])
        for c in ['AL_freq_eur', 'EK_gerp']:
            df.loc[np.random.rand(n) < 0.10, c] = np.nan
        df['Label'] = y
        return df
    tr = make(2000, 800); tr.to_csv('/tmp/master_train.csv', index=False)
    te = make(500, 3000); te.to_csv('/tmp/master_test.csv', index=False)
    print("Demo veri oluşturuldu: /tmp/master_train.csv (2000/800), /tmp/master_test.csv (500/3000)")
    return '/tmp/master_train.csv', '/tmp/master_test.csv'

# ══════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HELIXAI MASTER Pipeline v1')
    parser.add_argument('--train', type=str, default=None)
    parser.add_argument('--test', type=str, default=None)
    parser.add_argument('--label', type=str, default='Label')
    parser.add_argument('--output', type=str, default='master_results')
    parser.add_argument('--test-prior', type=float, default=DEFAULT_TEST_PRIOR)
    parser.add_argument('--spw', type=float, default=None)
    parser.add_argument('--skip-ablation', action='store_true')
    parser.add_argument('--ablation-repeats', type=int, default=ABLATION_N_REPEATS)
    parser.add_argument('--calib-method', type=str, default=None, choices=['sigmoid', 'isotonic'])
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    if args.demo or args.train is None:
        print("[DEMO MODU]")
        train_p, test_p = generate_demo_data()
    else:
        train_p, test_p = args.train, args.test

    run_master_pipeline(
        train_path=train_p, test_path=test_p, output_dir=args.output, label_col=args.label,
        test_prior=args.test_prior, spw=args.spw, run_ablation=not args.skip_ablation,
        ablation_repeats=args.ablation_repeats, calib_method=args.calib_method,
    )
