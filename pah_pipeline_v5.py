"""
HELIXAI — PAH Pipeline v5
============================
helixai_common.py'ye bağımlıdır (AYNI klasörde olmalı). v4'ten v5'e geçişte
pipeline ortak helixai_common.py altyapısına TAŞINMIŞTIR (4 panel arasında
tutarlılık ve bakım kolaylığı için) — metodoloji DEĞİŞMEMİŞTİR.

EL YAZISI NOTU YORUMU (panel: PAH):
  "PAH → benign öğrenme yetersizliği riski. Dikkatli oversample veya
   cost-sensitive learning uygula."

  Bu notun v4 -> v5'e YANSIMASI:
  v4 zaten 'cost-sensitive learning'i (scale_pos_weight ablation) uygulamıştı.
  Not AYRICA "oversample"ı bir ALTERNATİF/EK seçenek olarak işaret ediyor.
  Bu yüzden v5'te:
    1. [YENİ] hc.oversample_minority() — harici kütüphane GEREKTİRMEYEN,
       basit random-oversampling + küçük Gaussian jitter — train_fold()
       içine, imputation+scaling SONRASI, feature selection ÖNCESİ adım
       olarak entegre edildi.
    2. [YENİ] ablate_imbalance_strategy() — v4'ün TEK-BOYUTLU spw
       ablation'ı, artık {spw candidates} x {oversample: açık/kapalı}
       ORTAK bir ızgaraya genişletildi. "VEYA" diyen not, AMPİRİK olarak
       hangi kombinasyonun (cost-sensitive YALNIZ, oversample YALNIZ, her
       ikisi, veya hiçbiri) en iyi CV F1'i verdiğini bulmamızı sağlıyor —
       tek bir yöntemi teoriden seçmek yerine.
    3. [DÜZELTME] label_col varsayılanı 'label' (küçük harf) -> 'Label'
       (büyük harf) olarak düzeltildi: gerçek YARISMA_TRAIN_PAH.csv dosyası
       'Label' kolonunu kullanıyor; v4'teki varsayılan, CLI'da --label
       açıkça verilmezse gerçek dosyada KeyError'a yol açardı.

v3 -> v4 değişiklikleri (TARİHSEL KAYIT — bkz. pah_pipeline_v4.py):
  get_base_estimator güvenli düşüş, EM tabanlı test-prior sağlık kontrolü,
  kalibrasyon şeffaflığı loglaması, scale_pos_weight ablation (yön sorgulu).
  Bu v5, tüm bu davranışları KORUR (artık helixai_common üzerinden).

RESMİ PARAMETRELER (Şartname v2.0, Bölüm 3.2):
  Eğitim (yaklaşık) : 300 Patojenik / 50 Benign
  Test   (yaklaşık) : 100 Patojenik / 250 Benign
  Gerçek eğitim dosyası: 310 Patojenik / 62 Benign (n=372).

Kullanım:
  python pah_pipeline_v5.py --demo
  python pah_pipeline_v5.py --train YARISMA_TRAIN_PAH.csv --test pah_test.csv
"""

import argparse, os, json, time
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold

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

DEFAULT_TEST_PRIOR = 100 / (100 + 250)   # ≈ 0.2857
DEFAULT_SPW         = 1.0

W_XGB, W_LGB, W_CAT = 0.35, 0.30, 0.35

# [v5-YENİ] cost-sensitive x oversample ORTAK ızgarası
SPW_CANDIDATES_JOINT     = (0.3, 1.0, 2.0, 3.0)
OVERSAMPLE_CANDIDATES    = (False, True)
ABLATION_N_REPEATS       = 2

# [v5-YENİ] PAH'ta N=372 satırla 371 özelliğin tamamını taramak hem çok
# yavaş (~2 saat) hem de N'ye kıyasla aşırı geniş bir arama uzayı — aşırı
# öğrenme riskini artırır. CFTR'deki k_max mantığıyla aynı gerekçeyle bir
# tavan konuyor; amaç hız/overfitting kontrolü, model seçimi değil.
K_MAX_FEATURES = 40

# ══════════════════════════════════════════════
# MODEL YAPILARI
# ══════════════════════════════════════════════
def build_xgb(spw=DEFAULT_SPW):
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric='aucpr',
        early_stopping_rounds=30, random_state=SEED, verbosity=0,
    )

def build_lgb(spw=DEFAULT_SPW):
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, metric='average_precision',
        random_state=SEED, verbose=-1,
    )

def build_cat(spw=DEFAULT_SPW):
    return CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=4,
        scale_pos_weight=spw, eval_metric='PRAUC',
        early_stopping_rounds=30, random_seed=SEED, verbose=False,
    )

# ══════════════════════════════════════════════
# TEK FOLD EĞİTİMİ
# ══════════════════════════════════════════════
def train_fold(X_tr_raw, y_tr, X_val_raw, y_val, train_prior, test_prior,
               spw=DEFAULT_SPW, oversample=False, fold_id="?"):
    num_cols = X_tr_raw.select_dtypes(include=[np.number]).columns.tolist()

    imp = SimpleImputer(strategy='median')
    X_tr, X_val = X_tr_raw.copy(), X_val_raw.copy()
    X_tr[num_cols]  = imp.fit_transform(X_tr_raw[num_cols])
    X_val[num_cols] = imp.transform(X_val_raw[num_cols])

    scaler = RobustScaler()
    X_tr[num_cols]  = scaler.fit_transform(X_tr[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])

    # [v5-YENİ] Oversample — imputation+scaling SONRASI, feature selection ÖNCESİ.
    # SADECE X_tr üzerinde (X_val asla oversample edilmez/dokunulmaz — leakage olur).
    if oversample:
        X_tr, y_tr = hc.oversample_minority(X_tr, y_tr, seed=SEED)

    X_tr_f, dropped_corr = hc.correlation_filter(X_tr)
    X_val_f = X_val.drop(columns=dropped_corr, errors='ignore')

    sel = hc.shap_feature_selection(X_tr_f, y_tr, X_val_f, y_val, k_max=K_MAX_FEATURES)
    X_tr_s, X_val_s = X_tr_f[sel], X_val_f[sel]

    xgb_m = build_xgb(spw)
    xgb_m.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
    lgb_m = build_lgb(spw)
    lgb_m.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    cat_m = build_cat(spw)
    cat_m.fit(X_tr_s, y_tr, eval_set=(X_val_s, y_val))

    n_ben_val, n_pat_val = int((y_val == 0).sum()), int((y_val == 1).sum())
    print(f"    [Kalibrasyon | {fold_id} | spw={spw} | oversample={oversample}] val seti: "
          f"{len(y_val)} örnek ({n_pat_val} patojenik / {n_ben_val} benign)")

    models_raw = {'xgb': xgb_m, 'lgb': lgb_m, 'cat': cat_m}
    weights = {'xgb': W_XGB, 'lgb': W_LGB, 'cat': W_CAT}
    models_cal = {name: hc.calibrate_fitted_model(mdl, X_val_s, y_val, method='sigmoid')
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
# [v5-YENİ] ORTAK ABLATION: cost-sensitive (spw) x oversample
# ══════════════════════════════════════════════
def ablate_imbalance_strategy(X, y, test_prior,
                               spw_candidates=SPW_CANDIDATES_JOINT,
                               oversample_candidates=OVERSAMPLE_CANDIDATES,
                               repeats=ABLATION_N_REPEATS):
    print("\n" + "="*60)
    print(f"ADIM 6: Dengesizlik Stratejisi Ablation (panel: PAH, {repeats}x{N_SPLITS}-Fold)")
    print(f"  spw adayları: {spw_candidates} | oversample adayları: {oversample_candidates}")
    print("  [Not] 'VEYA' diyen el yazısı notu AMPİRİK olarak çözülüyor — "
          "cost-sensitive, oversample, ikisi birden veya hiçbiri taranıyor.")
    print("="*60)
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=repeats, random_state=SEED)
    results = {}
    for spw in spw_candidates:
        for ov in oversample_candidates:
            f1s = []
            for tr_idx, val_idx in rskf.split(X, y):
                X_tr, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
                y_tr, y_val = y.iloc[tr_idx].copy(), y.iloc[val_idx].copy()
                tr_prior = hc.compute_train_prior(y_tr)
                res = train_fold(X_tr, y_tr, X_val, y_val, tr_prior, test_prior,
                                  spw=spw, oversample=ov, fold_id=f"ablation_spw{spw}_ov{ov}")
                f1s.append(res['metrics']['f1'])
            key = (spw, ov)
            results[key] = {'spw': spw, 'oversample': ov,
                             'mean_f1': float(np.mean(f1s)), 'std_f1': float(np.std(f1s))}
            print(f"  spw={spw:<4} oversample={str(ov):<5} | F1={results[key]['mean_f1']:.4f} "
                  f"± {results[key]['std_f1']:.4f}")

    best_key = max(results, key=lambda k: results[k]['mean_f1'])
    best = results[best_key]
    print(f"\n  [Ablation Sonucu] Kazanan: spw={best['spw']}, oversample={best['oversample']} "
          f"(F1={best['mean_f1']:.4f})")
    results_str_keys = {f"spw{k[0]}_ov{k[1]}": v for k, v in results.items()}
    return best['spw'], best['oversample'], results_str_keys

# ══════════════════════════════════════════════
# ANA PIPELINE
# ══════════════════════════════════════════════
def run_pah_pipeline(train_path, test_path=None, output_dir='pah_results',
                      label_col='Label', test_prior=DEFAULT_TEST_PRIOR,
                      spw=None, oversample=None, run_ablation=True,
                      ablation_repeats=ABLATION_N_REPEATS):
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    print("\n" + "#"*60)
    print("HELIXAI -- PAH Pipeline v5")
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

    print("\nADIM 4: Amino Asit Özellik Mühendisliği + Allel Frekans Log Dönüşümü")
    tr_feats = hc.engineer_aa_features(tr_feats)
    tr_feats = hc.log_transform_af(tr_feats)
    if te_feats is not None:
        te_feats = hc.engineer_aa_features(te_feats)
        te_feats = hc.log_transform_af(te_feats)

    y_train = train_df[label_col].astype(int)
    X_train = hc.to_numeric_df(tr_feats, label_col=label_col)
    X_test  = hc.to_numeric_df(te_feats, label_col=label_col) if te_feats is not None else None
    print(f"\n  Final özellik sayısı : {X_train.shape[1]}")
    print(f"  Patojenik / Benign   : {(y_train==1).sum()} / {(y_train==0).sum()}")

    ablation_results = None
    if spw is not None and oversample is not None:
        chosen_spw, chosen_oversample = spw, oversample
        print(f"\nADIM 6: Dengesizlik stratejisi manuel: spw={chosen_spw}, oversample={chosen_oversample}")
    elif run_ablation:
        chosen_spw, chosen_oversample, ablation_results = ablate_imbalance_strategy(
            X_train, y_train, test_prior, repeats=ablation_repeats
        )
    else:
        chosen_spw, chosen_oversample = DEFAULT_SPW, False
        print(f"\nADIM 6: Ablation atlandı, varsayılan: spw={chosen_spw}, oversample={chosen_oversample}")

    fold_results, cv_summary, train_prior = hc.run_repeated_cv(
        X_train, y_train, test_prior, train_fold, N_SPLITS, N_REPEATS,
        fold_kwargs={'spw': chosen_spw, 'oversample': chosen_oversample}, panel_name="PAH"
    )

    final_threshold = hc.compute_oof_pooled_threshold(fold_results)
    stress_test_results = hc.run_stress_test_comparison(fold_results, test_prior, panel_name="PAH")

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
            pred_path = os.path.join(output_dir, 'pah_test_predictions.csv')
            pd.DataFrame({'prob_pathogenic': test_probs,
                          'predicted_label': (test_probs >= final_threshold).astype(int)}
                         ).to_csv(pred_path, index=False)
            print(f"\n  Etiketsiz test tahminleri kaydedildi: {pred_path}")

    shap_imp, shap_method = hc.run_shap(fold_results, X_train, output_dir, panel_name="PAH", prefix="pah")

    elapsed = time.time() - t_start
    print("\n" + "#"*60)
    print("HELIXAI PAH v5 -- FINAL RAPOR")
    print("#"*60)
    print(f"  Süre: {elapsed/60:.1f} dk | spw: {chosen_spw} | oversample: {chosen_oversample} | "
          f"Eşik: {final_threshold:.4f}")
    for k, v in cv_summary.items():
        print(f"  {k:<16} {v['mean']:>8.4f} +/-{v['std']:>8.4f}")

    report = {
        'panel': 'PAH', 'version': 'v5',
        'config': {'n_splits': N_SPLITS, 'n_repeats': N_REPEATS,
                   'weights': {'xgb': W_XGB, 'lgb': W_LGB, 'cat': W_CAT},
                   'train_prior_computed': train_prior, 'test_prior_param': test_prior,
                   'final_threshold_oof': final_threshold,
                   'chosen_scale_pos_weight': chosen_spw, 'chosen_oversample': chosen_oversample,
                   'shap_method': shap_method},
        'ablation_imbalance_strategy': ablation_results,
        'cv_summary': {k: {'mean': v['mean'], 'std': v['std']} for k, v in cv_summary.items()},
        'stress_test': {'before': {k: v for k, v in stress_test_results['before'].items() if isinstance(v, (int, float))},
                        'after': {k: v for k, v in stress_test_results['after'].items() if isinstance(v, (int, float))},
                        'bootstrap_used': stress_test_results['bootstrap_used']},
        'em_test_prior_check': em_check, 'test': test_metrics,
        'shap_top10': shap_imp.head(10).to_dict(), 'elapsed_min': round(elapsed / 60, 2),
    }
    rep_path = os.path.join(output_dir, 'pah_report_v5.json')
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
    tr = make(300, 50); tr.to_csv('/tmp/pah_train.csv', index=False)
    te = make(100, 250); te.to_csv('/tmp/pah_test.csv', index=False)
    print("Demo veri oluşturuldu: /tmp/pah_train.csv (300/50), /tmp/pah_test.csv (100/250)")
    return '/tmp/pah_train.csv', '/tmp/pah_test.csv'

# ══════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HELIXAI PAH Pipeline v5')
    parser.add_argument('--train', type=str, default=None)
    parser.add_argument('--test', type=str, default=None)
    parser.add_argument('--label', type=str, default='Label')
    parser.add_argument('--output', type=str, default='pah_results')
    parser.add_argument('--test-prior', type=float, default=DEFAULT_TEST_PRIOR)
    parser.add_argument('--spw', type=float, default=None)
    parser.add_argument('--oversample', type=str, default=None, choices=[None, 'true', 'false'])
    parser.add_argument('--skip-ablation', action='store_true')
    parser.add_argument('--ablation-repeats', type=int, default=ABLATION_N_REPEATS)
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    if args.demo or args.train is None:
        print("[DEMO MODU]")
        train_p, test_p = generate_demo_data()
    else:
        train_p, test_p = args.train, args.test

    ov_arg = None if args.oversample is None else (args.oversample.lower() == 'true')

    run_pah_pipeline(
        train_path=train_p, test_path=test_p, output_dir=args.output, label_col=args.label,
        test_prior=args.test_prior, spw=args.spw, oversample=ov_arg,
        run_ablation=not args.skip_ablation, ablation_repeats=args.ablation_repeats,
    )
