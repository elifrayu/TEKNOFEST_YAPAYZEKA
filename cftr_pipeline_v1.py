"""
HELIXAI — CFTR (Kistik Fibrozis Gen Paneli) Pipeline v1
===========================================================
helixai_common.py'ye bağımlıdır (AYNI klasörde olmalı).

EL YAZISI NOTU (fotoğraf, danışman):
  "CFTR → LOOCV zorunlu. Overfitting riski max. Regularization max'a çek.
   Feature sayısını minimize et. CFTR için basit robust model ensembleden
   daha iyi sonuç verebilir."

BU NOTUN PIPELINE'A YANSIMASI — 4 SOMUT MİMARİ KARAR:

  1. [MİMARİ DEĞİŞİKLİĞİ] CV stratejisi artık RepeatedStratifiedKFold DEĞİL,
     sklearn.model_selection.LeaveOneOut. Gerekçe: not "zorunlu" diyor VE
     CFTR'de toplam örnek sayısı sadece 111 (90 patojenik / 21 benign) —
     bu ölçekte k-fold'un val fold'ları (özellikle benign için, fold başına
     ~4 örnek) aşırı gürültülü olurdu; LOOCV her örneği tam olarak bir kez
     test eder, mevcut en az-varyanslı tahmindir.

  2. [MİMARİ DEĞİŞİKLİĞİ — ÖNEMLİ] LOOCV'de validation seti TEK ÖRNEKTİR;
     bu nedenle Platt kalibrasyonu PAH/Kanser/Master'daki gibi "her fold
     içinde val seti üzerinde" fit EDİLEMEZ (n=1 ile sigmoid fit etmek
     anlamsızdır). Bunun yerine: (a) 111 LOO iterasyonundan gelen HAM
     (kalibrasyonsuz) OOF olasılıkları TEK BİR havuzda toplanır, (b) bu
     havuz üzerinde TEK BİR Platt (log-odds -> logistic regression)
     kalibrasyon haritası fit edilir, (c) bu harita hem OOF değerlendirmesi
     hem de nihai test tahmini için kullanılır. [DÜRÜSTLÜK NOTU — PDR'da
     belirtilmelidir] Bu, kalibrasyonun değerlendirildiği havuzla fit
     edildiği havuzun ÖRTÜŞMESİ nedeniyle hafif optimistik olabilir; tam
     bağımsız bir iç LOOCV (nested LOOCV, 111x110 iterasyon) hesaplama
     açısından orantısızdır ve bu basitleştirme BİLİNÇLİ bir tercihtir.

  3. [MİMARİ DEĞİŞİKLİĞİ] Nihai TEST tahmini, PAH/Kanser/Master'daki gibi
     "tüm fold-modellerinin ortalaması" şeklinde YAPILMAZ. Gerekçe: LOOCV'nin
     111 modeli birbirinden sadece TEK satır farklı (110/111 örtüşme) —
     bunları ensemble etmek neredeyse hiçbir çeşitlilik katmadan 111x
     çıkarım maliyeti getirir. Bunun yerine, SEÇİLEN strateji (adım 4)
     TÜM eğitim verisiyle (111 satır) YENİDEN eğitilir; LOOCV'nin rolü
     YALNIZCA (a) genelleme performansını dürüstçe tahmin etmek ve
     (b) kalibrasyon haritasını/eşiği belirlemek içindir — bu standart ve
     kabul gören bir LOOCV kullanım şeklidir.

  4. [YENİ] ablate_model_strategy(): notun "basit robust model ensembleden
     daha iyi olabilir" önerisini TEORİDEN kabul ETMEZ — hafif bir
     RepeatedStratifiedKFold(5x2) taramasıyla {ensemble (max-regularize
     edilmiş 3'lü XGB+LGB+CAT), basit (L2-cezalı Lojistik Regresyon)} x
     {birkaç sınıf-ağırlığı ayarı} ızgarasını dener; kazanan, F1'e göre
     AMPİRİK olarak seçilir ve ancak ondan sonra tam LOOCV'ye geçilir.

  Ayrıca: SHAP özellik seçimi LOOCV döngüsünden ÖNCE, TÜM veriyle BİR KEZ
  yapılır ve k_max=12 ile SIKI bir tavana sahiptir ("feature sayısını
  minimize et" notuna doğrudan yanıt). Bu, KESİN sıfır-leakage değildir
  (seçim, her LOO iterasyonunda dışarıda tutulacak noktayı da görür) —
  111 iterasyonun her birinde yeniden SHAP araması hesaplama açısından
  orantısız olduğundan BİLİNÇLİ bir basitleştirmedir (PDR'da belirtilmeli).

GERÇEK VERİ (doğrulanmış, YARISMA_TRAIN_CFTR.csv):
  Eğitim (gerçek)             : 90 Patojenik / 21 Benign (n=111)
  Eğitim (şartname, yaklaşık) : 100 Patojenik / 20 Benign
  Test   (şartname, yaklaşık) : 20 Patojenik / 100 Benign

Kullanım:
  python cftr_pipeline_v1.py --demo
  python cftr_pipeline_v1.py --train YARISMA_TRAIN_CFTR.csv --test cftr_test.csv
"""

import argparse, os, json, time
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold, train_test_split

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import helixai_common as hc

# ══════════════════════════════════════════════
# SABİTLER
# ══════════════════════════════════════════════
SEED            = hc.SEED
THRESHOLD_STEP  = 0.005
K_MAX_FEATURES  = 12          # [v1-YENİ] "feature sayısını minimize et"

DEFAULT_TEST_PRIOR = 20 / (20 + 100)   # ≈ 0.1667

W_XGB, W_LGB, W_CAT = 0.35, 0.30, 0.35

# [v1-YENİ] model-strateji + dengesizlik-ayarı ızgarası (hafif ablation için)
STRATEGY_CANDIDATES = [
    ('ensemble', 1.0),
    ('ensemble', 0.233),   # standart formül: sum(neg)/sum(pos) = 21/90 ≈ 0.233
    ('ensemble', 2.0),
    ('simple',   None),
    ('simple',   'balanced'),
]
ABLATION_N_SPLITS  = 5
ABLATION_N_REPEATS = 2

# ══════════════════════════════════════════════
# MODEL YAPILARI — MAKSİMUM REGULARİZASYON
# ══════════════════════════════════════════════
def build_ensemble_models(spw=1.0):
    """[v1] max_depth=2, yüksek L1/L2, düşük n_estimators — 'regularization max'a çek'."""
    xgb_m = xgb.XGBClassifier(
        n_estimators=80, learning_rate=0.05, max_depth=2,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=5.0,
        scale_pos_weight=spw, eval_metric='aucpr',
        early_stopping_rounds=15, random_state=SEED, verbosity=0,
    )
    lgb_m = lgb.LGBMClassifier(
        n_estimators=80, learning_rate=0.05, max_depth=2,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=5.0, min_child_samples=5,
        scale_pos_weight=spw, metric='average_precision',
        random_state=SEED, verbose=-1,
    )
    cat_m = CatBoostClassifier(
        iterations=80, learning_rate=0.05, depth=2,
        l2_leaf_reg=10.0, scale_pos_weight=spw, eval_metric='PRAUC',
        early_stopping_rounds=15, random_seed=SEED, verbose=False,
    )
    return {'xgb': xgb_m, 'lgb': lgb_m, 'cat': cat_m}

def build_simple_model(class_weight=None):
    """[v1] 'basit robust model' — güçlü L2-cezalı Lojistik Regresyon."""
    return LogisticRegression(penalty='l2', C=0.3, max_iter=3000,
                               solver='lbfgs', class_weight=class_weight,
                               random_state=SEED)

# ══════════════════════════════════════════════
# [v1-YENİ] Platt kalibrasyonu — LOOCV için BAĞIMSIZ (base-estimator'a
# bağlı olmayan) 1D log-odds -> logistic regression haritası.
# ══════════════════════════════════════════════
def fit_platt_1d(raw_probs, labels, eps=1e-6):
    logits = np.log(np.clip(raw_probs, eps, 1-eps) / np.clip(1-raw_probs, eps, 1-eps)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver='lbfgs')  # minimal ceza — sadece yeniden ölçekleme
    lr.fit(logits, labels)
    return lr

def apply_platt_1d(platt_model, raw_probs, eps=1e-6):
    logits = np.log(np.clip(raw_probs, eps, 1-eps) / np.clip(1-raw_probs, eps, 1-eps)).reshape(-1, 1)
    return platt_model.predict_proba(logits)[:, 1]

# ══════════════════════════════════════════════
# TEK İTERASYON EĞİTİMİ (LOO veya ablation fold'u için ortak)
# ══════════════════════════════════════════════
def fit_strategy_get_raw_prob(X_tr_raw, y_tr, X_val_raw, strategy, param, seed=SEED):
    """
    X_tr_raw/X_val_raw: SADECE global_sel_feats kolonlarını içerir (çağıran
    tarafından önceden seçilmiş). Impute+scale burada (fold-içi, leakage-free)
    yapılır. 'ensemble' XGB/LGB/CAT için early-stopping amacıyla X_tr İÇİNDEN
    küçük bir iç val ayrılır (gerçek X_val asla bu amaçla kullanılmaz).
    Döner: (raw_prob_on_X_val: np.array, fitted_models: dict, weights: dict)
    """
    num_cols = X_tr_raw.select_dtypes(include=[np.number]).columns.tolist()
    imp = SimpleImputer(strategy='median')
    X_tr, X_val = X_tr_raw.copy(), X_val_raw.copy()
    X_tr[num_cols]  = imp.fit_transform(X_tr_raw[num_cols])
    X_val[num_cols] = imp.transform(X_val_raw[num_cols])
    scaler = RobustScaler()
    X_tr[num_cols]  = scaler.fit_transform(X_tr[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])

    if strategy == 'simple':
        m = build_simple_model(class_weight=param)
        m.fit(X_tr, y_tr)
        p_val = m.predict_proba(X_val)[:, 1]
        return p_val, {'simple': m}, {'simple': 1.0}, imp, scaler

    # strategy == 'ensemble'
    # Erken durdurma için X_tr'den KÜÇÜK bir iç val ayır (gerçek X_val'e DOKUNULMAZ)
    if len(X_tr) >= 20 and y_tr.nunique() == 2 and y_tr.value_counts().min() >= 2:
        Xi_tr, Xi_val, yi_tr, yi_val = train_test_split(
            X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=seed)
    else:
        Xi_tr, Xi_val, yi_tr, yi_val = X_tr, X_tr, y_tr, y_tr  # çok küçükse aynısını kullan (kabul edilebilir, sadece erken durdurma sinyali için)

    models = build_ensemble_models(spw=param)
    models['xgb'].fit(Xi_tr, yi_tr, eval_set=[(Xi_val, yi_val)], verbose=False)
    models['lgb'].fit(Xi_tr, yi_tr, eval_set=[(Xi_val, yi_val)],
                       callbacks=[lgb.early_stopping(15, verbose=False), lgb.log_evaluation(-1)])
    models['cat'].fit(Xi_tr, yi_tr, eval_set=(Xi_val, yi_val))

    weights = {'xgb': W_XGB, 'lgb': W_LGB, 'cat': W_CAT}
    p_val = hc.ensemble_predict_proba(models, weights, X_val)
    return p_val, models, weights, imp, scaler

# ══════════════════════════════════════════════
# [v1-YENİ] HAFİF ABLATION: model stratejisi + dengesizlik ayarı seçimi
# ══════════════════════════════════════════════
def ablate_model_strategy(X, y, test_prior, candidates=STRATEGY_CANDIDATES,
                           n_splits=ABLATION_N_SPLITS, n_repeats=ABLATION_N_REPEATS):
    """
    [Hakem notuna yanıt: 'CFTR için basit robust model ensembleden daha iyi
    sonuç verebilir'] Bu önermeyi TEORİDEN kabul ETMİYORUZ — hafif bir
    RepeatedStratifiedKFold(5x2) taramasıyla ampirik olarak test ediyoruz.
    Tam LOOCV (111 iterasyon) yerine bu hafif tarama kullanılır (hesaplama
    maliyeti); kazanan strateji daha sonra TAM LOOCV ile değerlendirilir.
    """
    print("\n" + "="*60)
    print(f"ADIM 6: Model Stratejisi Ablation (panel: CFTR, {n_repeats}x{n_splits}-Fold)")
    print(f"  Adaylar: {candidates}")
    print("="*60)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    results = {}
    for strategy, param in candidates:
        f1s = []
        for tr_idx, val_idx in rskf.split(X, y):
            X_tr, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
            y_tr, y_val = y.iloc[tr_idx].copy(), y.iloc[val_idx].copy()
            p_val, _, _, _, _ = fit_strategy_get_raw_prob(X_tr, y_tr, X_val, strategy, param)
            best_t, best_f1 = hc.find_best_threshold(y_val.values, p_val, step=THRESHOLD_STEP)
            f1s.append(best_f1)
        key = f"{strategy}|{param}"
        results[key] = {'strategy': strategy, 'param': param,
                         'mean_f1': float(np.mean(f1s)), 'std_f1': float(np.std(f1s))}
        print(f"  {key:<22} | F1={results[key]['mean_f1']:.4f} ± {results[key]['std_f1']:.4f}")

    best_key = max(results, key=lambda k: results[k]['mean_f1'])
    best = results[best_key]
    print(f"\n  [Ablation Sonucu] Kazanan: strategy={best['strategy']}, param={best['param']} "
          f"(F1={best['mean_f1']:.4f})")
    if best['strategy'] == 'simple':
        print("  Not: 'Basit robust model' notu bu ablation ile DOĞRULANDI.")
    else:
        print("  Not: Ensemble, basit modeli geride bıraktı — bu, notun varsayımının "
              "TERSİ; PDR'da bu beklenmedik sonuç açıkça tartışılmalıdır.")
    return best['strategy'], best['param'], results

# ══════════════════════════════════════════════
# GLOBAL ÖZELLİK SEÇİMİ (LOOCV öncesi, tek sefer)
# ══════════════════════════════════════════════
def select_global_features(X, y, k_max=K_MAX_FEATURES, seed=SEED):
    print("\n" + "="*60)
    print(f"ADIM 5: Global Özellik Seçimi (LOOCV öncesi, TEK SEFER, k_max={k_max})")
    print("="*60)
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    imp, sc = SimpleImputer(strategy='median'), RobustScaler()
    X2 = X.copy()
    X2[num_cols] = imp.fit_transform(X2[num_cols])
    X2[num_cols] = sc.fit_transform(X2[num_cols])
    X2_f, _ = hc.correlation_filter(X2)
    Xs_tr, Xs_val, ys_tr, ys_val = train_test_split(X2_f, y, test_size=0.2, stratify=y, random_state=seed)
    sel = hc.shap_feature_selection(Xs_tr, ys_tr, Xs_val, ys_val, k_max=k_max)
    print(f"  Seçilen {len(sel)} özellik: {sel}")
    print("  [DÜRÜSTLÜK NOTU] Bu seçim TÜM veriyle (LOO'dan önce) yapılmıştır — "
          "kesin sıfır-leakage değildir, PDR'da belirtilmelidir (bkz. modül başı not).")
    return sel

# ══════════════════════════════════════════════
# TAM LOOCV
# ══════════════════════════════════════════════
def run_full_loocv(X, y, sel_feats, strategy, param, test_prior):
    print("\n" + "="*60)
    print(f"ADIM 7: Leave-One-Out CV (n={len(X)}, strategy={strategy}, param={param})")
    print("="*60)

    train_prior = hc.compute_train_prior(y)
    print(f"  TRAIN_PRIOR: {train_prior:.4f} ({(y==1).sum()} patojenik / {(y==0).sum()} benign)")

    X_sel = X[sel_feats]
    loo = LeaveOneOut()
    raw_probs = np.zeros(len(X_sel))
    labels    = y.values.copy()

    for i, (tr_idx, val_idx) in enumerate(loo.split(X_sel)):
        X_tr, X_val = X_sel.iloc[tr_idx], X_sel.iloc[val_idx]
        y_tr = y.iloc[tr_idx]
        p_val, _, _, _, _ = fit_strategy_get_raw_prob(X_tr, y_tr, X_val, strategy, param)
        raw_probs[val_idx[0]] = p_val[0]
        if (i+1) % 20 == 0 or (i+1) == len(X_sel):
            print(f"  {i+1}/{len(X_sel)} LOO iterasyonu tamamlandı...")

    # [v1] Kalibrasyon: TEK Platt haritası, havuzlanmış OOF üzerinde
    platt = fit_platt_1d(raw_probs, labels)
    oof_calibrated = apply_platt_1d(platt, raw_probs)
    oof_corrected  = hc.prior_correction(oof_calibrated, train_prior, test_prior)
    best_t, best_f1 = hc.find_best_threshold(labels, oof_corrected, step=THRESHOLD_STEP)
    pooled_metrics = hc.compute_metrics(labels, oof_corrected, threshold=best_t)

    print(f"\n  [LOOCV OOF-HAVUZLANMIŞ SONUÇ] F1={pooled_metrics['f1']:.4f} | "
          f"MCC={pooled_metrics['mcc']:.4f} | Brier={pooled_metrics['brier']:.4f} | eşik={best_t:.4f}")

    return {
        'raw_probs': raw_probs, 'labels': labels, 'platt': platt,
        'oof_calibrated': oof_calibrated, 'oof_corrected': oof_corrected,
        'threshold': best_t, 'pooled_metrics': pooled_metrics, 'train_prior': train_prior,
    }

# ══════════════════════════════════════════════
# ANA PIPELINE
# ══════════════════════════════════════════════
def run_cftr_pipeline(train_path, test_path=None, output_dir='cftr_results',
                       label_col='Label', test_prior=DEFAULT_TEST_PRIOR,
                       strategy=None, param=None, run_ablation=True, k_max=K_MAX_FEATURES):
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    print("\n" + "#"*60)
    print("HELIXAI -- CFTR Pipeline v1 (LOOCV)")
    print(f"Şartname v2.0 | Test Prior={test_prior:.4f}")
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
    print(f"\n  Ham özellik sayısı : {X_train.shape[1]}")
    print(f"  Patojenik / Benign : {(y_train==1).sum()} / {(y_train==0).sum()}")

    sel_feats = select_global_features(X_train, y_train, k_max=k_max)

    ablation_results = None
    if strategy is not None:
        chosen_strategy, chosen_param = strategy, param
        print(f"\nADIM 6: Model stratejisi manuel: {chosen_strategy}, param={chosen_param}")
    elif run_ablation:
        chosen_strategy, chosen_param, ablation_results = ablate_model_strategy(
            X_train[sel_feats], y_train, test_prior
        )
    else:
        chosen_strategy, chosen_param = 'ensemble', 1.0
        print(f"\nADIM 6: Ablation atlandı, varsayılan: ensemble, spw=1.0")

    loo_result = run_full_loocv(X_train, y_train, sel_feats, chosen_strategy, chosen_param, test_prior)

    # Stres testi (OOF havuzu üzerinde, ortak altyapı ile)
    stress_test_results = hc.run_stress_test_comparison(
        [{'val_probs_raw': loo_result['oof_calibrated'],
          'val_probs_corrected': loo_result['oof_corrected'],
          'val_labels': loo_result['labels']}],
        test_prior, panel_name="CFTR"
    )

    print("\nADIM 8: Nihai Model — TÜM eğitim verisiyle yeniden eğitiliyor")
    num_cols = X_train[sel_feats].select_dtypes(include=[np.number]).columns.tolist()
    final_imp, final_scaler = SimpleImputer(strategy='median'), RobustScaler()
    X_full = X_train[sel_feats].copy()
    X_full[num_cols] = final_imp.fit_transform(X_full[num_cols])
    X_full[num_cols] = final_scaler.fit_transform(X_full[num_cols])

    if chosen_strategy == 'simple':
        final_models = {'simple': build_simple_model(class_weight=chosen_param)}
        final_models['simple'].fit(X_full, y_train)
        final_weights = {'simple': 1.0}
    else:
        final_models = build_ensemble_models(spw=chosen_param)
        Xi_tr, Xi_val, yi_tr, yi_val = train_test_split(X_full, y_train, test_size=0.15,
                                                          stratify=y_train, random_state=SEED)
        final_models['xgb'].fit(Xi_tr, yi_tr, eval_set=[(Xi_val, yi_val)], verbose=False)
        final_models['lgb'].fit(Xi_tr, yi_tr, eval_set=[(Xi_val, yi_val)],
                                 callbacks=[lgb.early_stopping(15, verbose=False), lgb.log_evaluation(-1)])
        final_models['cat'].fit(Xi_tr, yi_tr, eval_set=(Xi_val, yi_val))
        final_weights = {'xgb': W_XGB, 'lgb': W_LGB, 'cat': W_CAT}

    test_metrics, em_check = None, None
    if X_test is not None:
        X_test_s = X_test[sel_feats].copy()
        X_test_s[num_cols] = final_imp.transform(X_test_s[num_cols])
        X_test_s[num_cols] = final_scaler.transform(X_test_s[num_cols])
        raw_test = hc.ensemble_predict_proba(final_models, final_weights, X_test_s)
        calib_test = apply_platt_1d(loo_result['platt'], raw_test)
        corr_test  = hc.prior_correction(calib_test, loo_result['train_prior'], test_prior)

        est_prior = hc.estimate_test_prior_em(calib_test, loo_result['train_prior'])
        prior_diff = abs(est_prior - test_prior)
        em_check = {'sartname_prior': test_prior, 'em_estimated_prior': est_prior, 'abs_diff': prior_diff}
        print(f"\n  [Sağlık Kontrolü - EM] Şartname={test_prior:.4f} | EM-tahmini={est_prior:.4f} | Fark={prior_diff:.4f}")
        if prior_diff > 0.05:
            print("  [UYARI] Fark >0.05 — PDR'da not edilmeli (CFTR'de test seti çok küçük, n≈120, "
                  "bu fark tek bir varyantın bile etkisiyle oluşabilir).")

        if test_df is not None and label_col in test_df.columns:
            y_test = test_df[label_col].astype(int)
            test_metrics = hc.compute_metrics(y_test, corr_test, threshold=loo_result['threshold'])
            print("\n  TEST SETİ SONUÇLARI:")
            for k, v in test_metrics.items():
                if isinstance(v, float): print(f"    {k:<16}: {v:.4f}")
        else:
            pred_path = os.path.join(output_dir, 'cftr_test_predictions.csv')
            pd.DataFrame({'prob_pathogenic': corr_test,
                          'predicted_label': (corr_test >= loo_result['threshold']).astype(int)}
                         ).to_csv(pred_path, index=False)
            print(f"\n  Etiketsiz test tahminleri kaydedildi: {pred_path}")

    print("\nADIM 9: SHAP Açıklanabilirlik (nihai model, kalibrasyon-sarmalsız — doğrudan)")
    try:
        import shap as _shap
        if chosen_strategy == 'simple':
            explainer = _shap.LinearExplainer(final_models['simple'], X_full)
            sv = explainer.shap_values(X_full)
            shap_method = "simple_linear"
        else:
            sv = np.zeros((X_full.shape[0], X_full.shape[1]))
            for name, mdl in final_models.items():
                sv = sv + final_weights[name] * _shap.TreeExplainer(mdl).shap_values(X_full)
            shap_method = "weighted_ensemble_direct"
        importance = pd.Series(np.abs(sv).mean(axis=0), index=X_full.columns).sort_values(ascending=False)
        print(f"  [SHAP yöntemi: {shap_method}] Top-{min(10,len(importance))}:")
        for feat, val in importance.head(10).items():
            print(f"  {feat:30s} {val:.4f}")
        importance.reset_index().rename(columns={'index': 'feature', 0: 'shap_mean_abs'}
                                         ).to_csv(os.path.join(output_dir, 'cftr_shap.csv'), index=False)
    except Exception as e:
        print(f"  [UYARI] SHAP başarısız oldu ({type(e).__name__}: {e}) — atlanıyor, pipeline devam ediyor.")
        importance, shap_method = pd.Series(dtype=float), "failed"

    elapsed = time.time() - t_start
    print("\n" + "#"*60)
    print("HELIXAI CFTR v1 -- FINAL RAPOR")
    print("#"*60)
    print(f"  Süre: {elapsed/60:.1f} dk | Strateji: {chosen_strategy} (param={chosen_param}) | "
          f"Eşik: {loo_result['threshold']:.4f}")
    print(f"  [LOOCV OOF] F1={loo_result['pooled_metrics']['f1']:.4f} | "
          f"MCC={loo_result['pooled_metrics']['mcc']:.4f} | Brier={loo_result['pooled_metrics']['brier']:.4f}")

    report = {
        'panel': 'CFTR', 'version': 'v1',
        'config': {'cv_strategy': 'LeaveOneOut', 'n': len(X_train),
                   'chosen_model_strategy': chosen_strategy, 'chosen_param': chosen_param,
                   'k_max_features': k_max, 'selected_features': sel_feats,
                   'train_prior': loo_result['train_prior'], 'test_prior_param': test_prior,
                   'final_threshold': loo_result['threshold'], 'shap_method': shap_method},
        'ablation_model_strategy': ablation_results,
        'loocv_oof_metrics': loo_result['pooled_metrics'],
        'stress_test': {'before': {k: v for k, v in stress_test_results['before'].items() if isinstance(v, (int, float))},
                        'after': {k: v for k, v in stress_test_results['after'].items() if isinstance(v, (int, float))},
                        'bootstrap_used': stress_test_results['bootstrap_used']},
        'em_test_prior_check': em_check, 'test': test_metrics,
        'shap_top10': importance.head(10).to_dict() if len(importance) else {},
        'elapsed_min': round(elapsed / 60, 2),
    }
    rep_path = os.path.join(output_dir, 'cftr_report_v1.json')
    with open(rep_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Rapor kaydedildi: {rep_path}")
    return report, loo_result

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
    tr = make(100, 20); tr.to_csv('/tmp/cftr_train.csv', index=False)
    te = make(20, 100); te.to_csv('/tmp/cftr_test.csv', index=False)
    print("Demo veri oluşturuldu: /tmp/cftr_train.csv (100/20), /tmp/cftr_test.csv (20/100)")
    return '/tmp/cftr_train.csv', '/tmp/cftr_test.csv'

# ══════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HELIXAI CFTR Pipeline v1 (LOOCV)')
    parser.add_argument('--train', type=str, default=None)
    parser.add_argument('--test', type=str, default=None)
    parser.add_argument('--label', type=str, default='Label')
    parser.add_argument('--output', type=str, default='cftr_results')
    parser.add_argument('--test-prior', type=float, default=DEFAULT_TEST_PRIOR)
    parser.add_argument('--strategy', type=str, default=None, choices=[None, 'ensemble', 'simple'])
    parser.add_argument('--param', type=str, default=None,
                         help="strategy=ensemble icin spw (float); strategy=simple icin class_weight ('balanced' veya bos)")
    parser.add_argument('--skip-ablation', action='store_true')
    parser.add_argument('--k-max', type=int, default=K_MAX_FEATURES)
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    if args.demo or args.train is None:
        print("[DEMO MODU]")
        train_p, test_p = generate_demo_data()
    else:
        train_p, test_p = args.train, args.test

    parsed_param = None
    if args.param is not None:
        try:
            parsed_param = float(args.param)
        except ValueError:
            parsed_param = args.param if args.param != '' else None

    run_cftr_pipeline(
        train_path=train_p, test_path=test_p, output_dir=args.output, label_col=args.label,
        test_prior=args.test_prior, strategy=args.strategy, param=parsed_param,
        run_ablation=not args.skip_ablation, k_max=args.k_max,
    )
