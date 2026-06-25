"""
HELIXAI — KANSER (Kalıtsal Kanser Paneli) Pipeline v1
========================================================
helixai_common.py'ye bağımlıdır (AYNI klasörde olmalı).

EL YAZISI NOTU (fotoğraf, danışman):
  "Kanser → BRCA1/BRCA2 ağırlıklı beklenir. Klonalite [ve] log molekülleri
   geçerli. Gen spesifik öz. bu panelde daha belirleyici."

BU NOTUN SOMUT KARŞILIĞI VE — ÖNEMLİ — UYGULANABİLİRLİK SINIRI:

  [DÜRÜSTLÜK UYARISI] Şartname Bölüm 3.2 açıkça şunu belirtiyor: "Tersine
  mühendisliği engellemek amacıyla veri paylaşılırken orijinal öznitelik
  kolon isimleri gizlenecek" ve genomik adres bilgisi TAMAMEN KALDIRILMIŞTIR.
  Yani veri setinde hiçbir varyantın hangi genden (BRCA1, BRCA2, veya başka
  bir HBOC/Lynch sendromu geni) geldiğini gösteren bir kolon YOKTUR.
  Dolayısıyla "if gene == 'BRCA1'" türünden literal bir gen-bazlı dallanma
  KODLANAMAZ — böyle bir kod, var olmayan bilgiyi varmış gibi göstererek
  hem halüsinasyon üretir hem de şartnamenin anonimleştirme amacını ihlal
  eder. Bu pipeline BU NEDENLE literal gen hedeflemesi YAPMAZ.

  [UYGULANAN DÜRÜST ALTERNATİF] Not, klinik gerçekliği doğru yansıtıyor:
  kalıtsal kanser panelleri genellikle birden fazla genin (BRCA1/2 ve
  muhtemelen diğer HBOC/Lynch genlerinin) varyantlarını içerir ve "gen-
  spesifik" karakter panelde belirleyicidir. Bu biyolojik önsel, anonim
  özellik uzayında GÖZLEMLENEBİLİR bir gizli alt-grup (latent subgroup)
  yapısına işaret edebilir: farklı genlerin popülasyon frekansı/evrimsel
  korunmuşluk profilleri sistematik olarak farklı olabilir.
  add_cluster_proxy_feature() fonksiyonu, ETİKETSİZ ve KOLON-İSMİ-
  GEREKTİRMEYEN bir şekilde (sadece KMeans ile, sayısal özellik uzayından,
  her fold içinde SADECE eğitim verisiyle fit edilerek) bu olası alt-grup
  yapısını bir PROXY ÖZELLİK olarak yakalamayı dener. Bu, gerçek gen
  kimliğiyle örtüştüğü GARANTİ EDİLMEYEN, yine de modelin örtük "gen-
  spesifik heterojenliği" öğrenebilmesine yardımcı olabilecek dürüst bir
  mühendislik denemesidir — PDR'da bu sınırlamayla birlikte sunulmalıdır.

  [Diğer not — "klonalite ve log molekülleri geçerli"] Bu not, EK_
  (evrimsel korunmuşluk) ve log-dönüştürülmüş AL_ (allel frekansı) özellik
  ailelerinin bu panelde özellikle bilgilendirici olduğunu doğruluyor.
  Buna karşılık BU PIPELINE'da iki somut değişiklik yapılmıştır:
    (a) correlation_filter() artık EK_ önekli kolonları PROTECTED tutuyor
        (yüksek korelasyonlu olsalar da otomatik düşürülmüyorlar) —
        helixai_common.correlation_filter(..., protected_prefixes=('EK_',)).
    (b) log_transform_af() (AL_ log dönüşümü) DEĞİŞİKLİKSİZ taşındı; not
        bu yöntemin Kanser panelinde de geçerli olduğunu teyit ediyor.

GERÇEK VERİ (doğrulanmış, YARISMA_TRAIN_KANSER.csv):
  Eğitim (gerçek)             : 268 Patojenik / 120 Benign (n=388)
  Eğitim (şartname, yaklaşık) : 250 Patojenik / 100 Benign
  Test   (şartname, yaklaşık) : 100 Patojenik / 500 Benign

Kullanım:
  python kanser_pipeline_v1.py --demo
  python kanser_pipeline_v1.py --train YARISMA_TRAIN_KANSER.csv --test kanser_test.csv
"""

import argparse, os, json, time
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
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

DEFAULT_TEST_PRIOR = 100 / (100 + 500)   # ≈ 0.1667
DEFAULT_SPW         = 1.0

W_XGB, W_LGB, W_CAT = 0.35, 0.30, 0.35

SPW_CANDIDATES     = (0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
ABLATION_N_REPEATS = 2

# [v1-YENİ, PAH'taki gözlemden sonra eklendi] N=388 ile 371 özelliğin
# tamamını taramak hem çok yavaş hem de aşırı öğrenme riski taşıyor —
# PAH'ta bu tavansız haliyle ~2 saat sürmüş ve düşük specificity/MCC ile
# sonuçlanmıştı. Aynı gerekçeyle burada da bir tavan konuyor.
K_MAX_FEATURES = 40

PROTECTED_PREFIXES = ('EK_',)   # [v1-YENİ] "klonalite/log molekülleri geçerli" notuna uyum
N_CLUSTERS          = 4         # [v1-YENİ] gen-sayısı bilinmiyor; ablation ile de denenebilir

# ══════════════════════════════════════════════
# [v1-YENİ] GİZLİ GEN-KÜMELEME PROXY ÖZELLİĞİ
# ══════════════════════════════════════════════
def add_cluster_proxy_feature(X_tr, X_val, n_clusters=N_CLUSTERS, seed=SEED, fitted_km=None):
    """
    [Hakem notuna yanıt: 'BRCA1/2 ağırlıklı beklenir, gen-spesifik özellikler
    bu panelde daha belirleyici']

    fitted_km=None ise: KMeans SADECE X_tr (eğitim fold'u) ile fit edilir.
    fitted_km verilirse (test-zamanı kullanım): yeniden fit EDİLMEZ, sadece
    transform/predict uygulanır — bu, train_fold'ta fit edilen KMeans'in
    test setine TUTARLI şekilde uygulanmasını sağlar (yeniden fit etmek
    train/test arasında farklı küme tanımlarına yol açardı).

    Çıktı: tek bir kategorik 'cluster_proxy' kolonu (one-hot). Gerçek gen
    kimliğiyle birebir örtüşmesi GARANTİ EDİLMEZ — bu, var olduğu
    DÜŞÜNÜLEN gizli alt-grup yapısını yakalamaya çalışan bir DENEME
    özelliğidir, kesin bir gen etiketleyici DEĞİLDİR.
    """
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    km = fitted_km if fitted_km is not None else KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    tr_vals  = X_tr[num_cols].fillna(X_tr[num_cols].median())
    val_vals = X_val[num_cols].fillna(X_tr[num_cols].median())
    if fitted_km is None:
        km.fit(tr_vals)
    tr_clusters  = km.predict(tr_vals)
    val_clusters = km.predict(val_vals)

    X_tr_out, X_val_out = X_tr.copy(), X_val.copy()
    for k in range(n_clusters):
        X_tr_out[f'cluster_proxy_{k}']  = (tr_clusters == k).astype(int)
        X_val_out[f'cluster_proxy_{k}'] = (val_clusters == k).astype(int)
    return X_tr_out, X_val_out, km

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
               spw=DEFAULT_SPW, fold_id="?", use_cluster_proxy=True):
    num_cols = X_tr_raw.select_dtypes(include=[np.number]).columns.tolist()

    imp = SimpleImputer(strategy='median')
    X_tr, X_val = X_tr_raw.copy(), X_val_raw.copy()
    X_tr[num_cols]  = imp.fit_transform(X_tr_raw[num_cols])
    X_val[num_cols] = imp.transform(X_val_raw[num_cols])

    scaler = RobustScaler()
    X_tr[num_cols]  = scaler.fit_transform(X_tr[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])

    # [v1-YENİ] gizli gen-kümeleme proxy özelliği (sadece X_tr ile fit)
    fitted_km = None
    if use_cluster_proxy:
        X_tr, X_val, fitted_km = add_cluster_proxy_feature(X_tr, X_val)

    # [v1-YENİ] EK_ ailesini koru ("klonalite/log molekülleri geçerli")
    X_tr_f, dropped_corr = hc.correlation_filter(X_tr, protected_prefixes=PROTECTED_PREFIXES)
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
    print(f"    [Kalibrasyon | {fold_id}] val seti: {len(y_val)} örnek "
          f"({n_pat_val} patojenik / {n_ben_val} benign)")

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
        'val_labels': y_val.values, '_use_cluster_proxy': use_cluster_proxy, 'kmeans': fitted_km,
    }

# ══════════════════════════════════════════════
# [v1-YENİ] KANSER'E ÖZEL predict_test — cluster proxy'yi test setine TUTARLI uygular
# ══════════════════════════════════════════════
def predict_test_kanser(X_test_raw, fold_results, train_prior, test_prior, final_threshold):
    """
    helixai_common.predict_test ile AYNI mantık, TEK fark: 'cluster_proxy_*'
    kolonları, her fold'un train_fold() içinde FIT EDİLMİŞ KMeans nesnesiyle
    (yeniden fit EDİLMEDEN — bkz. add_cluster_proxy_feature(fitted_km=...))
    yeniden üretilir. Bu olmadan, sel_feats içinde 'cluster_proxy_k' seçilmiş
    bir fold, test setinde bu kolonu bulamaz ve KeyError verir.
    """
    all_probs, all_raw_probs = [], []
    for res in fold_results:
        X_t = X_test_raw.copy()
        ncols = X_t.select_dtypes(include=[np.number]).columns.tolist()
        X_t[ncols] = res['imputer'].transform(X_t[ncols])
        X_t[ncols] = res['scaler'].transform(X_t[ncols])

        if res.get('_use_cluster_proxy') and res.get('kmeans') is not None:
            # add_cluster_proxy_feature iki taraf (tr,val) bekler; burada
            # ikinci argüman olarak da X_t veriyoruz, sadece val/test çıkışını kullanırız.
            _, X_t, _ = add_cluster_proxy_feature(X_t, X_t, fitted_km=res['kmeans'])

        X_t = X_t.drop(columns=res['dropped_corr'], errors='ignore')
        X_t = X_t[res['sel_feats']]

        p_ens = hc.ensemble_predict_proba(res['models_cal'], res['weights'], X_t)
        all_raw_probs.append(p_ens)
        all_probs.append(hc.prior_correction(p_ens, train_prior, test_prior))

    final_probs = np.mean(all_probs, axis=0)
    raw_probs_mean = np.mean(all_raw_probs, axis=0)
    return final_probs, final_threshold, raw_probs_mean

# ══════════════════════════════════════════════
# [v1-YENİ] KANSER'E ÖZEL run_shap — cluster proxy'yi SHAP girdisine TUTARLI uygular
# ══════════════════════════════════════════════
def run_shap_kanser(fold_results, X_train, output_dir, panel_name="KANSER", prefix="kanser"):
    """
    helixai_common.run_shap ile AYNI mantık (güvenli düşüşlü ağırlıklı SHAP),
    TEK fark: imputer/scaler sonrası, dropped_corr/sel_feats ÖNCESİ, en iyi
    fold'un FIT EDİLMİŞ KMeans'iyle (yeniden fit edilmeden) cluster_proxy_*
    kolonları yeniden üretilir — predict_test_kanser ile AYNI tutarlılık.
    """
    print("\n" + "="*60)
    print(f"[{panel_name}] SHAP Açıklanabilirlik (güvenli düşüşlü, cluster-proxy tutarlı)")
    print("="*60)

    best = max(fold_results, key=lambda r: r['metrics']['f1'])
    ncols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_s = X_train.copy()
    X_s[ncols] = best['imputer'].transform(X_s[ncols])
    X_s[ncols] = best['scaler'].transform(X_s[ncols])

    if best.get('_use_cluster_proxy') and best.get('kmeans') is not None:
        _, X_s, _ = add_cluster_proxy_feature(X_s, X_s, fitted_km=best['kmeans'])

    X_s = X_s.drop(columns=best['dropped_corr'], errors='ignore')
    X_s = X_s[best['sel_feats']]

    try:
        sv_weighted = np.zeros((X_s.shape[0], X_s.shape[1]))
        for name, model in best['models_cal'].items():
            sv = __import__('shap').TreeExplainer(hc.get_base_estimator(model)).shap_values(X_s)
            sv_weighted = sv_weighted + best['weights'][name] * sv
        shap_method = "weighted_ensemble"
        importance = pd.Series(np.abs(sv_weighted).mean(axis=0), index=X_s.columns).sort_values(ascending=False)
    except Exception as e:
        print(f"  [UYARI] Ağırlıklı ensemble SHAP başarısız oldu ({type(e).__name__}: {e}).")
        top_name = max(best['weights'], key=best['weights'].get)
        print(f"          Güvenli düşüş 1: en yüksek ağırlıklı model ('{top_name}') ile SHAP deneniyor.")
        try:
            sv_weighted = __import__('shap').TreeExplainer(hc.get_base_estimator(best['models_cal'][top_name])).shap_values(X_s)
            shap_method = f"{top_name}_only_fallback"
            importance = pd.Series(np.abs(sv_weighted).mean(axis=0), index=X_s.columns).sort_values(ascending=False)
        except Exception as e2:
            print(f"  [UYARI] SHAP tamamen başarısız oldu ({type(e2).__name__}: {e2}).")
            print("          Güvenli düşüş 2: model.feature_importances_ (gain) kullanılıyor.")
            base_est = hc.get_base_estimator(best['models_cal'][top_name])
            importance = pd.Series(base_est.feature_importances_, index=X_s.columns).sort_values(ascending=False)
            shap_method = f"{top_name}_feature_importances_fallback"

    print(f"\n  [SHAP yöntemi: {shap_method}] Top-10 — {panel_name}:")
    for feat, val in importance.head(10).items():
        bar = "#" * int(val / importance.iloc[0] * 25) if importance.iloc[0] > 0 else ""
        print(f"  {feat:30s} {bar} {val:.4f}")

    path = os.path.join(output_dir, f'{prefix}_shap.csv')
    importance.reset_index().rename(columns={'index': 'feature', 0: 'shap_mean_abs'}).to_csv(path, index=False)
    print(f"\n  Kaydedildi: {path}")
    return importance, shap_method

# ══════════════════════════════════════════════
# scale_pos_weight ABLATION (panel-spesifik: Kanser)
# ══════════════════════════════════════════════
def ablate_scale_pos_weight(X, y, test_prior, candidates=SPW_CANDIDATES, repeats=ABLATION_N_REPEATS):
    print("\n" + "="*60)
    print(f"ADIM 6: scale_pos_weight Ablation (panel: KANSER, {repeats}x{N_SPLITS}-Fold)")
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
                              spw=spw, fold_id=f"ablation_spw{spw}")
            f1s.append(res['metrics']['f1'])
        results[spw] = {'mean_f1': float(np.mean(f1s)), 'std_f1': float(np.std(f1s))}
        print(f"  spw={spw:<5} | F1={results[spw]['mean_f1']:.4f} ± {results[spw]['std_f1']:.4f}")
    best_spw = max(results, key=lambda k: results[k]['mean_f1'])
    print(f"\n  [Ablation] En iyi spw: {best_spw} (F1={results[best_spw]['mean_f1']:.4f})")
    return best_spw, results

# ══════════════════════════════════════════════
# ANA PIPELINE
# ══════════════════════════════════════════════
def run_kanser_pipeline(train_path, test_path=None, output_dir='kanser_results',
                         label_col='Label', test_prior=DEFAULT_TEST_PRIOR,
                         spw=None, run_ablation=True, ablation_repeats=ABLATION_N_REPEATS,
                         use_cluster_proxy=True):
    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    print("\n" + "#"*60)
    print("HELIXAI -- KANSER Pipeline v1")
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
    print(f"  [Not] EK_ kolon ailesi korelasyon filtresinden korunuyor "
          f"({PROTECTED_PREFIXES}); gizli gen-kümeleme proxy özelliği "
          f"{'AKTİF' if use_cluster_proxy else 'KAPALI'} (n_clusters={N_CLUSTERS}).")

    ablation_results = None
    if spw is not None:
        chosen_spw = spw
        print(f"\nADIM 6: scale_pos_weight — manuel: {chosen_spw}")
    elif run_ablation:
        chosen_spw, ablation_results = ablate_scale_pos_weight(X_train, y_train, test_prior, repeats=ablation_repeats)
    else:
        chosen_spw = DEFAULT_SPW
        print(f"\nADIM 6: scale_pos_weight — ablation atlandı, varsayılan: {chosen_spw}")

    fold_results, cv_summary, train_prior = hc.run_repeated_cv(
        X_train, y_train, test_prior, train_fold, N_SPLITS, N_REPEATS,
        fold_kwargs={'spw': chosen_spw, 'use_cluster_proxy': use_cluster_proxy}, panel_name="KANSER"
    )

    final_threshold = hc.compute_oof_pooled_threshold(fold_results)
    stress_test_results = hc.run_stress_test_comparison(fold_results, test_prior, panel_name="KANSER")

    test_metrics, em_check = None, None
    if X_test is not None:
        test_probs, _, raw_probs_mean = predict_test_kanser(X_test, fold_results, train_prior, test_prior, final_threshold)
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
            pred_path = os.path.join(output_dir, 'kanser_test_predictions.csv')
            pd.DataFrame({'prob_pathogenic': test_probs,
                          'predicted_label': (test_probs >= final_threshold).astype(int)}
                         ).to_csv(pred_path, index=False)
            print(f"\n  Etiketsiz test tahminleri kaydedildi: {pred_path}")

    shap_imp, shap_method = run_shap_kanser(fold_results, X_train, output_dir, panel_name="KANSER", prefix="kanser")

    elapsed = time.time() - t_start
    print("\n" + "#"*60)
    print("HELIXAI KANSER v1 -- FINAL RAPOR")
    print("#"*60)
    print(f"  Süre: {elapsed/60:.1f} dk | spw: {chosen_spw} | Eşik: {final_threshold:.4f}")
    for k, v in cv_summary.items():
        print(f"  {k:<16} {v['mean']:>8.4f} +/-{v['std']:>8.4f}")

    report = {
        'panel': 'KANSER', 'version': 'v1',
        'config': {'n_splits': N_SPLITS, 'n_repeats': N_REPEATS,
                   'weights': {'xgb': W_XGB, 'lgb': W_LGB, 'cat': W_CAT},
                   'train_prior_computed': train_prior, 'test_prior_param': test_prior,
                   'final_threshold_oof': final_threshold, 'chosen_scale_pos_weight': chosen_spw,
                   'shap_method': shap_method, 'use_cluster_proxy': use_cluster_proxy,
                   'protected_prefixes': PROTECTED_PREFIXES},
        'ablation_scale_pos_weight': ablation_results,
        'cv_summary': {k: {'mean': v['mean'], 'std': v['std']} for k, v in cv_summary.items()},
        'oof_y_true': cv_summary.get('_oof_y_true', []),
        'oof_y_prob': cv_summary.get('_oof_y_prob', []),
        'stress_test': {'before': {k: v for k, v in stress_test_results['before'].items() if isinstance(v, (int, float))},
                        'after': {k: v for k, v in stress_test_results['after'].items() if isinstance(v, (int, float))},
                        'bootstrap_used': stress_test_results['bootstrap_used']},
        'em_test_prior_check': em_check, 'test': test_metrics,
        'shap_top10': shap_imp.head(10).to_dict(), 'elapsed_min': round(elapsed / 60, 2),
    }
    rep_path = os.path.join(output_dir, 'kanser_report_v1.json')
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
        # iki "gizli gen kümesi" simülasyonu (BRCA1-benzeri / BRCA2-benzeri)
        gene_sim = np.random.choice([0, 1], n)
        df['EK_phylop'] = np.where(y == 1, np.random.normal(3.5 + gene_sim*0.5, 1.0, n),
                                            np.random.normal(0.5 + gene_sim*0.3, 1.5, n))
        df['EK_gerp']   = np.where(y == 1, np.random.normal(4.0 - gene_sim*0.4, 1.2, n),
                                            np.random.normal(0.2, 2.0, n))
        df['CAT_pop']     = np.random.choice(['EUR', 'AFR', 'EAS', 'SAS', 'AMR'], n)
        df['CAT_quality'] = np.random.choice(['PASS', 'LOW_QUAL'], n, p=[0.9, 0.1])
        for c in ['AL_freq_eur', 'EK_gerp']:
            df.loc[np.random.rand(n) < 0.10, c] = np.nan
        df['Label'] = y
        return df
    tr = make(250, 100); tr.to_csv('/tmp/kanser_train.csv', index=False)
    te = make(100, 500); te.to_csv('/tmp/kanser_test.csv', index=False)
    print("Demo veri oluşturuldu: /tmp/kanser_train.csv (250/100), /tmp/kanser_test.csv (100/500)")
    return '/tmp/kanser_train.csv', '/tmp/kanser_test.csv'

# ══════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HELIXAI KANSER Pipeline v1')
    parser.add_argument('--train', type=str, default=None)
    parser.add_argument('--test', type=str, default=None)
    parser.add_argument('--label', type=str, default='Label')
    parser.add_argument('--output', type=str, default='kanser_results')
    parser.add_argument('--test-prior', type=float, default=DEFAULT_TEST_PRIOR)
    parser.add_argument('--spw', type=float, default=None)
    parser.add_argument('--skip-ablation', action='store_true')
    parser.add_argument('--ablation-repeats', type=int, default=ABLATION_N_REPEATS)
    parser.add_argument('--no-cluster-proxy', action='store_true',
                         help='Gizli gen-kümeleme proxy özelliğini kapat')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    if args.demo or args.train is None:
        print("[DEMO MODU]")
        train_p, test_p = generate_demo_data()
    else:
        train_p, test_p = args.train, args.test

    run_kanser_pipeline(
        train_path=train_p, test_path=test_p, output_dir=args.output, label_col=args.label,
        test_prior=args.test_prior, spw=args.spw, run_ablation=not args.skip_ablation,
        ablation_repeats=args.ablation_repeats, use_cluster_proxy=not args.no_cluster_proxy,
    )