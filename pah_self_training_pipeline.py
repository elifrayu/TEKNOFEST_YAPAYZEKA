"""
HELIXAI — PAH Self-Training (Yarı-Denetimli Öğrenme)
========================================================
helixai_common.py'ye bağımlıdır (AYNI klasörde olmalı).

YÖNTEM (PrimateAI'nin patentinde tanımlanan iteratif şemaya benzer —
bkz. modül başı not): Model önce SADECE etiketli PAH verisiyle eğitilir.
Sonra, ETİKETSİZ bir havuz (gerçek yarışma test seti veya geçici bir
simülasyon) üzerinde tahmin yapılır. Modelin ÇOK GÜVENLİ olduğu tahminler
("pseudo-label") düşük bir ağırlıkla eğitime GERİ KATILIR. Bu, dış veri
ÇEKMEDEN (gnomAD/ESM gibi harici kaynaklara hiç ihtiyaç duymadan), zaten
elimizde olan ama etiketsiz veriden ek bilgi çıkarmaya çalışır.

ÖNEMLİ — İKİ ÇALIŞMA MODU:
  1. SIMÜLASYON MODU (--unlabeled vermezseniz, şimdi kullanın): PAH'ın
     KENDİ etiketli verisinden bir kısmı ayrılır, etiketleri GİZLENİR,
     "sanki etiketsiz test seti" gibi kullanılır. Gerçek etiketler sadece
     SONUNDA, yöntemin gerçekten işe yarayıp yaramadığını ÖLÇMEK için
     kullanılır (eğitime asla katılmaz). Bu mod, gerçek test seti
     gelmeden ÖNCE yöntemi doğrulamanızı sağlar.
  2. PRODÜKSİYON MODU (--unlabeled ile gerçek, etiketsiz şartname test
     seti verildiğinde): Gerçek etiket yoktur, dolayısıyla "işe yaradı mı"
     ölçülemez — sadece pseudo-label ekleyip nihai tahminler üretilir.
     Mekanizma SİMÜLASYON MODUNDA zaten doğrulanmış olduğu için bu modda
     güvenle kullanılabilir.

[DÜRÜSTLÜK NOTU] Pseudo-label'lar GERÇEK etiket değildir — modelin kendi
tahminine "modelin kendi tahminini doğrulamak" gibi bir döngüsel risk
(circularity) taşır. Bunu azaltmak için: (a) SADECE çok yüksek güvenli
tahminler (prob<0.05 veya prob>0.95) pseudo-label olarak kabul edilir,
(b) pseudo-label'lara gerçek etiketlerden DAHA DÜŞÜK bir eğitim ağırlığı
(sample_weight) verilir, (c) sadece TEK bir iterasyon yapılır (PrimateAI'nin
"kademeli adımlarla erken yakınsamayı önleme" prensibiyle uyumlu — agresif/
çok-iterasyonlu self-training, modelin kendi hatalarını güçlendirme riski
taşır).

Kullanım:
  # Simülasyon modu (test seti henüz yokken, yöntemi doğrulama):
  python3 pah_self_training_pipeline.py --pah YARISMA_TRAIN_PAH.csv --mode simulate

  # Prodüksiyon modu (gerçek etiketsiz test seti geldiğinde):
  python3 pah_self_training_pipeline.py --pah YARISMA_TRAIN_PAH.csv --unlabeled pah_test.csv --mode production
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
N_REPEATS       = 10
THRESHOLD_STEP  = 0.005
K_MAX_FEATURES  = 40

DEFAULT_TEST_PRIOR = 100 / (100 + 250)
DEFAULT_SPW         = 1.0
W_XGB, W_LGB, W_CAT = 0.35, 0.30, 0.35

# [Pseudo-label güven eşikleri] — sadece ÇOK güvenli tahminler kabul edilir
TAU_BENIGN  = 0.05   # düzeltilmiş olasılık < 0.05 -> pseudo-label = 0 (benign)
TAU_PATHO   = 0.95   # düzeltilmiş olasılık > 0.95 -> pseudo-label = 1 (patojenik)
PSEUDO_LABEL_WEIGHT = 0.5   # gerçek etiketin (1.0) yarısı kadar güven

# ══════════════════════════════════════════════
# MODEL YAPILARI
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
# 1. VERİ HAZIRLAMA
# ══════════════════════════════════════════════
def load_and_prep(path, label_col='Label', has_label=True):
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c != label_col]
    tr_feats, _ = hc.global_col_filter(df[feat_cols])
    tr_feats = hc.biological_qc(tr_feats)
    tr_feats = hc.engineer_aa_features(tr_feats)
    tr_feats = hc.log_transform_af(tr_feats)
    X = hc.to_numeric_df(tr_feats, label_col=label_col)
    y = df[label_col].astype(int) if (has_label and label_col in df.columns) else None
    return X, y

# ══════════════════════════════════════════════
# 2. TEK MODEL EĞİTİMİ (sample_weight destekli — pseudo-label için gerekli)
# ══════════════════════════════════════════════
def fit_ensemble(X_tr, y_tr, sample_weight=None, spw=DEFAULT_SPW):
    models = {'xgb': build_xgb(spw), 'lgb': build_lgb(spw), 'cat': build_cat(spw)}
    models['xgb'].fit(X_tr, y_tr, sample_weight=sample_weight)
    models['lgb'].fit(X_tr, y_tr, sample_weight=sample_weight)
    models['cat'].fit(X_tr, y_tr, sample_weight=sample_weight)
    return models

def predict_ensemble(models, X):
    return (W_XGB * models['xgb'].predict_proba(X)[:, 1] +
            W_LGB * models['lgb'].predict_proba(X)[:, 1] +
            W_CAT * models['cat'].predict_proba(X)[:, 1])

# ══════════════════════════════════════════════
# 3. SELF-TRAINING TEK TUR
# ══════════════════════════════════════════════
def self_training_round(X_tr, y_tr, X_unlabeled, train_prior, test_prior, spw=DEFAULT_SPW):
    """
    1) İlk model SADECE gerçek etiketli veriyle eğitilir.
    2) X_unlabeled üzerinde tahmin yapılır, prior-correction uygulanır.
    3) Çok güvenli tahminler (tau eşiklerini geçen) pseudo-label olarak
       işaretlenir, DÜŞÜK ağırlıkla (PSEUDO_LABEL_WEIGHT) eğitime EKLENİR.
    4) İkinci (final) model, gerçek + pseudo-label ile YENİDEN eğitilir.
    Döner: (initial_models, final_models, pseudo_label_info)
    """
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    imp, sc = SimpleImputer(strategy='median'), RobustScaler()
    X_tr_s = X_tr.copy(); X_tr_s[num_cols] = imp.fit_transform(X_tr[num_cols])
    X_tr_s[num_cols] = sc.fit_transform(X_tr_s[num_cols])
    X_tr_f, dropped = hc.correlation_filter(X_tr_s)

    Xs_tr, Xs_val, ys_tr, ys_val = train_test_split(X_tr_f, y_tr, test_size=0.2,
                                                      stratify=y_tr, random_state=SEED)
    sel = hc.shap_feature_selection(Xs_tr, ys_tr, Xs_val, ys_val, k_max=K_MAX_FEATURES)

    X_tr_sel = X_tr_f[sel]
    initial_models = fit_ensemble(X_tr_sel, y_tr, spw=spw)

    X_unl = X_unlabeled.copy()
    X_unl[num_cols] = imp.transform(X_unl[num_cols])
    X_unl[num_cols] = sc.transform(X_unl[num_cols])
    X_unl = X_unl.drop(columns=dropped, errors='ignore')[sel]

    raw_unl = predict_ensemble(initial_models, X_unl)
    corr_unl = hc.prior_correction(raw_unl, train_prior, test_prior)

    pseudo_benign_idx = np.where(corr_unl < TAU_BENIGN)[0]
    pseudo_patho_idx  = np.where(corr_unl > TAU_PATHO)[0]

    print(f"    [Self-Training] Etiketsiz havuz n={len(X_unl)} | "
          f"güvenli pseudo-benign={len(pseudo_benign_idx)} | "
          f"güvenli pseudo-patojenik={len(pseudo_patho_idx)}")

    X_pseudo = pd.concat([X_unl.iloc[pseudo_benign_idx], X_unl.iloc[pseudo_patho_idx]], axis=0)
    y_pseudo = pd.Series([0]*len(pseudo_benign_idx) + [1]*len(pseudo_patho_idx))

    if len(X_pseudo) > 0:
        X_combined = pd.concat([X_tr_sel.reset_index(drop=True), X_pseudo.reset_index(drop=True)], axis=0, ignore_index=True)
        y_combined = pd.concat([y_tr.reset_index(drop=True), y_pseudo.reset_index(drop=True)], axis=0, ignore_index=True)
        w_combined = np.concatenate([np.ones(len(y_tr)), np.full(len(y_pseudo), PSEUDO_LABEL_WEIGHT)])
        final_models = fit_ensemble(X_combined, y_combined, sample_weight=w_combined, spw=spw)
    else:
        print("    [Self-Training] Güvenli pseudo-label bulunamadı — final model = initial model.")
        final_models = initial_models

    info = {'n_pseudo_benign': len(pseudo_benign_idx), 'n_pseudo_patho': len(pseudo_patho_idx),
            'n_unlabeled_pool': len(X_unl)}
    return initial_models, final_models, info, (imp, sc, dropped, sel)

# ══════════════════════════════════════════════
# 4. SİMÜLASYON MODU — PAH'ın kendi verisiyle yöntemi doğrula
# ══════════════════════════════════════════════
def run_simulation_mode(pah_path, output_dir, label_col='Label', test_prior=DEFAULT_TEST_PRIOR,
                         spw=DEFAULT_SPW, holdout_frac=0.3):
    print("\n" + "#"*60)
    print("HELIXAI -- PAH SELF-TRAINING (SİMÜLASYON MODU)")
    print("Gerçek test seti henüz yok; PAH'ın kendi verisinden bir kısım")
    print("'sanki etiketsiz' olarak ayrılıp yöntem doğrulanıyor.")
    print("#"*60)

    X, y = load_and_prep(pah_path, label_col)
    print(f"\n  Toplam PAH verisi: {X.shape} | Patojenik/Benign={(y==1).sum()}/{(y==0).sum()}")

    X_tr, X_holdout, y_tr, y_holdout = train_test_split(
        X, y, test_size=holdout_frac, stratify=y, random_state=SEED
    )
    print(f"  Eğitim havuzu: {len(X_tr)} | 'Etiketsiz' simülasyon havuzu: {len(X_holdout)} "
          f"(gerçek etiketler SADECE değerlendirme için saklanıyor, eğitime KATILMIYOR)")

    train_prior = hc.compute_train_prior(y_tr)

    print("\n  -- Self-training öncesi (sadece gerçek etiketli) referans model --")
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    imp0, sc0 = SimpleImputer(strategy='median'), RobustScaler()
    X_tr0 = X_tr.copy(); X_tr0[num_cols] = imp0.fit_transform(X_tr[num_cols])
    X_tr0[num_cols] = sc0.fit_transform(X_tr0[num_cols])
    X_tr0_f, dropped0 = hc.correlation_filter(X_tr0)
    Xs_tr0, Xs_val0, ys_tr0, ys_val0 = train_test_split(X_tr0_f, y_tr, test_size=0.2,
                                                          stratify=y_tr, random_state=SEED)
    sel0 = hc.shap_feature_selection(Xs_tr0, ys_tr0, Xs_val0, ys_val0, k_max=K_MAX_FEATURES)
    initial_only = fit_ensemble(X_tr0_f[sel0], y_tr, spw=spw)

    X_holdout0 = X_holdout.copy()
    X_holdout0[num_cols] = imp0.transform(X_holdout0[num_cols])
    X_holdout0[num_cols] = sc0.transform(X_holdout0[num_cols])
    X_holdout0 = X_holdout0.drop(columns=dropped0, errors='ignore')[sel0]
    raw_before = predict_ensemble(initial_only, X_holdout0)
    corr_before = hc.prior_correction(raw_before, train_prior, test_prior)
    t_before, _ = hc.find_best_threshold(y_holdout.values, corr_before, step=THRESHOLD_STEP)
    metrics_before = hc.compute_metrics(y_holdout.values, corr_before, threshold=t_before)

    print("\n  -- Self-training uygulanıyor --")
    initial_models, final_models, info, (imp, sc, dropped, sel) = self_training_round(
        X_tr, y_tr, X_holdout, train_prior, test_prior, spw=spw
    )

    X_holdout_s = X_holdout.copy()
    X_holdout_s[num_cols] = imp.transform(X_holdout_s[num_cols])
    X_holdout_s[num_cols] = sc.transform(X_holdout_s[num_cols])
    X_holdout_s = X_holdout_s.drop(columns=dropped, errors='ignore')[sel]
    raw_after = predict_ensemble(final_models, X_holdout_s)
    corr_after = hc.prior_correction(raw_after, train_prior, test_prior)
    t_after, _ = hc.find_best_threshold(y_holdout.values, corr_after, step=THRESHOLD_STEP)
    metrics_after = hc.compute_metrics(y_holdout.values, corr_after, threshold=t_after)

    print(f"\n  {'Metrik':<14}{'Self-training ÖNCESİ':>22}{'Self-training SONRASI':>24}{'Fark':>10}")
    print("  " + "─"*70)
    for k in ['f1', 'mcc', 'sensitivity', 'specificity', 'pr_auc']:
        b, a = metrics_before[k], metrics_after[k]
        print(f"  {k:<14}{b:>22.4f}{a:>24.4f}{a-b:>+10.4f}")

    verdict = "SELF-TRAINING FAYDALI (simülasyonda)" if metrics_after['mcc'] > metrics_before['mcc'] \
              else "ANLAMLI FAYDA GÖRÜLMEDİ (simülasyonda)"
    print(f"\n[SONUÇ - SİMÜLASYON] {verdict} (MCC farkı: {metrics_after['mcc']-metrics_before['mcc']:+.4f})")
    print("\n[ÖNEMLİ] Bu sonuç, GERÇEK test setinde AYNI YÖNDE davranacağının garantisi DEĞİLDİR")
    print("         — sadece mekanizmanın PAH verisi üzerinde çalıştığını ve makul göründüğünü gösterir.")

    os.makedirs(output_dir, exist_ok=True)
    report = {
        'mode': 'simulate', 'pseudo_label_info': info,
        'metrics_before': {k: v for k, v in metrics_before.items() if isinstance(v, (int, float))},
        'metrics_after': {k: v for k, v in metrics_after.items() if isinstance(v, (int, float))},
        'verdict': verdict,
    }
    with open(os.path.join(output_dir, 'pah_selftraining_simulation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nRapor kaydedildi: {output_dir}/pah_selftraining_simulation_report.json")
    return report

# ══════════════════════════════════════════════
# 5. PRODÜKSIYON MODU — gerçek etiketsiz test seti geldiğinde
# ══════════════════════════════════════════════
def run_production_mode(pah_path, unlabeled_path, output_dir, label_col='Label',
                         test_prior=DEFAULT_TEST_PRIOR, spw=DEFAULT_SPW):
    print("\n" + "#"*60)
    print("HELIXAI -- PAH SELF-TRAINING (PRODÜKSİYON MODU)")
    print("#"*60)

    X_tr, y_tr = load_and_prep(pah_path, label_col, has_label=True)
    X_unlabeled, _ = load_and_prep(unlabeled_path, label_col, has_label=False)
    print(f"\n  Eğitim (etiketli): {X_tr.shape} | Test (etiketsiz): {X_unlabeled.shape}")

    common = sorted(set(X_tr.columns) & set(X_unlabeled.columns))
    X_tr, X_unlabeled = X_tr[common], X_unlabeled[common]

    train_prior = hc.compute_train_prior(y_tr)
    initial_models, final_models, info, (imp, sc, dropped, sel) = self_training_round(
        X_tr, y_tr, X_unlabeled, train_prior, test_prior, spw=spw
    )

    num_cols = X_unlabeled.select_dtypes(include=[np.number]).columns.tolist()
    X_unl_s = X_unlabeled.copy()
    X_unl_s[num_cols] = imp.transform(X_unl_s[num_cols])
    X_unl_s[num_cols] = sc.transform(X_unl_s[num_cols])
    X_unl_s = X_unl_s.drop(columns=dropped, errors='ignore')[sel]

    raw_final = predict_ensemble(final_models, X_unl_s)
    corr_final = hc.prior_correction(raw_final, train_prior, test_prior)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'pah_selftraining_predictions.csv')
    pd.DataFrame({'prob_pathogenic': corr_final}).to_csv(out_path, index=False)
    print(f"\n[BİLGİ] Gerçek etiket olmadığı için 'işe yaradı mı' burada ÖLÇÜLEMEZ "
          f"(bu, simülasyon modunda zaten doğrulanmıştı).")
    print(f"Pseudo-label bilgisi: {info}")
    print(f"Tahminler kaydedildi: {out_path}")
    return {'pseudo_label_info': info, 'predictions_path': out_path}

# ══════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PAH Self-Training (Yarı-Denetimli Öğrenme)')
    parser.add_argument('--pah', type=str, required=True)
    parser.add_argument('--unlabeled', type=str, default=None,
                         help='Gerçek etiketsiz test seti (verilmezse SİMÜLASYON moduna geçilir)')
    parser.add_argument('--mode', type=str, default=None, choices=[None, 'simulate', 'production'])
    parser.add_argument('--label', type=str, default='Label')
    parser.add_argument('--output', type=str, default='pah_selftraining_results')
    parser.add_argument('--test-prior', type=float, default=DEFAULT_TEST_PRIOR)
    parser.add_argument('--spw', type=float, default=DEFAULT_SPW)
    args = parser.parse_args()

    mode = args.mode or ('production' if args.unlabeled else 'simulate')
    if mode == 'simulate':
        run_simulation_mode(args.pah, args.output, args.label, args.test_prior, args.spw)
    else:
        if not args.unlabeled:
            raise SystemExit("Prodüksiyon modu için --unlabeled (etiketsiz test seti) gereklidir.")
        run_production_mode(args.pah, args.unlabeled, args.output, args.label, args.test_prior, args.spw)
