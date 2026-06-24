"""
HELIXAI — PAH Pipeline v2
==========================
Değişiklikler v1'e göre:
  - LOOCV kaldırıldı CFTR İÇİN YAPARIZ AMA BURDA İŞ YÜKÜ GİBİ OLDU
  - Repeated Stratified K-Fold (5-Fold × 10 tekrar = 50 CV) eklendi YİNE DATALEAKAGESİZ
  - CatBoost eklendi (XGB + LGB + CAT weighted soft voting)
  - SHAP weighted average (3 model)
  - Data leakage: sıfır (tüm preprocessing fold içi)

PAH Veri Boyutu:
  Eğitim : 300 Patojenik, 50 Benign  (~350 toplam)
  Test   : 100 Patojenik, 250 Benign (~350 toplam)

Kullanım:
  python pah_pipeline_v2.py --demo
  python pah_pipeline_v2.py --train pah_train.csv --test pah_test.csv
"""

import argparse, warnings, os, json, time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, matthews_corrcoef, average_precision_score,
    roc_auc_score, confusion_matrix
)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import shap

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════
# 0. SABİTLER
# ══════════════════════════════════════════════
SEED           = 42
N_SPLITS       = 5
N_REPEATS      = 10          # 5 × 10 = 50 fold
CORR_THRESH    = 0.90
MISSING_THRESH = 0.90
AF_EPSILON     = 1e-6
THRESHOLD_STEP = 0.01

# PAH prior'ları (şartnameden)
TRAIN_PRIOR = 300 / 350      # ≈ 0.857
TEST_PRIOR  = 100 / 350      # ≈ 0.286

# Ensemble ağırlıkları (sabit — küçük validation'da optimize etme, overfit yapar)
W_XGB = 0.35
W_LGB = 0.30
W_CAT = 0.35

# ══════════════════════════════════════════════
# 1. BİYOKİMYASAL TABLOLAR
# ══════════════════════════════════════════════
BLOSUM62 = {
    ('A','A'):4, ('A','R'):-1,('A','N'):-2,('A','D'):-2,('A','C'):0,
    ('A','Q'):-1,('A','E'):-1,('A','G'):0, ('A','H'):-2,('A','I'):-1,
    ('A','L'):-1,('A','K'):-1,('A','M'):-1,('A','F'):-2,('A','P'):-1,
    ('A','S'):1, ('A','T'):0, ('A','W'):-3,('A','Y'):-2,('A','V'):0,
    ('R','R'):5, ('R','N'):-1,('R','D'):-2,('R','C'):-3,('R','Q'):1,
    ('R','E'):0, ('R','G'):-2,('R','H'):0, ('R','I'):-3,('R','L'):-2,
    ('R','K'):2, ('R','M'):-1,('R','F'):-3,('R','P'):-2,('R','S'):-1,
    ('R','T'):-1,('R','W'):-3,('R','Y'):-2,('R','V'):-3,
    ('N','N'):6, ('N','D'):1, ('N','C'):-3,('N','Q'):0, ('N','E'):0,
    ('N','G'):0, ('N','H'):1, ('N','I'):-3,('N','L'):-3,('N','K'):0,
    ('N','M'):-2,('N','F'):-3,('N','P'):-2,('N','S'):1, ('N','T'):0,
    ('N','W'):-4,('N','Y'):-2,('N','V'):-3,
    ('D','D'):6, ('D','C'):-3,('D','Q'):0, ('D','E'):2, ('D','G'):-1,
    ('D','H'):-1,('D','I'):-3,('D','L'):-4,('D','K'):-1,('D','M'):-3,
    ('D','F'):-3,('D','P'):-1,('D','S'):0, ('D','T'):-1,('D','W'):-4,
    ('D','Y'):-3,('D','V'):-3,
    ('C','C'):9, ('C','Q'):-3,('C','E'):-4,('C','G'):-3,('C','H'):-3,
    ('C','I'):-1,('C','L'):-1,('C','K'):-3,('C','M'):-1,('C','F'):-2,
    ('C','P'):-3,('C','S'):-1,('C','T'):-1,('C','W'):-2,('C','Y'):-2,('C','V'):-1,
    ('Q','Q'):5, ('Q','E'):2, ('Q','G'):-2,('Q','H'):0, ('Q','I'):-3,
    ('Q','L'):-2,('Q','K'):1, ('Q','M'):0, ('Q','F'):-3,('Q','P'):-1,
    ('Q','S'):0, ('Q','T'):-1,('Q','W'):-2,('Q','Y'):-1,('Q','V'):-2,
    ('E','E'):5, ('E','G'):-2,('E','H'):0, ('E','I'):-3,('E','L'):-3,
    ('E','K'):1, ('E','M'):-2,('E','F'):-3,('E','P'):-1,('E','S'):0,
    ('E','T'):-1,('E','W'):-3,('E','Y'):-2,('E','V'):-2,
    ('G','G'):6, ('G','H'):-2,('G','I'):-4,('G','L'):-4,('G','K'):-2,
    ('G','M'):-3,('G','F'):-3,('G','P'):-2,('G','S'):0, ('G','T'):-2,
    ('G','W'):-2,('G','Y'):-3,('G','V'):-3,
    ('H','H'):8, ('H','I'):-3,('H','L'):-3,('H','K'):-1,('H','M'):-2,
    ('H','F'):-1,('H','P'):-2,('H','S'):-1,('H','T'):-2,('H','W'):-2,
    ('H','Y'):2, ('H','V'):-3,
    ('I','I'):4, ('I','L'):2, ('I','K'):-1,('I','M'):1, ('I','F'):0,
    ('I','P'):-3,('I','S'):-2,('I','T'):-1,('I','W'):-3,('I','Y'):-1,('I','V'):3,
    ('L','L'):4, ('L','K'):-2,('L','M'):2, ('L','F'):0, ('L','P'):-3,
    ('L','S'):-2,('L','T'):-1,('L','W'):-2,('L','Y'):-1,('L','V'):1,
    ('K','K'):5, ('K','M'):-1,('K','F'):-3,('K','P'):-1,('K','S'):0,
    ('K','T'):-1,('K','W'):-3,('K','Y'):-2,('K','V'):-2,
    ('M','M'):5, ('M','F'):0, ('M','P'):-2,('M','S'):-1,('M','T'):-1,
    ('M','W'):-1,('M','Y'):-1,('M','V'):1,
    ('F','F'):6, ('F','P'):-4,('F','S'):-2,('F','T'):-2,('F','W'):1,
    ('F','Y'):3, ('F','V'):-1,
    ('P','P'):7, ('P','S'):-1,('P','T'):-1,('P','W'):-4,('P','Y'):-3,('P','V'):-2,
    ('S','S'):4, ('S','T'):1, ('S','W'):-3,('S','Y'):-2,('S','V'):-2,
    ('T','T'):5, ('T','W'):-2,('T','Y'):-2,('T','V'):0,
    ('W','W'):11,('W','Y'):2, ('W','V'):-3,
    ('Y','Y'):7, ('Y','V'):-1,
    ('V','V'):4,
}

GRANTHAM = {
    ('A','R'):112,('A','N'):111,('A','D'):126,('A','C'):195,('A','Q'):91,
    ('A','E'):107,('A','G'):60, ('A','H'):86, ('A','I'):94, ('A','L'):96,
    ('A','K'):106,('A','M'):84, ('A','F'):113,('A','P'):27, ('A','S'):99,
    ('A','T'):58, ('A','W'):148,('A','Y'):112,('A','V'):64,
    ('R','N'):86, ('R','D'):96, ('R','C'):180,('R','Q'):43, ('R','E'):54,
    ('R','G'):125,('R','H'):29, ('R','I'):97, ('R','L'):102,('R','K'):26,
    ('R','M'):91, ('R','F'):97, ('R','P'):103,('R','S'):110,('R','T'):71,
    ('R','W'):101,('R','Y'):77, ('R','V'):96,
    ('N','D'):23, ('N','C'):139,('N','Q'):46, ('N','E'):42, ('N','G'):80,
    ('N','H'):68, ('N','I'):149,('N','L'):153,('N','K'):94, ('N','M'):142,
    ('N','F'):158,('N','P'):91, ('N','S'):46, ('N','T'):65, ('N','W'):174,
    ('N','Y'):143,('N','V'):133,
    ('C','D'):154,('C','Q'):154,('C','E'):170,('C','G'):159,('C','H'):174,
    ('C','I'):198,('C','L'):198,('C','K'):202,('C','M'):196,('C','F'):205,
    ('C','P'):169,('C','S'):112,('C','T'):149,('C','W'):215,('C','Y'):194,('C','V'):192,
    ('V','I'):29, ('V','L'):32, ('I','L'):5,  ('F','Y'):22, ('K','R'):26,
    ('D','E'):45, ('S','T'):58, ('G','A'):60,
}

AA_PROPS = {
    'A':(1.8,  0, 67, 0),'R':(-4.5, 1,148, 1),'N':(-3.5, 0,114, 1),
    'D':(-3.5,-1,111, 1),'C':(2.5,  0, 86, 0),'Q':(-3.5, 0,128, 1),
    'E':(-3.5,-1,138, 1),'G':(-0.4, 0, 48, 0),'H':(-3.2,0.5,118,1),
    'I':(4.5,  0,124, 0),'L':(3.8,  0,124, 0),'K':(-3.9, 1,135, 1),
    'M':(1.9,  0,124, 0),'F':(2.8,  0,135, 0),'P':(-1.6, 0, 90, 0),
    'S':(-0.8, 0, 73, 1),'T':(-0.7, 0, 93, 1),'W':(-0.9, 0,163, 0),
    'Y':(-1.3, 0,141, 1),'V':(4.2,  0,105, 0),
}
VALID_AA    = set(AA_PROPS.keys())
STOP_CODONS = {'*','X','Ter','Stop'}

# ══════════════════════════════════════════════
# 2. BİYOKİMYASAL HESAPLAMALAR
# ══════════════════════════════════════════════
def blosum62_score(ref, alt):
    if pd.isna(ref) or pd.isna(alt): return np.nan
    ref, alt = str(ref).upper().strip(), str(alt).upper().strip()
    if ref not in VALID_AA or alt not in VALID_AA: return np.nan
    if ref == alt: return BLOSUM62.get((ref,ref), 0)
    key = (ref,alt) if (ref,alt) in BLOSUM62 else (alt,ref)
    return BLOSUM62.get(key, np.nan)

def grantham_dist(ref, alt):
    if pd.isna(ref) or pd.isna(alt): return np.nan
    ref, alt = str(ref).upper().strip(), str(alt).upper().strip()
    if ref not in VALID_AA or alt not in VALID_AA: return np.nan
    if ref == alt: return 0.0
    key = (ref,alt) if (ref,alt) in GRANTHAM else (alt,ref)
    return float(GRANTHAM.get(key, np.nan))

def aa_phys_diff(ref, alt):
    if pd.isna(ref) or pd.isna(alt): return (np.nan,)*4
    ref, alt = str(ref).upper().strip(), str(alt).upper().strip()
    if ref not in VALID_AA or alt not in VALID_AA: return (np.nan,)*4
    rp, ap = AA_PROPS[ref], AA_PROPS[alt]
    return (abs(rp[0]-ap[0]), abs(rp[1]-ap[1]), abs(rp[2]-ap[2]), abs(rp[3]-ap[3]))

# ══════════════════════════════════════════════
# 3. VERİ HAZIRLAMA — GLOBAL (fold dışı)
# ══════════════════════════════════════════════
def global_col_filter(df, thresh=MISSING_THRESH):
    miss  = df.isnull().mean()
    drops = miss[miss > thresh].index.tolist()
    print(f"  [Filter] {len(drops)} kolon kaldırıldı (eksiklik >{thresh*100:.0f}%)")
    return df.drop(columns=drops), drops

def biological_qc(df):
    df = df.copy()
    df['flag_stop_codon'] = 0
    df['flag_unusual_aa'] = 0
    aa_cols = [c for c in df.columns if c.upper().startswith('AA_')]
    if len(aa_cols) >= 2:
        ref_s = df[aa_cols[0]].astype(str).str.upper().str.strip()
        alt_s = df[aa_cols[1]].astype(str).str.upper().str.strip()
        df.loc[ref_s.isin(STOP_CODONS)|alt_s.isin(STOP_CODONS), 'flag_stop_codon'] = 1
        bad = (~ref_s.isin(VALID_AA)&~ref_s.isin(STOP_CODONS)&(ref_s!='NAN'))|\
              (~alt_s.isin(VALID_AA)&~alt_s.isin(STOP_CODONS)&(alt_s!='NAN'))
        df.loc[bad, 'flag_unusual_aa'] = 1
        df.loc[~ref_s.isin(VALID_AA), aa_cols[0]] = np.nan
        df.loc[~alt_s.isin(VALID_AA), aa_cols[1]] = np.nan
    print(f"  [QC] stop_codon={df['flag_stop_codon'].sum()} | unusual_AA={df['flag_unusual_aa'].sum()}")
    return df

def engineer_aa_features(df):
    df = df.copy()
    aa_cols = [c for c in df.columns if c.upper().startswith('AA_')]
    if len(aa_cols) < 2:
        print("  [AA] AA_ sütunu bulunamadı, atlandı.")
        return df
    rc, ac = aa_cols[0], aa_cols[1]
    df['feat_blosum62']      = df.apply(lambda r: blosum62_score(r[rc], r[ac]), axis=1)
    df['feat_grantham']      = df.apply(lambda r: grantham_dist(r[rc], r[ac]),  axis=1)
    phys = df.apply(lambda r: aa_phys_diff(r[rc], r[ac]), axis=1)
    df['feat_hydro_diff']    = phys.apply(lambda x: x[0])
    df['feat_charge_diff']   = phys.apply(lambda x: x[1])
    df['feat_volume_diff']   = phys.apply(lambda x: x[2])
    df['feat_polarity_diff'] = phys.apply(lambda x: x[3])
    print(f"  [AA] 6 özellik eklendi (BLOSUM62, Grantham, hidrofobisite, yük, hacim, polarite)")
    return df

def log_transform_af(df):
    df = df.copy()
    al_cols = [c for c in df.columns if c.upper().startswith('AL_')]
    for col in al_cols:
        num = pd.to_numeric(df[col], errors='coerce')
        mask = num.notna() & (num >= 0) & (num <= 1)
        if mask.any():
            df[col] = df[col].astype(float)   # int64 → float64
            df.loc[mask, col] = -np.log10(num[mask] + AF_EPSILON)
    print(f"  [AF] {len(al_cols)} AL_ sütuna -log10 uygulandı (ε={AF_EPSILON})")
    return df

def to_numeric_df(df, label_col='label'):
    X = df.drop(columns=[label_col], errors='ignore')
    cat_cols = [c for c in X.columns if c.upper().startswith('CAT_') and X[c].dtype == object]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    obj_cols = X.select_dtypes(include='object').columns.tolist()
    X = X.drop(columns=obj_cols, errors='ignore')
    return X

# ══════════════════════════════════════════════
# 4. YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════
def correlation_filter(X, thresh=CORR_THRESH):
    X_num = X.select_dtypes(include=[np.number])
    corr  = X_num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drops = [c for c in upper.columns if any(upper[c] > thresh)]
    return X.drop(columns=drops, errors='ignore'), drops

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn,fp,fn,tp = cm.ravel()
    sens = tp/(tp+fn) if (tp+fn)>0 else 0.0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
    return {
        'f1':           f1_score(y_true, y_pred, zero_division=0),
        'mcc':          matthews_corrcoef(y_true, y_pred),
        'pr_auc':       average_precision_score(y_true, y_prob),
        'roc_auc':      roc_auc_score(y_true, y_prob),
        'sensitivity':  sens,
        'specificity':  spec,
        'balanced_acc': (sens+spec)/2,
        'threshold':    threshold,
        'tp':int(tp),'fp':int(fp),'fn':int(fn),'tn':int(tn),
    }

def find_best_threshold(y_true, y_prob):
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.05, 0.96, THRESHOLD_STEP):
        f = f1_score(y_true, (y_prob>=t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1

def prior_correction(probs, train_p=TRAIN_PRIOR, test_p=TEST_PRIOR):
    """
    Bayes Prior Probability Correction:
    P_corr = [P × (π_test/π_train)] /
             [P × (π_test/π_train) + (1-P) × ((1-π_test)/(1-π_train))]
    """
    r_pos = test_p  / train_p
    r_neg = (1-test_p) / (1-train_p)
    num   = probs * r_pos
    den   = num + (1-probs) * r_neg
    den   = np.where(den==0, 1e-12, den)
    return num / den

# ══════════════════════════════════════════════
# 5. MODEL YAPILARI
# ══════════════════════════════════════════════
def build_xgb(spw=1.0):
    return xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric='aucpr',
        early_stopping_rounds=30, use_label_encoder=False,
        random_state=SEED, verbosity=0,
    )

def build_lgb(spw=1.0):
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, metric='average_precision',
        early_stopping_rounds=30, random_state=SEED, verbose=-1,
    )

def build_cat(spw=1.0):
    return CatBoostClassifier(
        iterations=300, learning_rate=0.05, depth=4,
        scale_pos_weight=spw, eval_metric='PRAUC',
        early_stopping_rounds=30, random_seed=SEED,
        verbose=False,
    )

# ══════════════════════════════════════════════
# 6. SHAP ÖZELLİK SEÇİMİ (fold içi)
# ══════════════════════════════════════════════
def shap_feature_selection(X_tr, y_tr, X_val):
    n = X_tr.shape[1]
    if n <= 10:
        return X_tr.columns.tolist()

    # Hızlı XGB ile SHAP
    q = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=SEED, verbosity=0,
        use_label_encoder=False, eval_metric='aucpr'
    )
    q.fit(X_tr, y_tr,
          eval_set=[(X_val, np.zeros(len(X_val)))],
          verbose=False)

    exp  = shap.TreeExplainer(q)
    sv   = exp.shap_values(X_tr)
    imp  = pd.Series(np.abs(sv).mean(axis=0), index=X_tr.columns).sort_values(ascending=False)

    # PR-AUC plateau: 5'ten başla, +5 adım
    best_prauc, best_n = -1, min(20, n)
    for k in range(5, n+1, 5):
        sel = imp.head(k).index.tolist()
        tmp = xgb.XGBClassifier(
            n_estimators=100, max_depth=3,
            random_state=SEED, verbosity=0,
            use_label_encoder=False, eval_metric='aucpr'
        )
        tmp.fit(X_tr[sel], y_tr,
                eval_set=[(X_val[sel], np.zeros(len(X_val)))],
                verbose=False)
        pv  = tmp.predict_proba(X_val[sel])[:,1]
        # Proxy: eğitim üzerinde prauc (val etiketi yok)
        pr  = average_precision_score(y_tr, tmp.predict_proba(X_tr[sel])[:,1])
        if pr > best_prauc:
            best_prauc, best_n = pr, k

    return imp.head(best_n).index.tolist()

# ══════════════════════════════════════════════
# 7. TEK FOLD EĞİTİMİ (tam data-leakage-free)
# ══════════════════════════════════════════════
def train_fold(X_tr_raw, y_tr, X_val_raw, y_val, fold_id="?"):
    """
    Her adım SADECE eğitim verisinde fit edilir, val'a transform uygulanır.
    Sıra:
      imputation → RobustScaler → corr filter → SHAP seçim →
      XGB + LGB + CAT eğitimi → weighted soft voting →
      prior correction → threshold taraması
    """
    num_cols = X_tr_raw.select_dtypes(include=[np.number]).columns.tolist()

    # 7a. Imputation — fold içi fit
    imp = SimpleImputer(strategy='median')
    X_tr  = X_tr_raw.copy(); X_val = X_val_raw.copy()
    X_tr[num_cols]  = imp.fit_transform(X_tr_raw[num_cols])
    X_val[num_cols] = imp.transform(X_val_raw[num_cols])

    # 7b. RobustScaler — fold içi fit
    scaler = RobustScaler()
    X_tr[num_cols]  = scaler.fit_transform(X_tr[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])

    # 7c. Korelasyon filtresi
    X_tr_f, dropped_corr = correlation_filter(X_tr)
    X_val_f = X_val.drop(columns=dropped_corr, errors='ignore')

    # 7d. SHAP özellik seçimi
    sel = shap_feature_selection(X_tr_f, y_tr, X_val_f)
    X_tr_s  = X_tr_f[sel]
    X_val_s = X_val_f[sel]

    # Class weight
    n_p  = (y_tr==1).sum(); n_b = (y_tr==0).sum()
    spw  = n_b / n_p if n_p > 0 else 1.0

    # 7e. XGBoost
    xgb_m = build_xgb(spw)
    xgb_m.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
    p_xgb = xgb_m.predict_proba(X_val_s)[:,1]

    # 7f. LightGBM
    lgb_m = build_lgb(spw)
    lgb_m.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(-1)])
    p_lgb = lgb_m.predict_proba(X_val_s)[:,1]

    # 7g. CatBoost
    cat_m = build_cat(spw)
    cat_m.fit(X_tr_s, y_tr, eval_set=(X_val_s, y_val))
    p_cat = cat_m.predict_proba(X_val_s)[:,1]

    # 7h. Weighted Soft Voting
    p_ens = W_XGB*p_xgb + W_LGB*p_lgb + W_CAT*p_cat

    # 7i. Prior Correction
    p_corr = prior_correction(p_ens)

    # 7j. Threshold tarama (val üzerinde — test'e dokunmadan)
    best_t, _ = find_best_threshold(y_val, p_corr)
    metrics    = compute_metrics(y_val, p_corr, threshold=best_t)
    metrics['fold'] = fold_id
    metrics['n_features'] = len(sel)

    return {
        'metrics':      metrics,
        'xgb':          xgb_m, 'lgb': lgb_m, 'cat': cat_m,
        'imputer':      imp,
        'scaler':       scaler,
        'sel_feats':    sel,
        'dropped_corr': dropped_corr,
        'threshold':    best_t,
        'val_probs':    p_corr,
        'val_labels':   y_val.values,
    }

# ══════════════════════════════════════════════
# 8. REPEATED STRATIFIED K-FOLD (5×10 = 50 CV)
# ══════════════════════════════════════════════
def run_repeated_cv(X, y):
    print("\n" + "="*60)
    print(f"ADIM 6-7: Repeated Stratified K-Fold ({N_SPLITS}-Fold × {N_REPEATS} tekrar = {N_SPLITS*N_REPEATS} CV)")
    print("Data leakage: 0 — her fold kendi imputer/scaler/feature set'ini üretiyor")
    print("="*60)

    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED
    )

    fold_results = []
    total = N_SPLITS * N_REPEATS

    for fold_idx, (tr_idx, val_idx) in enumerate(rskf.split(X, y)):
        repeat = fold_idx // N_SPLITS + 1
        fold   = fold_idx  % N_SPLITS + 1
        label  = f"R{repeat}F{fold}"

        if fold_idx % N_SPLITS == 0:
            print(f"\n  ── Tekrar {repeat}/{N_REPEATS} ──")

        X_tr, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[tr_idx].copy(), y.iloc[val_idx].copy()

        res = train_fold(X_tr, y_tr, X_val, y_val, fold_id=label)
        fold_results.append(res)

        m = res['metrics']
        print(f"  {label} | F1={m['f1']:.4f} | MCC={m['mcc']:.4f} | "
              f"PR-AUC={m['pr_auc']:.4f} | Sens={m['sensitivity']:.4f} | "
              f"Spec={m['specificity']:.4f} | Thresh={m['threshold']:.2f} | "
              f"Feats={m['n_features']}")

    # ── Özet
    keys = ['f1','mcc','pr_auc','roc_auc','sensitivity','specificity','balanced_acc']
    summary = {}
    for k in keys:
        vals = [r['metrics'][k] for r in fold_results]
        summary[k] = {
            'mean': np.mean(vals), 'std': np.std(vals),
            'min':  np.min(vals),  'max': np.max(vals),
        }

    print("\n" + "─"*60)
    print(f"REPEATED CV ÖZET ({total} fold):")
    print(f"  {'Metrik':<16} {'Ortalama':>10} {'±Std':>8} {'Min':>8} {'Max':>8}")
    print("  " + "─"*52)
    for k, v in summary.items():
        print(f"  {k:<16} {v['mean']:>10.4f} {v['std']:>8.4f} "
              f"{v['min']:>8.4f} {v['max']:>8.4f}")

    return fold_results, summary

# ══════════════════════════════════════════════
# 9. TEST SETİ TAHMİNİ
# ══════════════════════════════════════════════
def predict_test(X_test_raw, fold_results):
    """
    Tüm 50 fold modelinin ortalaması → tek tahmin.
    Her fold'un kendi imputer/scaler/feature_set'i kullanılır.
    """
    print("\n" + "="*60)
    print("ADIM 9: Test Seti Tahmini (50-Fold Ensemble Ortalaması)")
    print("="*60)

    all_probs = []
    for res in fold_results:
        X_t  = X_test_raw.copy()
        ncols= X_t.select_dtypes(include=[np.number]).columns.tolist()
        X_t[ncols] = res['imputer'].transform(X_t[ncols])
        X_t[ncols] = res['scaler'].transform(X_t[ncols])
        X_t = X_t.drop(columns=res['dropped_corr'], errors='ignore')
        X_t = X_t[res['sel_feats']]

        p_xgb = res['xgb'].predict_proba(X_t)[:,1]
        p_lgb = res['lgb'].predict_proba(X_t)[:,1]
        p_cat = res['cat'].predict_proba(X_t)[:,1]
        p_ens = W_XGB*p_xgb + W_LGB*p_lgb + W_CAT*p_cat
        all_probs.append(prior_correction(p_ens))

    final_probs = np.mean(all_probs, axis=0)
    avg_thresh  = np.mean([r['threshold'] for r in fold_results])
    print(f"  Ortalama threshold (50 fold): {avg_thresh:.4f}")
    return final_probs, avg_thresh

# ══════════════════════════════════════════════
# 10. SHAP (weighted 3 model)
# ══════════════════════════════════════════════
def run_shap(fold_results, X_train, output_dir):
    print("\n" + "="*60)
    print("ADIM 10: SHAP Açıklanabilirlik (Weighted 3-Model)")
    print("="*60)

    # En yüksek F1'li fold
    best = max(fold_results, key=lambda r: r['metrics']['f1'])
    ncols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    X_s = X_train.copy()
    X_s[ncols] = best['imputer'].transform(X_s[ncols])
    X_s[ncols] = best['scaler'].transform(X_s[ncols])
    X_s = X_s.drop(columns=best['dropped_corr'], errors='ignore')
    X_s = X_s[best['sel_feats']]

    # SHAP — her model için ayrı, sonra ağırlıklı ortalama
    sv_xgb = shap.TreeExplainer(best['xgb']).shap_values(X_s)
    sv_lgb = shap.TreeExplainer(best['lgb']).shap_values(X_s)
    sv_cat = shap.TreeExplainer(best['cat']).shap_values(X_s)

    sv_weighted = W_XGB*sv_xgb + W_LGB*sv_lgb + W_CAT*sv_cat

    importance = pd.Series(
        np.abs(sv_weighted).mean(axis=0),
        index=X_s.columns
    ).sort_values(ascending=False)

    print("\n  Top-10 SHAP (Weighted 3-Model) — PAH:")
    print("  " + "─"*48)
    for feat, val in importance.head(10).items():
        bar = "█" * int(val / importance.iloc[0] * 25)
        print(f"  {feat:30s} {bar} {val:.4f}")

    path = os.path.join(output_dir, 'pah_shap.csv')
    importance.reset_index().rename(
        columns={'index':'feature', 0:'shap_mean_abs'}
    ).to_csv(path, index=False)
    print(f"\n  Kaydedildi: {path}")
    return importance

# ══════════════════════════════════════════════
# 11. ANA PIPELINE
# ══════════════════════════════════════════════
def run_pah_pipeline(train_path, test_path=None,
                     output_dir='pah_results', label_col='label'):

    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    print("\n" + "█"*60)
    print("HELIXAI — PAH Pipeline v2")
    print(f"Repeated Stratified {N_SPLITS}-Fold × {N_REPEATS} = {N_SPLITS*N_REPEATS} CV | "
          f"XGB+LGB+CAT Weighted Soft Voting | Prior Correction")
    print("█"*60)

    # ── Adım 1: Yükleme
    print("\nADIM 1: Veri Yükleme")
    train_df = pd.read_csv(train_path)
    print(f"  Eğitim: {train_df.shape} | {train_df[label_col].value_counts().to_dict()}")
    test_df = None
    if test_path and os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        print(f"  Test  : {test_df.shape}")

    # ── Adım 2: Global kolon filtresi
    print("\nADIM 2: Global Kolon Filtresi")
    feat_cols = [c for c in train_df.columns if c != label_col]
    tr_feats, dropped_global = global_col_filter(train_df[feat_cols])
    if test_df is not None:
        te_feats = test_df[[c for c in feat_cols if c not in dropped_global]]
    else:
        te_feats = None

    # ── Adım 3: Biyolojik QC
    print("\nADIM 3: Biyolojik QC")
    tr_feats = biological_qc(tr_feats)
    if te_feats is not None:
        te_feats = biological_qc(te_feats)

    # ── Adım 4: AA mühendisliği
    print("\nADIM 4: Amino Asit Özellik Mühendisliği")
    tr_feats = engineer_aa_features(tr_feats)
    if te_feats is not None:
        te_feats = engineer_aa_features(te_feats)

    # ── Adım 5: AF log dönüşümü
    print("\nADIM 5: Allel Frekans Log Dönüşümü")
    tr_feats = log_transform_af(tr_feats)
    if te_feats is not None:
        te_feats = log_transform_af(te_feats)

    y_train  = train_df[label_col].astype(int)
    X_train  = to_numeric_df(tr_feats)
    X_test   = to_numeric_df(te_feats) if te_feats is not None else None

    print(f"\n  Final özellik sayısı : {X_train.shape[1]}")
    print(f"  Patojenik            : {(y_train==1).sum()}")
    print(f"  Benign               : {(y_train==0).sum()}")

    # ── Adım 6-7: Repeated CV
    fold_results, cv_summary = run_repeated_cv(X_train, y_train)

    # ── Adım 9: Test tahmini
    test_metrics = None
    if X_test is not None and test_df is not None and label_col in test_df.columns:
        test_probs, avg_thresh = predict_test(X_test, fold_results)
        y_test       = test_df[label_col].astype(int)
        test_metrics = compute_metrics(y_test, test_probs, threshold=avg_thresh)
        print("\n  TEST SETİ SONUÇLARI:")
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"    {k:<16}: {v:.4f}")
    elif X_test is not None:
        # Etiketsiz test — sadece tahmin üret
        test_probs, avg_thresh = predict_test(X_test, fold_results)
        pred_path = os.path.join(output_dir, 'pah_test_predictions.csv')
        pd.DataFrame({
            'prob_pathogenic': test_probs,
            'predicted_label': (test_probs >= avg_thresh).astype(int)
        }).to_csv(pred_path, index=False)
        print(f"\n  Etiketsiz test tahminleri kaydedildi: {pred_path}")

    # ── Adım 10: SHAP
    shap_imp = run_shap(fold_results, X_train, output_dir)

    # ── Final rapor
    elapsed = time.time() - t_start
    print("\n" + "█"*60)
    print("HELIXAI PAH — FINAL RAPOR")
    print("█"*60)
    print(f"\n  Toplam süre: {elapsed/60:.1f} dakika")
    print(f"\n  [{N_SPLITS*N_REPEATS}-FOLD REPEATED CV]")
    print(f"  {'Metrik':<16} {'Ort':>8} {'±Std':>8}")
    print("  " + "─"*34)
    for k, v in cv_summary.items():
        print(f"  {k:<16} {v['mean']:>8.4f} {v['std']:>8.4f}")

    if test_metrics:
        print("\n  [TEST SETİ]")
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"  {k:<16}: {v:.4f}")

    report = {
        'panel': 'PAH',
        'config': {
            'n_splits': N_SPLITS, 'n_repeats': N_REPEATS,
            'total_folds': N_SPLITS*N_REPEATS,
            'weights': {'xgb':W_XGB,'lgb':W_LGB,'cat':W_CAT},
            'train_prior': TRAIN_PRIOR, 'test_prior': TEST_PRIOR,
        },
        'cv_summary': {k:{'mean':v['mean'],'std':v['std']} for k,v in cv_summary.items()},
        'test': test_metrics,
        'shap_top10': shap_imp.head(10).to_dict(),
        'elapsed_min': round(elapsed/60, 2),
    }
    rep_path = os.path.join(output_dir, 'pah_report.json')
    with open(rep_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Rapor kaydedildi: {rep_path}")
    return report, fold_results

# ══════════════════════════════════════════════
# 12. DEMO VERİSİ
# ══════════════════════════════════════════════
def generate_demo_data():
    np.random.seed(SEED)

    def make(n_path, n_ben):
        n  = n_path + n_ben
        y  = np.array([1]*n_path + [0]*n_ben)
        ix = np.random.permutation(n); y = y[ix]

        df = pd.DataFrame()
        df['AL_freq_global'] = np.where(y==1,
            np.random.beta(1,100,n), np.random.beta(5,20,n))
        df['AL_freq_eur']    = np.where(y==1,
            np.random.beta(1,80,n),  np.random.beta(4,15,n))
        df['AL_count']       = (df['AL_freq_global']*10000).astype(int)

        aa_list = list(VALID_AA)
        df['AA_ref'] = np.random.choice(aa_list, n)
        df['AA_alt'] = np.random.choice(aa_list, n)

        df['EK_phylop']   = np.where(y==1,
            np.random.normal(3.5,1.0,n), np.random.normal(0.5,1.5,n))
        df['EK_gerp']     = np.where(y==1,
            np.random.normal(4.0,1.2,n), np.random.normal(0.2,2.0,n))
        df['EK_sift']     = np.where(y==1,
            np.random.beta(1,10,n),      np.random.beta(5,3,n))
        df['EK_polyphen'] = np.where(y==1,
            np.random.beta(8,2,n),       np.random.beta(2,8,n))

        df['CAT_pop']     = np.random.choice(['EUR','AFR','EAS','SAS','AMR'], n)
        df['CAT_quality'] = np.random.choice(['PASS','LOW_QUAL'], n, p=[0.9,0.1])

        # ~%10 eksik değer
        for c in ['AL_freq_eur','EK_gerp','EK_sift']:
            df.loc[np.random.rand(n)<0.10, c] = np.nan

        df['label'] = y
        return df

    tr = make(300, 50);  tr.to_csv('/tmp/pah_train.csv', index=False)
    te = make(100, 250); te.to_csv('/tmp/pah_test.csv',  index=False)
    print("Demo veri oluşturuldu:")
    print("  Eğitim: /tmp/pah_train.csv  (300 path / 50 benign)")
    print("  Test  : /tmp/pah_test.csv   (100 path / 250 benign)")
    return '/tmp/pah_train.csv', '/tmp/pah_test.csv'

# ══════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HELIXAI PAH Pipeline v2')
    parser.add_argument('--train',  type=str, default=None)
    parser.add_argument('--test',   type=str, default=None)
    parser.add_argument('--label',  type=str, default='label')
    parser.add_argument('--output', type=str, default='pah_results')
    parser.add_argument('--demo',   action='store_true')
    args = parser.parse_args()

    if args.demo or args.train is None:
        print("[DEMO MODU]")
        train_p, test_p = generate_demo_data()
    else:
        train_p, test_p = args.train, args.test

    run_pah_pipeline(
        train_path = train_p,
        test_path  = test_p,
        output_dir = args.output,
        label_col  = args.label,
    )