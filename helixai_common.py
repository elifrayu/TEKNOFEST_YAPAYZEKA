"""
HELIXAI — Ortak Yardımcı Modül (helixai_common.py)
====================================================
Master / Kanser / CFTR / PAH panellerinin TÜMÜNÜN ortak kullandığı:
  - Biyokimyasal tablolar ve hesaplamalar (BLOSUM62, Grantham, AA özellikleri)
  - Veri hazırlama (kolon filtresi, QC, AA feature engineering, AF log dönüşümü)
  - Metrikler, eşik arama, prior hesaplama
  - Kalibrasyon (Platt, sklearn sürüm-güvenli), Bayes Prior Correction
  - [v4-kökenli] EM tabanlı (Saerens 2002) etiketsiz test-prior sağlık kontrolü
  - [YENİ] Genel (model-sayısı-agnostik) CV orkestrasyonu: run_repeated_cv,
    run_loocv, ablate_scale_pos_weight, predict_test, run_shap, stres testi.

ÖNEMLİ TASARIM KARARI: Bu modüldeki orkestrasyon fonksiyonları (run_repeated_cv,
predict_test, run_shap, vb.) belirli bir model SAYISINA (3'lü ensemble) bağımlı
DEĞİLDİR. Her panel kendi train_fold() fonksiyonunu yazar ve şu STANDART
sözleşmeye uyan bir 'fold_result' sözlüğü döndürür:

    {
        'metrics':            dict (compute_metrics çıktısı + 'fold','n_features'),
        'models_cal':         {model_adi: kalibre_edilmis_model, ...},
        'weights':            {model_adi: agirlik, ...}  (sum=1.0),
        'imputer':            fit edilmiş SimpleImputer,
        'scaler':             fit edilmiş RobustScaler,
        'sel_feats':          seçilen kolon listesi,
        'dropped_corr':       korelasyon filtresiyle düşürülen kolonlar,
        'threshold':          bu fold için (rapor amaçlı) en iyi eşik,
        'val_probs_corrected':np.array,
        'val_probs_raw':      np.array,
        'val_labels':         np.array,
    }

Bu sayede CFTR'nin "3'lü ensemble" VEYA "tek basit model" adayları arasında
ablation yapması, PAH'ın "cost-sensitive + oversample" ızgara taraması yapması
gibi panel-spesifik kararlar train_fold() içinde kalırken, CV/threshold/SHAP/
stres-testi/EM-kontrolü mantığı TEK YERDEN (burada) yönetilir ve dört panelde
de AYNI ŞEKİLDE doğrulanmış olur.
"""

import warnings, os
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, LeaveOneOut
from sklearn.metrics import (
    f1_score, matthews_corrcoef, average_precision_score,
    roc_auc_score, confusion_matrix, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV

import shap

warnings.filterwarnings("ignore")

SEED           = 42
THRESHOLD_STEP = 0.005
CORR_THRESH    = 0.90
MISSING_THRESH = 0.90
AF_EPSILON     = 1e-6

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
    print("  [AA] 6 özellik eklendi (BLOSUM62, Grantham, hidrofobisite, yük, hacim, polarite)")
    return df

def log_transform_af(df):
    """
    AL_ grubu sadece frekans değil 'sayım' (count) kolonları da içerebilir
    (örn. büyük ölçekli bir AL_ kolonu muhtemelen bir alel sayımıdır).
    Bu fonksiyon [0,1] aralığı dışındaki satırları dönüşüm dışı bırakır —
    bu BİLİNÇLİ bir tasarım kararıdır (sayım kolonlarını bozmamak için).
    """
    df = df.copy()
    al_cols = [c for c in df.columns if c.upper().startswith('AL_')]
    for col in al_cols:
        num = pd.to_numeric(df[col], errors='coerce')
        mask = num.notna() & (num >= 0) & (num <= 1)
        if mask.any():
            df[col] = df[col].astype(float)
            df.loc[mask, col] = -np.log10(num[mask] + AF_EPSILON)
    print(f"  [AF] {len(al_cols)} AL_ sütuna -log10 uygulandı (ε={AF_EPSILON}); "
          f"[0,1] dışı kalan satırlar (muhtemel sayım kolonları) ham bırakıldı")
    return df

def to_numeric_df(df, label_col='Label'):
    X = df.drop(columns=[label_col], errors='ignore')
    cat_cols = [c for c in X.columns if c.upper().startswith('CAT_') and X[c].dtype == object]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    obj_cols = X.select_dtypes(include='object').columns.tolist()
    X = X.drop(columns=obj_cols, errors='ignore')
    return X

def correlation_filter(X, thresh=CORR_THRESH, protected_prefixes=()):
    """
    [common] protected_prefixes: bu önekle başlayan kolonlar korelasyon
    filtresinden ASLA düşürülmez (örn. Kanser panelinde EK_ grubunu —
    'klonalite/log molekülleri geçerli' notuna uyarak — korumak için).
    """
    X_num = X.select_dtypes(include=[np.number])
    corr  = X_num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drops = [c for c in upper.columns if any(upper[c] > thresh)]
    if protected_prefixes:
        drops = [c for c in drops if not any(c.upper().startswith(p.upper()) for p in protected_prefixes)]
    return X.drop(columns=drops, errors='ignore'), drops

# ══════════════════════════════════════════════
# 4. METRİKLER / EŞİK / PRIOR
# ══════════════════════════════════════════════
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
        'brier':        brier_score_loss(y_true, y_prob),  # [v4-genel] kalibrasyon kalitesi
        'threshold':    threshold,
        'tp':int(tp),'fp':int(fp),'fn':int(fn),'tn':int(tn),
    }

def find_best_threshold(y_true, y_prob, step=THRESHOLD_STEP):
    best_t, best_f1 = 0.5, -1
    for t in np.arange(0.01, 0.99, step):
        f = f1_score(y_true, (y_prob>=t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1

def compute_train_prior(y_train):
    """Eğitim önselini SABİT KODLAMAK yerine veriden hesapla."""
    return float((y_train == 1).mean())

def prior_correction(probs, train_p, test_p):
    """
    Bayes Prior Probability Correction (Saerens et al., 2002):
    P_corr = [P x (pi_test/pi_train)] /
             [P x (pi_test/pi_train) + (1-P) x ((1-pi_test)/(1-pi_train))]
    KRİTİK ÖN KOŞUL: `probs` KALİBRE EDİLMİŞ olasılıklar olmalıdır.
    """
    r_pos = test_p / train_p
    r_neg = (1 - test_p) / (1 - train_p)
    num   = probs * r_pos
    den   = num + (1 - probs) * r_neg
    den   = np.where(den == 0, 1e-12, den)
    return num / den

def estimate_test_prior_em(calibrated_probs_on_test, train_prior, max_iter=100, tol=1e-6):
    """
    [OPSİYONEL SAĞLIK KONTROLÜ] Saerens et al. (2002) EM tabanlı
    "quantification": test setinin GERÇEK patojenik oranını, ETİKETSİZ
    test verisi üzerindeki kalibre olasılıklardan iteratif tahmin eder.
    Gerçek test etiketleri GİZLİ olduğundan (şartname Bölüm 3.2), tek
    güvenilir etiketsiz sapma-kontrolü budur. BİRİNCİL prior kaynağı
    DEĞİLDİR — şartname değeri (--test-prior) öyle kullanılmalıdır.
    """
    pi_test = train_prior
    p = np.clip(calibrated_probs_on_test, 1e-6, 1 - 1e-6)
    for _ in range(max_iter):
        r_pos = pi_test / train_prior
        r_neg = (1 - pi_test) / (1 - train_prior)
        num  = p * r_pos
        den  = num + (1 - p) * r_neg
        post = num / den
        pi_new = float(post.mean())
        if abs(pi_new - pi_test) < tol:
            pi_test = pi_new
            break
        pi_test = pi_new
    return pi_test

# ══════════════════════════════════════════════
# 5. KALİBRASYON (sklearn sürüm-güvenli)
# ══════════════════════════════════════════════
def calibrate_fitted_model(fitted_model, X_val, y_val, method='sigmoid'):
    """
    Zaten eğitilmiş bir modeli, val fold üzerinde kalibre eder.
    sklearn >=1.6 'cv=prefit'i kaldırdı (FrozenEstimator ile değiştirildi).
    Jürinin sklearn sürümü bilinmediğinden (reproducibility şartı), her iki
    sürümle de çalışacak try/except ile geriye dönük uyumluluk sağlanmıştır.
    """
    try:
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(FrozenEstimator(fitted_model), method=method).fit(X_val, y_val)
    except ImportError:
        return CalibratedClassifierCV(fitted_model, method=method, cv='prefit').fit(X_val, y_val)

def get_base_estimator(calibrated_model):
    """
    calibrate_fitted_model() çıktısından SHAP için orijinal ağaç-tabanlı
    (veya doğrusal) modeli geri çıkarır. KASITLI olarak fallback YOK —
    hata olduğu yerde açıkça patlamalı; güvenli düşüş çağıran tarafta
    (run_shap) ele alınır.
    """
    est = calibrated_model.calibrated_classifiers_[0].estimator
    return est.estimator if hasattr(est, 'estimator') else est

# ══════════════════════════════════════════════
# 6. SHAP ÖZELLİK SEÇİMİ (fold içi, validation PR-AUC ile)
# ══════════════════════════════════════════════
def shap_feature_selection(X_tr, y_tr, X_val, y_val, k_max=None, k_step=5, seed=SEED):
    """
    k_max: seçilebilecek MAKSİMUM özellik sayısı (CFTR'de 'feature sayısını
    minimize et' notu doğrultusunda küçük bir tavan verilir; diğer panellerde
    None = n_features tavanı, yani sınırsız tarama).
    """
    n = X_tr.shape[1]
    if n <= 10:
        return X_tr.columns.tolist()

    cap = min(k_max, n) if k_max else n

    q = __import__('xgboost').XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=seed, verbosity=0, eval_metric='aucpr'
    )
    q.fit(X_tr, y_tr, verbose=False)

    # [DÜZELTME] shap<->xgboost sürüm uyumsuzluğuna karşı güvenli düşüş.
    # Bazı shap/xgboost sürüm kombinasyonlarında TreeExplainer, xgboost'un
    # internal base_score formatını okuyamayıp hata fırlatabilir (örn.
    # "could not convert string to float: '[8.57e-1]'"). Takım üyeleri
    # farklı makinelerde farklı sürümler kullanabileceğinden, pipeline bu
    # yüzden ÇÖKMEMELİ — SHAP başarısız olursa otomatik olarak XGBoost'un
    # kendi (gain-tabanlı, versiyon-bağımsız) feature_importances_'ına düşülür.
    try:
        exp = shap.TreeExplainer(q)
        sv  = exp.shap_values(X_tr)
        imp = pd.Series(np.abs(sv).mean(axis=0), index=X_tr.columns).sort_values(ascending=False)
    except Exception as e:
        print(f"  [UYARI] SHAP özellik sıralaması başarısız oldu ({type(e).__name__}: {e}). "
              f"XGBoost feature_importances_'a (gain) düşülüyor.")
        imp = pd.Series(q.feature_importances_, index=X_tr.columns).sort_values(ascending=False)

    best_prauc, best_n = -1, min(20, cap)
    for k in range(k_step, cap+1, k_step):
        sel = imp.head(k).index.tolist()
        tmp = __import__('xgboost').XGBClassifier(
            n_estimators=100, max_depth=3,
            random_state=seed, verbosity=0, eval_metric='aucpr'
        )
        tmp.fit(X_tr[sel], y_tr, verbose=False)
        pv = tmp.predict_proba(X_val[sel])[:, 1]
        pr = average_precision_score(y_val, pv)
        if pr > best_prauc:
            best_prauc, best_n = pr, k

    return imp.head(best_n).index.tolist()

# ══════════════════════════════════════════════
# 7. DENGESİZLİK STRATEJİLERİ — oversample (genel altyapı)
# ══════════════════════════════════════════════
def oversample_minority(X, y, seed=SEED, jitter_std=0.01):
    """
    [Genel altyapı] Basit, BAĞIMSIZ-KÜTÜPHANESİZ random oversampling
    (with replacement) + küçük Gaussian jitter. Harici bir paket (örn.
    imbalanced-learn) KASITLI olarak kullanılmıyor: jüri ortamında bu paket
    kurulu olmayabilir, bu da %100 reproducibility şartını riske atar.

    jitter_std: kopyalanan minority satırlara, sayısal kolonlarda
    std*jitter_std büyüklüğünde gürültü eklenir (tam aynı satırın birebir
    kopyalanması yerine hafif çeşitlilik — modelin "ezbere" eğilimini azaltır).
    Yalnızca sayısal kolonlara uygulanır.
    """
    rng = np.random.RandomState(seed)
    counts = y.value_counts()
    minority_label = counts.idxmin()
    n_minority, n_majority = counts.min(), counts.max()
    n_needed = n_majority - n_minority
    if n_needed <= 0:
        return X.copy(), y.copy()

    min_idx = y[y == minority_label].index
    sampled_idx = rng.choice(min_idx, size=n_needed, replace=True)

    X_extra = X.loc[sampled_idx].copy()
    num_cols = X_extra.select_dtypes(include=[np.number]).columns
    stds = X.loc[min_idx, num_cols].std(ddof=0).fillna(0.0)
    noise = rng.normal(loc=0.0, scale=(stds.values * jitter_std), size=X_extra[num_cols].shape)
    X_extra[num_cols] = X_extra[num_cols].values + noise

    y_extra = y.loc[sampled_idx].copy()
    X_new = pd.concat([X, X_extra], axis=0).reset_index(drop=True)
    y_new = pd.concat([y, y_extra], axis=0).reset_index(drop=True)
    return X_new, y_new

# ══════════════════════════════════════════════
# 8. GENEL CV ORKESTRASYONU (model-sayısı-agnostik)
# ══════════════════════════════════════════════
def ensemble_predict_proba(models_cal, weights, X):
    """models_cal/weights: {ad: nesne} sözlükleri. Ağırlıklı ortalama olasılık döner."""
    p = np.zeros(len(X))
    for name, model in models_cal.items():
        p = p + weights[name] * model.predict_proba(X)[:, 1]
    return p

def run_repeated_cv(X, y, test_prior, train_fold_fn, n_splits, n_repeats,
                     fold_kwargs=None, panel_name="?"):
    """
    [Genel] Panel-spesifik train_fold_fn(X_tr,y_tr,X_val,y_val,train_prior,
    test_prior,fold_id=...,**fold_kwargs) -> standart fold_result sözlüğü
    bekler (bkz. modül başı sözleşme). RepeatedStratifiedKFold ile çalışır
    (CFTR için bkz. run_loocv — orada LeaveOneOut kullanılır).
    """
    fold_kwargs = fold_kwargs or {}
    print("\n" + "="*60)
    print(f"[{panel_name}] Repeated Stratified K-Fold ({n_splits}-Fold x {n_repeats} tekrar "
          f"= {n_splits*n_repeats} CV)")
    print("="*60)

    train_prior = compute_train_prior(y)
    print(f"  TRAIN_PRIOR (veriden): {train_prior:.4f} "
          f"({(y==1).sum()} patojenik / {(y==0).sum()} benign)")
    print(f"  TEST_PRIOR (parametre): {test_prior:.4f}")

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    fold_results = []

    for fold_idx, (tr_idx, val_idx) in enumerate(rskf.split(X, y)):
        repeat = fold_idx // n_splits + 1
        fold   = fold_idx  % n_splits + 1
        label  = f"R{repeat}F{fold}"
        if fold_idx % n_splits == 0:
            print(f"\n  -- Tekrar {repeat}/{n_repeats} --")

        X_tr, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[tr_idx].copy(), y.iloc[val_idx].copy()

        res = train_fold_fn(X_tr, y_tr, X_val, y_val, train_prior, test_prior,
                             fold_id=label, **fold_kwargs)
        fold_results.append(res)

        m = res['metrics']
        print(f"  {label} | F1={m['f1']:.4f} | MCC={m['mcc']:.4f} | "
              f"PR-AUC={m['pr_auc']:.4f} | Thresh={m['threshold']:.3f} | Feats={m['n_features']}")

    keys = ['f1','mcc','pr_auc','roc_auc','sensitivity','specificity','balanced_acc','brier']
    summary = {}
    for k in keys:
        vals = [r['metrics'][k] for r in fold_results]
        summary[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                      'min': float(np.min(vals)), 'max': float(np.max(vals))}

    print("\n" + "─"*60)
    print(f"[{panel_name}] REPEATED CV ÖZET ({len(fold_results)} fold):")
    for k, v in summary.items():
        print(f"  {k:<16} {v['mean']:>10.4f} +/-{v['std']:>8.4f}  [{v['min']:.4f}, {v['max']:.4f}]")

    # OOF tahminlerini fold_results'a ekle (görselleştirme için)
    oof_y_true = np.concatenate([r['val_labels']          for r in fold_results])
    oof_y_prob = np.concatenate([r['val_probs_corrected'] for r in fold_results])
    # summary'e de ekle — rapor JSON'una geçsin
    summary['_oof_y_true'] = oof_y_true.tolist()
    summary['_oof_y_prob'] = oof_y_prob.tolist()

    return fold_results, summary, train_prior

def run_loocv(X, y, test_prior, train_fold_fn, fold_kwargs=None, panel_name="?",
              feature_select_once=True, k_max=None):
    """
    [v4-YENİ, CFTR için] Leave-One-Out CV.

    [DÜRÜSTLÜK NOTU — PDR'da belirtilmelidir] feature_select_once=True ise,
    SHAP özellik seçimi LOO döngüsünden ÖNCE, TÜM eğitim verisiyle (n=111)
    BİR KEZ yapılır ve bütün LOO iterasyonlarında SABİT kalır. Bu, KESİN
    olarak sıfır-leakage değildir (seçim, her iterasyonda dışarıda
    tutulacak noktayı da görmüş olur) — ANCAK 111 iterasyonun her birinde
    SHAP-tabanlı aramayı yeniden çalıştırmak hesaplama açısından
    orantısızdır ve "regularization max'a çek / feature sayısını minimize
    et" notunun ruhuna ZATEN hizmet eder (küçük, sabit bir özellik kümesi).
    Bu, bu kod tabanının önceki sürümlerindeki "StratifiedGroupKFold yerine
    StratifiedKFold" şeffaflığıyla AYNI ruhtaki bilinçli bir basitleştirmedir.
    """
    fold_kwargs = fold_kwargs or {}
    print("\n" + "="*60)
    print(f"[{panel_name}] Leave-One-Out CV (n={len(X)})")
    print("="*60)

    train_prior = compute_train_prior(y)
    print(f"  TRAIN_PRIOR (veriden): {train_prior:.4f} "
          f"({(y==1).sum()} patojenik / {(y==0).sum()} benign)")
    print(f"  TEST_PRIOR (parametre): {test_prior:.4f}")

    global_sel_feats = None
    if feature_select_once:
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import RobustScaler
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        imp_g, sc_g = SimpleImputer(strategy='median'), RobustScaler()
        X_g = X.copy()
        X_g[num_cols] = imp_g.fit_transform(X_g[num_cols])
        X_g[num_cols] = sc_g.fit_transform(X_g[num_cols])
        X_g_f, _ = correlation_filter(X_g, thresh=CORR_THRESH)
        # validation-PRAUC ölçümü için basit bir 80/20 iç bölme (sadece seçim için)
        from sklearn.model_selection import train_test_split
        Xs_tr, Xs_val, ys_tr, ys_val = train_test_split(
            X_g_f, y, test_size=0.2, stratify=y, random_state=SEED
        )
        global_sel_feats = shap_feature_selection(Xs_tr, ys_tr, Xs_val, ys_val, k_max=k_max)
        print(f"  [Global özellik seçimi — LOO öncesi, TEK SEFER] {len(global_sel_feats)} özellik seçildi "
              f"(k_max={k_max}). PDR'da bu basitleştirme belirtilmelidir.")

    loo = LeaveOneOut()
    fold_results = []
    for i, (tr_idx, val_idx) in enumerate(loo.split(X)):
        X_tr, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
        y_tr, y_val = y.iloc[tr_idx].copy(), y.iloc[val_idx].copy()

        res = train_fold_fn(X_tr, y_tr, X_val, y_val, train_prior, test_prior,
                             fold_id=f"LOO{i+1}", global_sel_feats=global_sel_feats,
                             **fold_kwargs)
        fold_results.append(res)
        if (i+1) % 20 == 0 or (i+1) == len(X):
            print(f"  {i+1}/{len(X)} LOO iterasyonu tamamlandı...")

    keys = ['f1','mcc','pr_auc','roc_auc','sensitivity','specificity','balanced_acc','brier']
    # LOOCV'de her fold'un val seti n=1 olduğundan fold-bazlı F1 anlamsızdır;
    # bu yüzden özet, OOF HAVUZUNDAN (pooled) hesaplanır (bkz. compute_oof_pooled_threshold).
    all_probs  = np.concatenate([r['val_probs_corrected'] for r in fold_results])
    all_labels = np.concatenate([r['val_labels'] for r in fold_results])
    best_t, best_f1 = find_best_threshold(all_labels, all_probs)
    pooled_metrics = compute_metrics(all_labels, all_probs, threshold=best_t)
    summary = {k: {'mean': pooled_metrics[k], 'std': 0.0, 'min': pooled_metrics[k], 'max': pooled_metrics[k]}
               for k in keys}
    print(f"\n[{panel_name}] LOOCV OOF-HAVUZLANMIŞ SONUÇ: F1={pooled_metrics['f1']:.4f} | "
          f"MCC={pooled_metrics['mcc']:.4f} | eşik={best_t:.4f}")
    return fold_results, summary, train_prior

def compute_oof_pooled_threshold(fold_results):
    """
    TÜM OOF tahminleri tek havuzda toplanıp TEK bir eşik bu havuzda aranır
    (N ayrı fold-threshold'unun gürültülü ortalaması yerine).
    """
    all_probs  = np.concatenate([r['val_probs_corrected'] for r in fold_results])
    all_labels = np.concatenate([r['val_labels'] for r in fold_results])
    best_t, best_f1 = find_best_threshold(all_labels, all_probs)
    print(f"\n  [OOF-havuzlanmış nihai eşik] {best_t:.4f} (havuz F1={best_f1:.4f}, "
          f"n={len(all_labels)} OOF tahmin)")
    return best_t

def simulate_clinical_stress_test(oof_probs, oof_labels, target_pi_patho, seed=SEED):
    """
    Fiziksel bir train/test bölmesi YAPMAZ — OOF havuzu yeniden örneklenir.
    [Gerekçe] Küçük panellerde (CFTR/PAH) toplam benign örnek sayısı, test
    priorını taklit eden bir holdout ayırmaya yetmez; bu, eğitim setindeki
    zaten kıt benign havuzunu tüketir. OOF-yeniden-örnekleme (her tahmin,
    onu üreten fold/LOO-modelinin GÖRMEDİĞİ bir örnekten gelir) korunur.
    """
    rng = np.random.RandomState(seed)
    idx_b = np.where(oof_labels == 0)[0]
    idx_p = np.where(oof_labels == 1)[0]
    n_benign_avail = len(idx_b)
    n_patho_needed = int(round(n_benign_avail * target_pi_patho / (1 - target_pi_patho)))
    bootstrap_used = n_patho_needed > len(idx_p)
    idx_p_sample = rng.choice(idx_p, size=n_patho_needed, replace=bootstrap_used)
    idx_final = np.concatenate([idx_p_sample, idx_b])
    rng.shuffle(idx_final)
    return oof_probs[idx_final], oof_labels[idx_final], bootstrap_used

def run_stress_test_comparison(fold_results, test_prior, panel_name="?"):
    print("\n" + "="*60)
    print(f"[{panel_name}] SİMÜLE EDİLMİŞ KLİNİK STRES TESTİ (hedef pi_patojenik={test_prior:.4f})")
    print("="*60)
    raw_pool   = np.concatenate([r['val_probs_raw']       for r in fold_results])
    corr_pool  = np.concatenate([r['val_probs_corrected'] for r in fold_results])
    label_pool = np.concatenate([r['val_labels']          for r in fold_results])

    p_raw,  y_raw,  bs1 = simulate_clinical_stress_test(raw_pool,  label_pool, test_prior)
    p_corr, y_corr, bs2 = simulate_clinical_stress_test(corr_pool, label_pool, test_prior)
    t_raw,  _ = find_best_threshold(y_raw,  p_raw)
    t_corr, _ = find_best_threshold(y_corr, p_corr)
    m_raw  = compute_metrics(y_raw,  p_raw,  threshold=t_raw)
    m_corr = compute_metrics(y_corr, p_corr, threshold=t_corr)

    print(f"\n  {'Metrik':<14} {'Düzeltme Öncesi':>16} {'Düzeltme Sonrası':>18} {'Fark':>10}")
    print("  " + "─"*60)
    for k in ['f1', 'mcc', 'sensitivity', 'specificity', 'pr_auc']:
        diff = m_corr[k] - m_raw[k]
        print(f"  {k:<14} {m_raw[k]:>16.4f} {m_corr[k]:>18.4f} {diff:>+10.4f}")
    if bs1 or bs2:
        print("\n  [UYARI] Bootstrap (with-replacement) örnekleme kullanıldı — "
              "PDR'da belirtilmelidir.")
    return {'before': m_raw, 'after': m_corr, 'bootstrap_used': bool(bs1 or bs2)}

def predict_test(X_test_raw, fold_results, train_prior, test_prior, final_threshold):
    """[Genel] models_cal/weights sözlük yapısını kullanır — model sayısından bağımsız."""
    all_probs, all_raw_probs = [], []
    for res in fold_results:
        X_t  = X_test_raw.copy()
        ncols= X_t.select_dtypes(include=[np.number]).columns.tolist()
        X_t[ncols] = res['imputer'].transform(X_t[ncols])
        X_t[ncols] = res['scaler'].transform(X_t[ncols])
        X_t = X_t.drop(columns=res['dropped_corr'], errors='ignore')
        X_t = X_t[res['sel_feats']]

        p_ens = ensemble_predict_proba(res['models_cal'], res['weights'], X_t)
        all_raw_probs.append(p_ens)
        all_probs.append(prior_correction(p_ens, train_prior, test_prior))

    final_probs    = np.mean(all_probs, axis=0)
    raw_probs_mean = np.mean(all_raw_probs, axis=0)
    return final_probs, final_threshold, raw_probs_mean

def run_shap(fold_results, X_train, output_dir, panel_name="?", prefix="shap"):
    """
    [Genel, sklearn-versiyon-güvenli] get_base_estimator() başarısız olursa
    (sklearn sürüm farkı), ağırlıklı çoklu-model SHAP'tan tek (en yüksek
    ağırlıklı) modele otomatik düşülür — pipeline ÇÖKMEZ.
    """
    print("\n" + "="*60)
    print(f"[{panel_name}] SHAP Açıklanabilirlik (güvenli düşüşlü)")
    print("="*60)

    best = max(fold_results, key=lambda r: r['metrics']['f1'])
    ncols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_s = X_train.copy()
    X_s[ncols] = best['imputer'].transform(X_s[ncols])
    X_s[ncols] = best['scaler'].transform(X_s[ncols])
    X_s = X_s.drop(columns=best['dropped_corr'], errors='ignore')
    X_s = X_s[best['sel_feats']]

    try:
        sv_weighted = np.zeros((X_s.shape[0], X_s.shape[1]))
        for name, model in best['models_cal'].items():
            sv = shap.TreeExplainer(get_base_estimator(model)).shap_values(X_s)
            sv_weighted = sv_weighted + best['weights'][name] * sv
        shap_method = "weighted_ensemble"
        importance = pd.Series(np.abs(sv_weighted).mean(axis=0), index=X_s.columns).sort_values(ascending=False)
    except Exception as e:
        print(f"  [UYARI] Ağırlıklı ensemble SHAP başarısız oldu ({type(e).__name__}: {e}).")
        top_name = max(best['weights'], key=best['weights'].get)
        print(f"          Güvenli düşüş 1: en yüksek ağırlıklı model ('{top_name}') ile SHAP deneniyor.")
        try:
            sv_weighted = shap.TreeExplainer(get_base_estimator(best['models_cal'][top_name])).shap_values(X_s)
            shap_method = f"{top_name}_only_fallback"
            importance = pd.Series(np.abs(sv_weighted).mean(axis=0), index=X_s.columns).sort_values(ascending=False)
        except Exception as e2:
            # [DÜZELTME] SHAP/xgboost sürüm uyumsuzluğu (örn. base_score format
            # hatası) tüm SHAP çağrılarını kırabilir. Bu durumda raporlama adımı
            # PIPELINE'I ÇÖKERTMEMELİ — modelin kendi feature_importances_'ına
            # (gain-tabanlı, versiyon-bağımsız) düşülür.
            print(f"  [UYARI] SHAP tamamen başarısız oldu ({type(e2).__name__}: {e2}).")
            print("          Güvenli düşüş 2: model.feature_importances_ (gain) kullanılıyor.")
            base_est = get_base_estimator(best['models_cal'][top_name])
            importance = pd.Series(base_est.feature_importances_, index=X_s.columns).sort_values(ascending=False)
            shap_method = f"{top_name}_feature_importances_fallback"
    print(f"\n  [SHAP yöntemi: {shap_method}] Top-10 — {panel_name}:")
    for feat, val in importance.head(10).items():
        bar = "#" * int(val / importance.iloc[0] * 25) if importance.iloc[0] > 0 else ""
        print(f"  {feat:30s} {bar} {val:.4f}")

    path = os.path.join(output_dir, f'{prefix}_shap.csv')
    importance.reset_index().rename(columns={'index':'feature', 0:'shap_mean_abs'}).to_csv(path, index=False)
    print(f"\n  Kaydedildi: {path}")
    return importance, shap_method