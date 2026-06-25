"""
HELIXAI — Rapor Analiz Aracı
==============================
Herhangi bir panelin *_report_v*.json dosyasını okuyup, confusion matrix'i
ve sınıf-bazlı (Patojenik/Benign) dökümü AÇIKÇA gösterir.

Neden gerekli: F1 skoru SADECE pozitif (Patojenik) sınıfa bakar; büyük
sınıf küçük sınıfı "ezberleyerek" F1'i şişirebilir. Bu script, ham
TP/FP/FN/TN sayılarını ve sınıf-bazlı precision/recall'u yan yana koyarak
bu şişmeyi yakalamanızı sağlar.

Kullanım:
  python3 analyze_report.py pah_results/pah_report_v5.json
"""
import json, sys

def print_confusion_block(metrics, title):
    if metrics is None:
        print(f"\n[{title}] -- Bu blok mevcut değil (None) --")
        return
    tp, fp, fn, tn = metrics['tp'], metrics['fp'], metrics['fn'], metrics['tn']
    n = tp + fp + fn + tn
    n_patho  = tp + fn   # gerçekte patojenik olan toplam
    n_benign = fp + tn   # gerçekte benign olan toplam

    print(f"\n{'='*60}")
    print(f"[{title}]  (n={n} | gerçek patojenik={n_patho} / gerçek benign={n_benign})")
    print(f"{'='*60}")
    print(f"{'':18}{'Tahmin: Patojenik':>20}{'Tahmin: Benign':>18}")
    print(f"{'Gerçek: Patojenik':18}{tp:>20}{fn:>18}   <- {tp+fn} örnek")
    print(f"{'Gerçek: Benign':18}{fp:>20}{tn:>18}   <- {fp+tn} örnek")

    prec_patho = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    rec_patho  = tp / (tp + fn) if (tp + fn) > 0 else float('nan')   # = sensitivity
    prec_ben   = tn / (tn + fn) if (tn + fn) > 0 else float('nan')
    rec_ben    = tn / (tn + fp) if (tn + fp) > 0 else float('nan')   # = specificity

    print(f"\n  {'Sınıf':<12}{'Precision':>12}{'Recall':>12}{'Destek (n)':>14}")
    print(f"  {'Patojenik':<12}{prec_patho:>12.4f}{rec_patho:>12.4f}{n_patho:>14}")
    print(f"  {'Benign':<12}{prec_ben:>12.4f}{rec_ben:>12.4f}{n_benign:>14}")

    print(f"\n  F1 (sadece Patojenik'e bakar) : {metrics.get('f1', float('nan')):.4f}")
    print(f"  MCC (her iki sınıfı da hesaba katar): {metrics.get('mcc', float('nan')):.4f}")

    diff = metrics.get('f1', 0) - metrics.get('mcc', 0)
    if diff > 0.25:
        print(f"\n  [UYARI] F1-MCC farkı büyük ({diff:.3f}) — Benign sınıfının zayıf")
        print(f"          öğrenildiği, ama az destek (n={n_benign}) yüzünden F1'e")
        print(f"          az yansıdığı görülüyor. Recall(Benign)={rec_ben:.2f} bunu doğruluyor.")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'pah_results/pah_report_v5.json'
    with open(path, encoding='utf-8') as f:
        report = json.load(f)

    print(f"\nPANEL: {report.get('panel')} | VERSİYON: {report.get('version')}")

    # 1) Stres testi öncesi (ham, düzeltmesiz)
    st = report.get('stress_test', {})
    print_confusion_block(st.get('before'), "STRES TESTİ ÖNCESİ (ham olasılık)")

    # 2) Stres testi sonrası (Bayes correction uygulanmış) -- GÜVENİLİR OLAN BU
    print_confusion_block(st.get('after'), "STRES TESTİ SONRASI (correction uygulanmış, test dağılımı simüle edilmiş)")

    # 3) Gerçek test seti (varsa)
    print_confusion_block(report.get('test'), "GERÇEK TEST SETİ (varsa)")

    print(f"\n{'─'*60}")
    print("OKUMA REHBERİ:")
    print("  - 'STRES TESTİ ÖNCESİ' = test setinin gerçek (benign-ağırlıklı)")
    print("    dağılımını YOKSAYAN, ham CV performansı.")
    print("  - 'STRES TESTİ SONRASI' = şartnamenin test dağılımını SİMÜLE")
    print("    EDEREK ölçülen, gerçek yarışma sonucuna en yakın tahmin.")
    print("  - PDR'da raporlanacak/güvenilecek sayı genellikle İKİNCİSİ olmalı.")
