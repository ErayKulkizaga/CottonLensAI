# Risk Lab doğrulayıcı ön kaydı — 13 Kasım 2026

Profil: `risk-lab-confirmatory-t1-v1`. Durum: **PREREGISTERED_NOT_RUN**.
Kod: `research/risk_lab.py`. Testler: `research/tests/test_risk_lab.py`.
Kod, dondurulmuş `research/range_ewma.py` (`e8255b5d…`) modülünü kullanır.
Hash'ler `research/evidence/risk-lab-prereg-20261113.json` dosyasındadır.

Veri: `range-ewma-confirmatory-t1-v1` kabulünde **zaten kabul edilmiş 7 piyasa**
(ZW, KE, ZL, SB, KC, CC, HE). Dosyalar ve hash'leri aynıdır
(`research/evidence/range-ewma-admission-20261113.json`). Yeni piyasa eklenmez,
yeni veri indirilmez. Bu piyasalarda daha önce yalnız range-EWMA ile EWMA'nın
QLIKE karşılaştırması skorlandı (`8701743`). GARCH, HAR ve aralık sonuçları
bu piyasalarda hiç hesaplanmadı.

Fiyat kapıları, EWMA referansı ve önceki sonuçlar değişmez. Bu ön kayıt
tarihsel proxy veri içindir; ileri doğrulama değildir.

## Hipotezler (tek aile, 4 test)

Her hipotezde kazanç, günlük kayıp farkıdır (referans − range tabanlı aday).
Pozitif değer, adayın daha iyi olduğunu gösterir.

| Kod | Soru | Kayıp | Piyasalar |
|---|---|---|---|
| H2a | Range-EWMA, GARCH(1,1)'i geçiyor mu? | QLIKE, r²(t+1) | 7 |
| H2b | Range-EWMA, HAR'ı geçiyor mu? | QLIKE, r²(t+1) | 7 |
| H3 | Range tabanlı sert hareket olasılığı, EWMA tabanlıdan daha iyi mi? | Brier | **5** (ZW, KE, ZL, CC, HE) |
| H4 | Range tabanlı %80 aralık, EWMA tabanlıdan daha iyi mi? | Winkler (interval score) | 7 |

**H3'te SB ve KC birincil kümeden çıkarıldı.** Ön kayıttan önce bu iki piyasada
sert hareket keşfi yapıldı (Brier ve AUC görüldü). Onların sonucu ayrıca
raporlanır ama karara girmez.

## Dondurulmuş tanımlar

- **GARCH(1,1):** Normal QMLE, ortalama sıfır.
  - Her takvim yılı için parametreler yalnız **1 Ocak'tan önceki** getirilerle
    (genişleyen pencere, en az 1.000 gözlem) L-BFGS-B ile kestirilir.
  - Kalıcılık (α+β) < 0,999.
  - Yıl içinde filtre sabit parametrelerle ileri yürür.
- **HAR:** r²(t+1) ~ 1 + r²ₜ + ort₅(r²) + ort₂₂(r²), OLS.
  - Aynı yıllık refit kuralı geçerlidir; eğitim satırlarının hedef tarihi
    1 Ocak'tan önce olmalıdır.
  - Tahmin, eğitim ortalamasının 0,1 katıyla alttan sınırlanır.
- **Aday:** Dondurulmuş range-EWMA, yani `0,5·EWMA + 0,5·RangeEWMA`.
- **Sert hareket olayı:** |r(t+1)| > t anına kadar gözlenen |r| değerlerinin
  genişleyen %90 kantili (en az 500 gözlem).
- **Olasılık:** Standartlaştırılmış |z|'nin geçmiş-only ampirik aşılma oranı.
  z = getiri / tahmin σ; yalnız olgunlaşmış z'ler kullanılır, en az 500 tane.
  Aynı yöntem EWMA σ ve aday σ için ayrı ayrı uygulanır.
- **Aralıklar:** Filtrelenmiş tarihsel simülasyon. Geçmiş-only z kantilleri ×
  tahmin σ. %80 birincil, %90 raporlanır.
- **Origin'ler:** Piyasa başına, bütün tahminlerin tanımlı olduğu ve hedef
  tarihi en geç **2026-10-30** olan günler. Hipotezler arasında aynı ortak küme
  kullanılır.

## Karar kuralı

Kural A koluyla aynıdır: eşit ağırlıklı havuzlanmış ortalama ve ortak tarih
bloklu bootstrap (20/60, 10.000 örnek, seed 42). Bir hipotez şu koşullarla
**geçer**:

- havuzlanmış ortalama > 0,
- CI20 ve CI60 alt sınırları > 0,
- piyasaların en az ⌈2/3⌉'ünde pozitif ortalama.

Her hipotez iki düzeyde raporlanır: %95 ve aile için Bonferroni düzeltmeli
%98,75. **"Doğrulandı"** etiketi yalnız Bonferroni düzeyinde geçen hipoteze
verilir. %95'te geçip Bonferroni'de geçmeyen hipotez "zayıf destek" olarak
yazılır.

**Betimleyici raporlar (kapı değildir):**

- piyasa başına AUC (EWMA / range),
- climatology'ye göre Brier skill,
- %80/%90 kapsama oranları,
- %90 Winkler kazancı,
- ortalama QLIKE'lar.

Olumsuz sonuçlar silinmez. Sonuçlara bakarak yeni tanım veya varyant
skorlanmaz.

## Bu tasarımdan önce görülenler (açık beyan)

- **CT=F (görülmüş, panele girmez):** `risk_lab` duman testi, 2015–2026 Eylül
  arasında 2.945 origin üzerinde çalıştırıldı.
  - QLIKE kazancı: GARCH'a karşı +0,0185, HAR'a karşı +0,0230.
  - Brier kazancı: +0,00046.
  - AUC: 0,738.
  - %80 kapsama: 0,787.
- **CT keşfi:** Sert hareket ve aralık keşifleri yapıldı. Kalibre aralıklar
  kısa ufukta koşulsuz bantlardan %4–7 iyiydi; range'in ek katkısı ~%1 idi.
- **SB ve KC:** Sert hareket keşfinde Brier ve AUC görüldü (yukarıdaki dışlama).
- **Tasarım tercihleri:** Kantil %90, yıllık refit ve HAR alt sınırı bu
  görülenlerden sonra yazıldı. Hiçbiri 7 piyasanın GARCH, HAR veya aralık
  sonucuna bakılarak seçilmedi.
