# Eski loss aramasının sınırı — sıfır-fit yeniden kurma

Profil `legacy-loss-lineage-v1`; mevcut `research-v2-tf-placement` XGBoost
T+1 makbuzlarını ve frozen kaynak `73a92265…` inceler. Yeni deney/model
eğitimi değildir. 2016–2023 yıllarındaki 126'şar origin, toplam 1.008;
güncel 2.006-origin full-year değerlendirmesiyle karıştırılmaz.

- Kamuya açık eski arşiv manifestinin SHA-256'sı ve kullanılan her dosya
  doğrulanır; değiştirilmiş/sanitized girdi bu replay'e kabul edilmez.
- Özgün 123 dosyalık source manifesti doğrulanır. Kod kaynak ZIP'inden
  çalıştırılmaz; eski 128-aday üreticinin XGBoost dalı açıkça yeniden kurulur.
- Kayıtlı parametreler kimliğin kaynağıdır. Buradaki yeniden üreticide 19
  parametrede gözlenen son-bit farkları kaydedilir; yapısal değişiklik veya
  8 ULP üzeri fark reddedilir. Tarihsel fit kimlikleri yuvarlanmaz/değiştirilmez.
- Her dönemde 128 seed42 adayın 3×63 geçmiş skoru, top5'in üç-seed onayı,
  karmaşıklık eşitlik kuralı ve seçilen üç outer makbuzu yeniden kurulur.
  OOS sonuçla yeni seçim/ağırlık yapılmaz; özgün log-getiri ortalaması kullanılır.
- Eğitim kohortları, warmup, H5 olgunlaşma ve frame kimliği makbuzla eşleşir.
  Origin/hedef/fiyatlar korunur; eksik kayıtla sessiz kesişim veya fallback yoktur.
- Naive paired fiyat-MAE, yön, yıllar ve yıl-koruyan blok20/60 bootstrap
  (10.000 tekrar, seed42) raporlanır. Üç seed aynı origin'leri tekrarlar;
  3.024 seed-satırı bağımsız 3.024 origin değildir.

İki ayrı kanıt sorusu vardır: eski seçilmiş programın Naive karşısındaki
sonucu ve kontrollü loss etkisinin gerçekten sınanıp sınanmadığı. İkincisi
için target/loss/hiperparametreler birlikte değişiyorsa ablation sonucu
çıkarılmaz. Model payload'larının yokluğu açıkça korunur; yeni model
çıkarımı doğrulanmış gibi sunulmaz. T+5'e çıkarım yapılmaz.

Mevcut CPU bağımlılık ortamında proje komutu:

```bash
python ml/legacy_loss_lineage.py --input-root ORIGINAL_PROJECT_OR_RESTORED_ROOT --archive-manifest evidence-manifest.json.gz --output NEW_OUTPUT
```

Output girdi kökünün dışında olmalıdır. Kaynak/ayar/rapor kimlikleri ve
CSV değişmeden tekrar üretilebilir. Python `-O` ile kontrollerin devre dışı
bırakılması reddedilir. Yeni fit/bağımlılık/veri indirme veya canlı yayın yoktur.
