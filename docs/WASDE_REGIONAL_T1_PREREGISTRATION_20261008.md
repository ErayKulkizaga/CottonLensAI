# Bölgesel WASDE T+1: kilitlenen katkı deneyi

**VERIFIED:** ön kayıt tamamlandı; gerçek piyasa fit'i **0**. Bu bir performans
sonucu veya kaynak kabulü değildir. Amaç, sayısal olarak kontrol edilmiş 26
bölgesel WASDE alanının mevcut Cotton çekirdeğine T+1 katkısını tek sabit tarifle
ölçmektir. Model/ufuk/gecikme sonuçlara bakılarak seçilmeyecek.

## Eski denemeden farkı

`python ml/history.py check --query wasde` ön kayıt öncesinde 21 ilişkili kayıt
ve 14 tamamlanmış fit tarifi buldu; yeni tam kimlikte eşleşme yok. Bu,
WASDE'nin hiç denenmediği anlamına gelmez. Eski World balance/text denemeleri
T+5 ve sınırlı temsillerdi; bu bölgesel 26 alanlı T+1 karşılaştırması değildir.
Eski negatifler bütün WASDE bilgisine veya T+1'e genellenmez.

## Sabit tasarım

| Unsur | Kilitlenen karar |
|---|---|
| Profil / deney | `wasde-regional-t1-pilot-v1` / `research-wasde-regional-t1-pilot-v1` |
| Girdi | Dondurulmuş `research-full-year-v1-r2` çekirdeği; mevcut 95 raporlu aday ve iki sayısal mutabakat |
| Ortak değerlendirme | 2019–2023: 252, 253, 252, 251, 246; toplam **1.254 origin/kol** |
| Kollar | `mask_D0`, `numeric_D0`, `mask_D1`, `numeric_D1` |
| Model / hedef | Ridge alpha=1, seed42, pencere1; ayrı T+1 standardized log-return, mevcut fiyat dönüşümü |
| Öğrenme | 2016-01-13'ten genişleyen geçmiş; eğitimde median/mean/std, ortak warmup ve `target_date_5 < cutoff` |
| Seçim / refit | Sabit tarif; yalnız geçmiş 3×63 doğrulamada küçültme `[0,.25,.5,.75,1]`, eşitlikte küçük ağırlık; 21 gözlemde refit |
| Birincil karşılaştırma | Seçilmiş `numeric_D0` / `mask_D0` eşleştirilmiş fiyat-MAE; ayrıca aynı origin'lerde Naive |
| Bütçe | **180 iç + 244 dış = 424 fit**, 20 yıllık çıktı, 5.016 tahmin satırı; 1 süreç, en çok 2 thread, 30 dakikalık oturum |

2016–2018, mevcut kaynak protokolünün 500 geçmiş olgun kaynak satırı ve üç iç
blok koşulunu karşılamıyor. Bu yıllar sessizce eksik sayılmadı. Beş görülmüş
yıl bağımsız holdout veya korunmuş **6/8** yayın koşulunun karşılığı değildir.
2024+ kullanılmaz. Kaynak eksikliği nedeniyle origin silinmez.

## Karşılaştırmayı gerçekten ne değiştiriyor?

Her kol 24 çekirdek özellik, ortak rapor yaşı/erişilememe bilgisi ve 26 kaynak
ya da kontrol sütunu içerir: 52 mantıksal, mevcut otomatik eksiklik göstergeleri
ile 104 dönüştürülmüş sütun. Kontrolün her ek sütunu, sayısal değer biliniyorsa
**sabit 0**, bilinmiyorsa **NaN** taşır. Bunlar karşı-olgusal kontrol sütunlarıdır;
kaynak değerleri sıfırlanmadı veya doldurulmadı. Böylece iki kolun otomatik
eksiklik göstergeleri birebir aynıdır. Sayısal katkı, eksiklik/takvim bilgisinin
fazladan verilmesiyle karışmaz.

1.254 günlük origin'de **60 ayrı rapor vintage'ı** var: 2019–2023 yeni raporları
ve taşınan Aralık 2018 raporu. En erken iç eğitimde yalnız **27 ayrı rapor**
bulunur. Günlük satır sayısı bağımsız temel bilgi sayısı değildir. Ocak 2019
raporu uydurulmadı; 117 bilinmeyen revizyon hücresi korunur. Negatif sonuç,
bütün temsillerde bilginin yokluğunu kanıtlamayacak.

## Saat varsayımı ve leakage sınırı

- Karar: Cotton kaynak tarihini izleyen gün **00:15 UTC**.
- D0 varsayımı: ilgili sayısal WASDE sürümü, rapor tarihini izleyen gün
  **00:00 UTC** itibarıyla kullanılabilir. **Tarihsel olarak doğrulanmış değil.**
- D1: D0'ın ilk uygun kararından **bir ek kayıtlı Cotton gözlemi** sonra.
- Azami yaş 40 Cotton gözlemi; iki gecikmede aynı D0 başlangıcına bağlı.
  Ek gecikme kaynağın son kullanma sınırını uzatmaz.
- Kaynak tarihleri ve `assumed_available_at` korunur. Bunlar `published_at`
  veya doğrulanmış `available_at` yapılmaz. Gelecek rapor geçmişe doldurulmaz.

Saat varsayımının yanlış olması hâlinde gerçek erişilebilirlik leakage'i hâlâ
mümkündür. Bu nedenle tarihsel kabul kapısı, vintage/saat onayı ve runtime
özellikleri açılmadı. Bu teslimatın modülü **yalnız offline ön kayıt** üretir;
engine'e çalıştırılabilir profil veya sahte kabul makbuzu eklemez.

## Sonuca bakmadan kilitlenen yorum

Fark işareti pozitifte sayısal kol daha iyi. Kazanç
`100*(1 - numeric_MAE/comparator_MAE)`; aynı origin, hedef tarihi ve gerçekleşen
değerler zorunlu. Yıl içi paired bootstrap: 10.000 tekrar, seed42,
blok20/60. D1 ana sonucun yerine seçilemez. Ham tahmin, aktif oran/yön, yıllık
kazanç, en büyük ceil(%1) Naive hatası çıkarımı ve kaynak yaşı 0–4 tanıları
zorunlu, birincil ölçütün yerine kullanılamaz.

Aşağıdaki kurallar sırayla uygulanır; makine tarafından okunabilir sözleşme
ön kayıt dosyasının hash'ini bağlar ve fit öncesinde ayrı dondurulmuştur:

| Önceden belirlenen sonuç | Karar |
|---|---|
| Kimlik, kaynak saati, payload veya ortak hedef kontrolü başarısız | Bilimsel sonuç çıkarılmaz; ilgili kontrol düzeltilmeden fit yok |
| D0/D1 kontrol karşısında her iki blokta alt sınır >0; D0 Naive nokta kazancı ≥%5 ve tüm-origin yön ≥%53 | Varsayıma bağlı aday; gerçek erişim ve ileri doğrulama hazırlanır, yayın yapılmaz |
| Her iki sayısal kolun Naive kazancında her iki blokta üst sınır <%5 | Bu seçilmiş sabit tarif pratik hedefi kurtarmıyor; Naive korunur, program büyütülmez |
| D0/D1 katkısı her iki blokta desteklenir ama pratik nokta eşikleri sağlanmaz | Katkı kaydedilir, hedef çözülmüş sayılmaz; otomatik grid/kaynak ekleme yok |
| Yalnız D0 desteklenir veya aralıklar ayıramaz | Belirsiz/gecikmeye duyarlı; iyi gecikme sonradan seçilmez, otomatik ek fit yok |

Sıfır küçültme ve `[0,0]` farkı kaynakta sinyal yokluğu değildir; ham sonuç
ayrı raporlanır. Bu aralıklar bütün tarihsel araştırma seçimlerini kapsamaz.
%5, %53/%55 ve 6/8 koşulları gevşetilmedi.

## Kimlik, koruma ve yeniden kontrol

- ML kaynak kimliği: `86329dfba9a5fb2090e90b85e078323181c2b1ef498f85500a2ca375c6b23fd0`.
- Ön kayıt kimliği: `985c35f68fde70b7060a71852d08b907cc4a5f22b50660f81ea22167d4c68063`.
- Karar sözleşmesi SHA-256: `ae7a5f33bb9636db52a0cf5cba066c055994ca2a5a3988b0da4e4c7b1693b020`.
- 229 tamamlanmış payload + completion dosyası; eski **2.441 dosya** değişmedi.
- 71 dar test ve tüm yerel ML testleri **691 geçti / 3 atlandı**; yeni iki
  dosyada Ruff geçti. GPU kontrolü/eğitimi yapılmadı. Backend/UI kodu değişmedi.

[Kanıt özeti](../research/evidence/wasde-regional-preregistration-20261008.json),
[234 dosyalı Release envanteri](../research/evidence/wasde-regional-preregistration-release-20261008.json)
ve [ek GitHub paketi](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/wasde-regional-preregistration-20261008)
tam girdileri, ön kaydı, kaynak snapshot'ını, karar sözleşmesini ve bağımsız
satır bazlı `verify.py` kontrolünü saklar. Arşivi kendi klasörüne çıkarıp pinned
CPU ortamında `verify.py` çalıştırılır; bu işlem fit yapmaz.

Yeni ön kayıt için, mevcut tamamlanmış klasörü değiştirmeden:

```powershell
$env:PYTHONPATH = 'ml/src;ml'
python -m cottonlens_ml.research.wasde_regional_pilot --repo . --output NEW_NAMESPACE --reference FROZEN_FULL_YEAR_R2 --candidate REGIONAL_CANDIDATE --numeric-audit PDF_XML_AUDIT --csv-audit AS_REPORTED_AUDIT
```

**Tek sonraki iş:** bu kilitli tasarımı mevcut Experiment/ledger CPU yoluna
bağlayıp, source/manifest/engine kimliğini yeni yürütme namespace'inde yeniden
dondurmak; ardından yalnız 424 fit'lik deneyi çalıştırmak. Ön kayıt v1
değiştirilmeyecek. Yürütme kodu değişikliği yeni source kimliği gerektirir;
bu manifestin “tamamlandı” etiketi tamamlanmış deney veya cache izni değildir.
