# Sabit T+5 named-kontrat kapasite kontrolü — 10 Ekim 2026

## Soru ve kapsam

PR39 tamamlandı: aynı örneklemde doğru kontrat etiketi CT etiketiyle
eğitime göre ham fiyat-MAE'yi azaltıyor; named Ridge yine Naive'den
%7,938310 kötü. Bu fark bütün proje başarısızlığının roll kaynaklı
olduğunu kanıtlamıyor. Kalan dar soru: aynı bilgilerde doğrusal Ridge'in
çıkaramadığı doğrusal olmayan bir koşullu ortalama mevcut mu?

Sicil tarandı (55 ilgili XGBoost/T+5 çalışma, 117 tarif nesnesi). En yakın
eski native kayıt `0324c461bf69df913453f420278baa41fa6a44e74e92ca2a8c48e5c3f78818c6`
eski `research-full-year-v1-r2`, `price-outer-2018-8`: CT etiketi, 24 özellik,
76 ağaç. Kaynak/adapter/model checksum'ları doğrulandı. Bu, mevcut named
etiket/28 özellik/2023 ortak train karşılaştırmasını sınamıyor. Sicilde
eşleşme bulunmaması tek başına yenilik kanıtı değildir. Eski CT grid'i,
3380-fit Ridge zayıf-sinyal kontrolü veya tamamlanmış 44+1 label deneyi
tekrarlanmayacak.

## Dondurulacak tek tarif

Profil `named-capacity-control-t5-v1`, deney
`research-named-capacity-control-t5-v1`. Mevcut Experiment/Ledger/CPU
başlatıcı kullanılır; ayrı namespace ve kaynak kimliği gerekir.

| Unsur | Sabit tasarım |
|---|---|
| Referans | PR39 tamamlanmış named Ridge native kayıtları; yeniden fit yok |
| Veri/hedef | Aynı ikinci, origin'de seçilmiş kontrat; aynı derived history ve standardized log-return T+5 |
| Örneklem | Aynı 2023 yılının 246 dış origin'i, 242 eşleşmiş hedef, 4 açık bilinmeyen |
| İç doğrulama | Aynı geçmiş kaynak-olgun 3×63 origin ve 3/4/3 iç refit |
| Dış refit | Aynı 12 iş, aynı 21 Cotton gözlemlik takvim ve train tarihleri |
| Bilgi | Aynı 28 numeric D0 özellik; gelecek kontratı/hedef metadata özellik değildir |
| Ön işleme | Aynı train-only median/scaler; window1, seed42; aynı hedef dönüşümü |
| Tek değişiklik | Ridge alpha1 yerine sabit XGBoost depth2, eta0,03, min_child_weight20, alpha0, lambda1, 100 ağaç |
| Seçim | Ağaç/grid/early-stop seçimi yok; geçmiş shrinkage [0;0,25;0,5;0,75;1] aynı, eşit skorda en küçük ağırlık |
| Birincil | Ham Ridge−XGBoost eşleştirilmiş aynı-kontrat fiyat-MAE farkı |
| İkincil | Naive kazancı, ham/seçilmiş yön ve aktif oran; shrinkage sonucu birincili değiştiremez |
| Bütçe | 22 piyasa+en fazla 4 sentetik fit; tek süreç, en fazla 2 thread, 30 dakika/oturum |

100 ağaç sonuç görülmeden kilitlenen tek ihtiyatlı boosting bütçesidir;
geçmiş 2023 skorundan seçilmez. Yeni 63-label early-stop ayırmak en erken
528 train örneğini 500 altına düşüreceği için ek iç seçim yapılmaz.

Sentetik kontroller ayrı ledger'dadır: sabit tarifte güçlü doğrusal ve
`0,02*(x0²−1)` doğrusal olmayan sinyal (her ikisi ≥%50 MAE kazancı),
etiket blok kaydırma (≤%20), ayrıca aile ezberleme kontrolü (128 aynı
train/test, depth8/400 ağaç, ≥%90). Ezberleme tarifi piyasa tarifi değildir.
Her sonuç native tahminlerden yeniden değerlendirilir; cached `passed`
yetmez. Tek kontrol başarısızsa piyasa fit'i durur, otomatik parametre
değişikliği/yerine yeni fit yok. Bu güçlü sinyaller %5'lik piyasa etkisini
bulma gücünü veya gerçek piyasa becerisini kanıtlamaz.

## Zaman ve veri sınırları

Karar Cotton kaynak tarihi+1 gün 00:15 UTC. Arşiv kaynak takvimi+1 gün
00:00 UTC ve final vintage varsayımı değişmez. Tarihsel UTC/ilk sürüm
kanıtlanmış değildir; deney yalnız varsayıma bağlı tarihsel duyarlılıktır.
Train hedef tarihi cutoff'tan önce ve named hedef kaynağı varsayımsal
erişim saati refit kararından önce olmalıdır. Orijinal boş özellikler,
origin/target tarihleri korunur; geç/bilinmeyen origin fiyatından fiyat
tahmini üretilmez, eksik gerçekleşmeye sıfır hata verilmez. Kaynak/ön
işleme/target/shared model kodu referans snapshot'ıyla byte eşit olmalı.

2023 daha önce görülmüş geliştirme verisidir. Tek yıl eski 6/8 yıl, %5
MAE ve %53/%55 yön yayın koşullarını değerlendiremez. 2024+ seçimde yok.
Veri indirme, canlı writer, GPU, başka model/grid, merge veya model yayını yok.

## Önceden belirlenen karar

10.000 tekrar, seed42, blok20 ve60; 246 ordinal yuva korunur, 4 bilinmeyen
yuvada yalnız bookkeeping ağırlığı0; her çekim bilinen örnek sayısına
bölünür. Recentered basic aralıklar araştırma/model/insan seçimlerinin
tamamını veya varsayımsal vintage yanlılığını kapsamaz. Naive hata en büyük
%1 (3 origin) çıkarılmış duyarlılık ayrıca raporlanır, seçimde kullanılmaz.

- İki ham Ridge−XGB aralığının alt sınırı pozitif ve ham XGB Naive
  kazancı ≥%5: yalnız dar araştırma adayı; bağımsız doğrulama gerekir.
- İki ham XGB−Naive kazanç aralığının üst sınırı <%5: bu tek sabit sığ
  kapasite tarifi pratik hedefi kurtarmıyor. Bütün doğrusal olmayan
  modeller/bütün kaynaklar hakkında negatif hüküm verilmez.
- Diğer durum: INCONCLUSIVE; otomatik ek fit/grid başlamaz.

Kod/test/veri/gerçek paket sürümleri ve karar sözleşmesi dondurulup
checksum Release yayımlanacak; taze GitHub restore doğrulanmadan **sentetik
profil kontrolleri dahil yeni fit yok**. Ardından tek kayıtlı deney
sonuçlandırılır; native tahmin, bağımsız Decimal fiyat hesabı, scalar
bootstrap ve tutarlı bozulma kontrolleriyle denetlenir. Eski kanıtlar
silinmez/değiştirilmez; sonuç sicile ayrı ek kayıt olarak girer.
