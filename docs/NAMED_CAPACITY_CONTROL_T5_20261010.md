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

## 11 Ekim: çalıştırma tamamlandı, piyasa kararı açık

PR40 ve sıfır-fit ön kayıt GitHub'da yayımlandı; taze indirmede 768 dosya
doğrulandı. Kaynak `4a2feff2817bf57db64970c94c8b6a7bbd55cbdc97c8c872225a0756a9dd15af`,
256 ML dosyası. 22 piyasa+4 sentetik fit tamamlandı; eski Ridge yeniden
eğitilmedi. Bilinen doğrusal/nonlinear sentetik MAE kazançları %92,469824/
%77,639833; negatif kontrol −%3,510613, aile ezberleme %99,953439.
Bunlar **piyasa becerisi değildir**.

22 yeni, 22 referans ve 4 sentetik native tahmin çıkarımı tekrar doğrulandı;
veri/recipe/train kimliği, label olgunluğu ve 246/242/4 örneklem korunuyor.
Ek doğrulama fit'i0. Genel ML suite 1064 geçti/3 skip/2 eski uyarı; son
kaynak dar regresyon suite 34 geçti (17 yeni kapasite testi). Son kaynak
koruması geniş test collection'ından sonra eklendi ve dar çalıştırmada
sınandı. Ruff geçti. GPU kontrolü çalıştırılmadı.

11 Ekim heartbeat'i “Performans skoru ... başlatma” kısıtı içerdiğinden bu
tetiklemede yeni piyasa skor raporu başlatılmadı. Bu ek kayıt yalnız
**VERIFIED fit yürütme/native bütünlük** kanıtıdır; çalışma performans
sınıfı INCONCLUSIVE ve pratik karar henüz verilmedi. Ön kayıt ve eski
kanıtlar değişmez. Model eğitimi tekrarlanmayacak.

Tek sonraki iş, skor kapsamı izin verdiğinde zaten dondurulmuş
`report.py → verify.py → controls.py` sıfır-fit zincirini çalıştırıp mevcut
22+4 deneyi kapatmak. Yeni model/grid/tarif yok. Rapor/verifier checksum'ları
ön kayıt Release'indedir; fit sonrası birincil metrik veya eşik seçilemez.

[Native yürütme Release'i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/named-capacity-execution-20261011)
taze GitHub indirmesinde 902 dosya/48 native çıkarımla doğrulandı; yeni fit0,
piyasa skor raporu yok. [Teslimat kanıtı](../research/evidence/named-capacity-execution-delivery-20261011.json).
Sicil 170 çalışma aşaması/523 tarif nesnesi içerir; 29.688 fit makbuzu
kopyası bağımsız deney sayısı değildir. Eski 168 çalışma, 520 tarif ve94
proof korunur. Paketleyicinin R1 son kontrolü append sonrası aynı nesnenin
523 uzunluğunu donmuş520 ile yanlış karşılaştırdı; bağımsız R2 frozen-prefix
kontrolü eski kayıtların değişmediğini doğruladı. Model/veri/analiz değişmedi.

## 11 Ekim: sıfır-fit raporun sayısal düzeltmesi

Hedefin devam talimatıyla aynı tarihsel deney raporu sonuçlandırılıyor;
canlı126 cohort performansı kapalı kalır. R1 rapor, piyasa sonuçları
hesaplanmadan önce sentetik native replay ölçümünde durdu: `np.exp`
float32 log tahmininde çalışırken kayıtlı `evaluate` float64'e çevirir.
Bilinen-sinyal gain farkı 3,1339628×10⁻⁷ idi; bu başarısız öğrenme veya
farklı native tahmin değildir. R2 yalnız bu replay log dizisini exp'ten
önce float64'e çevirir. Dört kayıtlı sentetik native veriyle regresyon
geçti; aynı1e−13 tolerans ve tüm tahmin değerleri korunur, yeni fit0.

R1/hash/başarısız log korunur. [Ek sayısal kayıt](../research/evidence/named-capacity-report-amendment-20261011.json)
ve checksum paketi herhangi bir yeni piyasa skorundan önce yayımlanır.
ML kaynak4a2feff…, veri, tarif, cohort, hedef, birincil metrik, bootstrap
ve karar eşikleri değişmez. Ön kayıttaki R1 hash'i değiştirilmez;
final sonuç R2 hash'ini ve bu açık düzeltmeyi birlikte taşımak zorundadır.
