# CottonLens — kanonik araştırma ve uygulama planı

Kabul edilen plan: 2026-10-01. D0 = 2026-10-01T20:06:59.863911Z;
D21 kontrolü = 2026-10-22T20:06:59.863911Z. Bunlar ilk full-year ön kaydından
gelir; yeni kaynak/deney kimlikleri bu takvimi yeniden başlatmaz.
Önceki protokoller, sonuçlar ve release'ler değiştirilmez. Bu belge onların
yerine geriye dönük protokol yazmaz; sonraki araştırmanın kararlarını belirler.

## Sabit sözleşmeler

Kalibrasyon düzeltmesi (uygulama denetimi): ilk pilotta yıllık HAR/GARCH seçimini
aynı yıl öncesindeki kalibrasyon tahminlerine uygulamak seçim zamanını ihlal eder.
Bu aralık çıktıları seçim/ürün kanıtı değildir ve korunarak geçersiz işaretlenir.
İlk geçerli normalize aralık pilotu, hem geçmiş skor hem güncel ölçek için sabit
λ=0,94 EWMA kullanır; hiçbir sonraki model seçimi bu skorları değiştiremez.
HAR/GARCH QLIKE karşılaştırması ayrı kalır. Tarihsel EWMA yeniden oynatımı canlıda
önceden arşivlenmiş tahmin gibi adlandırılmaz. Düzeltme ayrı analiz kimliği taşır;
fiyat/varyans fit'leri ve eski protokol yeniden yazılmaz.

- CT=F günlük fiyat proxy'si; T+1/T+5 sonraki kayıtlı Cotton gözlemleri.
  Fiziksel spot, doğrulanmış borsa seansı veya intraday gerçekleşmiş varyans değildir.
- Fiyat gate: ≥%5 Naive fiyat-MAE kazancı, yön T+1 ≥%53/T+5 ≥%55,
  yeni sekiz blokta ≥6 kazanım; eski dört blokta ≥3 kazanım.
- 2024+ sadece seen historical audit; seçim, feature, cadence tuning girdisi değildir.
  Yeniden kullanılan 2016–2023 tarihsel blokları da bağımsız holdout değildir.
- Yerel CPU: Ridge/ElasticNet/küçük XGBoost/EWMA/HAR/GARCH, tek iş ve ≤2 thread.
  Açık `COTTONLENS_ALLOW_LOCAL_CPU_TABULAR=1` gerekir. CI gerçek eğitim yapamaz.
  GPU/sequence ve `cottonlens-train` yalnız Colab; CPU'ya sessiz GPU dönüşü hata.
- Eski DB, preservation backup, R2, snapshot ve deneyler korunur. Commit/push yok.
  Veri bütçesi 0 TL; ayrıca onaysız ücretli kaynak yok. Backend eğitim paketi almaz.

## 2026-10-02 — dış incelemeler sonrası sonraki yürütme kararı

ChatGPT/Perplexity ekleri öneri olarak değerlendirildi. Birincil kaynaklar ve
gerçek kodla uyuşmazlıklar `RESEARCH_DIRECTIVES_REVIEW_20261002.md` içinde.
**Bu bölüm bundan sonraki iş sırasını belirler; aşağıdaki eski ön kayıtlar,
kararlar ve ölçümler geçmiş kanıt olarak aynen korunur.** Ana hedef, başarı
gate'leri, Kademe A/B, 2024+ seçimsiz audit ve bütçe sözleşmesi değişmez.

### Yeniden başlatılmayacak işler

Full-year fiyat/shrinkage ve düzeltilmiş aralık pilotu tamamlandı: P−/U−.
AMS, FAS, NASS, WASDE, CFTC, FX, mısır/soya ve hava paketleri yalnız parser
değil, modellenmiş ve değerlendirilmiş kontrollere sahiptir. Kendi dondurulmuş
kurallarında kaynak önceliğini geçmediler. AMS/WASDE kısa cohort'ları ayrı kalır.
Bu sonuç tüm model/kaynak evreninin veya T+1'in işe yaramadığını göstermez.
CFTC pilotu managed-net/OI, producer-net/OI ve managed-share değişimini zaten
denedi; bunlar yeniden “yeni feature” sayılmaz. Yeni türev/model/loss ancak
eski denemeden somut farkı ve nedenini yazan ayrı hipotezle açılır.
Yeni adla aynı split'i kurmak holdout üretmez. E0 yeniden fit edilmez; aynı
kimlikte kayıt yüklenir. Gerçek veri/cohort değişirse yeni karşılaştırmanın
base/control fit'leri gerekebilir ve eski skorlarla aynı sıralamaya konmaz.

### 1. Canlı kanıt engeli ve iki sınırlı kaynak kontrolü

Öncelik, mevcut Windows görevinin Access denied engelidir. Yeni logger/runner,
sunucu veya paralel scheduler yazılmaz. Mevcut task/collector için kayıt,
başarılı gerçek invocation, cutoff, lock, duplicate ve missing receipt'i
doğrulanmadan “otomatik toplama aktif” denmez. Naive + sabit EWMA referansları
korunur; HAR veya fiyat challenger ayrı lock/start_origin ile ancak doğrulanınca
eklenir. Kaçırılmış tahminler backfill edilmez; 126-origin hedefi olgunlaşmadan
ara performans aday seçimini yönlendirmez. Bu operasyon engeli, kaynakların
salt-okuma erişim incelemesini durdurmaz.

İlk kaynak işi **Cotton On-Call**: bir resmî indeks ve en fazla altı eski/normal/
gecikmeli rapor; 0 TL, 0 fit. Ön kayıt
`output/reports/weather-exploration-20261002/post-diagnosis/next-hypothesis.json`.
As-of, gerçek release UTC, indirme receipt'i ve delivery_month ayrı tutulur.
DST, kontrat birimi/toplam, sıfır OI, revizyon ve eksikler kontrol edilir.
Printed release mevcut payload'ın ilk sürümü olduğunu tek başına kanıtlamaz.
İlk örnek başarılı diye bütün arşiv temiz veya model katkısı var sayılmaz.

Ayrı, bounded kontrat erişim kontrolü: önce mevcut smoke kayıtları tekrar
kullanılır; gerekirse en fazla üç tekil kontratta küçük public sample ile
close/settlement ayrımı, eşzamanlı en az iki vade, tarih kapsamı, volume/OI
yayın zamanı ve kullanım hakları belirlenir. Ücretsiz tam 2010+ geçmiş varsayılmaz.
Barchart/Yahoo üyeliği, trial veya ödeme otomatik açılmaz. Eğri eksikse açık
eksik olarak kaydedilir; On-Call için zorunlu bağımlılık değildir. Tek kontrol
raporunda erişilebilir alanlar, eksik tarih/alanlar ve devam/dur kararı yer alır.

### 2. İlk kullanılabilir yeni bilgiyle küçük karşılaştırma

2026-10-02 uygulama kontrolü: canlı Windows görevi yalnız mevcut kullanıcıya
bağlı logon trigger ile kaydedildi; native stderr nedeniyle süreç kesilmesi
ayrıca düzeltildi. İki gerçek scheduled invocation exit 0; aynı forward origin
tekrarlanmadı, kaçırılmış pencere `missing` kaldı. Yerel USDA API secrets hâlâ
eksik; public kaynak toplaması bu durumdan bağımsızdır. Görev kullanıcı oturumu
açık ve bilgisayar çalışırken yürür; 24/7 veya cloud-sync garantisi değildir.

On-Call feasibility tamamlandı: bir resmî indeksin bytes'ı ve altı rapor
(2001 legacy, 2016 kış/yaz, 2020, 2023, gecikmeli 2025) SHA receipt ile saklandı;
her raporda altı kontrat kolonunun toplamı doğrulandı. 2016 yaz OI başlığında
tarih yazım hatası; 2020/2023 footer'larında çelişen ek tarih; legacy raporda
revize geçmiş değişimler ve 50/100 kontrat reporting threshold farkı bulundu.
`release after` yalnız exclusive lower bound'dur; exact publication veya
specific-vintage availability kanıtı değildir. Bütün örnekler model admission
kapalı/Kademe A kalır; 2025 örneği yalnız format/gecikme kontrolüdür.
`sources/oncall.py` audit CLI ve 28 focused test geçti. Kanıt raporu:
`output/reports/live-oncall-feasibility-20261002/completion.md`.

Üç mevcut Yahoo kontrat snapshot'ı yeniden kullanıldı: ortak 503 tarih,
2024-10-02–2026-10-01, OHLCV var/OI yok/settlement doğrulanmadı. Bu 2024+
metadata kontrolü model seçimi yapmaz; 2016–2023 tarihsel eğri kanıtı değildir.
Eğri hattı açılmadı. Sonraki iş, On-Call'ın 2016–2023 erişim/gap/revizyon
envanteri ve gecikme stresli **ayrı Kademe A** küçük T+5 pilotunun ön kaydıdır;
tam arşiv/feature freeze doğrulanmadan eğitim başlamaz. Diğer kaynak pilotları
tekrarlanmaz; cadence/gate/bütçe aynı kalır.

On-Call tarihsel envanter sonucu (2026-10-02): 2010–2023 resmî indeksteki
710 rapor indirildi; 700 rapor ayrıştırıldı, 10 kaynak biçim/tarih sorunu
karantinada. 253 rapor girdisi maskeli (229 çelişen footer tarihi,18 OI tarihi,
6 eksik release footer);447 raporun seviye girdisi kaldı. Ham sürümler korunur.
2016–2023 aynı2006 T+5 origin/sekiz yıl donduruldu;lag1/2/6,yedi grup,
Ridge alpha1/küçük XGBoost,5 purge/21 refit/üç63 geçmiş iç blok değişmedi.
Maksimum2534 fit işi yalnız bütçe hesabıdır;gerçek fit sayısı0.
**Kaynak readiness blocked:** lag1 seviye kapsamı2018–2023 yıllarında
54/251,46/252,36/253,25/252,65/251,63/246;2021 haftalık değişim kapsamı0/252.
Bu tabloyla uzun pilot açılmaz;prepare başarısı eğitim izni değildir.
Motor,pilot öncesi ready/table hash'ine bağlı onaylı source-readiness kaydı
ister;eksik/blocked/değişmiş kayıt hiçbir fit başlatmaz. İlk metadata-only
freeze,coverage kontrolü eklenirken ayrı source-staging kimliğinde korundu.
Sonraki sınırlı iş çelişen footer alanlarının anlamını kaynak düzeyinde
doğrulamaktır;URL günüyle yayın tarihi tahmin edilmez. Çözülemeyen kapsam
engeli sürerse bu bilgi hattı bekletilir,önceden planlanan kayıtlı tahmin
hata teşhisine dönülür;aynı veride geniş GPU araması yapılmaz.
Kanıt:`output/reports/oncall-exploration-20261002/completion.md`.

On-Call availability varsayım revizyonu (2026-10-02):229 footer uyuşmazlığında
ek tarihlerin221'i release tarihinden1 gün,8'i2 gün önce;bir raporda ayrıca
5 gün sonraki tarih var. Etiketsiz alanın anlamı doğrulanamadı;oluşturma/yayın
tarihi diye yeniden adlandırılmaz. İlk maske sayısal girdileri de gereksiz yere
eliyordu. Ayrı `oncall-exploration-v2` yalnız **Kademe A** varsayımıyla
`max(asof+7 gün,release lower-bound UTC günü+1,her ek footer günü+2)` kullanır;
ilk kesin sonraki Cotton observation ve lag1/2/6 korunur. +2,varsayılan Eastern
günün sonrasını UTC'de de geçer;first-vintage/gerçek publication kanıtı değildir.
Tarih uyuşmazlık işaretleri/ham receipts tutulur;OI/quantity/threshold/eksik
release maskeleri ve10 karantina aynen kalır. Eski v1 paket/blocked kaydı değişmez.
700 rapordan676 seviye girdisi kaldı;24 maske(18 OI,6 footer eksik).
2016–2023 lag1 seviye kapsamı235/250,246/251,206/251,207/252,253/253,252/252,
223/251,246/246;2021 haftalık değişim252/252. Aynı2006 origin,tarif/gate/cadence
değişmedi. Kaynak readiness sadece sınırlı keşif pilotuna geçti;release kapalı.
İlk CPU çalışması iki dakika/tek süreç/en fazla iki thread ile süre/resume
ölçümü içindir;tamamlanmamış pilotun skoruyla aday seçilmez.
Kanıt:`output/reports/oncall-exploration-v2-20261002/completion.md`.

On-Call için ilk paket net satış/alış-OI oranı ve haftalık değişim gibi küçük
kontrat temelli bilgi olacaktır; değişken/lag sırası fit öncesi sabitlenir.
Kontrat eğrisi gerçekten erişilebilirse base/base+curve ayrı hipotezdir.
Mevcut 24-feature base, age/missing kontrolü ve aday aynı origin'lerde; ilk
pilot mevcut Ridge alpha1/küçük XGBoost ve geçmiş shrinkage bütçesini korur.
Coverage, kısa ortak cohort ve doğal availability ayrı raporlanır.
T+5 ilk kontrol; T+1 sonucu bununla var sayılmaz, kendi ön kaydı gerekir.
Kaynak eklendikten sonra dört başka kaynağı peş peşe otomatik açma kuralı yok.

Futures curve: seçim yalnız karar anında bilinen takvim/likiditeye dayanır.
OI/volume final değeri aynı gün biliniyormuş gibi kullanılmaz. Kontrat roll
edince farklı sembollerden fiyat farkı, aynı kontrat getirisi gibi hesaplanmaz.
`s12=log(F1/F2)`; signed yıllık slope `365*s12/(tau2-tau1)` açık bir proxy'dir.
Maturity aralıkları eşit değilse curvature için normalize slope farkı gerekir.
Spot-futures basis, futures spread ve gerçekleşen roll getirisi ayrı kavramlar.
Beş/20 günlük spread değişimi, Boons–Prado'nun 12 aylık nearby-strateji momentum
farkı değildir; adı proxy olur. Monthly-return literatürü günlük Cotton gate'ini
garantilemez. CT=F primary korunur; aynı-kontrat label veya T+21 fikri kabul
edilirse hedef/purge/benchmark ayrı protokol ister ve şu anda açılmamıştır.

P+ ve kaynak öncelik eşikleri değişmez. Roll duyarlılığı zorunlu rapordur;
ex-post kötü origin çıkarılarak yeni gate yazılmaz. Kademe A'nın +1/+5 gecikme
stresleri ve vintage sınırı korunur; gate/release kanıtı olamaz. Ön kayda rağmen
yeniden kullanılan dış bloklar bağımsız test olmaz. Dış sonuçlar geçmiş bir
fold'un reçetesini değiştiremez; yeni fikir yeni research kimliği ve sınırlı
bütçe taşır. Bilgiye veya açık model-varsayımı hipotezine katkı yoksa arama büyümez.

### 3. Koşullu modeller, panel ve risk

Önce en fazla üç expert (Naive dahil) ve statik/global shrinkage kontrolü;
ardından tek geçmiş-olgun-kayıp ağırlığı hipotezi. Expert'ın geçmiş OOF tahmini
o tarihte seçilebilir recipe ile üretilmiş olmalı; sonradan seçilen modeli
eski origin'lere uygulayıp canlı/causal OOF diye sunma. Ex-post roll veya gelecek
büyük-hareket etiketi online gate girdisi değildir. Az örnekli regime'de global
ağırlık kullanılır; adaptation'ın statik kontrol karşısındaki katkısı ayrı ölçülür.

Lead-lag ile panel pooling ayrıdır. Yeni emtia feature'ı Cotton satır sayısını
artırmaz; pooling eğitim satırını artırsa da bağımsız Cotton test sayısı aynı
kalır. İlk küçük panel üç emtia (Cotton/mısır/soya), yalnız erişim/target kimliği
kontrolleri ve katkı gerekçesi sonrası; Ridge/ElasticNet önce gelir. Bütün
varlıkların bar/etiketleri aynı dış UTC cutoff'tan önce bilinmelidir.

Yeni bilgi/risk hattı gerekçelendirirse DLinear ve tek exogenous aday (örneğin
TimeXer), ardından uygun tek foundation adayı açılır. Kronos fiyat veya TTM
varyans için koşullu keşiftir; aynı anda dokuz aile/7 günlük A100 işi değildir.
Model repo commit'i, checkpoint revision/SHA, kod ve ağırlık lisansı, pretraining
kapsamı ayrı kaydedilir. TimesFM ≤2.5 ayrı aday olabilir; indirilen 3.0 ağırlıkları
non-commercial/non-production olduğu için otomatik demo/runtime yolu açılmaz.
Pretraining kapsamı belirsiz tarihsel sonuç keşiftir; bağımsız forward gerekir.
L4/A100 seçimi ancak smoke throughput/bellek ölçümüyle; mevcut tabular CPU sınırı
değişmez. Fine-tune büyük paneli veya lisansı varsayarak otomatik başlamaz.

Risk hattında yalnız açık yeni hipotez (örneğin doğrulanmış On-Call + HAR-X)
eski EWMA/HAR/GARCH ve düzeltilmiş aralıklarla aynı hedefte karşılaştırılır.
Günlük kare-getiri vekili intraday RV değildir. Pinball/coverage/width/interval
score birlikte; iyi aralık fiyat başarısı yerine geçmez. NOAA geçmiş sürümleri
revize edildiğinden POWER yerine NOAA almak kendiliğinden Kademe B sağlamaz.

### Teslim ve sınırlar

Bu değişiklikte araştırma kodu/notebook/deney kimliği değiştirilmez. İki workbench,
tek ledger, immutable kaynak ve ayrı demo DB/artifact sınırı korunur. 0 TL veri
bütçesi geçerli; 1000 TL ek belge önerisi kabul edilmedi. Cotcast'e mesaj veya
demo scraping başlatılmaz. Ortak hedef/origin/cutoff olmadan sayısal uzaklık veya
üstünlük sözü yok. D21 raporu yalnız tamamlanan işleri, kaynak/operasyon açıklarını
ve sınırlı sonraki hipotezi içerir; tasarlanan işler yapılmış gibi anlatılmaz.

## Aşama 1 — D0–D5: kanıt, protokol, ileri kayıt

`research/review.py --extended` kayıtlı tahminleri okur; yeni model fit etmez.
Fiyat/log-return OOS R², Spearman IC, Naive/median-return, geçmiş majority-direction,
eşleştirilmiş yön farkı, tahmin dağılımı; ay/yıl, fiyat, volatilite, eksiklik ve
büyük hareket kırılımları çıkarılır. Fold sınırları korunarak 10.000 paired bootstrap;
ana blok 20, duyarlılık 10/40. Güç/MDE kayıp farkının bağımlılığından hesaplanır;
normal MDE yalnız yaklaşık, post-hoc ve model seçimine göre düzeltilmemiştir.
0,92 SE ve 2,6 puan MDE sabit gerçekler değildir. Eşik değiştirmek için kullanılmaz.
Clark–West genel gate değildir; ancak önceden tanımlanmış uygun iç içe lineer MSE
karşılaştırmalarında tamamlayıcıdır. Ridge otomatik uygun sayılmaz.

Roll denetimi: OHLC, duplicate, sıfır hacim ve geçmişten tanımlanan sıçrama işaretleri.
Doğrulanmış kontrat değişimi ve şüpheli roll ayrıdır. Ana cohort bütünü korunur;
işaretli hedef pencerelerini dışlamak yalnız ex-post duyarlılıktır, predictor değildir.
Fiyat geriye düzeltilmez, büyük hareket/sıfır hacim otomatik silinmez.
ICE listesi March/May/July/October/December kontrat aylarını belirtir;
bu aylar veya vade sonları Yahoo'nun gerçek roll tarihi sayılmaz.
[ICE ürün sözleşmesi](https://www.ice.com/products/254).
OHLC frozen feature dosyasında yoksa bu açıkça raporlanır; GK başarı ölçüsü uydurulmaz.

Yeni `research-full-year-v1`:

- 2016–2023 bütün geçerli fiyat ve olgun T+5 hedef origin'leri; hedef 2024'e taşmaz.
  Feature eksikliği origin silmez; preprocessing training'de öğrenilir.
- Sekiz yıllık dış blok; her yıl başından önce üç ardışık 63-origin iç blok.
  Beş kayıtlı gözlem purge; etiket cutoff'tan önce bilinmelidir.
- Yıl başında geçmiş iç sonuçlarla seçim; yılın dış sonuçları reçeteyi değiştirmez.
- Cadence 21 kayıtlı gözlem, yaklaşık aylık operasyon varsayımı; pilotta aranmaz.
- Kesin tarih listeleri, label/source/lock hash'leri ve feature sırası fit'ten önce
  `ready.json` ile dondurulur. Karar saati sonraki kaynak bar tarihinin 00:15 UTC'si.
- Kısa kaynak ayrı ortak cohort kullanır; cohort'lar tek sıralamada karıştırılmaz.

Canlı ilk referanslar Naive + EWMA λ=0,94. `research/prospective.py` v2 ayrı lock,
ilk 126 uygun yeni kaynak origin'i, `missing` kayıtları, input checksum ve hash zinciri
tutar. Origin'ler ileri kaynak akışıyla çözülür; bilinmeyen gelecek borsa takvimi
uydurulmaz. Eksik origin başka tarihle değiştirilmez. Gecikmeli yeni eski origin
tespit edilirse cohort review gerekir. Tahmin cutoff 00:15; yayın 00:15–00:30 UTC.
Bu saatten önce gözlenmiş tamamlanmış kaynak barı yoksa yayın yapılmaz. Bilgisayar
kapalıyken kaçırılan kayıtlar missing olur, tahmin backfill edilmez. Performans bütün
126 origin'in T+5 hedefleri olgunlaşınca açılır; öncesinde yalnız durum/kapsama izlenir.

`research/live.py` ve Windows task: logon + saatlik (:05 UTC) kaynak toplama,
ayrıca 00:20 UTC yayın. Tek süreç, IgnoreNew; yerel veri önce, Drive kopyası sonra.
CFTC/NASS/WASDE/FAS/AMS ve tekil Cotton kontratları mümkün oldukça arşivlenir.
API anahtarları yalnız ortam/Colab Secrets; yerelde yoksa kaynak status'ta açıkça eksik.
Ham sürüm + request/download zamanları + checksum; aynı içerik deduplicate,
erişimler ayrı receipt. Bugünkü indirme tarihi ilk yayın tarihi değildir.
O sürümün gelecekte kullanımı için üst availability sınırıdır; geçmişe taşınmaz.
Arşiv kaynak erişimi, vintage/model admission ve yeniden dağıtım hakkı ayrı kararlardır.
Hash zinciri dış attestation değildir. Drive readback bulut senkronizasyon onayı değildir.

Aşama sonu: ek ölçümler, roll/erişim belirsizlikleri, frozen protocol, canlı arşiv,
kesin pilot iş bütçesi. Veri/etiket hizası problemi varsa pilot açılmaz.

## Aşama 2 — D6–D14: altı hipotezli küçük pilot

Aynı mevcut 24 feature, aynı origin, seed 42:
Ridge α={0,1;1;10}; ElasticNet α={0,001;0,01}, l1_ratio=0,5;
tek XGBoost depth2/eta0,03/child20/L1=0/L2=1, max600/price-MAE patience50.
Training-only target scaling/inverse; shrink w={0;0,25;0,5;0,75;1} yalnız geçmiş
iç tahminlerde seçilir. Eşitlik basit model, ardından küçük w. Eski dış kazanan
ayarları daha erken tarihlerde geçmişten seçilmiş gibi kullanılmaz.

Varyans: sabit EWMA, Log-HAR(1/5/22 günlük kare getiri özetleri), sıfır ortalama
Student-t GARCH(1,1). Ana hedef gelecekteki h günlük kare getiri toplamı.
Geçerli ortak OHLC varsa GK sadece duyarlılık. Geçersiz/non-convergent aday
seçilemez; eksik iş başarısız hipotez sayılmaz.
[arch modeli](https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_modeling.html).

Aralıklar: Normal-EWMA, son252 olgun h-getirisinin ampirik kuantilleri,
geçmişte hesaplanmış tahmin σ'sına normalize252 olgun skor.
%80/%90: gözlenen getiride pinball/coverage/genişlik/interval score; fiyat karşılığı
ayrı rapor. Calibrator geleceği veya henüz olgunlaşmamış hedefi kullanmaz.
252 geçmiş skor olmadan tam sonuç yok. QLIKE = log(v̂)+v/v̂; sıfır v geçerli,
v̂ pozitif olmalı. QLIKE'nin negatif olabilmesi nedeniyle yüzde-kayıp oranı kullanılmaz.

| Hipotez | Karşılaştırma | Ölçüm |
|---|---|---|
| H1/H2 | Shrunk price / Naive, T+1/T+5 | price-MAE |
| H3/H4 | İçte seçilen HAR/GARCH / EWMA, T+1/T+5 | QLIKE |
| H5/H6 | Normalize aralık / Normal-EWMA, T+1/T+5 | %90 interval score |

Bootstrap CI + altı hipotez BH düzeltmesi; %80 ve ampirik referans zorunlu ikincil.
Ön kayıt görülmüş veriyi bağımsız yapmaz. Her horizon ayrı karar:
P+: iç MAE ≥%0,5 kazanım, dış aggregate pozitif, ≥5/8 kazanım.
U+: coverage90 %87–93 ve80 %77–83; intervalscore90 ≥%2 iyi, ≥5/8 kazanım,
ampirik kuantilden kötü değil. Coverage CI ayrıca; koşullu calibration garantisi yok.
Volatilite kararı ayrıca, aralık sonucu onun üstünlüğü değildir.

| Karar | Sonraki iş |
|---|---|
| P+U+ | Fiyat + risk |
| P+U− | Fiyat, sınırlı aralık düzeltmesi |
| P−U+ | Naive fiyat + doğrulanmış risk |
| P−U− | Geniş arama yok; tek bilgi/target teşhisi |
| Eksik | Karar bekliyor; negatif sayılmaz |

Tek rapor: tüm adaylar, dış yıllar, süreler, model/seed/feature/source kimlikleri,
altı test, belirsizlikler, sonraki sınırlı bütçe. P+/U+ ürün gate'i değildir.

## Aşama 3 — D15+, D21 kontrolü: koşullu büyütme ve release

İlk negatif pilot sonrası tek bilgi kontrolü `ams-exploration-v1` profilidir.
999 AMS 2020–2023 alıntısından 998'i MMN içerik hash'iyle eşleşir; eksik metadata
raporu varsayımla tamamlanmaz. UTC/ilk sürüm belirsizliği devam ettiği için yalnız
Kademe A'dır: rapor ve görüntülenen yayın gününün büyüğü sonrasındaki 1/2/6 Cotton
gözlemi; böylece temel gecikmeye +1/+5 stres uygulanır. Bunlar gerçek yayın zamanı
olarak kaydedilmez. Üç ek gözlemden eski taşıma eksik işaretlenir. Her gecikmede
aynı 24-feature base + age/missing kontrolü + spot/proxy quotation spread kıyası.
Ridge alpha1 ve tek küçük XGBoost reçetesi geçmiş iç bloklarda seçilir; geniş arama
yoktur. Kaynak başlangıcından üç63 iç blokta en az500 geçmiş satır şartı korunur;
bu arşivde yalnız2023 tam-yıl dış bloğu yeterlidir. Bu ayrı kısa cohort, sekiz
yıllık gate veya release kanıtı değildir. As-published compiler/importer sözleşmesi
gevşetilmez; bu profil lock/export'u reddeder ve Naive birincil kalır.

Tek kaynak grubu sırası AMS/FAS → NASS/WASDE → CFTC → emtia/FX → iklim/uydu/metin.
Erken kaynak doğrulanamıyorsa diğer araştırma sonsuza kadar beklemez.
Kademe A belirsiz vintage/takvim/lag: yalnız keşif ve +1/+5 gözlem stress;
release/gate/forward başarısı kanıtı olamaz. Kademe B belirli sürüm için gerçek
yayın/availability kanıtı; bugünkü gözlem ancak sonrası için kullanılabilir.
Base/base+group/drop-group/missingness-only aynı origin'lerde. İlk16 aday/aile/horizon/fold;
iç göreli kayıp ≥%0,5 katkı ve ≥5/8 iç blok olumluysa büyüt.
[CFTC istisnaları](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm).

32→128 aday; en iyi3 seed17/42/101. İki turda ≥%0,5 iç ilerleme yoksa aynı alan
büyütülmez. L4 ilk; A100 ölçülen süre/bellek veya panel/sequence gerekçesiyle.
DLinear → gerekçeli exogenous. TTM/Kronos zero-shot keşif; pretraining bilinmiyorsa
temiz bağımsız tarihsel kanıt sayılmaz. Panel/fine-tune ucuz kontrol katkısından sonra;
başka emtiaların gelecek etiketleri dış Cotton cutoff'una sızmaz.
TTM+Log-HAR birincil hipotez değildir; intraday RV bulgusu günlük kare getiriye eşit değildir.

Fiyat gate geçerse birincil; aksi Naive. Audit reçete değiştirmez. Yeni aday ayrı
forward başlangıcı; eski cohort/predictions korunur.126 origin güç garantisi değildir.
Aralık/coverage → Naive skill → kaynak durumu → live coverage → açıklama/olay paneli.
Nokta/aralık ayrı model ID; fiyat yönü sınıflandırıcı olasılığı gibi gösterilmez.
Yeni schema gerekiyorsa exporter/checksum/importer/runtime/contracts birlikte değişir;
eski24kolon/release korunur; demo ayrı DB/artifact. Backend training paketi almaz.
Cotcast metriği hakkında çıkarım yok;1−MAPE kendi tahminlerimizin yalnız ek tablosudur.

AMS küçük pilotu (2026-10-02): 2023'te 246 ortak origin, yedi paket ve 231 tamamlanmış
fit. Geçmiş iç seçim bütün paketlerde shrinkage w=0 seçti; yayımlanan fiyatlar Naive
ile aynı. Spread paketlerinin küçültülmemiş MAE kazancı −%1,435/−%1,205/−%0,054
(lag 1/2/6); bunlar yalnız teşhis, yeni aday seçimi değildir. Aynı AMS alanı büyütülmez.
Tek sonraki bilgi hipotezi FAS haftalık satış/sevkiyat değişimleridir; vintage kanıtı
yoksa yalnız Kademe A, gecikme stresleriyle ve ayrı ön kayıt/cohort kimliğiyle açılır.

FAS ön kayıt (2026-10-02): `research-fas-exploration-v1`, 752 haftalık güncel API
kaydı, yalnız 2024 öncesi gözlemler; T+5 için sekiz tam yıl ve yedi paket.
Net satış/sevkiyat seviyeleri ve aynı pazarlama yılı içindeki 7-günlük değişimler;
yıl sınırları/gap'ler üzerinden fark alınmaz, country/week çiftleri tekildir.
Varsayımsal availability hafta sonu+7 takvim günü, sonra Cotton observation lag1/2/6;
bu gerçek yayın zamanı kanıtı değildir. Base ve age/missing kontrolleri aynı origin'lerdedir.
AMS ile aynı tek Ridge/fixed küçük XGBoost ve geçmiş shrinkage seçimi; maksimum2534 fit,
tek CPU süreç/iki thread. Hiçbir dış sonuç ayar değiştirmez; vintage belirsizliğinden
release/gate kapalıdır. Tamamlanan deneyler yeniden eğitilmez.

NASS ön kayıt (2026-10-02): `research-nass-exploration-v1`; 2010–2023 arşivinden
311 ulusal Upland Cotton haftası (308 sayısal eşleşme, 3 PDF'de gerçek sıfır).
Good+excellent, poor+very-poor ve yalnız aynı sezonun ardışık haftalarında
good+excellent değişimi; yüzdeler sabit oran birimine çevrilir. Yayın sayfasının
gününden sonraki Cotton lag1/2/6; UTC/ilk vintage doğrulanmadığı için Kademe A.
Rapor en çok 10 ek Cotton gözlemi taşınır, kışa/yıl sınırına taşınmaz; bütün
origin'ler korunur. T+5 için aynı base ve age/missing kontrolleri, sabit iki
küçük model ve geçmiş shrinkage seçimi. İlk fit öncesi protokol/kod/veri ve
araştırma önceliği kuralı dondurulur: her lag'de base ve kontrol karşısında
≥%0,5 iç kazanç ve ≥5/8 iç kazanım, pozitif dış kazanç ve ≥5/8 dış kazanım.
Bu release gate değildir. Aynı alan genişletilmezse sonraki tek kaynak WASDE'dir.

FAS ve NASS kararları (2026-10-02): iki ayrı pilotun her biri 56/56 çıktı ve
1855 tamamlanmış fit ile kapandı; kendi dondurulmuş base/kontrol kıyaslarında
araştırma önceliği koşulları geçilmedi. NASS kondisyon lag1/2/6 için T+5 Naive
MAE kazancı −%0,692/−%1,057/−%0,879; sekiz yılda kazanım 1/3/2 yıl.
5565 NASS payload hash'i doğrulandı, cache replay yeni fit üretmedi. Bu
tarif/protokolde katkı kanıtı yok; kaynağın genel olarak faydasız olduğu iddiası
değildir. Aynı satış/kondisyon arama alanı büyütülmez. Sonraki tek hipotez WASDE
pamuk stok/kullanım ve arz-talep revizyonları; önce içerik/availability incelemesi,
sonra ayrı ön kayıtla küçük kontrol. Kademe A hiçbir release/gate kanıtı değildir.
Kanonik deney kayıtları: `output/reports/fas-exploration-20261002/completion.md`
ve `output/reports/nass-exploration-20261002/completion.md`. Frozen kaynak ZIP'leri
ilk fit öncesindeki belgeyi korur; bu sonuç notu o paketlerin üzerine yazılmaz.

WASDE ön kayıt (2026-10-02): `research-wasde-exploration-v1`, mevcut 2016–2023
arşivinden World pamuk stok/kullanım ve üretim/kullanım oranları, aynı mahsul
yılındaki geçmiş sürüme göre stok/kullanım değişimi. 98 sayısal kontrolü geçmiş
bağlantı, aynı tarih/ayda değişmemiş içerik tekrarları birleştirildiğinde 95 gözlem;
2019-01 için rapor uydurulmaz. Güncel rapor ayının son mahsul yılı projeksiyonu
seçilir; eski yılın gerçekleşmiş değeri veya yeni mahsul yılını aşan fark kullanılmaz.
Yayın sayfası günü sonrası Cotton lag1/2/6, en çok40 ek gözlem taşıma, yalnız
Kademe A. En az500 geçmiş kaynak dönemi satırı ve üç63 iç blok şartı nedeniyle
2019–2023 ayrı beş yıllık cohort; eski sekiz yıl skorlarıyla tek sıralama yapılmaz.
Aynı iki küçük sabit model ve shrinkage; yedi base/age-missing/balance paketi.
Kaynak-özel araştırma bütçesi koşulu ilk fit'ten önce dondurulur: her lag'de
base ve matching-control karşısında ≥%0,5 iç kazanç, ≥4/5 iç kazanım, pozitif
dış Naive kazancı ve ≥4/5 dış kazanım. Bu kısa cohort release gate'i değildir;
sabit %5/%53/%55/6-of-8 ürün eşikleri değişmez. Dış sonuçlarla ayar değiştirilmez.

WASDE kararı (2026-10-02): beş yıl/1254 origin için 35/35 çıktı ve1162 fit
tamamlandı;3486 payload hash'i doğrulandı, cache replay yeni fit üretmedi.
Bilanço lag1/2/6 paketlerinin T+5 Naive MAE kazancı −%0,746/−%0,297/−%0,619;
yıl kazanımı0/5,1/5,0/5. Üç lag'de base karşısında iç kazanç da negatif;
araştırma önceliği koşulu geçilmedi. Bu tarif/cohort için katkı kanıtı bulunmadı;
kaynağın her hedef veya modelde faydasız olduğu iddiası değildir. Mevcut World
bilanço arama alanı büyütülmez; Naive birincil kalır. Sonraki tek bilgi hipotezi
CFTC pozisyonlarıdır: önce pinned kapsam/içerik/availability kontrolü, ardından
kısıtlı ön kayıtlı kıyas. Sonuçların kimlik/protokol ayrımı:
`output/reports/wasde-exploration-20261002/completion.md`. İlk fit öncesindeki
source ZIP ve eski sonuçlar değişmeden korunur; yeni sonuç notu onları değiştirmez.

## Çalıştırma, doğrulama ve D21 teslimi

FX ön kaydı (2026-10-02): `fx-exploration-v1`, mevcut2010–2023 ECB
BRL/CNY/INR/USD yıllık arşivi; yeni satın alma/indirme yok. PKR bu kaynakta
yoktur. Aynı tarihli currency/EUR, USD/EUR'ya bölünerek currency/USD üretilir;
farklı tarihli kurlar çaprazlanmaz. Yinelenen currency/date vintage belirsizliği
compiler'ı durdurur. Genel16:00 yayın takvimi her sürümün tarihsel kanıtı değildir;
Tier A/keşif, release ve başarı gate'i kapalı.
Özellikler üç kur için1/5/21 Cotton-observation log getirisi ve21-observation
volatilite; fiyat tarihleri sıkıştırılmaz. Sonraki Cotton observation/lag1/2/6,
maksimum3 ek gözlem taşıma; boş tatil kaydı yaşı tazelemez. Kur/özellik eksikleri
train-only preprocessing'e kalır. Her kurun age/missing göstergeleri matching
kontrolde aynen korunur. Yedi paket, aynı iki küçük model, T+5/seed42/21 refit,
beş purge/üç63 geçmiş iç blok. Kaynak tek bir FX hipotezidir; emtia eklemesi ayrı.
Araştırma önceliği her lag'de base VE kontrol karşısında ≥%0,5 iç kazanç ve≥5/8
iç kazanım, pozitif dış Naive kazancı ve≥5/8 dış kazanım; üç lag birlikte aranır.
Sabit ürün eşikleri değişmez; dış sonuçlarla ikinci tarif seçilmez.

CFTC ön kaydı (2026-10-02): `cftc-exploration-v1`, Cotton033661,
Disaggregated Futures Only/all maturities. Resmî2010–2023 yıllık ZIP'ler
checksum ile sabitlenir; bunlar ilk yayımlanmış vintage değildir. Fon ve
producer/merchant net kontratları aynı raporun open interest'ına oranlanır;
fon payı farkı yalnız ardışık7 günlük, maskelenmemiş raporlar arasında.
OI tekil kontrat verisi diye adlandırılmaz. Varsayımsal erişim rapor+7 takvim
günü, ardından Cotton lag1/2/6; maksimum10 ek gözlem taşıma. Bilinen2013,
2018–19 kapanış/catch-up ve2023 ION pencereleri ile2012-11-27 ve2019-03-26
revizyon kayıtları feature olarak maskelenir; yaş tazelenmez, fiyat origin'i
silinmez. Gecikme/tatil/vintage geçmişi eksiksiz doğrulanmış sayılmaz.
Tier A, yalnız keşif; yayın/başarı/release gate'i kapalı.
Yedi base/missingness/position paketi aynı dondurulmuş cohort'ta; tek Ridge
ve küçük XGBoost, T+5/seed42/21 refit/5 purge/üç63 iç blok.
Araştırma önceliği her lag için base VE matching-control karşısında ≥%0,5
iç kazanç ve≥5/8 iç kazanım, pozitif dış Naive kazancı ve≥5/8 dış kazanım;
üç lag birlikte geçmeden değişmeyen arama alanı büyütülmez. Bu, sabit fiyat
ürün gate'inin yerine geçmez; dış sonuçla yeni tarif seçilmez.

CFTC kararı (2026-10-02): sekiz yıl/2006 ortak origin,56/56 çıktı ve1855 fit
tamamlandı;5565 payload hash'i doğrulandı, cache replay yeni fit üretmedi.
Pozisyon lag1/2/6 paketlerinin T+5 Naive MAE kazancı −%0,854/−%0,768/−%1,783;
üç lag'in iç katkısı da negatif, araştırma önceliği koşulu geçilmedi.
Kapsam730 resmî haftalık rapor,29 maskelenmiş girdi; eski cache'le ortak469
fon pozisyon kaydı eşleşti. Bu kapsam/tarif için katkı kanıtı bulunmadı;
kaynağın her hedef veya modelde faydasız olduğu söylenmez. Değişmeyen CFTC
arama alanı büyütülmez; Naive birincil kalır. Sonraki tek bilgi hipotezi ücretsiz
emtia/FX bağlamıdır: önce pinned kapsam/availability ve ortak cohort kontrolü,
ardından kısıtlı ön kayıtlı kıyas. Kesinti sonrası45 çıktı/1524 tamamlanmış fit
korundu; ölü same-host writer kilidi kontrol edilip silinmeden arşivlendi.
Kanıt: `output/reports/cftc-exploration-20261002/completion.md`.

FX kararı (2026-10-02): sekiz tam yıl/2006 ortak origin,56/56 çıktı,
1855 tamamlanmış fit ve5565 doğrulanmış payload hash'i.
Cache replay yeni fit üretmedi; bu fresh reproduction değildir.
Kur lag1/2/6 T+5 Naive MAE kazançları -0.528%/-0.566%/-0.540%;
yıl kazanımları 1/8,3/8,2/8. Üç lag birlikte araştırma önceliği: False.
Tier A/yayın-vintage kanıtı eksik; ürün gate'i değerlendirilmedi, release kapalı.
Bu kaynak/tarif için sonuç, bütün FX modelleri hakkında genelleme değildir.
Değişmeyen FX araması büyütülmez. Sonraki tek bilgi hipotezi ücretsiz mısır/soya bağlamıdır; önce kaynak/availability ve ortak cohort incelemesi, ardından sınırlı ön kayıtlı kontrol. Naive birincil kalır.
Kanıt: `output/reports/fx-exploration-20261002/completion.md`.

Mısır/soya ön kaydı (2026-10-02): `crop-exploration-v1`, Yahoo ZC=F/ZS=F
2010–2023 günlük fiyat proxy'leri; fiziksel ürün fiyatı veya temiz tekil kontrat
zinciri değildir. Ham JSON/receipt ve tablo checksum ile dondurulur. Sağlayıcının
bildirdiği timezone bar tarihi yalnız takvim varsayımıdır; tarihsel first-vintage,
yayın/close erişimi ve gerçek roll doğrulanmış değildir. Tier A/keşif; release
ve ürün gate'i kapalı, yeniden dağıtım izni varsayılmaz. Mevcut USDA/FX özellikleri
birleştirilmez: tek yeni bilgi hipotezi mısır/soya fiyat-getiri/risk bağlamıdır.
1/5/21 Cotton-observation log getirisi/21 volatilite; ilk sonraki Cotton observation
ve lag1/2/6, maksimum3 ek observation taşıma. Boş/nonpositive fiyat yaş tazelemez;
close maskelenir, sıfır hacim doldurulmaz. OHLC tutarsızlığı/sıçrama işaretlenir;
origin silme/geriye düzeltme yok. Sekiz tam yıl2016–2023,2006 ortak T+5 origin,
yedi base/missing/price paketi, tek Ridge(alpha1) ve küçük XGBoost(depth2),seed42.
Üç63 geçmiş iç blok,5 purge,21 refit, train-only preprocessing/shrinkage seçimi.
Her lag'de base VE matching-control karşısında ≥%0,5 iç MAE kazancı ve≥5/8 iç
kazanım, pozitif dış Naive kazancı ve≥5/8 dış kazanım; üç gecikme birlikte aranır.
Dış sonuçlara bakıp alternatif tarif seçilmez; ürün eşikleri değişmez.

Mısır/soya kararı (2026-10-02): sekiz tam yıl/2006 ortak origin,56/56 çıktı,
1855 tamamlanmış fit ve5565 doğrulanmış payload hash'i.
Cache replay yeni fit üretmedi; bu fresh reproduction değildir.
Mısır/soya lag1/2/6 T+5 Naive MAE kazançları -1.880%/-2.506%/-1.149%;
yıl kazanımları 1/8,0/8,1/8. Üç lag birlikte araştırma önceliği: False.
Tier A/yayın-vintage kanıtı eksik; ürün gate'i değerlendirilmedi, release kapalı.
Bu kaynak/tarif için sonuç, bütün FX modelleri hakkında genelleme değildir.
Değişmeyen mısır/soya araması büyütülmez. Sonraki sınırlı bilgi hipotezi bölgesel yağış/sıcaklık bağlamıdır; önce kaynak/availability, bölge/sezon ve ortak cohort kontrolü. Otomatik geniş GPU araması açılmaz. Naive birincil kalır.
Kanıt: `output/reports/crop-exploration-20261002/completion.md`.

Teksas hava durumu ön kaydı (2026-10-02): `weather-exploration-v1`, NASA
POWER UTC günlük PRECTOTCORR(mm/day)/T2M_MAX(C),2010–2023. Southern High Plains
Lubbock(33.6,-101.9),Hale(34.2,-101.9),Gaines(32.75,-102.65) sabit eşit ağırlıklı
örnekleri; bağımsız yer istasyonu veya eyalet/ülke üretim ağırlığı değildir.
MERRA2/POWER yeniden analiz/assimilation sürümü, tarihsel first-vintage kanıtı
yok. Genel2–3 günlük NASA latency yeterli timestamp kanıtı sayılmaz; varsayım
ölçüm UTC günü+3 takvim günü, ilk sonraki Cotton observation, ardından lag1/2/6;
en fazla3 ek observation taşıma. Tier A, release ve başarı gate'i kapalı.
7/30 tam takvim günü yağış toplamı,7 gün Tmax ortalaması ve35C üstü7 gün
excess-heat toplamı. Eşik veApril–November season flag eksploratif sabittir;
agronomik zarar/hasat modeli diye sunulmaz. Aynı dört özelliğin sezon
etkileşimleri; season flag/age/missing matching kontrolde de aynen korunur.
Bir nokta eksikse bölgesel footprint değiştirilmez; gün atma/backfill/anomali
normalini bütün geçmişten öğrenme yok. Ürün ve UTC/calendar sınırları sabitlenir.
Sekiz tam yıl/2006 ortak T+5 origin, yedi grup,35 kolon, tek Ridge(alpha1)/
küçük XGBoost(depth2),seed42; üç63 geçmiş iç blok,5 purge,21 refit ve train-only
preprocessing/shrinkage. Her lag base VE kontrol karşısında≥%0,5 iç MAE kazancı,
≥5/8 iç kazanım, pozitif dış Naive kazancı ve≥5/8 dış kazanım; üç lag birlikte
geçmeden aynı arama büyütülmez. Dış skorla tarif/eşik/coğrafya seçimi yapılmaz.
Diğer USDA/FX/emtia paketleri karıştırılmaz; tek hava durumu hipotezidir.

Teksas hava durumu kararı (2026-10-02): sekiz tam yıl/2006 ortak origin,56/56 çıktı,
1855 tamamlanmış fit ve5565 doğrulanmış payload hash'i.
Cache replay yeni fit üretmedi; bu fresh reproduction değildir.
Hava durumu lag1/2/6 T+5 Naive MAE kazançları -1.012%/-1.184%/-1.886%;
yıl kazanımları 2/8,1/8,1/8. Üç lag birlikte araştırma önceliği: False.
Tier A/yayın-vintage kanıtı eksik; ürün gate'i değerlendirilmedi, release kapalı.
Bu kaynak/tarif için sonuç, bütün iklim feature’ları veya hava durumu modelleri hakkında genelleme değildir.
Değişmeyen hava durumu araması büyütülmez. Sonraki sınırlı iş kayıtlı tahminlerde hareket/sezon/volatilite ve roll işaretleriyle hata teşhisidir; bu teşhisten tek target/model hipotezi ön kaydedilir. Otomatik yeni kaynak/geniş GPU araması açılmaz. Naive birincil kalır.
Kanıt: `output/reports/weather-exploration-20261002/completion.md`.

İki workbench korunur, küçük yeni notebook üretilmez. Research:
status → review → prepare → pilot → compare; reproduction kilitli araştırma reçetesini
yeniden doğrular, olumlu sonuç şartı değildir. Ürün lock/export ancak olumlu kanıt
ve desteklenen release sözleşmesiyle. Data: kaynak/arşiv işlemleri.
Run All default status, eğitim başlatmaz. Aynı Experiment/Ledger, checksum/resume;
CPU group TensorFlow/CUDA'sız ayrı kilitlidir. İlk tam işten süre tahmini;
4 saati aşarsa kalan aynı kimlik/ortam/ledger ile Colab'a taşınır.
Notebook/ZIP sürümlü, eski paket silinmez. Bilinçli tekrar yalnız reproduction,
new_seed veya bugfix_verification gerekçesiyle.

Zorunlu test: geleceği değiştirince geçmiş feature/split/transform/calibration/prediction
değişmez; purge/olgunluk/common-origin/missingness; interruption/checksum/transfer/lock;
forward duplicate/no backfill;CPUallowlist/2thread/CIblock/GPUfallback;
fiyat reproduction+CPUparity ≤1e−6 log-return. Varyans/aralık ayrı ölçekli tolerans.
Import/API geçici DB; UI değişirse build ve1440/1024/390px taşma0.

D21 tek karar raporu: sicil, kaynak kartları, negatif hipotezler, fiyat/aralık kararı,
live status, sınırlı sonraki iş. Doğrulanmış release ayrı demo; eksik doğrulama
araştırma raporu olarak kalır. Obsidian yalnız doğrulanmış kalıcı ders/link alır.


## Kaynak pilotları sentezi ve sonraki sınırlı hipotez — 2026-10-03

On-Call v2,56/56 çıktı ve2006 ortak T+5 origin ile tamamlandı. Naive MAE kazancı lag1/2/6 için −%0,272/−%0,026/−%0,385; bütün eşleştirilmiş ana aralıklar sıfırı kapsıyor. Yayın/vintage kanıtı hâlâ yok; release kapalı.

Dokuz kaynak pilotunun63 grubu/434 grup-yıl kaydı aynı değerlendirmede incelendi. Yedi kaynakta date/price/target vektörleri aynı2006 origin; AMS246 ve WASDE1254 ayrı cohort. Kaynaklı küçültülmemiş tahminler de aggregate MAE kaybediyor; sıfır ağırlığı kaldırmak çözüm değil. XGB seçimleri genel olarak tek ağaç değil. Sakin hareketlerdeki fazla hata yalnız teşhis; sonradan origin seçimine dönüşemez.

Tek sonraki hipotez: aynı24 feature/altı mevcut küçük fiyat reçetesi/iki horizon/inner seçim/purge/cadence ile eğitim geçmişini sabit üç takvim yılına sınırla. Tam geçmiş kontrolünün16 mevcut fiyat çıktısı hash ile donduruldu ve tekrar eğitilmeyecek. Yeni model, veri ve ortam kimlikleri ilk fit öncesi ayrı deneyde bağlanmalı. Tasarım kaydı henüz çalıştırılabilir runner değildir. Başarı gate’leri değişmez; negatif sonuçta başka history pencereleri denenerek alan genişletilmez. Bütçe ve kontrol hash’leri: `output/reports/research-synthesis-20261003/recency-preregistration.json`; kanıt: aynı dizindeki `completion.md`. D0/D21 değişmez.


### Recency runner hazır — 2026-10-03
`recency-pilot-v1` mevcut Experiment/Ledger ve full_year fiyat fonksiyonlarına bağlandı. Kontrol16 çıktı/karar/history byte ve hash’leriyle taşınır; control fit=0. Aynı workbench pilot akışında prepare ve compare otomatik, Run All varsayılanı status. En çok1106 yeni küçük fit/16 output; bir süreç/iki thread/60 dakikalık oturum.26 contract test ve scoped Ruff geçti; prepare/status/pilot-plan salt metadata ile doğrulandı, henüz gerçek yeni fit yok. Kaynak paketi `source-recency-pilot-v1-20261003.zip`; işlem kimlikleri/teslim kanıtı `output/reports/recency-implementation-20261003/completion.json`.

### Recency Colab sonucu ve karar — 2026-10-03
Deneme Colab’da16/16 çıktı ile tamamlandı; her horizon2006 aynı origin. Naive MAE kazancı T+1 −%0,1983 (2/8 yıl), T+5 −%1,4770 (1/8 yıl). Tam geçmişe karşı iç kazanç −%0,0987/+%0,3661; iç yıl kazanımı4/8 ve3/8. Araştırma önceliği ve fiyat gate’leri iki horizon’da da geçmedi. Üç yıllık geçmiş hipotezi kapanır; alternatif history pencereleriyle aynı arama büyütülmez. Küçültülmemiş tahminler de −%2,6523/−%10,2841 MAE kazancı veriyor: ağırlığı kaldırmak çözüm değil. Naive birincil; release kapalı.

18 metadata paketi/archive-member checksum’ı,16 karar/tahmin, ortak origin ve kontrol bağları doğrulandı. Paylaşılan çıktı kayıtlı raporla aynı; mevcut compare ile yeniden hesaplamada en büyük fark1,11e−16, kategorik kararlar aynı. Model payload’ları yeniden doğrulanmadı; fresh reproduction/release parity yapılmadı. İnceleme sırasında yeni eğitim ve kontrol refit’i0. Kanıt: `output/reports/recency-colab-review-20261003/completion.md`. Sonraki sınır: yeni fit bütçesi açmadan mevcut iç kararların zaman içindeki istikrarı ve iç/dış farkının değerlendirilmesi; tek farklı hipotez ve eşlenmiş kontrol hazırlanması. D0/D21 ve canlı kayıt değişmez.

### Seçim istikrarı ve tek yeni temsil hipotezi — 2026-10-03
162 mevcut fit kaydından tam geçmiş kontrolünün48 iç bloğu/16 yıllık seçimi yeniden kuruldu, seçilmiş iç skorlar doğrulandı; eğitim0. T+1 2017: üç iç blokta kazanan seçim içte+%1,472, dışta−%1,518. T+5 2018: üç iç blokta kazanan seçim içte+%4,379, dışta+%0,235. Üç blok kazanımı tek başına taşınabilirlik kanıtı değil. Sonradan yalnız bu koşulu geçen kayıtları kullanma teşhisi T+5’te yalnız+%0,0249 verir; yeni aday veya başarı kanıtı sayılmaz. Recency iç fit ayrıntıları Drive taraması yavaş olduğu için yeniden kurulmadı; o denemenin blok istikrarına ilişkin çıkarım yapılmaz.

Sonraki tek hipotez `return-path-pilot-v1`: base24 kolon karşısında base+59 önceki günlük Cotton getirisi (mevcut günün getirisi base’de, toplam60 gözlem/83 kolon). Eski genişletilmiş paket birkaç dağınık lag içeriyordu; bu temsil tam yolu denetler. Her iki kolda sabit Ridge(alpha10), tam geçmiş ve aynı target/seed/purge/21 refit/yıllık iç seçim; alpha/window taraması yok. Her kol past-only shrinkage’ı aynı grid ile seçer. Aynı2006 origin/horizon; eşlenmiş kontrol eski altı-reçete kazananı yerine sabit Ridge olduğundan yeni kimlikte hesaplanır. Üst bütçe676 küçük fit, bir süreç/iki thread,60 dakika oturum. Araştırma önceliği/gate’ler korunur; negatifse yeni lag uzunlukları aranmaz. Feature helper ve4 sentetik test hazır; çalıştırıcı, fit kaynak/ortam kimliği ve mevcut workbench entegrasyonu henüz hazır değil, eğitim başlatılmaz. Ön kayıt: `output/reports/selection-stability-20261003/return-path-preregistration.json`; teşhis: aynı dizindeki `completion.md`. Önceki deneyler ve D21 değişmez.
