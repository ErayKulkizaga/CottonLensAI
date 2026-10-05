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

### Return-path pilot tamamlandı — 2026-10-03
Sabit Ridge24/83 eşlenmiş karşılaştırma676 fit/32 çıktı ile onaylı yerel CPU yolunda tamamlandı; her horizon2006 origin. Path Naive MAE kazancı T+1−%0,0433/T+5−%0,9516; yıl kazanımları0/8 ve0/8. Base24 karşısında iç kazanç−%0,1628/−%0,0669; iç yıl kazanımları0/8 ve2/8. Araştırma önceliği ve fiyat gate'leriFalse; Naive birincil, release kapalı. Aynı lineer lag-window araması büyütülmez.

2028 payload hash'i doğrulandı; cache replay0 fit ekledi.77 küçük doğrulanmış paket Drive klasörüne aktarıldı;34 metadata paketinden temiz açılış aynı düzeltilmiş raporu model indirmeden üretti. Bu fresh reproduction veya bulut senkronizasyon teyidi değildir. Çoğunluk-yön yüzde alanı raporlama hatası düzeltildi; eski rapor ve fit kaynak kimliği korunur, modeller tekrar eğitilmez. Read-only workbench yeni report-v2 kaynak paketini, prepare/pilot eski dondurulmuş fit paketini kullanır.

Kanıt: `output/reports/return-path-implementation-20261003/completion.json`; düzeltilmiş rapor `path-e088c7c2849ea996.json`. Ayrıntılı kamuya açık durum `docs/RESEARCH_STATUS_20261003.md`. Sonraki sınırlı iş kayıtlı küçültülmemiş path tahminlerinin eşlenmiş base kontrolüne göre teşhisidir; bu aşamada otomatik yeni fit/geniş GPU araması açılmaz. D0/D21 ve canlı kayıt değişmez.

### Return-path ham tahmin teşhisi ve tek transfer hipotezi — 2026-10-03
Yeni eğitim0.288 mevcut iç fit kaydıyla32 yıllık seçimin skor/ağırlıkları1e-12 içinde doğrulandı. Ham path iç/dış Naive MAE kazancı T+1−%5,026/−%4,424; T+5−%7,167/−%5,971. Dış Spearman IC−0,02695/−0,01303; sıfır-getiriye karşı gerçek OOS R²−0,0525/−0,0975. Küçültmeyi kaldırmak çözüm değil; aynı lineer lag araması kapanır. Küçük gerçekleşmiş hareketlerdeki yüksek hata açıklama içindir, origin eleme/canlı rejim seçimine dönüşmez.

Tek yeni hipotez `agri-transfer-pilot-v1`: geçmiş mısır/soya örnekleriyle ortak parametre öğrenimi, aynı altı causal kendi-seri feature'ı kullanan Cotton-only Ridge10 kontrolü. Önceki crop feature ablation'ından farklıdır; daha fazla örnek bağımsız bilgi veya başarı garantisi sayılmaz. Girdi10568 kayıtlı satır, aynı2006 Cotton origin/etiket. Eğitim dönüşümleri yalnız mature Cotton'da öğrenilir; iki kola aynen uygulanır. Pooled kayıp toplamı Cotton-control toplamına eşit; sabit seri payları0,50/0,25/0,25. Seed42/tam geçmiş/21 refit/beş Cotton gözlem purge/yıllık past-only shrinkage korunur; yeni tarama ekseni yok. Üst bütçe676 fit/32 çıktı. Auxiliary hedefler kendi sonraki1/5 kayıtlı gözlemi; yayın/roll/vintage kanıtı yok, Tier A ve release kapalı.

Causal input helper ve9 sentetik test/scoped Ruff geçti. Girdi/hipotez v2 donduruldu; ilk v1 kayıt korunur. Tasarım `752a333dfdcbfd34f28467128c48940506b96914cd4a88b0e1c1d7291607b7c2`. Training-ready=False: aynı Experiment/Ledger'da pooled-fit adapter, ağırlık/maturity/resume contract'ları ve fit kod/ortam kimliği bağlanmadan başlatılmaz. Sonraki somut iş bu adapter ve testlerdir; yeni notebook veya geniş GPU işi yok. Kanıt `output/reports/return-path-diagnosis-20261003/completion.md`; input packet `transfer-inputs-v2`. Eski deney/DB ve D21 korunur.

### Ortak tarımsal eğitim pilotu tamamlandı — 2026-10-03
Yukarıdaki hazırlık durumu tamamlandı: aynı Experiment/Ledger ve eşlenmiş runner içinde pooled-fit adapter uygulandı. Dönüşümler yalnız Cotton training'de öğrenilir; sabit0,50/0,25/0,25 seri paylarının toplam ağırlığı Cotton kontrolüne eşittir. Kaynak snapshot'ın 2024'e taşan son hedefleri yeni hazırlıkta maskelenir; eski dosya ve2006 evaluation origin/horizon değişmez. Beş gözlem purge, auxiliary maturity, ağırlık aktarımı, geleceği değiştirme, bozuk kayıt ve kesinti/resume testleri geçti. ML suite472 passed/1 skipped; scoped Ruff geçti.

Onaylı tek CPU süreç/iki thread yolunda676 fit/32 çıktı tamamlandı. Cotton-only kontrolün Naive MAE kazancı T+1−%0,0254/T+5−%0,0925; ortak model+%0,0617/+%0,1757. Ortak model yıl kazanımları5/8 ve4/8, all-origin yön%44,37/%39,23 (252/497 flat). İç katkı kontrole göre−%0,0206/−%0,1124 ve4/8 iç kazanım; araştırma önceliği ve fiyat gate'leriFalse. Eşlenmiş kontrole karşı güven aralıkları sıfırı içerir; BH p0,2216/0,2050. Pozitif küçük dış tarihsel kazanç geniş panel/mixture araması için yeterli değildir; Tier A ayrıca release'i kapatır. Naive birincil kalır.

2028 payload hash'i ve676 kaydın transform/weight/refit maturity sınırları doğrulandı. Cache replay0 fit;77 checksum paket Drive klasörüne kopyalandı;34 metadata paketinden temiz restore aynı raporu model indirmeden üretti. Bu fresh reproduction, Colab/GPU test, deployment parity veya cloud-sync teyidi değildir. Eski DB checksum'ı değişmedi. Mevcut research workbench güncellendi, Run All/status eğitim açmaz; yeni notebook yok. Fit kaynak kimliği `0c1cf837c4ebe4e12e2502acf187dac59e4f68c3173c605bf7d883f4abef3b53`, rapor `transfer-d0a1908023824705.json`. Kanıt `output/reports/agri-transfer-implementation-20261003/completion.md`; kamuya açık durum `docs/RESEARCH_STATUS_20261003.md`.

Sonraki sınırlı iş kayıtlı ham ortak model tahminlerinin eşlenmiş kontrole göre yeni eğitim olmadan teşhisidir. Dış sonuçlardan alpha/seri payı/cadence ayarlanmaz; aynı arama genişletilmez. D0/D21 ve ileri dönem kayıtları değişmez.

### Ortak model ham tahmin teşhisi tamamlandı — 2026-10-03
Yeni eğitim0;288 iç fit receipt'i32 yıllık seçim skor/ağırlığını1e-12 içinde doğruladı,355 kanıt hash'i kontrol edildi. Ham pooled model Cotton-only ham kontrole karşı T+1+%0,1985/T+5+%0,9126 iyileşme,8/8 ve7/8 yıl kazanımı sağladı; Naive ham kazancı yalnız+%0,0239/−%0,0990. T+5 ham kontrol farkının keşifsel bootstrap aralığı sıfırı dışlar, Naive karşısındaki iki aralık sıfırı içerir. Bu sonradan incelenen fark seçilmiş-strateji ön-kayıt gate'inin yerine geçmez, release kanıtı değildir.

Pooled dış tahmin standart sapması kontrolün%59/%58'i, iki model tahmin korelasyonu0,835/0,896. Daha düşük gürültüyle uyumludur; mekanizma/ek bilgi kanıtı değildir. Dış Spearman IC0,0252/0,0339 ve ham yön%50,15/%51,89; eşikler geçilmez. T+5 ham OOS log-return R² negatif. Shrinkage kaldırılmaz. Seçilmiş iç skor (eşit ağırlıklı blok göreli MAE) ile ham aggregate fiyat-MAE farklı ölçümlerdir; karıştırılmaz.

Sabit ortak-lineer hipotez gate için yetersiz olarak kapanır; küçük eşlenmiş kontrol katkısı korunur. Alpha/seri payı/cadence veya aynı panel araması büyütülmez. Sonraki sınırlı iş tamamlanan model/objective/feature kombinasyonlarını recipe üreticisinin seçeneklerinden ayıran sicil envanteri ve tek farklı hipotezin ön kaydıdır. Absolute-error eski recipe üreticisinde zaten vardır; gerçekten çalıştırılmış receipt'ler doğrulanmadan yeni denenmemiş fikir diye sunulmaz. Kanıt `output/reports/agri-transfer-diagnosis-20261003/completion.md`;12 dosya Drive klasöründe hash-readback ile doğrulandı. ML kaynak kimliği,676 fit/32 çıktı, eski DB ve D21 değişmedi.

### Yan sohbet incelemesi — koşullu takip sırası, 2026-10-03

Bu ek yalnız sonraki araştırma kuyruğunu günceller; ana sohbette hazırlanan
`agri-nonlinear-pilot-v1` sabit XGBoost ortak-eğitim karşılaştırmasını kesmez veya
yeniden tasarlamaz. O pilotun mevcut ön kaydı, eşlenmiş kontrolü ve bütçesi önceliklidir;
bu ek hazırlanan kodun tamamlandığı veya pilotun çalıştırıldığı anlamına gelmez.

Tamamlanan kombinasyon envanteri `output/reports/completed-combination-inventory-20261003/v2/inventory.json`
722 kayıtlı karar ve 8068 aday değerlendirmesi içerir. Değerlendirmeler yıllar/gruplar
arasında tekrar eder; bağımsız deney veya benzersiz fit sayısı değildir. Eski XGBoost/
CatBoost kare ve mutlak-hata hedefleri çalıştırılmış kayıtlarda bulunur; yeni fikir diye
tekrarlanmaz. Bu metadata envanteri bütün model payload'larının yeniden doğrulanması
veya inventory kapsamı dışındaki eski deneylerin hiç yapılmadığı kanıtı değildir.

Takip işleri sırayla ve ayrı kararlarla açılır; hepsini otomatik çalıştırma izni değildir:

1. **Yeni fit olmadan metrik köprüsü:** kayıtlı tahminlerden aynı veri/protokol/horizon/
   origin'lerde fiyat-MAE (fiyat birimi), MAPE (%), gerçek OOS R² ve Naive'a göre MAE
   kazancını tek tabloda göster. Mevcut review/raporlama kodunu kullan; farklı cohort'ları
   yarıştırma. `100 × (1 − MAPE oranı)` yalnız açıkça etiketlenmiş betimleyici dönüşümdür;
   R² veya yön doğruluğu değildir. Cotcast hedef/tarih/availability/protokolü bilinmeden
   eşitlik veya üstünlük iddiası kurulmaz. Bu tablo aday seçimi için yeni metrik açmaz.
2. **Aktif nonlinear pilotu tamamla ve karşılaştır:** test/kimlik/resume kapıları geçmeden
   eğitim açılmaz; mevcut bütçe korunur. Sonuç tek karar raporuna bağlanır. Negatif sonuç
   otomatik daha geniş GPU veya asset-weight/depth/cadence taraması başlatmaz.
3. **Eksik istatistiksel referans için küçük ARIMA fizibilitesi:** geçmiş deneylerde gerçekten
   çalıştırılmadığı doğrulandıktan sonra mevcut ledger içinde ayrı ön kayıt hazırlanır.
   Naive, drift ve küçük, en fazla altı önceden sabitlenmiş ARIMA order'ı aynı origin'lerde
   karşılaştırılır; differencing/order seçimi yalnız geçmiş iç validation'dadır. SARIMA
   yalnız training verisinde gerekçeli mevsimsellik varsa ayrı hipotezdir. Aynı purge/
   maturity/refit kuralları, başarısız fit kaydı ve sayısal bütçe ilk fit öncesi dondurulur.
   ARIMA için mevcut yerel CPU allowlist izni varsayılmaz; guard/dependency/test sözleşmesi
   ayrıca incelenir. Otomatik exhaustive `auto_arima` taraması açılmaz.
4. **Koşullu LightGBM pilotu:** önceki ve aktif ağaç deneylerinden somut farklı hipotez
   yazılabiliyorsa tek küçük eşlenmiş kontrol açılır. Aynı feature/cohort, loss, target ve
   refit korunarak model ailesi değiştirilir; önceden sınırlı reçete ve toplam fit bütçesi
   tanımlanır. Dependency/CPU-GPU desteği, ledger kimliği ve export fizibilitesi önce
   doğrulanır. İç-validation katkısı olmadan geniş Optuna araması açılmaz; LightGBM tek
   başına yeni bilgi veya başarı garantisi değildir.
5. **Sequence/panel seçimi koşullu kalır:** mevcut DLinear → gerekçeli exogenous sırası
   korunur. DeepAR olasılıksal ortak-seri hipotezi, N-BEATS ise ayrı univariate kontrol
   olabilir; ikisi aynı anda büyük taramaya girmez. PyTorch Forecasting'in standart
   N-BEATS'i dış feature kullanmaz; USDA/hava girdileri sessizce ona aktarılmaz. Yeni
   aile ancak önceki raporla gerekçelendirilmiş tek ön kayıt ve Colab smoke/export kontrolü
   sonrası açılır. Mevcut iki workbench/tek ledger kullanılır; yeni notebook hattı kurulmaz.

Veri yeniden başlatılmaz: günlük CT=F geçmişi ve USDA kaynak pilotları mevcut. Alpha
Vantage `COTTON` aylık/çeyreklik/yıllık global fiyat serisidir, günlük ICE kontratının
yerine geçmez. WASDE yayın günü tek başına eski sürümün erişilebilirliğini kanıtlamaz.
Kaynak/vintage/roll sınırları ve Kademe A/B değişmez. Günlük ICE kontratıyla birebir
rakip kıyası yapılamaması, mevcut CT=F hedefinde ücretsiz araştırmayı durdurmaz.

Başarı veya toplam maliyet garantisi verilmez: daha çok mimari/Optuna veya mevcut
donanım %99 doğruluk, <%1 hata veya Cotcast eşitliği sağlamaz; toplam GPU harcaması
otomatik $50 ile sınırlı değildir. Veri bütçesi0 TL, ayrı onay gerektiren harcamalar,
%5 MAE, T+1 %53/T+5 %55 yön ve 6/8 blok gate'leri, 2024+ seçimsiz audit ve D21 aynen korunur.

Kaynaklar: [Alpha Vantage Cotton API](https://www.alphavantage.co/documentation/),
[PyTorch Forecasting model kapsamı](https://pytorch-forecasting.readthedocs.io/en/v1.7.0/api/pytorch_forecasting.models.html),
[Cotcast'in kendi MAPE açıklaması](https://cotcast.ai/blog/global-cotton-trading-guide-strategies-profitability).


## Sabit nonlinear ortak-eğitim pilotu — 3 Ekim 2026

Tamamlanan kombinasyon envanteri: 722 imzalı karar, 8068 aday değerlendirmesi ve 48
örneklenmiş completed marker. Sayılar bağımsız deney/benzersiz fit değildir; bütün eski
payload'lar yeniden doğrulanmadı. Eski XGBoost/CatBoost squared/absolute loss kayıtları
vardır. Yeni hipotez loss değişimi değil, ortak tarımsal eğitimin sabit küçük XGBoost
yapısıyla eşlenmiş Cotton kontrolüne katkısıdır. Envanter ve ön kayıt:
`output/reports/completed-combination-inventory-20261003/v2/`.

`agri-nonlinear-pilot-v1`: iki kol, aynı altı nedensel own-series feature ve her horizon'da
aynı 2006 origin; Cotton-only preprocessing/target scaling; ortak kolda 50/25/25 seri
ağırlığı ve Cotton kontrolüyle aynı toplam ağırlık. Depth2, squared-error, seed42,
en fazla600 ağaç; sadece geçmiş63 Cotton validation gözleminde fiyat-MAE early stopping50.
Üç geçmiş iç bloktaki ağaç sayısının medyanı yıl başlamadan kilitlenir;21 gözlem refit'lerinde
değişmez. Yıllık shrinkage yalnız geçmiş iç sonuçlardan seçilir. Yeni ayar/feature/seed/history/
cadence/seri-pay araması açılmaz. Üst sınır772 fit (96 early stopping +676 refit),32 yıllık çıktı.

Tek mevcut Experiment/Ledger ve Research Workbench kullanılır. Yeni GPU source bundle:
`source-agri-nonlinear-pilot-v1-20261003.zip`; Colab-only CUDA/hist, CPU fallback hata.
Status/compare ayrı CPU ortamında; Run All varsayılanı status/RUN_TRAINING=False.
İlk15 dakikalık oturum süre ölçümü içindir; aynı kimlikle resume tamamlanmış işleri kullanır.
477 ML testi geçti/1 atlandı; GPU yolu mock ile test edildi. Gerçek CUDA/Colab çalışması,
yeni fiyat sonucu, reproduction ve deployment parity henüz doğrulanmadı.

Tier A release engeli, fiyat gate'leri,2024+ seçimsiz audit ve D21 tarihi değişmez.
Negatif sonuç aynı alanı büyütmez; sıradaki ayrı hipotez karar raporuyla açılır.
Eski DB,676 lineer fit, kaynak paketleri ve diğer tamamlanmış deneyler korunur.
Teslim raporu: `output/reports/agri-nonlinear-implementation-20261003/completion.md`.


## Sabit nonlinear pilot tamamlandı — 3 Ekim 2026

Colab research-agri-nonlinear-pilot-v1:772 ayrı durably_saved ID/32 yıllık çıktı tamamlandı.
34 metadata paketi ve32 sonuç/karar hash'i doğrulandı;32 checkpoint örneği NVIDIA L4/cuda:0,
sabit ağırlık, olgun etiket cutoff ve örneklenmiş refit ağaç sayısını doğruladı.96 payload
hash kontrolü tam772-model denetimi, reproduction veya deployment parity değildir.
Naive MAE kazancı Cotton kontrol:+0.038311%/+0.213315%; ortak kol:-0.051932%/-0.035006%.
Ortak iç katkı:+0.015967%/+0.373482%,3/8 ve4/8 iç kazanım. Priority/price gate'leri false;
Naive birincil/Tier A release engeli korunur. Ortak ham tahminler de Naive'ı geçmez;
shrinkage kaldırılmaz, aynı ortak alan genişletilmez. Flat tahminler yön yüzdesini düşürür;
nonflat alt küme ana gate yerine geçmez. Aynı-origin MAE/MAPE/gerçek OOS R² köprüsü kaydedildi.
Kanıt:output/reports/agri-nonlinear-colab-review-20261003/completion.md. Yeni fit yok.
Mevcut workbench status/RUN_TRAINING=False; GPU oturumu kapatılabilir.
Sıradaki sınırlı iş küçük Naive/drift/ARIMA fizibilitesinin ön kaydı; CPU izin/dependency
sözleşmesi varsayılmaz, bu incelemede yeni ARIMA eğitimi açılmaz. D21 değişmez.


## Küçük istatistiksel referans pilotu hazır — 3 Ekim 2026

statistical-pilot-v1/research-statistical-pilot-v1: önceki aynı2006 Cotton origin/horizon,
2016–2023 sekiz yıl, üç geçmiş63-origin iç blok,5-observation purge/olgun etiket ve21-refit.
Tek seri log fiyat; Naive, geçmiş günlük log-getiri ortalamalı drift, ARIMA(1,1,0)/(0,1,1).
Drift referans kolu w=1; seçilen kol Naive/drift/iki ARIMA ve önceki beş shrinkage ağırlığını
yalnız geçmiş iç fiyat-MAE ile seçer. Eşitlikte Naive, drift, AR10, MA01 sırası. Katsayılar
son olgun/purged training origin'inde öğrenilir; kaynakta gerçekten gözlenen her barla
durum filtrelenir, katsayı yeniden fit edilmez. Gelecek fiyat/etiket state update'e girmez.

Üst sınır964 ledger computation işi; en fazla482 sayısal ARIMA fit'i,32 yıllık çıktı.
statsmodels0.14.5 zaten CPU lock'tadır; optimizer100 iterasyon, auto_arima/seasonal search yok.
Yakınsamayan iç aday incomplete/numerical_failure kaydıyla dışlanır; kimlikli kayıt tekrar
denenmez. Seçilen modelin dış refit hatası yeni aday/Naive ile gizlice değiştirilmez.
Colab CPU-only/tek süreç/en fazla2 thread; yerel CPU allowlist genişletilmedi. Mevcut
Experiment/Ledger, matched runner ve tek Research Workbench. Run All status/eğitim kapalı.
Yeni kaynak paketi source-statistical-pilot-v1-20261003.zip; ilk çalışma15 dakikalık budget.
483 ML testi geçti/1 atlandı; statsmodels API gerçek sentetik seriyle de test edildi,
son helper-hash kontrolü sonrası5 odak testi geçti. Gerçek piyasa ARIMA fit'i başlatılmadı;
yeni skor, reproduction veya deployment parity yok. Release otomatik kapalı.
Gates/2024+ audit/D21, eski DB/source paketleri ve tamamlanmış Colab deneyleri korunur.
Ön kayıt/teslim:output/reports/statistical-implementation-20261003/.


## Statistical Colab outcome: complete; close the fixed order pilot

32/32 annual outputs verified on the same2006 Cotton origins/horizon.41 metadata
packages restored;932 distinct saved computation IDs in the console log (including
analytical references, not932 ARIMA fits). Six numerical inner candidates were excluded.
Past-selected Naive/drift/ARIMA strategy Naive MAE gains:-0.287484%/-0.087902%; direction
36.79%/23.73%, year wins1/8 for each horizon. Both gates and research-priority fail.
Raw strategy also loses Naive. Naive/flat choices explain low all-origin direction;
do not condition on nonflat forecasts or remove shrinkage.2017 T+1 inner score0.986502
transported to a2.919244% outside MAE loss. Original data/year cohort remains unchanged.
Saved bootstrap report replayed within1e-12 numeric tolerance; no new fit, full payload
audit, fresh reproduction or runtime parity. Naive stays primary; no automatic wider
ARIMA/LightGBM/sequence search. Next: consolidate completed evidence and register one
distinct bounded information/model hypothesis. D21 and2024+ selection prohibition
unchanged. Existing launcher returns to status/RUN_TRAINING=False; old DB preserved.
Evidence:output/reports/statistical-colab-review-20261003/completion.md.


## Bounded nonlinear temporal-representation contrast prepared — 3 October 2026

Four completed matched pilots were consolidated:128 signed annual outputs,2006 exact
common origins/prices/targets per horizon; no selected strategy meets the price gate.
Scoped completed inventory has no XGBoost83-column dense-return-path comparison. This
does not prove that no unavailable legacy experiment tried it. Linear60-observation
Ridge path and fixed nonlinear six-feature transfer tested separate representations.
One distinct mechanism test is nonlinear interactions in the same recorded path;
it adds no new information channel and is not a broad GPU/hyperparameter search.

nonlinear-path-pilot-v1 / research-nonlinear-path-pilot-v1: fixed CUDA XGBoost depth2,
eta0.03,child weight20,L1=0,L2=1,seed42,sampling1,scaled log target,squared loss;
base24 versus base+59 preceding returns83. Max600 trees/patience50, three past63-origin
inner blocks; yearly median tree count and shrinkage locked before outside results,
21-observation refits,5-observation purge/mature labels. Both arms use the same frozen
2016–2023 origins; no row dropping for missing features or time compression.
Use the preserved original24-feature snapshot, not the six-column transfer snapshot.
Terminal2024+ targets are masked only in a new packet; old snapshots untouched.
Budget772=96 stopping fits+676 refits,32 annual outputs; no new windows/recipes/seeds.
P1/P5 paired price-MAE,10000 bootstrap,20 main/10/40 sensitivity,BH; reused-history
evidence. Same research-priority and5%/53%/55%/6-of-8 price gates; no automatic release.
Negative result closes this fixed hypothesis without expanding the same path grid.

Existing Experiment/Ledger/matched runner and two workbenches retained. Colab GPU-only
fitting; CPU fallback error. Status/compare use CPU; defaultstatus/RUN_TRAINING=False.
489 ML tests passed/one skipped; final frozen-packet preparation test also passed.
Only mocked/synthetic fits locally; no new market score or actual CUDA/reproduction
claim. D21,2024+ no-selection audit, source availability limits,old DB and backups intact.
Evidence:output/reports/research-decision-20261003/evidence.json;
preregistration/implementation:output/reports/nonlinear-path-implementation-20261003/.

Frozen design:32ba7324583d22918562bfb9ed575d2a562224ca56e84afa68cccfdc7c1c5dd8.


## Fixed nonlinear return-path Colab outcome: complete; hypothesis closed

772 distinct saved fit IDs,32 outputs and34 metadata packages verified. Eight checkpoint
spot-checks confirm NVIDIA L4/cuda:0, payload hashes/mature labels and sampled locked
annual tree counts; not a full772-model audit, reproduction or deployment parity.
83-column Naive MAE gains:-0.004520%/-0.538577%; year wins3/8 /0/8.24-column control
gains:-0.105540%/-0.473757%. Inner path gains:-0.101730%/-0.084072%,5/8 /2/8 inner wins.
Both priority/price gates fail; BH P1/P5 p0.541146/0.692231. Raw path forecasts also lose
Naive(-0.120841%/-1.163883%); removing shrinkage is unsupported. Flat forecasts504/1002
explain selected all-origin direction36.89%/24.58%; do not replace the gate by nonflat
coverage. Close fixed nonlinear path, no wider lag/depth/seed/history search or release.
Naive primary,2024+ selection prohibition/D21 unchanged. Existing launcher status/
RUN_TRAINING=False; GPU may close. Zero new fits; old DB and unrelated working tree kept.
Evidence:output/reports/nonlinear-path-colab-review-20261003/completion.md.

### Execution update: one bounded session, 3 October 2026

Keep the two workbenches. Research pilot automatically continues planned pauses of
the same registered experiment within a total session budget:240 minutes by default,
maximum240 for CPU or720 for GPU. Frozen fit runners retain their60-minute segment
limit and code identity. Compare once when complete. Failures, interruption and no
saved progress stop without retrying or clearing locks. This is a soft budget at
fit/checkpoint boundaries, not a completion-time guarantee. Run All remains status.

The updated matched table verifies160 annual outputs from five completed pilots on
identical2006 origins/horizon; no selected price gate passes. Existing evidence is
preserved under its original identity; the new consolidation is version2. Do not
automatically queue more same-feature searches. Any next fit needs a distinct written
hypothesis and finite budget; unchanged D21 and release gates govern its decision.

### Next information hypothesis: WASDE cotton commentary

Source preflight produced90 checksummed cotton narrative reports for2016–2023;
existing TXT tables are not a new text channel. Eight ambiguous date/layout/version
entries excluded with reasons; no observation deleted. Tier A only: PDF cover date
does not prove first publication/vintage. Monthly information updates are not1254
independent news items. Corpus-v4 and exact extraction parser are pinned separately.

First proposed test is T+5 only, one fixed Ridge alpha1 recipe. Train-only TF-IDF128
unigrams/SVD8, fitted on unique eligible past documents (minimum12); no modern encoder
or broad model search. Three arms isolate text from report age/missingness and existing
World-balance ratios, using the preserved1254-origin2019–2023 cohort. Lag1 plus lag2/6
stress, three past63 inner blocks,21 refit, five-observation purge and locked shrinkage.
Planned45 outputs/954 refits. Priority needs>=0.5% inner gain against both controls,
four of five positive inner years, positive outside contribution and four of five
outside wins. Timing-fragile gain does not open wider search. No five-year/Tier-A
price gate or release claim. This proposal is not training-ready: implement/test the
train-only adapter and checkpoint identities, then freeze exact recipes and budget in
the existing workbench/ledger before any fit. D21 and original gates unchanged.

### Execution update: bounded narrative pilot ready, 3 October 2026

The WASDE commentary proposal above is implemented in the existing CPU workbench
and single ledger. Profile wasde-text-pilot-v1, separate experiment identity; T+5 only,
one Ridge alpha1 per arm, three timing/numeric/text arms at lag1/2/6. Preserve the
original1254-origin five-year cohort and all missing observations. The serializable
TF-IDF128/min_df2/SVD8 transform uses unique mature training documents, seed42;
scalers/target transform also remain train-only. Exact fitted state is checksummed.
Prepare is metadata-only. Fixed budget954 refits/45 outputs and the stated contribution
rule are frozen before the first fit. The existing launcher remains status/trainingFalse;
user deliberately enables this pilot once, with automatic checkpoint-segment resume.

547 ML tests passed,1 skipped; no real market fitting performed for this delivery.
Colab timings and predictive results are pending. Compare checks both matching
controls and every lag stress; a positive result authorizes review of this hypothesis,
not automatic search, admission, deployment or a revised gate. A negative result closes
this fixed text hypothesis. Monthly reports do not create1254 independent information
updates. Source vintage/publication remains unverified Tier A; Naive primary,2024+
selection prohibition and22 October decision checkpoint remain unchanged.

### WASDE narrative decision,4 October2026

The fixed T+5 Ridge/TF-IDF/SVD pilot completed45 outputs on1254 historical origins.
No lag condition meets the registered contribution rule; lag2 returns exact Naive.
Unshrunk predictions also lose Naive. Close this hypothesis; no automatic larger text
or same-feature search. This does not rule out every future text model/source. Keep
Naive primary, Tier-A restriction,2024+ selection ban and22 October checkpoint.
Detailed checked evidence:output/reports/wasde-text-colab-review-20261004/completion.md.

### Next bounded hypothesis: joint fundamental state,4 October2026

The reviewed source experiments tested FAS export sales, NASS national Cotton
condition and Texas weather separately. Register one combined representation in
`fundamental-joint-pilot-v1`; do not repeat those experiments or the failed text pilot.
Keep their exact shared2006 T+5 origins,2016–2023,21-observation refits, five-observation
purge/maturity and three63 past inner blocks. All missing origins remain scored.
Frozen parent histories contain3520 rows; only886 of2006 evaluation origins have
all three interaction inputs together at each lag. This is coverage, not a new cohort.

At lag1/2/6 compare timing/missingness/season controls(31 inputs), joint numerical
sources(46), and the joint sources plus three fixed products(49): sales change,
growing-season heat and growing-season rainfall, each multiplied by poor/very-poor
crop-condition fraction. One Ridge alpha1/scaled-log/seed42 per arm; past-only
shrinkage stays unchanged. No feature, model, cadence or seed search. Budget:
648 inner and873 outer refits,1521 total,72 annual outputs. Source evidence remains
Tier A; combining it does not establish publication timing or vintage validity.

Prioritization requires interaction contribution against BOTH timing and joint
controls: at least0.5% inner gain and5/8 inner wins, positive aggregate outer gain
and5/8 outer wins, at all three lag stresses. This is a research decision only;
Naive comparisons are still reported, no release gate or admission is evaluated.
If negative, close this fixed joint-state hypothesis; do not automatically expand
the interaction grid. CPU only, existing workbench/ledger/mirror, metadata-only
prepare and checksum-bound packet. Original gates,2024+ ban and D21 remain fixed.


## Availability-clock correction and completed pilot — 5 October 2026

The approved `availability-clock-pilot-v1` completed 676 CPU Ridge fits, 32 annual
outputs and 8,024 prediction rows on matched 2016–2023 origins. This is a Yahoo
availability-assumption sensitivity experiment, not verified publication timing.
T+1 selected control/available Naive gains: -0.328473% / -0.153805%; T+5:
-0.799971% / -1.339565%. Both available arms win 0/8 years against Naive.
T+1 improves slightly against control, but 87.54% of the loss reduction comes
from selecting zero weight in 2020. Raw available predictions are worse than
raw control overall. T+1 Naive-gain 95% upper bounds are 0.125691% (block20)
and 0.019337% (block60), far below the unchanged 5% practical gate.
Decision: retain Naive; no automatic grid/source/GPU expansion of this timing recipe.

Legacy LSTM's 498-versus-439 origin comparison is invalid: reconstructed matched
T+1 aggregate gain is -1.696141%, not +2.764668%. Full LSTM predictions are absent,
so no paired CI is claimed. Historical TCN known-signal gain -23.593133% failed
the learning control despite its old overall `passed`; all executed families
now require >=50% known-signal gain and cached controls are re-evaluated.
Old evidence remains immutable; corrections and 109-entry inventory are separate.
Synthetic controls are not market results; incomplete runs are not negatives;
T+5 source evidence does not settle T+1. Raw direction, selected direction and
active rate differ; zero-shrinkage [0,0] intervals do not establish no source signal.
Previously reviewed 2016–2023 and 2024+ history is not an independent holdout.

Details and exact evidence paths: [availability-clock result](AVAILABILITY_CLOCK_RESULT_20261005.md).
Existing release gates and the prohibition on 2024+ selection remain unchanged.


## User decision: futures direction and position — 5 October 2026

The user selected futures trade direction and position as the product decision.
The initial contract is [TRADING_RESEARCH_CONTRACT_20261005.md](TRADING_RESEARCH_CONTRACT_20261005.md).
Old price gates/results remain unchanged; the new objective does not certify old
models. The next bounded proposal is a no-fit T+1 saved-signal execution-timing
sensitivity, with fixed sign/flat decisions and no threshold search. Current CT=F
OHLCV lacks contract identity/session timestamps; any resulting PnL is a proxy,
not verified tradable net performance. Actual execution costs and risk budget
must be fixed before any net-performance or position-sizing approval.
No new fits, data acquisition or live trading were authorized by the goal choice.
