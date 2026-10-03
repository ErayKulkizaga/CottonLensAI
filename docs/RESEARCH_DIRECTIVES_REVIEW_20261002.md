# ChatGPT ve Perplexity incelemelerinin değerlendirilmesi

Tarih: 2026-10-02. Bu belge araştırma değerlendirmesidir; yeni execution plan
veya deney sonucu değildir. Kanonik plan `MASTER_PLAN_20261001.md` olarak kalır.
Eklerdeki “hemen uygula”, “bu belge canonical olsun”, bütçe ve deney sırası
ifadeleri kullanıcının değerlendirmemizi istediği öneriler olarak ele alındı.
Bu incelemede eğitim, veri satın alma, hesap açma, commit/push yapılmadı.

## Karar

Bilgi kalitesini ve piyasa temsilini model büyüklüğünden önce inceleme yönü
doğru. Fakat iki belge mevcut çalışmanın gerisinde kalıyor ve bazı literatür
sonuçlarını farklı hedeflere taşıyor. “Daha iyi model değil, yalnız veri sorunu”
teşhisi kanıtlanmış değil. Veri, hedef/kontrat tanımı ve model varsayımı ayrı
hipotezler olarak kalacak. Negatif küçük-model pilotu bir kaynağın her modelde
faydasız olduğunu göstermez; büyük aramayı da tek başına gerekçelendirmez.

Planın sonraki yürütme sırası güncellendi:

1. Mevcut forward collector'ın Windows görev engelini çöz; gerçek çalıştırma
   receipt'i, duplicate/no-backfill ve cutoff davranışını doğrula.
2. Cotton On-Call'ın sınırlı kaynak kontrolünü yap. Ayrı bir küçük kontrolle
   ücretsiz tekil kontrat verisinin tarih/alan/erişim haklarını belirle.
3. İlk kullanılabilir, gerçekten yeni bilgi için aynı-origin ablation'ı ön kaydet.
   Futures curve bulunamadığı için On-Call araştırması beklemeyecek.
4. Yalnız katkı görülen hattı büyüt. Statik shrinkage/ensemble'dan sonra causal
   online ağırlıklar; sonra gerekçeli sequence/panel/foundation araştırması.

Bu sıranın ayrıntısı kanonik planın 2026-10-02 karar bölümündedir. D0/D21 ve
başarı eşikleri yeniden başlatılmadı veya gevşetilmedi.

## Gerçek repository ve kayıtlarla uyuşmazlıklar

| Eklerdeki öneri/varsayım | Gerçek durum ve karar |
|---|---|
| USDA/CFTC çoğunlukla parser düzeyinde; şimdi modellerle bağla | AMS, FAS, NASS, WASDE ve CFTC için model özellikleri, prepare/pilot/compare ve tamamlanmış kayıtlar var. Aynı işi yeniden açma. |
| Full-year, Ridge/shrinkage, EWMA/HAR/GARCH ve aralık motorunu başlat | Tam yıl 2016–2023, 2006 origin/ufuk pilotu ve ayrı kalibrasyon düzeltmesi tamamlandı. Fiyat/aralık kararı P−/U−. Eski geçersiz aralıklar korunuyor. |
| COT managed-money/producer net-OI oranını ilk kez dene | `cftc_exploration.py` zaten managed-net/OI, producer-net/OI ve managed-share değişimini kullanıyor. Yeni z-score/lag ancak açıkça farklı, kayıtlı hipotezle denenebilir. |
| INR/CNY/BRL, soya/mısır ve hava bilgisi henüz yok | Bu gruplar ayrı Kademe A pilotlarında modele bağlandı. Hepsi mevcut tüm-gecikme öncelik kuralını geçemedi. INR/CNY/BRL ECB çapraz kurlarıdır; PKR bu pakette yok. |
| Forward log'u şimdi kur | Kod ve referans kilidi var; Windows görev kurulumu Access denied ile başarısız. Otomatik toplama çalışıyor denemez. 126 olgun ileri origin henüz yok. |
| Yeni full-year-v2/protokol ve E0 başlangıcı | Sadece yeni ad verilerek eski tarih görülmemiş olmaz. Gerçek hedef/split/availability değişikliği yoksa mevcut cohort ve kayıtları kullan. |
| Futures curve her şeyden önce zorunlu bağımlılık | Aynı anda bilinen tekil kontrat fiyatları/volume/OI ve tarihsel kapsam henüz doğrulanmadı. Ücretsiz erişim yoksa açık eksik olarak kaydet; başka ücretsiz hipotezi bloke etme. |
| 1000 TL veri bütçesi; önce trial kullan | Geçerli veri bütçesi 0 TL. Önceki 500 TL olasılığı ayrıca somut ihtiyaç/onay gerektirir. Ek belge bütçe veya abonelik yetkisi vermez. |

Kayıt kanıtları: `output/reports/full-year-completion-20261002/completion.md`,
`output/reports/{ams,fas,nass,wasde,cftc,fx,crop,weather}-exploration-20261002/`.
AMS yalnız 2023/246 origin, WASDE beş yıl/1254 origin; bunlar sekiz yıllık
cohort'larla tek skor sıralamasında karıştırılmadı. Diğer altı kaynak paketi
sekiz yıl/2006 ortak T+5 origin kullandı. Bu sonuçlar T+1 veya tüm kaynak/model
evreni hakkında hüküm değildir.

Son hava teşhisi: normal hareketlerde ve şüpheli roll pencerelerinin dışında
da kayıp var. İşaretler doğrulanmış kontrat değişimleri değil; bazı raw OHLC
provenance bilgileri de mevcut inceleme girdisinde eksik. Roll sorunu çözüldü
iddiası veya kötü origin'leri çıkararak başarı üretme yolu benimsenmedi.

## Birincil kaynak kontrolü

| Kaynak | Doğrulanan bulgu | CottonLens'e etkisi |
|---|---|---|
| [Boons–Prado, Basis-momentum](https://research.unl.pt/ws/portalfiles/portal/14263925/MM_Basis_Momentum_V24.pdf), §I.B, denklem 3 | İki nearby stratejisinin 12 aylık bileşik momentum farkı; aylık getiriler ve tanımlı kontrat taşıma kuralı | Eğri bilgisini incelemek için gerekçe. Beş günlük log-spread değişimi aynı tanım değildir; ayrı proxy adı gerekir. Günlük Cotton MAE üstünlüğü kanıtı değil. |
| [Wang–Zhang, yazarların araştırma özeti](https://www.bayes-cid.com/pdf/issues/2024-winter/publications/Pages-25_29-CID-Winter-2024-Tianyang-and-Shirui-021025.pdf) | 22 emtianın aylık continuous getirileri, commodity-specific/macro girdiler ve LightGBM portföy sonuçları | Aylık risk primi/portföy başarısı günlük T+1/T+5 fiyat-MAE ile aynı görev değil. Tree ailesini bırakmak için de gerekçe yok. |
| [Rahimikia–Ni–Wang](https://arxiv.org/abs/2511.18578) | Günlük excess-return deneyinde hazır TSFM'ler zayıf, finans verisinden pretraining daha iyi | Domain uyumu önemlidir. Bu çalışmanın büyük veri erişimi ve ekonomik sonuçları bizim ücretsiz Cotton pipeline'ına hazır çözüm sağlamaz. |
| [Brini](https://arxiv.org/html/2607.05291v1) | Beş dakikalık getiriden günlük realized variance; TTM'nin ham QLIKE oranında %1,3–1,8 avantajı, kısa ufukta önemli kalibrasyon bileşeni | Bizim günlük kare-getiri vekili aynı hedef değildir. TTM+HAR yalnız koşullu keşif; QLIKE tanımları farklıysa yüzde oranları taşınmaz. MCS üyeliği üstünlük olasılığı değildir. |
| [Kronos makalesi](https://arxiv.org/abs/2508.02739), [resmî repo](https://github.com/shiyu-coder/Kronos) | Büyük finansal K-line pretraining ve benchmark RankIC artışı; repo MIT | %93 göreli RankIC artışı %93 doğruluk veya Cotton fiyat-MAE kazancı değildir. Checkpoint lisansı, formatı ve eğitim kapsamı ayrıca doğrulanacak. |
| [TimesFM resmî repo/lisans notu](https://github.com/google-research/timesfm) | Kod ve ≤2.5 ağırlıkları Apache-2.0; indirilen 3.0 ağırlıkları non-commercial/non-production | 3.0 ağırlığı otomatik demo/runtime adayı yapılmaz. 2.5 veya izinleri uygun başka sürüm ayrı koşullu adaydır; ücretli Cloud yolu açılmaz. |
| [Aylık Cotton NNAR çalışması](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1628744/full), §3.2.1 | %1,19 MAPE in-sample fitted one-step değerlendirmeye ait | OOS Naive benchmark hedefimiz olamaz. |
| [NOAA nClimGrid-Daily](https://www.ncei.noaa.gov/products/land-based-station/nclimgrid-daily), latency bölümü | Ön veriler genellikle 2–3 günde gelir; günler/aylar boyunca yeni girdilerle revize edilir | NOAA'ya geçmek vintage sorununu kendiliğinden çözmez. Crop-area/past-acreage toplulaştırma farklı hipotezdir; otomatik yeni hava turu yok. |
| [Cotton On-Call örneği](https://www.cftc.gov/MarketReports/CottonOnCall/HistoricalCottonOn-Call/deaoncall061826.html), [gecikmeli örnek](https://www.cftc.gov/MarketReports/CottonOnCall/HistoricalCottonOn-Call/deaoncall100925.html) | Kontrat ayına göre unfixed satış/alış/OI; dönem, URL tarihi ve gerçek yayın zamanı ayrılabiliyor | En yararlı yeni ücretsiz aday. Tarihsel payload'ın ilk sürüm olduğu ayrıca kanıtlanmalı. Haftalık OI günlük volume/eğri yerine geçmez. |
| [ICE Cotton sözleşmesi](https://www.ice.com/products/254/Cotton-No-2-Futures/specs) | First notice ve last trading ayrı kurallar | Kontrat seçimi/expiry doğrulanır; bu kurallar Yahoo roll tarihini kanıtlamaz. |
| [Barchart tarihsel indirme açıklaması](https://www.barchart.com/my/price-history/download/INTU), [kullanım koşulları](https://www.barchart.com/terms) | Üyelikle tarihsel indirme ürünü var; erişim ve kullanım koşulları ürüne bağlı | Bütün expired Cotton kontratlarının 2010+ fiyat/volume/OI kapsamı ve ücretsiz trial hakkı doğrulanmadı. Realtime trial ile Premier koşulları birbirine karıştırılmaz; otomatik hesap/scraping yok. |

[Vitale–Robinson](https://www.mdpi.com/1911-8074/18/2/93) için publisher arama
kaydı December Cotton hedefini doğruluyor; tam metin bu oturumda 429 verdi.
5-fold/random-search workflow'undan kesin zaman sızıntısı hükmü çıkarmadım.
[Chandan–Kumari](https://www.sciencedirect.com/science/article/abs/pii/S2214579625000644)
tam metnine de erişemedim; eklerdeki split/sonuç ayrıntıları teyitsiz kaldı.
Random hyperparameter search tek başına random temporal split demek değildir.
Bu iki çalışmaya göre mimari veya kabul eşiği değiştirilmedi.

## Teknik olarak düzeltilen öneriler

- **Eğri:** `s12=log(F1/F2)`; yıllıklandırılmış signed slope
  `365*s12/(tau2-tau1)` olarak açık bir sözleşme taşımalı. Aralıklar eşit
  değilse ham iki-spread farkı olgunluğa göre curvature değildir; normalize
  slope farkı ayrı proxy olur. Spot-futures basis ve futures-futures spread
  farklı kavramlar. Literatürün kendi futures-basis tanımı kullanılırsa isim,
  işaret ve dönüşüm ayrıca kaydedilir. Bu proxy gerçekleşmiş roll P&L değildir.
- **Kontrat seçimi:** gelecekteki likiditeye göre “aktif kontrat” seçilmez.
  Güncel volume/OI'nin yayın gecikmesi ayrıca hesaba katılır. Aynı target için
  h gün sonra aynı kontrat mı, yeni nearby mı olduğu önceden sabitlenir.
  CT=F primary hedefi sessizce değiştirilmez; ayrı kontrat hedefi ayrı protokoldür.
- **Panel:** yan emtia feature'ları eklemek Cotton gözlem sayısını artırmaz.
  Çok-emtia supervision daha fazla eğitim satırı sağlar, fakat aynı tarihteki
  bağımlılık nedeniyle bağımsız Cotton test örneklemi büyümüş sayılmaz.
  Bütün varlıkların etiketleri aynı dış UTC cutoff'tan önce olgun olmalıdır.
- **Online ensemble:** geçmiş OOF etiketinin olgunluğu kadar, o OOF tahminini
  üreten recipe'nin o tarihte seçilebilir olması da gerekir. Geç seçilmiş
  modeli eski origin'lere uygulamak geçmişte yayımlanmış tahmin değildir.
  Önce Naive dahil en fazla üç expert ve statik/global kontrol; sonra tek
  past-loss weighting hipotezi. Ex-post roll/movement grupları gate girdisi olamaz.
- **Risk:** coverage ve genişlik interval score ile birlikte ölçülür;
  adaptive conformal koşullu coverage garantisi diye sunulmaz. Volatilite
  üstünlüğü fiyat gate'inin veya fiyat yönünün yerine geçmez.
- **Model sırası:** küçük düzenlileştirilmiş modeller ve aynı-budget kontroller
  korunur. Yeni bilgi katkısı veya açık model-varsayımı hipotezi varsa bir küçük
  alternatif aile eklenebilir; mevcut negatifi binlerce config ile kurtarma yok.
  Foundation/sequence adayları aynı anda açılmaz; kod/ağırlık lisansı ayrı kaydedilir.

## Açılmayan işler

Yeni full-year taraması, eski USDA/COT pilotlarının tekrarı, otomatik Barchart
trial, 1000 TL bütçe, 7 günlük zorunlu A100 takvimi, yeni T+21 gate'i, rakip demo
scraping'i ve mesaj gönderimi açılmadı. Cotcast hakkında sayısal uzaklık veya
başarı garantisi üretilemez; ortak hedef/origin/cutoff protokolü gerekir.
Mülakat anlatısı yalnız gerçekten tamamlanan, kimliği ve kanıtı olan işleri içerir.
