# Kontrat/vintage veri açığı için teklif ve örnek kabul şartları

Güncel takip: Kullanıcı6Kasım2026'da ICE talebini gönderdiğini bildirdi.
Gönderim/ekler bağımsız gözlenmedi; yanıt veya fiyat teklifi henüz doğrulanmadı.
Bu bildirim veri kabulü veya satın alma yetkisi değildir.

Bu belge satın alma talimatı değildir; gönderilen mesajın bağımsız makbuzu da değildir. Canlı teklif/fiyat
alınmadı; herhangi bir kaynak kabul edilmedi. Aşağıdaki resmi ürün sayfaları
6Kasım2026'da incelendi. GPU erişimi bu kaynak boşluklarını gidermez.

## Talep

ICE Futures US Cotton No.2 bireysel kontratları için2010–2023dahil günlük,
ayarlanmamış uzlaşma fiyatları ve kontrat referans verisi. Seans kapsamı,
listeleme/son işlem tarihi, kontrat kimliği, cent/lb birimi, settlement/last
ayrımı, null/no-trade/not-listed durumları ve düzeltme sürümleri gerekir.
Sürekli seri tek başına bu talebi karşılamaz. Gerekli ilk model durumu için
2010öncesi başlangıç gerekiyorsa ayrı kapsam olarak fiyatlandırılmalı; başlangıç
durumu performansa göre seçilmemeli.

Tarihsel first-publication timestamp/vintage, düzeltme zamanı ve arşiv erişim
kanıtı ayrıca sorulmalı. Sağlayıcı bunları sunamıyorsa açıkça NO bildirilmeli.
Güncel indirmenin UTC zamanı geçmiş erişilebilirliğin yerine geçmez. İki teklif
ayrılmalı: nihai tarihsel fiyat/kontrat paketi ve point-in-time/vintage paketi.
İlki fiyat araştırmasını destekleyebilir; tek başına sızıntısız PIT kanıtı olmaz.

Lisans: özel araştırma, model eğitimi, Colab'da işleme, türetilmiş model/tahmin
saklama ve privateGitHub'da ham veri yedekleme hakları ayrı belirtilmeli. Ham
yeniden dağıtım yasaksa şifreli olsa da GitHub'a yüklenebilir varsayılmayacak;
lisanslı arşiv yöntemi ayrı kararlaştırılacak. Ücret, exchange entitlement,
abonelik, geçmiş indirme limiti ve tekrar erişim şartları istenmeli.

## Mevcut açıkların somut talep kapsamı

[58 kontratlık talep matrisi](../research/evidence/source-request-matrix-listing-20261106/contract-evidence-request.csv)
ve [hash bağlı hesap kaydı](../research/evidence/source-request-matrix-listing-20261106/result.json)
mevcut katı readiness çıktılarından üretildi. Kontrat teslim ayları2012-12–2024-05;
gereken kaynak seansları2010-01-04–2023-11-23arasında3.108tarih.2024teslim ayı,
2024sonuçlarını seçime katmak anlamına gelmez:2023origin'lerinde kullanılan ileri
vadeli kontratın kimliğidir.

Güncel99.523hücre üst sınırının38.588'i kabul edilmiş tablosu bulunmayan
günlerden,1.625'i kısmi haftalık prefix'ten,551'i kimlik/fiyat çatışmasından,
58.759'u rapor/annual dışında listeli kontrat kapsamı kanıtı bulunmamasından gelir.
Bunlar kesin fiyat hatası veya işlem görmüş/listelenmiş kontrat sayısı değildir.
Hücre→origin bağımlılık sayısı4.778.562; iki CSV yönünden bağımsız toplam aynı.
Bu sayı bağımsız fit, hipotez veya tekil etkilenen origin sayısı değildir.

İlk40.450hücrelik [matris sürümü](../research/evidence/source-request-matrix-20261106/result.json)
korunur; raporun tamlığını listing kapsamı sayan önceki koşullu hesabı açıklar.
Güncel kabul kapsamı olarak kullanılmaz. [Resmi ICE expiry tablosu](https://www.ice.com/products/254/Cotton-No-2-Futures/expiry)
6Kasım2026'da15güncel kontrat ve FTD/LTD alanı gösteriyor. Bu canlı sayfa geçmiş
58kontratın listeleme tarihlerini sağlamıyor; güncel listeleme kuralı geçmişe
kopyalanmayacak. Bir USDA raporunda10quote bulunması tüm listeli kontratların
bunlardan ibaret olduğunu kanıtlamaz.

Önce her58kontrat için resmi `listed_on`, `last_trade_on`, exchange contract ID
ve bu tarihlerin kaynak belgesi/hash'i istenir. Veri sağlayıcısı kontratın o gün
listelenmediğini belgeleyebiliyorsa hayali settlement fiyatı istenmez. Ancak
listeleme tarihi, on-kontrat tablosundaki sıra veya teslim ayından tahmin edilmez;
metadata gelene kadar söz konusu hücre UNKNOWN kalır. Gerçekten listeli günler
için dated settlement/no-quote açıklaması; PIT/vintage kanıtı ayrıca gerekir.
Bu aşamalama talebi daraltır, kabul kapısını gevşetmez. Matrisin kendisi G1'i açmaz.

## Aday kaynaklar ve doğrulanmamış noktalar

| Aday | Resmi sayfanın desteklediği kapsam | Satın almadan kanıtlanacak |
|---|---|---|
| [ICE Report Center](https://www.ice.com/report-center) | Günlük settlement ve raporlar için resmi başlangıç noktası | İstenen tüm geçmiş kontratlar, makine erişimi, ilk sürüm/UTC ve lisans |
| [Nasdaq Data Link MWIS](https://data.nasdaq.com/databases/MWIS) | ICE borsalarında işlem gören futures kontratları için günlük settlement feed tanımı | CT geçmiş başlangıcı, erişilebilir ürün, kontrat metadata, PIT arşivi ve güncel teklif |
| [Portara Cotton ICE](https://portaracqg.com/futures/day/cte) | Cotton günlük/intraday ürün ve bireysel kontrat seçeneği | Daily alanın last mı settlement mı olduğu, ilk sürüm, listeleme takvimi ve haklar |

Bu tablo kaynak önerisi veya kalite sıralaması değildir. Reklam kapsamı
gerçek örnek veri ve alan sözleşmesi yerine geçmez. Henüz rakamsal teklif yok.

## Ücretsiz örnek için sonuçlardan bağımsız kabul testi

Sabit tarihler:2018-08-01,2019-08-05,2019-11-18,2020-04-24,
2020-12-04,2021-02-09,2022-07-01,2022-10-10,2023-06-26.
Her tarihin önceki/sonraki iki resmi Cotton seansı ve o seanstaki tüm listeli
kontratlar istenir. Tarihler bilinen kaynak/roll/alan çatışmalarını sınar;
başarı skoru seçmek için kullanılmaz. Örnek yalnız tanısaldır; tüm tarihi kabul
etmez. Tam paket için tüm seans/hücre denetimi ayrıca gerekir.

Zorunlu testler:

- Tekil kontrat/seans anahtarı, geçerli lifecycle, tutarlı birim ve açık null
  semantiği; expired/unlisted kontratın sessiz yeniden adlandırılması reddedilir.
-2019annual'daki yanlış kimlikler ve2021ön-kontrat eksikliği kaynak tanıklarıyla
  ayrıştırılır; referans kaynağın kendisi de hatalı olabilir, uyuşmazlık korunur.
- Published change/previous varsa Decimal aritmetik; settlement high–low
  dışında diye reddetme veya kırpma yok. Sağlayıcı hassasiyeti kaynak sözleşmesinde
  sabitlenir; fiyat eşleştirme toleransı sonuç görülmeden belirlenir.
-2020-04-24 gibi çözümsüz fark için sağlayıcının sürüm ve alan açıklaması gerekir.
- Roll serisi verilirse mapping ve adjustment ayrı olmalı; origin'de seçilen
  kontratın hedefteki fiyatı aynı kontrattan alınabilmeli.
- Aynı yanıtın tekrar indirilmesi hash/fark raporuyla kontrol edilir. Eşit hash
  fiyatın doğru veya tarihsel olarak erişilebilir olduğunu tek başına ispatlamaz.
- PIT iddiası varsa ilk yayın ve revizyon kanıtı örneklenir; eksik kanıtla
  `historical_availability_verified` verilemez.

Karar sırası: örnek → alan/lisans cevabı → tam kapsam ve maliyet teklifi →
kullanıcı satın alma kararı → tam veri kabulü. Otomatik iletişim/satın alma yok.

## Rule 10.07: conditional schedule, not actual historical listing admission

The primary ICE Cotton rulebook, Rule10.07, describes the default opening in the
35th month preceding delivery and includes a November16,2007effective annotation.
It also permits Board direction/override. The raw PDF and its hash are preserved
in the private evidence release; this is stronger context than current expiry
rows alone but does not establish that no historical exceptions occurred.

The [conditional matrix](../research/evidence/ice-listing-schedule-20261106/conditional-contract-schedule.csv)
partitions the same99,523UNKNOWNcells into83,312before the default listing month,
1,216within it and14,995after it. The arithmetic matches the FTD month in all15
observed current expiry rows. No exact first business day is inferred. None of
these historical cells is admitted or declared absent on this diagnostic alone.

Ask for actual dates and the completeness of exception notices first. Only a
validated answer can eliminate requests for genuinely unlisted dates; the
conditional partition is not permission to drop inconvenient observations.
The complete [ICE inquiry draft](ICE_DATA_REQUEST_DRAFT_20261106.md) requests
reference data, exceptions, a fixed diagnostic sample and separate license/price
quotes. It has not been sent; external-message authorization is pending. No
purchase or subscription is authorized by this draft.
