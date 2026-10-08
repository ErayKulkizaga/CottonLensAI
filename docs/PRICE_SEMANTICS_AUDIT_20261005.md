# Fiyat anlamı, kontrat geçişleri ve ispat sınırı — 5 Ekim 2026

**Karar:** CottonLensAI bir araştırma/ispat projesidir; gerçek alım/satım yapılmayacak.
ICE Cotton No. 2 araştırma referansıdır. Broker/komisyon seçimi tahmin araştırmasının
ön koşulu değildir. Ortak origin/hedeflerde Naive karşısında tahmin katkısı birincil,
pozisyon ve PnL yardımcı simülasyondur. Önceki %5 MAE, %53/%55 yön, 6/8 yıl
eşikleri değiştirilmedi. Geçmiş dönemler yeni holdout ilan edilmedi.

**Sonuç:** Close alanını high–low içine kırpmak veya şüpheli tarihleri silmek
yanlış olur. Mevcut bağımsız raporlar kapanışların büyük ölçüde gerçek yayımlanmış
vadeli fiyatlarla örtüştüğünü gösteriyor. Kontrat değişimleri bazı hedefleri
etkiliyor; kayıtlı modellerin Naive'i geçememesi geçiş olmayan tarihlerde de sürüyor.
Bu inceleme ne bütün etiketlerin hatasızlığını ne de kaynaklarda sıfır sinyali ispatlar.

## Dondurulmuş kaynak ve gerçek kapsam

`ml/src/cottonlens_ml/data.py::download_market_data` Cotton için `CT=F`,
DXY için `DX-Y.NYB`, WTI için `CL=F` kullanıyor. `yf.download` çağrısı
`auto_adjust=False`, `threads=False`; çıktı yalnız
`date, series, open, high, low, close, volume`. Tarihlerin timezone'u
`tz_localize(None)` ile kaldırılıyor; bu tek başına bir tarih kayması kanıtı değil.
[yfinance 0.2.54 kodunda](https://raw.githubusercontent.com/ranaroussi/yfinance/0.2.54/yfinance/multi.py)
günlük indirme varsayılanı timezone bilgisini zaten kaldırır. Kaydedilen tarih
etiketi kesin borsa/veri teslim zamanı yerine kullanılamaz.

Ham `market.parquet` SHA-256:
`23cfabb7ccb2553f0547bc3290ff4916ce91c788c98a94acdb2e5b9cd1f3151c`.
4.205 Cotton gözlemi vardır. Raw snapshot manifesti kaynak cache kimliği ve
dosya hash'lerini taşır; kontrat, tarihsel bar saati veya satır başına teslim
zamanını taşımaz. İlgili snapshot'ın oluşturulma zamanı 24 Eylül 2026'dır;
geçmiş yayın saati değildir.

Arşiv manifestindeki 199.142 **dosya yolu** da incelendi: iki market dosyası aynı
hash'e sahiptir; `data/` altında kontrat/curve/quote/intraday/session adını taşıyan
ayrı fiyat dosyası bulunmadı. Bu, bütün dosya içeriklerinin denetlendiği veya başka
bir dosyada kontrat bilgisi bulunamayacağı iddiası değildir.

## OHLC bulguları — VERIFIED

Kayıtlı `clock-t1-open-close-proxy-v1` çalışmasının 2.006 hedef barı, origin ve
gerçekleşen getiri eşitliği doğrulanarak incelendi. Tolerans `1e-5` cent/lb:

| Bayrak | Hedef sayısı |
|---|---:|
| Close high–low dışında | 268 (%13,36) |
| Bir 0,01 cent tick'ten daha fazla dışında | 261 |
| Open high–low dışında | 1 |
| Sıfır hacim | 126 |
| High = low | 248 |
| Close/open aralık, hacim veya range bayraklarının birleşimi | 461 (%22,98) |

Close dışarıda olanlarda medyan hacim 9; diğerlerinde 12.782.
Örnekler, fiyat birimi cent/lb:

| Hedef | Open / high / low | Close | Hacim | Aralık dışı uzaklık |
|---|---|---:|---:|---:|
| 2022-07-01 | 109 / 109 / 109 | 103,68 | 2 | 5,32 |
| 2023-06-26 | 80,85 / 80,85 / 80,55 | 77,07 | 116 | 3,48 |
| 2020-02-27 | 65,25 / 64,38 / 64,38 | 62,60 | 8 | 1,78 |

Bunlar normal işlem barı yorumunu doğrulamıyor. Ancak settlement/mark fiyatı
işlem aralığının dışında bulunabilir. [ICE Rule 4.34](https://www.ice.com/publicdocs/rulebooks/futures_us/4_Trading.pdf)
işlemsiz kontratlarda spread/bid–offer gibi bilginin settlement belirlemesine
izin verir. Bu kural, Yahoo Close'un her satırda settlement olduğu kanıtı değildir.

## Önceden arşivlenmiş USDA raporlarıyla çapraz doğrulama — VERIFIED

Yeni indirme yapılmadı. Mevcut `publication-content-review.json` içindeki 998
raporun **tamamının dosya hash'i** yeniden doğrulandı. 2021-10-20 raporunun
basılı tarihi `#########` olduğundan tarih doğrulamasına alınmadı; bu red açıkça
kaydedildi. Kalan 997 tarih doğrulanmış `FUTURES TODAY` tablosunun **996'sında**
ilk listelenen kontrat fiyatı Yahoo Close ile `1e-5` toleransta eşleşiyor.
Bu kapsamdaki high–low dışı **136 Close'un tamamı** da eşleşiyor.

Yukarıdaki üç örnekte bağımsız tabloda sırasıyla `Jul-22 103.68`,
`Jul-23 77.07`, `Mar-20 62.60` bulunuyor. Kaynaklar ve orijinal hash'ler
Release'teki `ams-v1/quotes.json` içinde kayıtlıdır. Örneğin mevcut
[1 Temmuz 2022 AMS raporu](https://mymarketnews.ams.usda.gov/filerepo/sites/default/files/3004/2022-07-01/908648/ams_3004_00594.txt)
bu fiyatı bildiriyor.

Tek fiyat uyuşmazlığı 2020-04-24: Yahoo Close 54,93, ilk AMS May-20 fiyatı
56,88; tablodaki hiçbir kontrat eşleşmiyor. Hangi kaynak/sürümün doğru olduğu
çözülmedi. **Hiçbir değer değiştirilmedi.** Bir ekli referans tarihin okunamaması
ile fiyat uyuşmazlığı birbirinden ayrı bulgulardır. İlk kaynak manifestindeki
999 satırlık spot veri kapsamı, burada mevcut 998 belge sürümüyle karıştırılmadı.

Bu karşılaştırma settlement yorumunu **STRONGLY SUPPORTED** yapar;
vendor alan sözleşmesi, tam kontrat haritası, geçmiş erişilebilirlik/vintage veya
gerçekleşebilir fiyat iddiasını VERIFIED yapmaz. AMS verisi bu işlemle özellik
olarak eğitime kabul edilmedi; geç yayımlanmış sürümler geçmiş bilgi sayılmadı.

## Kontrat geçişlerinin gerçek etkisi ve karşı kanıt

2020–2023 arasında ardışık Cotton gözlemlerinde ilk listelenen ve Close ile
eşleşen kontrat **20 kez değişiyor**. Önceki gün yeni kontrat da tabloda bulunduğu
için değişim şu şekilde ayrılabiliyor:

`yeni_t − eski_(t−1) = (yeni_t − yeni_(t−1)) + (yeni_(t−1) − eski_(t−1))`

İki geçişte bu fark hareketin işaretini değiştiriyor. 2022-10-10'da görülen
değişim −4,00 cent; yeni kontratın kendi değişimi +4,00 cent, önceki gün spread'i
−8,00 cent. **Geçişlerde hedef bileşimi sorunu var.** Bu bağımsız fiyat eşleşmesi
güçlü kanıttır; Yahoo'nun sertifikalı kontrat metadata'sının yerine geçmez.

Kayıtlı iki kolun tahminleri değiştirilmeden, bütün ara günleri AMS ile eşleşen
fiyat yolları sınıflandırıldı. Hiçbir yeni fit/strateji/eşik seçilmedi:

| Ufuk / tanısal altküme | Origin | Naive MAE | Müdahale seçilmiş kazanç | Müdahale ham kazanç |
|---|---:|---:|---:|---:|
| T+1 aynı listelenen kontrat | 961 | 1,279209 | %0,000000 | −%1,515876 |
| T+1 kontrat değişimi | 20 | 2,449501 | %0,000000 | −%0,542310 |
| T+5 aynı listelenen kontrat | 847 | 3,072385 | %0,017723 | −%3,217501 |
| T+5 kontrat değişimi | 94 | 3,290107 | −%0,126540 | −%8,489665 |

MAE birimi cent/lb. Kontrol kolunun aynı-kontrat seçilmiş kazancı T+1'de
−%0,282623, T+5'te −%0,090102. T+1'de 1.025, T+5'te 1.065 origin AMS
kapsamı dışında veya doğrulanmamış fiyat yolundadır; ayrıca raporlandı,
“roll yok” sayılmadı. 2.006 origin'in tamamı bu sınıflara muhasebeleştirildi.
Bilinen geçiş yolları tüm dönem Naive mutlak hata toplamının T+1'de %2,39,
T+5'te %6,54'ünü oluşturuyor. 2016–2019 geçişleri bu hesapla doğrulanmadı.

**NOT SUPPORTED:** “Sadece roll günlerini düzeltmek mevcut tahminleri kurtarır.”
Başarısızlık aynı-kontrat yollarında da var. **Çözülmeyen karşı açıklama:**
bu modeller karışık hedeflerle eğitilmişti; eğitimdeki geçiş/likidite
kontaminasyonunun bütün tahminleri etkilemesi bu sıfır-fit analizle dışlanamaz.
Altkümeler önceden bağımsız holdout değildir; yeni başarı iddiası veya uygun
origin tanımına dönüştürülmez.

Ham yardımcı PnL'nin de tamamı bar bayraklarından gelmiyor. Müdahale ham brüt
proxy 49.079,90 USD-equivalent; 461 bayraklı hedef dışında 42.804,89 kalıyor.
Close'u aralığa projekte etme **yalnız tanısal** hesabında 39.809,91 kalıyor.
Sabit short proxy'nin kazancı ise high–low dışı günlere belirgin biçimde bağımlı.
Bu rakamlar gerçek/net kazanç, yeni strateji seçimi veya model başarısı değildir.

## Saat denetimi ve sınırlar

[Güncel ICE saatinde](https://www.ice.com/products/254/Cotton-No-2-Futures)
normal açılış New York 21:00; yaz/kış UTC karşılığı 01:00/02:00'dır.
Doğru trade-date eşlemesi **varsayılırsa** 00:15 kararından 45/105 dakika sonra
olur. [2009 tarihli duyuru](https://www.ice.com/publicdocs/futures_us/exchange_notices/ExNot031909aghours.pdf)
da 21:00 açılışını bildirir; aradaki bütün yılların doğrulaması değildir.
[25 Kasım 2016 istisnası](https://www.ice.com/publicdocs/futures_us/exchange_notices/ExNot2016Thanksgiving.pdf)
08:00 New York açılışını gösterir. Dolayısıyla bütün günlük barlara sabit bir
açılış damgası yapıştırılmadı. 2016 resmi takvimindeki on kapalı tarih raw'da
yok; sekiz yıllık seans takviminin tamamı onaylanmış sayılmadı.

## Teslimat, doğrulama ve sonraki tek kontrol

Yeni read-only modüller: `research.execution_audit`, `research.price_reconciliation`.
Bozuk checksum, yanlış tarih/hedef/PnL, eksik kaynak ve mevcut çıktının üzerine
yazma reddedilir. Bayraklar fiyat düzeltmez; kaynak tarihleri yalnız tanısal
olarak sınıflanır. 25 dar kapsamlı test geçti, ML Ruff geçti. Bağımsız aritmetik
verifier 690 eski girdiyi ve toplam 1.694 girdiyi yeniden hash'ledi; orijinaller
değişmedi. Yeni eğitim, GPU, bağımlılık kurulumu, veri indirme ve canlı model
değişikliği yapılmadı. CI sonucu PR'dan ayrıca okunmalıdır.

[Makine kanıtı](../research/evidence/price-semantics-20261005.json) ve
[research-semantics-20261005 Release](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/research-semantics-20261005)
bayrakları, fiyat tablolarını, bağımsız kontrolü ve frozen kaynak kimliğini saklar.

Bir sonraki veri-kabul işi, daha önce çıkarılmış **95 rapor/26 alanlı bölgesel
WASDE adayının** mevcut orijinal raporlarla alan/değer, sürüm ve erişilebilirlik
eşleşmesini doğrulamaktır. Başarısız kontroller açık red olarak korunacak;
doğrulanmayan satır/alan otomatik eğitime alınmayacak. Bu, yeni bir model araması
değildir. Yeni T+1 kaynak deneyi ancak bu kontrol ve sicilde tekrar denetimi
sonrasında ayrı kilitli tarifle yapılabilir; mevcut T+5 negatiflerinden T+1
sonucu çıkarılmayacak. Broker araştırması ispat projesinin devamını engellemez.
