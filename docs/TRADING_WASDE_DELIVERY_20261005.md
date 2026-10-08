# İlk işlem tanısı ve bölgesel WASDE teslimatı — 5 Ekim 2026

Sonraki kullanıcı açıklaması: proje araştırma/ispat kapsamındadır, gerçek işlem
yoktur. Aşağıdaki maliyet/gerçekleşme engelleri net işlem iddiası içindir; tahmin
araştırmasının ön koşulu değildir. [Sonraki fiyat/seans denetimi](PRICE_SEMANTICS_AUDIT_20261005.md)
open/close semantiğine ilişkin yeni kanıt ekler; bu ilk raporun sayıları değişmedi.

**Karar: yeni model eğitimi başlamıyor; Naive ve canlı artifact korunuyor.**
Kayıtlı T+1 tahminlerinin sabit yön kuralı, uygulanabilir maliyet sonrası işlem
üstünlüğünü henüz göstermiyor. WASDE'nin dışarıda kalmış ülke alanları kurtarıldı;
tarihsel erişilebilirlik kanıtı olmadan eğitime kabul edilmedi.

## Yapılan düzeltmeler

- `research/protocol.py::attach_releases` artık sürümlü karar saatini kabul eder.
  Varsayılan `legacy-midnight-v1` önceki davranışı korur; yeni paket açıkça
  `cotton-next-day-0015-v1` seçerse kaynak tarihi +1 gün 00:15 UTC sınırı kullanılır.
  Tam sınırda bilinen kayıt alınır, sınırdan sonraki dışlanır. Bu karar saati
  kaynağın gerçek yayımlanma saatinin yerine geçmez.
- `research/publications.py` CLI, engine ile aynı `attach_package` yolundan geçer.
  Önceden CLI'da uygulanmayan `max_age_days`, yaş ve eksiklik kanalları artık
  aynıdır. Sırası bozuk/tekrarlı origin'ler reddedilir; sessiz yeniden sıralama
  ve farklı satıra özellik atama yapılmaz. Eski deney dosyaları düzeltilmez.
- Yeni analiz ve aday veri paketleri yalnız yeni çıktı dizininde oluşturulur.
  Girdi checksum'ları doğrulanır; tamamlanma kaydı payload'lardan sonra yazılır.
  Eski cache yeni kod kimliğiyle devam ettirilmez. Runtime artifact şeması değişmedi.

## Yeni fit olmadan T+1 işlem duyarlılığı

Kaynak: `research-availability-clock-pilot-v1` içindeki 16 T+1 yıllık tahmin ve
karar dosyası; iki kolun aynı **2.006 origin'i**, 2016–2023. Yeni fit sayısı **0**.
Kural önceden sabit: seçilmiş tahminin işareti, sıfır flat; ham işaret ayrı tanıdır.
Yıl veya eşik seçilmedi. Giriş/çıkış proxy'si sonraki kayıtlı Cotton barının
`open → close` hareketi; 50.000 lb kontrat **eşdeğeri** için 500 USD/(cent/lb).
Bu tutarlar sermaye getirisi değildir; gerçek kontrat PnL'si olarak sunulmaz.

| Kural | Brüt USD eşdeğeri | İşlem | Pozitif yıl | İşlem başına başa baş çift yönlü maliyet |
|---|---:|---:|---:|---:|
| Flat | 0 | 0 | 0/8 | — |
| Her gün short | 31.620,02 | 2.006 | 5/8 | 15,76 |
| Kontrol, seçilmiş | 7.090,02 | 504 | 2/8 | 14,07 |
| Müdahale, seçilmiş | 11.530,02 | 251 | 1/8 | 45,94 |
| Kontrol, ham | 21.599,88 | 2.006 | 5/8 | 10,77 |
| Müdahale, ham | 49.079,90 | 2.006 | 6/8 | 24,47 |

Seçilmiş müdahalenin **bütün 251 işlemi ve bütün brüt sonucu 2017'de**.
Kontrolün aktif yılları 2017/2020. Yüksek işlem başına ortalama, sekiz yıllık
istikrar gibi yorumlanamaz. Seçilmiş müdahalenin brüt drawdown'ı 3.719,99 USD
eşdeğeri; ham müdahaleninki 24.200,01. Maliyetler, marjin ve sermaye bilinmiyor.

Seçilmiş müdahale−kontrol farkı karar günü başına **+2,2134 USD eşdeğeri**.
Yılları aşmayan paired moving-block bootstrap, 10.000 tekrar/seed42:

| Blok | %95 fark aralığı, USD eşdeğeri/karar günü |
|---|---:|
| 20 | [−3,9006; +8,2313] |
| 60 | [−3,0739; +7,2505] |

Flat'a göre seçilmiş müdahalenin aralıkları pozitiftir: blok20
[+0,6518; +10,8240], blok60 [+2,1197; +10,0859]. Bu olguyu saklamıyoruz;
ancak aynı tek aktif yıldan gelir, kontrol katkısını veya gerçek giriş/maliyet
uygulanabilirliğini doğrulamaz. İncelenmiş yıllar ve sonraki tanısal inceleme,
bootstrap aralıklarını bağımsız strateji doğrulamasına dönüştürmez.

Ham müdahalenin 6/8 pozitif yılı, fiyat-MAE başarısızlığının **işlem hedefinde
herhangi bir sinyal yok** anlamına gelmediğini gösteren tanısal bir bulgudur.
Ham kol sonuçtan sonra üretim stratejisi seçilmedi; MAE için seçilmiş shrinkage'ın
işlem faydasını optimize ettiği varsayılmayacak. Henüz ekonomik beceri kanıtı yok.

Önceki kapanıştan giriş varsayılırsa seçilmiş müdahale 14.150,03 görünür;
bunun 2.620,01'i sonraki açılışa kadarki bileşendir. Bu giriş sinyalden öncedir.
Ayrıca CT=F günlük `open` değerinin **00:15 UTC kararından sonra** olduğu da
kanıtlanmamıştır. Her kayıt `execution_verified=false`; net PnL ve sermaye
getirisi alanları boş. Kontrat kimliği, seans zaman damgası ve işlem fiyatı yok.

## Dışarıda kalan WASDE alanları

Mevcut 98 kaynak sürümü/96 ayrı XML'den **95 aylık rapor, 26 alan** çıkarıldı:
World/ABD üretim-tüketim-stok; ABD ihracatı; Çin tüketimi, Hindistan ve Brezilya
üretimi; üç oran ve bunların aynı mahsul yılı içindeki rapor revizyonları.
Yeni veri indirilmedi, fiyat/model verisine join yapılmadı.

- World stock/use paydası domestic use; ABD için domestic use **+ exports**.
  Birimler milyon 480 lb balya; oranlar boyutsuz. FAS running-bale birimiyle karıştırılmaz.
- Yeni mahsul yılında veya 62 günü aşan rapor boşluğunda revizyon eksik bırakılır.
  2019 Ocak raporu yok; sahte rapor/interpolasyon eklenmedi. Eksiklik sıfır yapılmadı.
- 2019-11-08'in üç aynı kopyası birleştirildi. 2018-12-14, yalnız bütün
  bölgesel alanlar eşit olduğu doğrulanınca aynı ayın tekrarı olarak daraltıldı.
  Sadece World oranlarına bakarak ülke değişikliği taşıyan sürüm silinmez.
- World tarih/mahsul yılı ve iki ortak oran, eski dondurulmuş tabloyla `1e-12`
  toleransta eşleşti. Yeni ülke hücreleri her raporda eski XML parser üzerinden
  bağımsız seçimle kontrol edildi. Bu kontrol yeni alanların tam PDF mutabakatı değildir.

Paket **`wasde_regional_candidate`**, bir `as_published` eğitim paketi değil.
`model_eligible=false`, `release_allowed=false`,
`publication_timestamp_verified=false`, `first_version_verified=false`.
Rapor tarihi `available_at` olarak yeniden etiketlenmedi. Mevcut publication
loader adayı doğrulanmış kaynak gibi kabul etmez.

## Kanıt bütünlüğü ve yeniden çalışma

[Makine doğrulama kaydı](../research/evidence/trading-wasde-delivery-20261005.json)
girdileri, küçük sonuçları ve çıktı hash'lerini saklar. Önceki incelemenin 555
dosyası ve ek kaynak/karar girdileri dahil **690 orijinal dosyanın SHA-256'sı
değişmedi**. Bu, bütün yerel diskin veya geçmişte kaybolmuş dosyaların doğrulandığı
iddiası değildir. Bağımsız kontrol CSV'den tüm kuralların PnL, işlem, drawdown,
yıllık toplam ve maliyet başa baş değerlerini yeniden hesapladı.

Doğrulama: ML sentetik/sözleşme suite'i **615 passed, 3 skipped**; `ruff check ml`,
sicil doğrulaması ve diff boşluk kontrolü geçti. Gerçek GPU/sequence veya piyasa
eğitimi çalıştırılmadı. Backend/frontend kaynakları değiştirilmedi; GitHub CI
depo sözleşmelerini ayrıca denetler. İlk suite çalışırken test dosyasının import
sırası düzenlendiği için kaynak kimliği kontrolü doğru biçimde durdu; tek değişen
hash doğrulandı ve ML dosyaları sabitlendikten sonraki tam suite geçti.

Payload ve bağımsız kontrol betiği:
[research-delivery-20261005 Release](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/research-delivery-20261005).
Yeni ZIP eski evidence Release'i değiştirmez. Release manifesti ZIP SHA-256'sı,
dosya hash'leri ve çalıştırılmış ML kaynak kimliğini içerir; restore mevcut dosyanın
üzerine yazmamalıdır.

Mevcut CPU/test ortamında, `PYTHONPATH=ml/src`:

```bash
python ml/history.py check --query trading --horizon 1
python -m cottonlens_ml.research.trading_diagnostic --experiment output/full-year/experiments/research-availability-clock-pilot-v1 --market output/claude-audit/drive/data/raw/market.parquet --output output/new-trading-diagnostic
python -m cottonlens_ml.sources.wasde_regional --archive-root output/wasde-vintage --world-table output/wasde-exploration-inputs-20261002/world-balance-2016-2023.csv --output output/new-wasde-candidate
```

Girdi arşivleri önce eski evidence Release'ten ayrı dizine doğrulanarak geri
kurulur. Komutlar fit veya yeni veri indirmez; mevcut çıktı dizinini reddeder.
Doğrulanmış snapshot'lar eski kodun cache'i olarak otomatik devam ettirilmez.

## Tek sonraki iş

**İşlem fiyatı ve seans kanıtını tamamlamak.** Önce CT=F proxy yerine araştırma
sözleşmesindeki gerçek kontratın kimliği, bar/açılış zaman damgası ve 00:15 UTC'den
sonra uygulanabilir giriş tanımlanmalı; maliyet dayanağı aynı dosyada kilitlenmeli.
Bu başarılmadan ham sinyalin kârlılığı hakkında yeni model veya eşik aranmayacak.
İlk kontrol yalnız eldeki provenansı inceler; eksik kontrat/quote verisinin
edinimi bu teslimatta yapılmadı.

WASDE için kalan kabul engelleri: belirli sürümün tarihsel erişilebilirliği,
ilk/revize-vintage provenansı ve yeni alanların PDF mutabakatı/kullanım kapsamı.
Bu koşullar veri kaynağını silme gerekçesi de eğitim izni de değildir; aday paket
korunur. FAS/NASS/CFTC/POWER/FX için önceki plandaki kanıt sınırları değişmedi.
