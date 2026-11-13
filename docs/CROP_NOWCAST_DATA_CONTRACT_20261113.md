# Pamuk verimi nowcast: yapılabilirlik ve veri sözleşmesi taslağı — 13 Kasım 2026

Durum: **TASLAK — ön kayıt değil, deney değil.** Büyük indirme, model eğitimi
ve 2022–2025 test yılı sonucuna bakma yapılmadı. Bu belge, ayrı bir araştırma
hattının yapılabilir olup olmadığını ve hangi sızıntı kurallarıyla
yürütülmesi gerektiğini kayda geçirir. Fiyat protokolleri, Naive/EWMA
referansları ve önceki sonuçlar değişmez.

**Verim tahmini başarılı olsa bile, bu tek başına pamuk fiyatının daha iyi
tahmin edildiğini göstermez.** Fiyata katkı varsa, sonradan ayrı ve önceden
kaydedilmiş bir testle gösterilmelidir.

## Önceki planda düzeltilen hatalar

- Birim: USDA verimi **lb / hasat edilen acre** olarak verilir. Önceki metindeki
  "lb/dönüm" yanlıştır.
- Başarı olasılıkları (%80, %40–50, %20) hesaba dayanmayan tahminlerdi; geri
  çekildi. Başarı ölçütleri, test yılları açılmadan önce ayrı ön kayıtla
  yazılacak.
- Aynı yılın CDL tarla haritasını kullanmak sızıntıdır (aşağıda kural).
- Hedef "yalnız uydu + hava" iken yöntemde ürün durum endeksi de vardı. Bunlar
  artık ayrı bilgi kümeleri olarak tanımlanıyor.
- Yaklaşık 3.000 ilçe-yıl satırı 3.000 bağımsız gözlem değildir. Aynı yıldaki
  ilçeler ortak havadan etkilenir; çıkarım birimi **yıl** (ya da yıl blokları)
  olmalıdır.

## Yapılabilirlik kontrolleri (13 Kasım, yalnız metadata ve 2022 öncesi örnekler)

| Konu | Bulgu | Sonuç |
|---|---|---|
| MODIS MOD/MYD13Q1 v061 (Planetary Computer) | Anonim STAC erişimi çalışıyor. 2018 örnek öğelerinin hepsi `created` = Aralık 2021. | Arşiv **yeniden işlenmiş sürüm**; 2008–2021'de gerçek zamanlı erişilen sürüm (C5/C6) geri kurulamaz. Composite 16 günlük; `end_datetime` kullanılmalı, başlangıç tarihi değil. Üretim gecikmesi henüz belgeyle doğrulanmadı. |
| USDA CDL (Planetary Computer) | 2008–2021 `cropland` katmanları var. 2008 `frequency` ve 2021 `cultivated` çok yıllık katmanları da var. | Çok yıllık katmanlar geleceği içerir; **yasak**. 2022+ CDL Planetary Computer'da yok; CropScape servisi denemede zaman aşımına uğradı. |
| CDL yayın tarihleri | Yalnız 2025 CDL için resmi tarih bulundu: **27 Şubat 2026**. 2008–2024 için resmi sayfada tarih yok. | Kural: Y yılı nowcast'inde yalnız **CDL ≤ Y−1**. Eski yılların yayın tarihleri ayrıca bulunmalı. Bulunamazsa, Y−1 haritasının 1 Ağustos Y'den önce yayımlandığı **varsayım** diye etiketlenir. |
| NASA POWER | Meteoroloji kaynağı MERRA-2, uzantısı GEOS-IT (~2 gün gecikme). MERRA-2 geldiğinde GEOS-IT üzerine yazılıyor. | Tarihsel değerlendirme, gerçek zamanlı sürüm yerine sonradan gelen MERRA-2 ile yapılır. Bu bir **vintage farkıdır** ve sonuçta sınır olarak yazılır. Gerçek zamanlı test ancak ileriye dönük yapılabilir; ilk uygun tarih 1 Ağustos 2027. |
| USDA Ağustos Crop Production raporları | ESMIS'te bulundu: 2012-08-10, 2015-08-12, 2021-08-12. 2018 için yalnız özel rapor bulundu; ana rapor ayrıca aranmalı. | Raporlarda eyalet bazında hasat alanı, verim ve üretim tahmini var. Örnek: 2015 ABD upland Ağustos tahmini 7.749 bin acre, 784 lb/acre, 12.650 bin balya. 2021 raporu, Teksas'ın objective-yield anketinin dışında kaldığını yazıyor. |
| PDF ayrıştırma | `pdftotext -layout` eyalet satırlarında sütunları kaydırıyor (Teksas 2014/2015 değerleri karışıyor). | PDF tek başına kaynak olamaz. Birincil kaynak QuickStats olmalı; PDF yalnız çapraz kontrol için kullanılır. |
| NASS QuickStats API | Anahtarsız istek HTTP 401 döndü. | Ücretsiz API anahtarı gerekli. Anahtar yalnız ortam değişkeninde tutulur; dosyaya veya Git'e yazılmaz. |

İndirilen dört PDF ve hash'leri yerel `output/crop-nowcast-feasibility-20261113/`
klasöründe ve `research/evidence/crop-nowcast-feasibility-20261113.json`
dosyasında. 2022 CDL için yapılan bir CropScape alan sorgusu zaman aşımına
uğradı ve yanıtı görülmeden durduruldu; test yılı bilgisi okunmadı.

## Hedef tanımı

- **Birincil hedef:** NASS upland pamuk **üretimi** (480 lb balya), eyalet ve ABD
  düzeyinde. Değer, Y+1 Ocak yıllık özetindeki ilk yıllık tahmindir. Sonraki
  revizyonlar ayrı kaydedilir.
- **İkincil hedef:** verim (lb / hasat edilen acre). Bunu USDA'nın Ağustos
  verimiyle karşılaştırmak için, kendi hasat alanı tahminimizden hesaplanır.
- **Gelecekteki hasat alanı hiçbir ağırlıkta kullanılmaz.** Bu sızıntı olur.
  Ulusal toplama şöyle yapılır: Haziran sonu Acreage raporundaki eyalet ekim
  alanı (1 Ağustos'tan önce yayımlanır; tarihi doğrulanacak) × bizim tahmin
  ettiğimiz hasat oranı × bizim tahmin ettiğimiz verim.
- **Karşılaştırma:** Aynı hedef (ilk yıllık tahmin) ile bizim 1 Ağustos
  nowcast'imiz ve USDA Ağustos tahmini (~10–12 Ağustos) karşılaştırılır. Bilgi
  farkı tabloda açıkça yazılır:
  - USDA çiftçi ve tarla anketlerini kullanır; biz kullanmayız.
  - Biz ~10 gün daha erken tahmin yaparız.
  - Bizim uydu ve hava verimiz yeniden işlenmiş sürümdür.

## Bilgi kümeleri (her test yılı Y, kesim 31 Temmuz Y sonu)

| Model | İzin verilen girdiler |
|---|---|
| **S (uydu + hava)** | `end_datetime + L ≤ kesim` koşulunu sağlayan MODIS composite'leri (L ön kayıtta sabitlenecek; üretim gecikmesi belgelenene kadar muhafazakâr 8 gün önerisi); CDL ≤ Y−1 cotton maskesi; POWER günlük değerleri ≤ kesim − 3 gün; Y yılı Haziran Acreage ekim alanı; Y'den önceki yılların yayımlanmış verimleri |
| **S+C** | S + kesimden önceki son NASS Crop Progress durum raporu (yayın tarihiyle). İkincil modeldir; "yalnız uydu" iddiasına girmez. |
| Referans: trend | Eyalet doğrusal trendi, yalnız Y'den önceki yıllar |
| Referans: USDA | Ağustos Crop Production tahmini (rapor tarihi kayıtlı) |

Eğitim etiketleri yalnız kesim anında yayımlanmış olanlardır. Y−1 ilçe verimi
genellikle Y yılının ilk aylarında çıkar; her yıl için yayın tarihi kayıtla
doğrulanacak, doğrulanamazsa o etiket kullanılmaz.

## Değerlendirme düzeni

- **Geliştirme yılları:** 2010–2021, yuvarlanan origin (her yıl yalnız önceki
  yıllarla eğitilir). Model ve özellik seçimi yalnız burada yapılır.
- **Mühürlü test yılları: 2022–2025.** Ön kayıttan önce bu yılların verimi, USDA
  hatası, CDL'si ve uydu sonuçları **açılmaz**. Bir kez açılır; sonuç olumsuz
  da olsa raporlanır.
- **İstatistiksel güç sınırı:** 4 test yılı ulusal karşılaştırma için çok azdır.
  Eyalet × yıl hataları **yıl blok bootstrap** ile raporlanır. Ulusal sonuç
  betimleyici kalır; kesin üstünlük iddiası 4 yıldan çıkarılmaz.

## Sızıntı kontrolleri (kodda test olarak yazılacak)

1. Her girdi satırında `available_at` alanı bulunur. Model, `available_at >
   kesim` olan bir satır görürse durur.
2. Y yılı için CDL yılı < Y olmalıdır. `frequency` ve `cultivated` katmanları
   yasaktır.
3. Bir MODIS composite yalnız `end_datetime + L ≤ kesim` ise kullanılır;
   başlangıç tarihine bakılmaz.
4. Y yılının hasat alanı, üretimi veya verimi hiçbir özellikte ya da ağırlıkta
   bulunmaz (adversarial test: bu değerler değiştirilince tahmin değişmemelidir).
5. Test yılları kodda mühürlüdür: ön kayıt hash'i verilmeden 2022–2025 için
   skor fonksiyonu çalışmaz.
6. İlçe gizleme ("(D)" withheld) değerleri sıfır sayılmaz; eksik olarak kalır.

## Açık sorular (geniş indirmeden önce)

- 2008–2024 CDL yayın tarihleri ve Haziran Acreage raporlarının yayın tarihleri.
- MODIS v061 üretim gecikmesi (L değeri); MODIS kullanıcı rehberinden alınacak.
- QuickStats API anahtarı (kullanıcıdan).
- 2018 Ağustos ana raporu.
- Teksas'taki tarla terki: hasat oranını uydudan tahmin etmek bu hattın en zor
  kısmı. Geliştirme yıllarında ayrı ölçülmeli.

## Karar

Hat yapılabilir görünüyor: veri erişimi, resmi karşılaştırma ve sızıntı
kuralları tanımlanabiliyor. Ancak üç vintage sınırı (MODIS yeniden işleme,
MERRA-2 üzerine yazma, CDL yayın tarihleri) tarihsel testi gerçek zamanlı
testten ayırır. Sonraki adım, yukarıdaki açık soruların kapatılmasıdır. Ardından
test yıllarına bakmadan ön kayıt yazılır. Geniş indirme ve eğitim ayrı onay
ister.
