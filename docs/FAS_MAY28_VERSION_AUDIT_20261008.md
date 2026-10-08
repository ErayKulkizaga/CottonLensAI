# FAS 28 Mayıs: sürüm ve yuvarlama ayrımı

**Sonuç: veri bozulması veya revizyon nedeni henüz belirlenemiyor.** İki
aday açıklama kaldı: basılı alt sınıf hesaplama/yuvarlaması ve API ile raporun
farklı veri sürümleri. İki küçük fark da yuvarlama varsayımıyla matematiksel
olarak uyumlu. Bu uyumluluk, varsayımın gerçek yayın yöntemi olduğunu kanıtlamaz.
Kaynak kabulü, ±50 doğrudan kontrolü ve eski kanıtlar değiştirilmedi. Sıfır fit.

## Yapılan ayırıcı kontrol

Mevcut PDF'nin 31–34. sayfaları ve aynı pinned 2020 API snapshot'ı kullanıldı.
Pima, geçmiş pazarlama yılı ve ülke yerine bölge satırları kullanılmadı.
1401/1402/1403 için birim, dönem, başlık ve ülke satırı ayrı denetlendi.
Eksik veya `*` işaretli stoklar sıfırla doldurulmaz. Üç alt sınıfın bağımsız
en yakın 100 balyaya yuvarlandığı varsayımında, her stok için ±50 aralık
toplandı; sıfır stokun alt sınırı sıfırdır. Bu aralıklar **doğrudan All Upland
kontrolünün yeni toleransı değildir**, yalnız açıklayıcı hesaplamadır.

| Ülke | Üç basılı accumulated alt sınıfı, bin balya | Varsayımsal kesin toplam aralığı, balya | Pinned API | Doğrudan ±50 |
|---|---|---:|---:|---|
| Çin | 1.472,8 + 5,5 + 0,4 = 1.478,7 | [1.478.550, 1.478.850] | 1.478.645 | Başarısız: −55 |
| Pakistan | 1.720,6 + 40,4 + 0,0 = 1.761,0 | [1.760.900, 1.761.150] | 1.760.947 | Başarısız: −53 |

**VERIFIED:** iki API değeri bu varsayımsal aralıkların içindedir. Önceki
[doğrudan mutabakat](FAS_COUNTRY_REPORT_REVIEW_20261008.md) hâlâ başarısızdır.
**PLAUSIBLE:** alt sınıf yuvarlaması farkları açıklayabilir.
**NOT SUPPORTED:** yalnız −55/−53 farkından veri bozulması veya revizyon sonucuna gitmek.
Yayıncının hesaplama yöntemi ve aynı sürümdeki kesin alt sınıf değerleri yoktur.

## Özgün yayın erişimi

Aramada görünen USDA `Year2020/CAM-5-28-20.pdf` adresinin iki path yazımı ve
eski `weeklyHist.htm` adresi doğrudan anonim GET ile 404 döndü. İstek zamanları,
nihai URL ve yanıt checksum'ları yeni kanıtta saklandı. 404 bugünün durumudur;
2020'de raporun yayımlanmadığı veya hiçbir arşivde bulunamayacağı çıkarılmaz.
GovDelivery araması ikinci bir 28 Mayıs byte sürümü buldurmadı; kapsamlı
internet/arşiv yokluğu kanıtı sayılmaz. Birincil eski dosya korunur.

Mevcut ortamda `USDA_FAS_API_KEY` tanımlı değil. Kimlik doğrulama atlatılmadı;
yeni piyasa veri seti veya toplu API geçmişi indirilmedi. **Bugünün API anahtarı
ve güncel grade snapshot'ı, 2020 ilk sürümünü kanıtlamaz.** Yayın/ilk sürüm ve
karar anındaki erişim hâlâ doğrulanmamış durumda; hiçbir saat uydurulmadı.

## Sonraki tek minimal test

**Çin, 2020-05-28, 1404 accumulatedExports için özgün as-issued tam balya
değerini**, o yayın sürümüne bağlanan kimlik/erişim kanıtıyla karşılaştır:

- Özgün kesin değer **1.478.645** ise bu hücrede API sürüm farkı açıklaması
  dışlanır; basılı gösterim/hesaplama farkı araştırılır.
- Farklıysa bu hücrede sürüm/veri değişimi vardır; eski API tarihsel ilk sürüm
  yerine geçirilemez. Yuvarlama ayrıca eşlik edebilir.
- Yalnız bir ondalıklı PDF, güncel API, URL tarihi veya CreationDate gelirse
  test ayrım yapamaz. Belirsizlik açık kalır; daha fazla aynı tür arama/eğitim yok.

Bu **tek hücre** neden ayrımı için minimaldir; bütün ülke geçmişini kabul
etmeye yetmez. Pakistan ve kalan kapsam ancak bundan sonra aynı kanıt türüyle
denetlenir. Yeni ön kayıt/model yarışı otomatik sonraki adım değildir.

## Araç ve koruma

Mevcut çevrimdışı `ml/review_fas_archive.py` aracına isteğe bağlı
`--subtype-precision` eklendi; `--country-reference` gerektirir. Uyumlu aralık
sonucuna rağmen doğrudan stok farkı exit **2** olarak kalır. Mevcut eğitim,
özellik, hedef, yayın saati, runtime ve exporter akışları değişmedi.

[Tam kayıt](../research/evidence/fas-may28-subtype-review-20261008.json),
[küçük kanıt](../research/evidence/fas-may28-version-audit-20261008.json) ve
[Release](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/fas-may28-version-20261008)
kod/girdi kimliği ve her alt sınıf değerini korur. 8 inceleme girdisi, 24 eski
kanıt dosyası, 134 önceki sicil kaydı ve trials dosyası değişmedi. Yeni kayıt
piyasa performansı için **INCONCLUSIVE**; 435 tarif / 25.000 fit makbuzu aynı.
83 ilgili test, Ruff ve sicil doğrulaması geçti; gerçek piyasa/GPU eğitimi yok.
Arşiv yeni dizine açılıp 16 üyenin ve 216 kaynak dosyasının checksum'ı
doğrulandı. O kaynaktan çevrimdışı tekrar, beklenen exit 2'yi ve dondurulmuş
kayıtla tam JSON eşitliğini üretti; eski kaynak dosyaları yeniden yazılmadı.
