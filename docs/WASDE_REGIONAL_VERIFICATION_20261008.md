# Bölgesel WASDE: sayı, sürüm ve zaman denetimi — 8 Ekim 2026

**Karar:** Bölgesel bilgi korunur; tarihsel model kabulü verilmez. Sayıların doğru
aktarılması, bu değer sürümünün tahmin anında bilindiğini tek başına kanıtlamaz.
Bu çalışma sıfır fit veri doğrulamasıdır; piyasa performansı sonucu değildir.

## Denetlenen paket

`wasde-regional-candidate-v1`, 2016–2023'te 95 rapor ve 26 alan içerir:
10 Dünya/ABD/Çin/Hindistan/Brezilya seviyesi, 3 oran ve bunların 13 revizyonu.
Seviyeler milyon 480-pound balya; oranlar boyutsuzdur. ABD stocks/use paydası
domestic use + exports, Dünya paydası domestic use'dır.

- Kaynakta 98 rapor bağlantısı / 96 ayrı XML sürümü vardır. Aynı tarihli
  2019-11-08 kopyaları çelişki olmadan tek rapora indirgenmiştir.
- 2018-12-14 kopyasının **bütün 13 seviyesi ve mahsul yılı**, 2018-12-11 ile
  aynıdır; ayrıca aylık gözlem yapılmaz. Bu eşitlik ilk yayın saatini kanıtlamaz.
- Ocak 2019 bilinen raporsuz aydır; yeni rapor veya sıfır revizyon üretilmez.
- 13 seviye/oran alanında eksik hücre yoktur. 117 eksik revizyon hücresi korunur:
  ilk rapor ve sekiz Mayıs mahsul yılı geçişi, her birinde 13 alan. Toplam
  2.470 özellik hücresinin 2.353'ü sayısal, 117'si bilinmeyendir.

`sources/wasde_regional.py::regional_values` raporun yayımladığı en yeni mahsul
yılının **o ayki tahminini** seçer. Gelecek mahsul yılı için o gün yayımlanmış
tahmin, gelecekte gerçekleşmiş üretimle aynı şey değildir.
`add_revisions` yalnız aynı mahsul yılı ve 1–62 günlük önceki raporla fark alır.
Sonraki rapordaki düzeltilmiş eski değerler geçmiş snapshot'a geri yazılmaz.

## İki ayrı doğrulama

`sources/wasde_regional_validation.py` eski `passed` kayıtlarını kanıt yerine
koymaz: arşivdeki PDF'i yeniden çıkarır; ay, birim, tablo sütun sırası, mahsul
yılı, tahmin ayı ve yedi bölgenin tam hücre kapsamını XML ile karşılaştırır.
Sayısal uyuşmazlıkta farklı çıkarım ayarı denenerek hata silinmez. Sadece
desteklenmeyen metin düzeninde ikinci çıkarıcı kullanılabilir.

**VERIFIED:** 98 kaynak sürümü / 96 ayrı PDF için **19.208 hücre**, sıfır
sayısal uyuşmazlık. 81 sürüm pypdf, 17 sürüm pdfplumber (`x=3`, `y=3`)
çıkarımıyla doğrulandı. Bağımsız XML hesabı 98 sürümde 1.274 seviye/oran
kontrolü ve aday CSV'de 2.470 seviye/revizyon kontrolü yaptı. Önceki 1.694
girdi dahil **2.089 orijinal dosyanın SHA-256'sı değişmedi**. Bu bütün yerel
diskin veya geçmişte kaybolmuş payload'ların doğrulandığı iddiası değildir.

Nisan 2019 başlığındaki boşluklar, Eylül 2020'de görünmez `filler` metninin
Brezilya satırını bölmesi ve sayfa konumuna güvenmenin riski incelendi. Eylül
2020'nin 27. sayfası görsel olarak da kontrol edildi: Brezilya üretimi Ağustos
ve Eylül satırlarında 12,00'dır. Kaynak PDF veya XML değiştirilmedi.
Doğrulayıcı iki çıkarıcıda da bütün sayfaları kapsar; kısmi tabloya geçiş yoktur.

Bağımsız betik `verify_independently.py`, mevcut `cotton_rows`, `regional_values`
ve `add_revisions` fonksiyonlarını çağırmadan XML hücrelerinin üst düğümlerini
okur; 13 seviye/oran ve 13 revizyonu CSV ile yeniden hesaplayarak karşılaştırır.
Bu ikinci bir PDF çıkarımı iddiası değildir. Kesin hücre sayıları, dosya
hash'leri ve tamamlanma kimliği [makine kanıtında](../research/evidence/wasde-regional-verification-20261008.json)
ve [Release'te](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/wasde-verification-20261008)
korunur. Eski kanıtlar üzerine yazılmaz.

## Leakage korumasında somut düzeltme

Önceki WASDE compile/import sözleşmesi, vintage kanıt dosyasının paket içinde
bulunmasını kontrol ediyordu; dosyanın **tam değer/sürüm/saat iddiasını**
özellik satırına bağlamıyordu. Sentetik regresyonda sonradan değişmiş bir değer
ve yeniden checksum'lanmış parquet bu boşluğu gösterdi. Bunun tamamlanmış
bir piyasa deneyini fiilen etkilediği gösterilmedi; eski deneyler silinmez.

`sources/wasde_admission.py::require_review` artık compile ve load aşamalarında
aynı inceleme sözleşmesini uygular: kaynak SHA-256, vintage kimliği, tam
özellik değerleri hash'i, erişilebilirlik saati ve dayanak dosyaları eşleşmelidir.
Takvim, PDF creation/modification veya bugünkü indirmenin geçmişe atanması
kabul edilen kanıt türü değildir. Bu bir **inceleme sözleşmesi kontrolüdür**;
insanın kaynak kanıtını doğru değerlendirdiğini kriptografik olarak kanıtlamaz.

Mevcut 95 satırda tarihi belirlenmiş raporlar vardır, fakat bu belirli değer
sürümlerinin 00:15 UTC karar anından önce mevcut olduğunu doğrulayan inceleme
paketi yoktur. `available_at` boş, `first_version_verified=false`,
`publication_timestamp_verified=false`, `model_eligible=false` kalır.
2026 ingestion saatleri 2016–2023 yayın saatlerine dönüştürülmez.

Ayrıca generic `compile_review` tam ve sonlu özellik snapshot'ı ister.
117 bilinmeyen revizyon hücresi bulunan bu 26-alanlı aday doğrudan o sözleşmeye
sokulamaz. Kabul sırasında eksiklik açıkça temsil edilmelidir; origin silmek
veya bilinmeyen revizyonu sıfır yapmak çözüm değildir. Mevcut araştırma
preprocessing'i eğitim verisinde imputation + missingness uygular; burada
o aşamaya geçilmedi.

## Kaynağın bize öğrettiği ve tek sonraki kontrol

**VERIFIED:** Elimizde eski üç Dünya alanından daha geniş bir bilgi temsili
vardır. 95 aylık rapor vardır; günlük forward-fill yeni bağımsız rapor üretmez.
Revizyon, önceki rapora farktır; piyasa beklentisine göre sürpriz değildir.
Yeni temsilin T+1/T+5 katkısı henüz ölçülmedi. Eski T+5 kaynak testleri bu yeni
temsilin T+1 için negatif kanıtı sayılamaz.

USDA'nın [tarihsel “as reported” veri açıklaması](https://www.usda.gov/historical-wasde-report-data-3),
yayımlandığı dönemde bildirilen değerleri sonraki yeni bilgi revizyonlarından
ayırır. [Resmî düzeltme kaydı](https://www.usda.gov/historical-changes-revisions)
2018-12-14 repostunun süt bilançosuyla ilgili olduğunu söyler; bu bir Cotton
değişikliği iddiası değildir. “As reported” açıklaması da arşivlenen XML/PDF
baytlarının ilk sürüm olduğunu veya tek tek dosyaların teslim saatini kendiliğinden
kanıtlamaz. [WASDE sayfasındaki](https://www.usda.gov/about-usda/general-information/staff-offices/office-chief-economist/commodity-markets/wasde-report)
öğlen ET takvimi fiilî dosya erişim kaydı yerine kullanılamaz.

**Tek sonraki kontrol:** USDA'nın yayımlandığı dönemdeki değerleri koruduğunu
açıkça belirttiği tarihsel export ile bu 95 raporun 10 temel hücresini eşleştir;
2018-12-11/14 gibi kopyaları ayrıca ayır. Bu, PDF baytları değişti diye Cotton
bilgisi de değişti sanmayı önler. Eşleşen değer sürümü için karar anından önce
erişilebilirlik dayanağı ayrıca kayda girmeli; sonuç yalnız takvimle onaylanmaz.
Bu teslimatta o export indirilmedi. Zaman/sürüm engeli çözülmeden yeni fit,
model grid'i, kaynak ekleme veya otomatik yayın başlatılmaz.

Doğrulama: 53 dar kapsamlı test, tam ML suite'inde 649 geçti / 3 atlandı;
Ruff geçti. Bunlar sentetik/sözleşme kontrolleridir. GPU ve gerçek piyasa
eğitimi yapılmadı; API/runtime artifact şeması, backend ve frontend değişmedi.

Yeniden kontrol, ayrı çıktı dizini ve mevcut PDF inceleme ortamında
(`PYTHONPATH=ml/src`; `pypdf`/`pdfplumber` backend bağımlılığı değildir):

```bash
python ml/history.py check --query wasde-regional
python -m cottonlens_ml.sources.wasde_regional_validation --candidate output/trading-wasde-delivery-20261005/wasde-regional-v1 --archive-root output/wasde-vintage --output output/new-wasde-numeric-check
```

Tamamlanmış audit üzerine yazılmaz; yarım kalan kontrol ancak kod, PDF
çıkarıcı sürümü, aday ve kaynak/ingestion makbuz kimlikleri eşleşirse devam eder.
Değişmiş veri veya makbuz eski checkpoint'i kullanamaz. Bu komut bir veri
indirme, model eğitimi veya tarihsel erişilebilirlik onay komutu değildir.
