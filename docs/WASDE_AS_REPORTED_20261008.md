# WASDE: resmî geçmişle mutabakat ve erişim saati — 8 Ekim 2026

**VERIFIED:** 2016–2023 bölgesel adayındaki 95 raporun 950 temel değeri,
USDA'nın yayımlandığı biçimi koruyan resmî CSV geçmişiyle eşleşiyor.
Üç oran ve sezon içi revizyonlarla toplam **2.470 alan kontrolü, sıfır fark**.
117 bilinmeyen revizyon hücresi boş kalıyor. Bu çalışma piyasa tahmini değildir:
**sıfır fit, sıfır tarihsel model kabulü**. Naive ve başarı eşikleri değişmedi.

## Kanıt ve tekrar çalıştırma

- `wasde-as-reported-verification-v1`: 1 resmî ZIP ve 36 aylık CSV;
  36.711.505 ham byte, 468.362 CSV satırı. Bunlar 95 aylık bilgi güncellemesi;
  468 bin bağımsız eğitim gözlemi değildir.
- ZIP'in tek üyesi 2016–2020 için 59 rapor içeriyor. 2021–2023 için 36 rapor var.
  Ocak 2019 için satır uydurulmadı. CSV tarihleri hem ISO hem ABD ay/gün/yıl
  biçiminde; belirsiz tarih tahmini kullanılmıyor.
- Yalnız `World Cotton Supply and Use`, doğru bölge/alan, milyon 480-pound balya,
  rapor ayı ve rapordaki en yeni crop year seçiliyor. Reliability veya özet
  Cotton tabloları seçilmiyor. Başka alanlardaki negatif `Loss` değerleri bu
  on adet negatif olmayan seviye alanının bozuk olduğu şeklinde yorumlanmıyor.
- Ayrı stdlib CSV ayrıştırıcısı ve ayrı oran/revizyon hesabı aynı 2.470 kontrolü
  doğruladı. Önceki **2.089 dosya** yeniden hash'lendi; değişiklik yok.
- 11 Aralık 2018 resmî CSV'de var; 14 Aralık için ayrı Cotton raporu yok.
  Önceki PDF/XML denetiminde iki sürümün Cotton değerleri eşitti. USDA'nın
  düzeltme kaydı değişen alanın süt olduğunu belirtiyor. Bu, bütün PDF byte'larının
  ilk sürüm olduğunu kanıtlamaz. 8 Kasım 2019 adayındaki aynı-gün kopyaları tek
  resmî CSV raporuyla eşleşiyor.

Denetim modülü: `ml/src/cottonlens_ml/sources/wasde_as_reported.py`.
Girdi checksum, rapor tarihi, sezon, birim veya hücre sayısı uyuşmazlığında
sessiz kesişim/onarım yok. Sayısal uyuşmazlıklar karşılaştırma dosyasında görünür;
`numeric_verified=false`, CLI başarısız çıkış kodu verir. Her durumda model kabulü
kapalıdır. Eski çıktının üzerine yazılmaz, tamamlama makbuzu en son yazılır.

```bash
python ml/history.py check --query wasde-as-reported
python -m cottonlens_ml.sources.wasde_as_reported \
  --candidate-root output/trading-wasde-delivery-20261005/wasde-regional-v1 \
  --exports-root output/wasde-as-reported-20261008 \
  --output output/wasde-as-reported-repeat
```

İkinci komut ML kaynakları kurulu olduğunda veya `PYTHONPATH=ml/src` ile çalışır.
Eski kanıtların geri kurulması gerekir; ağ isteği/eğitim başlatmaz.
Yeni dizindeki makbuz aynı veri/kod kimliklerine bağlıdır; işletim sistemi yolu
bilimsel kimliğe katılmaz.

## Yeni öğrenilen zaman ayrımı

**VERIFIED:** `ReleaseDate` ve `ReleaseTime=12:00` alanları geçmiş raporu
tanımlar; CSV'nin gerçek teslim saatini kanıtlamaz. USDA'nın 2021 kitapçığı,
2010–2020 konsolide CSV geçmişinin ilk kez 2021'de sağlandığını ve aylık CSV'lerin
rapor sonrası iki iş günü içinde güncelleneceğini belirtir. 2025 toplantı kaydı
CSV'nin ertesi gün yayımlandığını, Mayıs 2025'ten itibaren raporla aynı gün yaklaşık
iki saat sonra yayımlanmasının planlandığını açıklar. Bu genel politika,
2016–2023'te tek bir dosyanın fiilî erişim makbuzu değildir.

Dolayısıyla CSV formatının kendisini 2016'daki bir kullanıcıya verilmiş kabul
edemeyiz. CSV burada **değer sürümünü kontrol eden kaynak**. O tarihte yayımlanmış
PDF'den aynı değerleri çıkarmak ayrı bir tarihsel bilgi yolu olabilir; onun
karar anından önce erişilebilirlik dayanağı ayrıca bağlanmalıdır.
2026 indirme zamanı, report date, planlanan saat ve ilk sürüm birbirinin yerine
kullanılmadı. Kaynaklara `available_at` atanmadı.

**STRONGLY SUPPORTED:** adayın on temel alanındaki başarısızlığı bozuk sayılar
veya bu arşivde görülen sonradan revize edilmiş değerlerle açıklamak için kanıt
yok. Önceki PDF/XML mutabakatına bağımsız resmî CSV mutabakatı eklendi.
Bu sonuç, bilgi kümesinin T+1/T+5 becerisi olduğunu veya olmadığını göstermez.

**Tek sonraki iş:** 11 Aralık 2018 Cotton değerleri için, ilk rapor ile sonraki
repostu ayıran **değer düzeyinde** bir erişilebilirlik incelemesi: checksum bağlı
PDF/CSV değerleri ve tarihsel resmî yayın kaydı/eşzamanlı arşiv yakalaması aynı
makbuzda bağlanmalı; ilk 00:15 UTC kararından önce bir üst sınır gerçekten
kanıtlanabiliyor mu? Bir raporda geçerli kanıt kurulmadan 95 rapora otomatik saat
ataması yapılmayacak. Kanıt bulunamazsa açık varsayımlı duyarlılık çalışması ile
doğrulanmış geçmiş backtest ayrımı korunacak; bu paketten yeni eğitim başlamaz.

## Teslimat ve doğrulama

21 yeni regresyon testi; ilgili WASDE kontrolleriyle 67 test başarılı.
Yerel tam ML: **670 başarılı, 3 atlandı**; değişen modül/test için Ruff başarılı.
GPU, gerçek piyasa eğitimi, canlı model veya UI değişikliği yok.
Kaynak/test değişiklikleri gerçek sayısal denetimden önce bitirildi; son dar test,
aynı kaynak bytes'ını kontrol etti.

Kimlikler ve karşılaştırma çıktıları
[`research/evidence/wasde-as-reported-20261008.json`](../research/evidence/wasde-as-reported-20261008.json)
ve [ek kanıt Release'i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/wasde-as-reported-20261008)
üzerinden bulunur. Eski Release/aday/sicil kayıtları değiştirilmedi; bu sıfır-fit
mutabakatı piyasa becerisi bakımından **INCONCLUSIVE** olarak ayrı kaydedildi.

Resmî dayanaklar:

- [USDA as-reported geçmişi ve revizyon politikası](https://www.usda.gov/historical-wasde-report-data-3).
- [USDA düzeltme kaydı](https://www.usda.gov/historical-changes-revisions).
- [2021 Data Users kitapçığı, basılı sayfa 24](https://data.nass.usda.gov/Education_and_Outreach/Meeting/2021/2021_Spring_Data_Users_Booklet.pdf).
- [2025 Data Users soru/yanıtları, basılı sayfa 8–9](https://data.nass.usda.gov/Education_and_Outreach/Meeting/2025/2025%20Spring%20Data%20Users%20Meeting%20Question%20and%20Answer%20Summary%20with%20Slides.pdf).
