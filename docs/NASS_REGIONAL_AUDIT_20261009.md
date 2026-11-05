# Texas/NASS: mevcut raporlardaki bölgesel bilgi ve sürüm denetimi

9 Ekim 2026 — **sıfır fit**, yeni indirme yok. Gece çalışan ileri kayıt görevi
ve otomasyonu değiştirilmedi. [Sayısal kanıt](../research/evidence/nass-regional-audit-20261009.json)
ve [yeniden üretim girdileri](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/nass-regional-audit-20261009).

## Ne doğrulandı?

**VERIFIED:** Eski ulusal kondisyon tablosunun 311 haftası, 14 yıllık içerik
denetimi ve bunların seçtiği resmî TXT/page/makbuz dosyaları yeniden doğrulandı:
**949 girdi**, önce/sonra aynı SHA256. Kapsam 30 Mayıs 2010–29 Ekim 2023
gözlem haftalarıdır; bütün yılın veya bütün Crop Progress sezonunun arşivi değildir.

Her rapordan Texas'ın beş kondisyon kategorisi ayrıldı; ulusal satır eski
tabloya eşit kalır. Bölgesel good+excellent ile ulusal good+excellent,
**311/311 haftada farklıdır**: medyan mutlak fark **13 yüzde puan**, Texas−ulusal
fark aralığı **−24…+8 puan**. Bu farklı bilgi, tahmin katkısı kanıtı değildir.

| Texas gelişim evresi | Tablosu mevcut rapor | 311 içinde tablo yok |
|---|---:|---:|
| Planted | 46 | 265 |
| Squaring | 148 | 163 |
| Setting Bolls | 157 | 154 |
| Bolls Opening | 173 | 138 |
| Harvested | 112 | 199 |

Toplam **636 gelişim tablosu**. Ekim tablosu yılda yalnız **1–5** raporda mevcut.
Arşiv, ulusal kondisyon haftaları seçilerek oluşturulmuş: erken ekim ve geç
hasat tarihinin tam olduğunu söyleyemeyiz. Tablo yokluğu, yüzde sıfır veya
tamamlanmış evre olarak kodlanmadı. Henüz günlük hizalama/forward-fill yoktur.

## İki somut ayrıştırma/sürüm tehlikesi

**VERIFIED:** 1 Haziran 2010 legacy raporunda dört sütun **bu hafta, önceki
hafta, önceki yıl, beş yıllık ortalama** sırasındadır. Modern tabloda sıra
**önceki yıl, önceki hafta, bu hafta, beş yıllık ortalama** olur. Yeni parser
sütunu sabit indeks varsayımından değil, üç tarih/yıl başlığının doğrulanmasından
seçer. Ortalama dönemi raporda yayımlanan önceki beş yıl olmalıdır; bütün
tarihten hesaplanmış klimatoloji eklenmedi.

**VERIFIED:** 11 Haziran 2012 raporu (10 Haziran gözlem haftası), önceki haftanın
Texas planted değerini **81→83**, squaring değerini **12→11** olarak `* Revised.`
işaretiyle değiştiriyor. 4 Haziran raporundaki özgün **81/12** korunur; yeni
raporun tekrar ettiği **83/11** ayrı hücrelerdir. Sonraki rapordaki revize geçmiş
sütunla eski haftayı yeniden yazmak leakage oluştururdu. Burada yapılmadı.

Bu iki riskin önceki **ulusal kondisyon T+5** deneyine zarar verdiğine dair
kanıt yok: o yol gelişim sütunlarını veya Texas satırını kullanmıyordu.
Bu nedenle geçmiş negatif sonucun nedeni diye sunulmuyorlar.

`NA`, `(NA)`, `(D)` null ve özgün qualifier olarak korunur. `-` ancak raporda
“Represents zero” tanımı varsa sıfırdır. `*` ancak “Revised” açıklaması varsa
kabul edilir. Bilinmeyen simge, tarih/sütun değişimi, çift tablo, checksum veya
ulusal değer uyuşmazlığında komut exit 2 verir; sessiz düzeltme yapmaz.

## Geçmiş test neyi elemedi?

Sicil kontrolü yapıldı: `nass-exploration-v1` yalnız ulusal kondisyonu T+5'te
denedi. Seçilmiş kondisyon lag1/2/6 Naive kazancı **−%0,6922 / −%1,0569 /
−%0,8788**. Bunlar Texas gelişim evrelerini veya T+1'i sınamış sayılmaz.
Yeni sicil kaydı `nass-texas-report-audit-v1`, **INCONCLUSIVE** sınıfında veri
denetimidir; eğitim/performance kaydı değildir ve metric alanı boştur.

## Kabul ve tek sonraki iş

Panel **karantinadadır**: `model_eligible=false`, `release_allowed=false`,
`available_at=null`, `availability_policy=UNSET`. TXT içindeki release günü,
HTML tarih alanı ve 2026 ingestion zamanı ayrı tutulur; bunlar belirli dosyanın
ilk sürümüne/tarihsel erişim saatine otomatik kanıt olmaz.

**Bir sonraki NASS işi:** mevcut **30 Mayıs 2023** raporunun bu byte sürümüne
bağlı tarihsel erişim tanığını incelemek. Texas planted **50**, raporun kendi
ortalaması **54**; squaring **5**, ortalaması **10**. İçerik eşleştirmesinin
yanına o sürümün en geç erişilebilir olduğu zaman eklenmeli. Tek örnek bütün
311 haftayı kabul ettirmez; yeterli kanıt yoksa bu durum açık tutulur.

Tam sezon varsayımı, yeni veri toplama veya gerçek PIT onayı olmadan bu panel
doğrudan eğitime aktarılmayacak. Varsayımlı duyarlılık pilotu ayrıca ön kayıt
gerektirir. FAS özgün hücre engeli ve gece ilk yayın kontrolü ayrı işlerdir;
NASS incelemesi bunları açmaz. Daha büyük model/grid başlatılmadı.

## Yeniden üretim ve koruma

Mevcut ML CPU ortamında; yeni paket kurulmaz. CLI yalnız okur ve stdout'a JSON
yazar. Yeni paket için ayrı namespace kullanın; eski dosyaların üzerine yazmayın.

```bash
python ml/history.py check --query nass-texas
python ml/review_nass_regions.py --table INPUTS/table/upland-condition-2010-2023.csv --audits INPUTS/audits
```

Frozen source **`ec003b687a0c5b164579950822b0bbeb482ea25ab303064d8436bc2ee6a583fa`**,
**229 ML dosyası**. Yeni parser/audit için **33 sentetik regresyon testi** geçti:
iki sütun biçimi, eksiklik/simge, tarih/ortalama, bozuk/değişen girdi, çevrimdışı
çalışma ve sonraki revizyonun eski raporu değiştirmemesi. Bunlar piyasa başarısı
veya tarihsel yayın doğrulaması değildir. GPU kontrolü/eğitimi yapılmadı.

**141 eski sicil kaydı**, **435 tarif / 25.000 tamamlanmış fit makbuzu** ve
`trials.json` korunur. Önceki **36 kanıt JSON'u + `.gitkeep`**, 13 ileri kilit/girdi/
origin dosyası ve ana dirty checkout değişmedi. Source ZIP, panel ve resmî
girdiler Release'te; Git'te kod ve küçük özet bulunur. Anahtar/QuickStats ham
API snapshot'ları, canlı quote verisi, görev XML'i ve makine yolları bu pakete girmez.
