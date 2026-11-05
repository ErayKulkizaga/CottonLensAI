# FAS ülke paneli: hazırlık ilerledi, tarihsel kabul açılmadı

**VERIFIED:** mevcut checksum bağlı 15 yıllık kaynak yeniden indirilmeksizin
752 hafta × dört kod = **3.008 satırlık karantina paneline** dönüştürüldü.
Gözlem kapsamı 6 Ağustos 2009–28 Aralık 2023; 2010–2024 kaynak adları
pazarlama yılı kimlikleridir. Yeni piyasa fit'i **0**.

`ml/prepare_fas_countries.py` ve `sources/fas_country_candidate.py:prepare`
eski alan denetimini önce/sonra çalıştırır; pinned kaynak, derlenmiş ulusal
tablo veya manifest değişirse çıktı üretilmez. CLI yalnız stdout JSON üretir.
Bu panel bir eğitim paketi veya yeni deney ön kaydı değildir.

## Hazırlanan temsil

Kodlar **4890, 5350, 5520, 5700**; dördünde de 752/752 rapor satırı vardır.
2020 isim tanıkları bütün tarihsel isim/kod geçerliliğine genişletilmedi.

- Net satış, haftalık sevkiyat ve bekleyen satış seviyeleri korunur.
- Sevkiyat/bekleyen satış paydası, o haftanın bütün bildirilen ülke satırlarıdır;
  tam rapor kapsamının doğrulandığı söylenmez.
- Dört haftalık akış toplamı, aynı pazarlama yılında dört ardışık takvim haftası
  ister. Her kodda 707 toplam tanımlı; yıl başındaki 45 bilinmeyen değer doldurulmaz.
- Eksik ülke/hafta sıfır değildir. Sıfır payda oranı tanımsızdır; net satış
  payı üretilmez. Dört kodda **413 negatif net satış** değeri korunur.
- Açıklanamayan `accumulatedExports` farkının bulunduğu alan ve onun cebirsel
  türevi `currentMYTotalCommitment` bu aday temsilde yoktur. Ham alanlar silinmedi;
  bunları çıkarmak diğer alanlara publication/vintage onayı vermez.

**Yeni kaynak durumu:** 29 Temmuz 2021, kod **2230**, `outstandingSales = −76`;
kaynak birikimli ihracat **26.767**, commitment **26.691**, net satış **−168**
bildiriyor. Muhasebe eşitliği geçer; ekonomik nedeni doğrulanmadı. Ham değer
değiştirilmez. Toplam stok pozitif olsa bile negatif bileşen bulunan o haftanın
dört bekleyen satış payı `null` kalır; satır düşürülmez.

## Eğitime kalan iş

| Aşama | Durum / kalan iş |
|---|---|
| CPU eğitim, purged seçim/refit, checkpoint ve paired rapor motoru | Çalışıyor; WASDE'de 424 piyasa fit'i tamamlandı. |
| Ülke aritmetiği / veri koruması | Bu teslimatta hazır; Cotton seanslarına hizalanmış girdi değil. |
| Belirli sürümün erişilebilirlik/kapsam kabulü | **Dış kanıt bekliyor**; yalnız FAS ülke hipotezini bloke eder. |
| Kabul kapsamına göre adapter / eşit eksiklik kontrolü | Kaynak politikası belli olduktan sonra tamamlanıp test edilmeli. |
| Tek deney ön kaydı / tam fit bütçesi / kod-veri kimliği | Henüz yapılmadı; eski ulusal T+5 tarifi tekrarlanmayacak. |
| Küçük CPU pilotu / bağımsız metrik doğrulaması | Ön kayıt sonrasında; 30 dakikalık oturum sınırı hedeflenir, süre garantisi değildir. |

Özgün Çin hücresi iki uyuşmazlık açıklamasını ayırır; **tek başına bütün 752
haftanın, haftalık akışların veya erişim sürümlerinin kabulünü sağlamaz**.
Tam saniye zorunlu değildir: kullanılacak belirli sürümü karar anından önceye
bağlayan güvenilir tarihsel üst sınır yeterli olabilir. Gözlem tarihi, genel
yayın takvimi ve bugünkü indirme tarihi bunun yerine geçmez.

Bu engel GPU/broker/anahtar ihtiyacı değildir. Kaynak kanıtına tarih verilemez;
başka modelleri aynı veride tekrar eğitmek eksikliği çözmez. Kapsam kabulü
açıldıktan sonra da adapter/test ve ön kayıt işi kaldığından “tek komutla şimdi
eğitim” denmez. [Mevcut kabul sırası](RESEARCH_DATA_REENTRY_PLAN_20261005.md) korunur.

**Somut sonuç zaten var:** [bölgesel WASDE T+1](WASDE_REGIONAL_T1_RESULT_20261008.md)
5.016 tahminde seçilmiş D0/D1 Naive kazancını −%0,5229 / −%0,0107 buldu;
pratik hedefi kurtarmadı. Bu panel başarı sonucu değildir. Pozitif beceriye
tarih sözü verilmez; görülmüş tarihsel yıllar bağımsız ileri doğrulama sayılmaz.

## Tekrar üretme ve sınırlar

Depo kökünde Python 3.12+; bu CLI üçüncü taraf paket istemez:

```bash
python ml/history.py check --query fas-country
python -S ml/prepare_fas_countries.py --raw-root FAS_RAW_ROOT --manifest MANIFEST --table TABLE
```

Girdiler eski FAS `source.json` ağacı ile
`weekly-sales-2010-2023.manifest.json` / `.csv` çiftidir. Kod ve pinned girdiler
checksum bağlı Release paketinde korunur. Çıktı **model_eligible=false,
release_allowed=false, availability_policy=UNSET** taşır. `release_allowed`
canlı model/kaynak kabulüdür; kanıt ZIP'inin saklanmasına dair lisans beyanı değildir.

Testler gelecekteki haftaların geçmiş aritmetiği etkilememesini, eksik haftaları
sıkıştırmama, yıl sınırı, sıfır/negatif stok bileşeni, negatif satış, kaynak
değişimi ve bağımlılıksız CLI davranışını doğrular. Bunlar **kaynak
revizyon/vintage leakage'ını ortadan kaldırmaz**.
[Sayısal kayıt](../research/evidence/fas-country-preparation-20261008.json)
piyasa performansı yerine hazırlık kapsamını ve korunan kimlikleri kaydeder.

Yerel doğrulama: 34 odaklı test; kapsamlı ML **779 geçti / 3 atlandı**;
son sınır testi güçlendirmesinden sonra 18 aday testi ve Ruff geçti.
24 ZIP üyesi/221 ML kaynak dosyası restore edildi; panel JSON'u birebir eşit.
18 eski girdi, 31 önceki kanıt/belge dosyası, trials ve 137 eski sicil kaydı
korundu; ana kullanıcı çalışma ağacının Git durumu değişmedi.
[Release paketi](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/fas-country-preparation-20261008)
ve [üye checksum envanteri](../research/evidence/fas-country-preparation-release-20261008.json)
tam tekrar üretme girdilerini sağlar; yeni model/tahmin içermez.
