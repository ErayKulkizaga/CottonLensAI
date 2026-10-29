# Sabit pamuk sezonu ve EWMA risk hatası

**NEGATIVE_ZERO_FIT_INFORMATION_SCREEN.** [USDA NASS 2010 ekim/hasat takvimi](https://www.nass.usda.gov/Publications/Todays_Reports/reports/fcdate10.pdf)
araştırma döneminden önce yayımlandı. Tek sabit gösterge origin ayı Nisan–Kasım
olduğunda1, diğer aylarda0'dır. Takvim tarımsal faaliyeti tarihler; fiyat
varyansının işaretini veya tahmin becerisini kanıtlamaz. Pozitif hipotez,
aktif aylarda ertesi Cotton gözleminin `actual/EWMA0.94−1` hatasının daha
yüksek olmasıydı. Yıl içi merkezlenmiş season/error covariance hesaplandı.

Kohort özgün CT T+1/2016–2023/2.006 origin; tarih, hedef ve fiyat değiştirilmedi.
Bootstrap yıl içinde ordinal20/60 blokla10.000 tekrar/seed42 yaptı.
Önceden dondurulan bilgi ekranı iki CI alt sınırının pozitifliğini, en az6/8
pozitif yılı ve EWMA kaybının en yüksek%1'i çıkarıldığında pozitif ilişkiyi
gerektiriyordu.

| Ölçü | Sonuç |
|---|---:|
| Yıl ağırlıklı covariance | +0,01411943 |
| CI20 | [−0,04719169; +0,06188183] |
| CI60 | [−0,02804097; +0,04924608] |
| Pozitif yıl | 5/8 |
| En yüksek %1 EWMA kaybı sonrası | −0,01017329 |

Ekran geçmedi. Bu covariance bir QLIKE kazanımı değildir; yeni model fit veya
Colab gerekçesi değildir. Geçmiş tekrar incelenmiştir, 2024+ seçime girmedi,
çoklu araştırma kararları düzeltilmedi. Kaynak saat/vintage ve CT sürekli
kontrat sınırları sürer. İlk sıfır-fit v1, yalnız Ruff C413 kod biçimi
düzeltmesi için teslim öncesi korunup v2 ile değiştirildi; ölçüm aynı kaldı.

[Özel kanıt sürümü](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/cotton-season-risk-moment-20261014)
8 dosya/SHA256 `2cf06c0d54ccd2f6765804bbfa374efdb00c59b80315e0376f3addc1e2b50830`.
Taze indirme/restore ve bağımsız Decimal2.006 origin,8 yıl covariance, iki
bootstrap CI ve1.985 trim origin denetimi geçti; makbuz byte eşit. İki
sentetik sınır testi ve Ruff geçti. [Sonuç kanıtı](../research/evidence/cotton-season-risk-moment-20261014.json)
sicile eklendi;203 aşama/543 tarif/30.244 tamamlanmış fit kopyası.
