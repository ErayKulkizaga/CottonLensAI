# Resmî günlük named settlement arşivi ve sekiz yıllık EWMA kontrolü

[USDA AMS tarihsel upland fiyat arşivindeki](https://apps.ams.usda.gov/Cotton/HistoricalUplandPrices/)
2012–2017 günlük DSQ ZIP dosyaları ile 2017–2024 sezon çalışma kitaplarının
yalnız `Futures` sayfası okundu. [USDA'nın fiyat verisi SSS belgesi](https://mymarketnews.ams.usda.gov/sites/default/files/resources/2025-08/FAQs-Cotton-Price-Reporting-and-Data-Access_508.pdf)
2025 Ağustos öncesi fiyatların yeni veritabanı/API üzerinden indirilmediğini,
eski rapor ve `apps.ams.usda.gov/Cotton/` yolunun kullanıldığını açıklar.
Kaynaklar 12 orijinal dosya ve 30.060 futures settlement satırından oluşur.
Kaynak tarihiyle uyuşmayan 50 satır, sent altı hassasiyetli 8 satır ve eş
10 tekrarlı anahtar açıkça korunur; ilk ikisi katı erişim hesabında kullanılmaz.
Bozuk ilk kontrat üçüncüyü ikinci sıraya yükseltmez.

Orijinal 2016–2023 T+1 Cotton takvimindeki 2.006 originin 1.989'unda,
origin gününde seçilmiş **ikinci teslimat kontratı** hedef günde de geçerli
fiyata sahiptir. Aynı kontratın en az 22 geçmiş ardışık Cotton çifti, her yıl
189 geçmiş doğrulama etiketi ve en dar 2016 eğitim penceresinde 619 olgun
etiket vardır. Sekiz yılın tamamı bu kaynak/temporal ön kontrolü geçer.
Yeni panelin eski bağımsız 2023 named paneliyle örtüşen 244 gününde kontrat
ayı ve gerçekleşen kare log getiri eşittir; farklı erişim günleri korunur.

Tarihsel erişim saati yalnız **varsayımdır**: kaynak günü+1 takvim günü
00:00 UTC; karar origin+1 gün 00:15 UTC. Orijinal ilk yayımlanan sürümün
bytes'ı ve gerçek publication UTC kanıtlanmadı. Konsolide çalışma kitaplarının
arşive konma tarihi işlem gününden sonraya düşebilir. Bu nedenle sonuç
koşullu tarihsel araştırmadır; canlı/backtest ayrımı ve 2024+ seçimsiz audit
korunur. Named settlement riski CT=F fiyat proxy'sine eşitlenmez.

Eski 2023 own-history HAR'ın seyrek geçmişinden farklı olarak bu kaynak
2012'ye uzanır. Bu somut veri farkı için aynı dört parametreli QLIKE
EWMA-relative HAR, EWMA λ=0,94, 21 gözlem refit, 500 minimum train,
189/3×63 geçmiş validation ve ağırlıklar 0/0,25/0,5/0,75/1 önceden
sabitlendi. 174 piyasa+1 sentetik fit, tek CPU proses/en fazla iki thread ile
aynı ledger'a yazıldı. Kaynak/tasarım kimlikleri ve fit bütçesi [özel ön kayıt
sürümünde](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/ams-named-ewma-har-prereg-20261014)
19 dosyadan taze geri kurularak doğrulandı.

| EWMA eksi aday QLIKE; pozitif aday lehine | Ortalama | Blok20 %95 GA | Blok60 %95 GA | Pozitif yıl | Üst %1 EWMA kaybı trim |
|---|---:|---|---|---:|---:|
| Ham HAR | +0,006775 | [−0,007987; +0,021795] | [−0,006128; +0,019890] | 6/8 | −0,012701 |
| Geçmişte seçilen karışım | +0,003735 | [−0,007086; +0,014693] | [−0,005258; +0,013716] | 4/8 | −0,008363 |

Önceden sabit kapı iki aralığın alt sınırı>0, en az 6/8 pozitif yıl ve
trimmed kazanç>0 idi. **Geçilmedi.** Dış 2.006 slotun 1.989'u scoreable;
eksik 17 slot istatistik takviminde tutuldu. Raporlama fit0; yerel bağımsız
denetim 1.989 ham kaynak-etkiket çiftini, 5.967 Decimal QLIKE kaybını ve
dört blok aralığını doğruladı. Native 174 modelin payload, kimlik,
optimizer gradyanı ve tahmini ayrıca yeniden oynatıldı. Ölçülmüş EWMA
üstünlüğü, bağımsız holdout, fiyat/yön kapısı veya Colab gerekçesi yoktur.

[Özel sonuç sürümü](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/ams-named-ewma-har-result-20261014)
1.249 çalışma dosyasını ve 12 orijinal kaynak dosyasını saklar. Sonuç paketi
taze indirilip 1.249 dosya hash'iyle geri kuruldu; bağımsız denetim byte
eşit yeniden oynadı. On beş özel sürüm asset'inin uzak SHA-256 özeti yerel
orijinal kaynak ve sonuç dosyalarıyla eşleşti. Sicil 204 aşama/545 tarif/
30.419 tamamlanmış fit kopyası olarak doğrulandı. Yerel yürütme:
`output/ams-named-ewma-har-20261014/execution-v1/`; kaynak ön kontrolü:
`output/ams-upland-price-preflight-20261014/full-source-v4/` ve
`named-own-readiness-v3/`. Sicilde aynı kapsamın tekrar fit'i açılmamalı.
