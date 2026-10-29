# Named T+1 EWMA: geçmişten sabit ölçek kontrolü

22 piyasa +1 sentetik analitik fit tamamlandı. Her refit'te tek parametre
`mean(mature_actual/origin_EWMA)`; yeni kaynak, optimizer veya grid yok.
Önceden kilitlenen same named etiket,246slot/243tahmin/242paired,
500minimum train,189validation ve21gözlem refit korundu. λ0,94 değişmedi.

| Pozitif değer scaled EWMA lehine | Delta QLIKE | CI20 | CI60 | Top1% trim |
|---|---:|---|---|---:|
| Ham | −0,00544751 | [−0,03075360;0,01921665] | [−0,02249733;0,01104304] | −0,02192979 |
| Past-selected shrinkage | −0,00183714 | [−0,02155502;0,01736700] | [−0,01518731;0,01104222] | −0,01467007 |

Sonuç INCONCLUSIVE: bu sabit scale tarifi pozitif üstünlük sağlamadı.
Training ratio1,16375–1,24659 bir OOS başarı sinyali değildi. Yeni scale,
recency/λ veya subset grid'i açılmaz; EWMA korunur. Negatif nokta sonucu
bütün risk modellerinde bilgi yokluğunu kanıtlamaz.

2023 görülmüş tek yıl; bağımsız/6-of-8 gate veya live kanıtı değildir.
Tarihsel UTC/vintage varsayımı ve sparse observed-history sınırı korunur.
Source `443102eccde8acc51bf2de6b882088ee69f4da9ff1f5ceeed07aea77412cdbcd`;
design `0405432905085d4e4ffc1a53af5bb17f2dfb87fedc24ca8a54c3810bff6b50f9`.

8 yeniunit+15ilgili test/Ruff geçti. Native22scale ve predicted EWMA
çarpımları yeniden kuruldu. Independent242EWMA/726Decimal loss/
2.988feature scalar/4CI/22cutoff/16.764train-row doğrulandı; report fit0.
Ön kayıt352private GitHub dosyasından taze geri kuruldu; eski kanıtlar
değişmedi. Native sonuçlar `output/named-ewma-scale-control-20261013/`.

Tek sonraki sıfır-fit soru: mevcut risk modelleri üç close-return HAR
özelliğiyle çalışırken bugünkü OHLC range bilgisi yalnız ayrı GK **hedef**
tanısında mı kullanılmış? Fiyat modelindeki cotton_range feature varlığı
risk feature katkısı testi değildir. Aynı origin/hedef/EWMA kimliğinde,
tek ön kayıtlı range/error moment tanısı model eğitimini gerekçelendirebilir;
yeni veri veya GPU araması kendiliğinden açılmaz.
