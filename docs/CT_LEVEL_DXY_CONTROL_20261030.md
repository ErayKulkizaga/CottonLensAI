# CT risk seviyesi ve DXY bilgisi: sabit karşılaştırma

Profil: `ct-risk-level-dxy-control-t1-v1`. Ön kayıt; henüz piyasa sonucu yok.

**Soru:** Ham DXY oynaklık durumu, Cotton'ın kendi risk seviyesi katsayısını
serbest bırakmaya göre ek tahmin bilgisi sağlıyor mu? İki kol aynı beş
parametreye sahiptir; dış sonuçlara bakarak kol, pencere veya gecikme seçilmez.

| Unsur | Kilitli karar |
|---|---|
| Veri/hedef | Mevcut CT proxy; sonraki seansın kare log-getirisi |
| Kohort | Aynı 2016–2023 / 2.006 origin; tarihler kesiştirilmez/düşürülmez |
| Referans | EWMA λ=0,94; eski 169 native dört-parametre HAR tekrar eğitilmez |
| `own_level` | Üç relative-HAR + serbest `log(origin CT EWMA)` |
| `dxy_state` | Üç relative-HAR + `log(max(raw DXY beş kare-getiri ortalaması,1e-12))` |
| Öğrenme | Aynı QLIKE/BFGS, expanding geçmiş, beş seans etiket olgunluğu |
| Seçim/refit | Geçmiş 3×63 validation; 21 gözlem refit; ağırlık 0/0,25/0,5/0,75/1 |
| Birincil | Geçmişte seçilmiş DXY kolu − EWMA QLIKE katkısı |
| Mekanizma | Ham DXY − ham own-level katkısı, aynı kapasitede |
| Bütçe | 338 piyasa + en fazla 1 native sentetik fit; tek CPU süreç/iki thread/30 dakika |

DXY istatistiği **önce ham DXY gözlem takviminde** hesaplanır. Ardından
`source_date + 1 gün 00:00 UTC <= CT_origin + 1 gün 00:15 UTC` koşuluyla
son bar alınır; en fazla üç Cotton gözlemi eskilik kabul edilir. Bu saat
**varsayımdır**: tarihsel teslim saati/ilk-vintage doğrulanmış değildir.
Geçersiz son bar atlanarak eski geçerli bar aranmaz. Kaynak tarihi, pencere
başlangıcı, availability, karar saati ve yaş tahmin payload'ında tutulur.
Eksik kaynak nedeniyle train veya OOS origin düşürülmez; kontrol durur.

Ham veri SHA-256:
`23cfabb7ccb2553f0547bc3290ff4916ce91c788c98a94acdb2e5b9cd1f3151c`.
[Kaynak denetimi](DXY_RISK_SOURCE_AUDIT_20261029.md) bunun mevcut private
GitHub arşivinden geri kurulmuş veri olduğunu kaydeder; yeni dataset indirilmez.

## Geçmişi tekrar etmeme

Sicilde `risk-level-source-t1-v1` vardır; eski DXY deneyi saklanır. Ancak
[AMS kaynak kusuru](AMS_REQUIRED_CELL_READINESS_20261029.md) ve farklı named
hedef nedeniyle bu deney temiz CT-DXY risk testi değildir. Eski fiyat-getiri
DXY testleri de risk katkısı hükmü vermez. Son [CT şok deneyi](CT_SHOCK_FORECAST_CONTROL_20261029.md)
338 piyasa fit'iyle tamamlandı; tekrar çalıştırılmaz. Yeni çalışma eski
girdileri/sonuçları değiştirmeyen ayrı namespace ve source kimliği kullanır.

## Önceden belirlenmiş karar

Pozitif ortalama QLIKE katkısı, yıl sınırlarını koruyan paired bootstrap'ın
**hem 20 hem 60 blok** %95 alt sınırının pozitifliği ve **6/8 pozitif yıl**
gerekir. Bootstrap 10.000 tekrar, seed 42. Top%1 EWMA kaybını çıkarmak
hassasiyettir; ek başarı kapısı veya ana kohortu değiştirme aracı değildir.
QLIKE negatif olabileceği için yüzde başarı yazılmaz.

Bu koşullar geçilse bile yalnız **tarihsel takip adayı** doğar. Geçmiş yıllar
çok kez incelendi; bağımsız holdout, ileri başarı, ekonomik/işlem başarısı veya
temiz named-contract risk sonucu sayılmaz. Birincil sonuç belirsizken ham
veya ikincil kol sonradan birincil ilan edilmez. Grid/ufuk taraması/GPU yoktur.

Kaynak, hedef veya native kimlik kontrolü başarısızsa performans hükmü
verilmez. Tamamlanmış payload ve bağımsız kontrol olmadan completed yazılmaz.
Raporlamada fit girişleri kapatılır ve eski/native dosya hash'leri korunur.

## Uygulama ve teslim sınırı

Mevcut `research.ct_shock_control` ve ortak QLIKE learner/Ledger kullanılır;
ikinci eğitim motoru kurulmaz. Bağımsız kontrol DXY'yi log-difference ve
`merge_asof` ile yeniden kurar; engine'in searchsorted hizalamasını çağırmaz.
49 ilgili sentetik/regresyon testi ve Ruff geçti; bu testler piyasa başarısı değildir.

Kaynak/test/veri/ortam kilidi, private Release taze indirme/restore ve
bağımsız preflight olmadan piyasa eğitimi başlamaz. Sonuç da native payload,
eşlenmiş loss ve bağımsız replay ile private Release/draft PR'da korunur.
Saatlik Windows görevi ve üç tetikleyici kapalı kalır; ileri yayın otomasyonu
yeniden kurulmaz. Live kaynak/cohort126, eski veriler ve Drive değiştirilmez.

## Dondurulan kimlik ve ön kayıt teslimi

Source: `5846f9cbb0f360731e84f251ed311f5f665510590594645204daa95c6438ab26`.
Design: `5dd87b3d117daac2e3bac87fae9ae4b7151f339b55cb7bbc5344e347ceef7df8`.
Data: `5ad179ef26e95bcc8b8fad8252275b717c377aef09c77cc2e89dd1391aa1dfb3`.

Private [ön kayıt Release'i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/ct-level-dxy-prereg-20261030)
1.799 dosya / sıfır dışlamayla taze geri kuruldu; frozen bağımsız preflight
byte eşit geçti. ZIP SHA-256:
`d0e2f4d1df387e404659aaf38a09c6337947ae6a2f9834ddb1830ddd19a3fed9`.
Bağımsız kontrol 3.520 close, 169 geçmiş iş ve minimum 1.191 olgun train
satırı doğruladı. Etiket/saat/ilk-vintage sınırları bu teslimle değişmez.
