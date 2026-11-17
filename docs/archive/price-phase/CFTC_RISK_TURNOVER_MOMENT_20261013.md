# CFTC pozisyon değişimi ve EWMA risk hatası: sıfır-fit kontrol

Tek ön kayıtlı gösterge `abs(cftc_managed_share_change_L1)` idi: managed-money
net/OI oranının geçmiş haftalık değişim büyüklüğü. Amaç pozisyon ayarlamasının
EWMA'nın ötesinde T+1 risk bilgisi taşıyıp taşımadığını taramaktı; yön veya
lag seçimi yapılmadı. CFTC'nin [2008 Cotton piyasa incelemesi](https://www.cftc.gov/sites/default/files/idc/groups/public/%40newsroom/documents/file/cottonfuturesmarketreport0110.pdf)
pozisyon/piyasa mekanizması için bağlamdır; bu göstergenin tahmin gücünü ispatlamaz.

Mevcut doğrulanmış 730 Cotton033661/FutOnly kayıt, report+7 gün/strict next
Cotton observation/L1, 10 ek observation carry ve 29 report maskesi kullanıldı.
Final historical vintage ve calendar clock varsayımdır; first-publication UTC
doğrulanmış değildir. Eski CFTC signed T+5 fiyat deneyleri tekrar fit edilmedi.
Risk ve HAR+CFTC sicil kontrolleri incelendi; eşleşme yokluğu tek başına yenilik
ispatı değildir. İlk prepare hatası (verified reader'ın record_id'yi çıkarması)
sonuç üretmeden durdu; makbuz ve ayrı prepared namespace korunur.

CT proxy T+1 kare-getiri/EWMA0,94, 2016–2023 ve 2.006 origin aynı kalır.
1.875 eligible/131 missing; ordinal slotlar sıkıştırılmadı. Yıl içi centering,
20/60 moving-block, 10.000 draw/seed42 ve her draw'da recenter kullanıldı.

| Kontrol | Sonuç |
|---|---:|
| Covariance | −0,0002501055 |
| CI20 | [−0,0040275945; +0,0027691750] |
| CI60 | [−0,0035496407; +0,0027824247] |
| Pozitif yıl | 2/8 |
| Top1% baseline-loss trim | −0,0013914897 |

Ön kayıtlı pozitif screen başarısız. İşareti sonradan çevirme, yeni lag/grid
veya learner açılmaz. Bu sonuç ilişki/kurulabilirlik tanısıdır; kaynak-geneli
bilgi yokluğu, measured forecast skill, bağımsız holdout ya da Colab gerekçesi
değildir. Araştırma geçmişi yeniden kullanılmıştır; multiplicity düzeltilmemiştir.

Bağımsız verifier 2.006 position clock/feature ve Decimal gerçek hedef/hata
satırını yeniden kurdu; iki aralığı ayrı `numpy.cov` yöntemiyle doğruladı.
Yeni fit0. Kaynak/clock kodu ve girdiler hash ile bağlandı. İlgili math/source
testleri21 passed, Ruff temiz. Native eski modeller, live126 ve eski fit sicili
değişmez. Yerel kanıt: `output/cftc-risk-turnover-moment-20261013/prepared-v1/`.

Private `cftc-risk-turnover-preflight-20261013` Release taze indirildi;15dosya
checksum ile geri kuruldu. Bağımsız feature/error/ikiCI replay geçti;
verification JSON byte eşit. Eski198aşama/541fit tarifi değişmedi.
Kanıt: `research/evidence/cftc-risk-turnover-preflight-20261013.json`.
