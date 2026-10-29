# Mevcut DXY verisi: ham kaynaktan risk girdisi kurulabilirliği

**VERIFIED — yalnız veri yeniden kurma.** Yeni veri indirilmedi ve DXY'nin
tahmin katkısı ölçülmedi. Daha önce GitHub'dan geri kurulmuş raw market
dosyası, mevcut AMS risk çalışmasının tutulmuş feature tablosuyla karşılaştırıldı.
Üç ayrı yerel eski kopya aynı raw market SHA-256 değerini taşıyor:
`23cfabb7ccb2553f0547bc3290ff4916ce91c788c98a94acdb2e5b9cd1f3151c`.

Bağımsız as-of yeniden kurma mevcut feature-builder'ı çağırmaz. Her Cotton
origin'i için yalnız **önceki Cotton gözleminden geç olmayan** son pozitif
DXY fiyatını kullanır; üç gözlem azami eskilik kuralı aynıdır. Getiriler
bu eşlenmiş fiyatların log oranıdır. Takvim boşluklarında satırlar silinmez,
gelecek kapanışı geçmişe doldurulmaz ve eski veri/tahminler değiştirilmez.

| Denetim | Sonuç |
|---|---:|
| Ham Cotton close ile aynı bulunan mevcut fiyatlar | 3.520 |
| DXY 1/5/20 getiri alanlarında bağımsız kontrol edilen finite scalar | 10.531 |
| Kaynak tarihi, availability ve eskiliği kontrol edilen satır | 3.520 |
| Getirilerde en büyük mutlak hesap farkı | 6,94e−18 |
| Yeni beş-gözlem kare-getiri ortalamasının hazır olduğu eval origin'i | 2.006 / 2.006 |
| Aynı girdinin hazır olduğu benzersiz geçmiş train origin'i | 2.787 / 2.787 |
| Değerlendirmede en büyük kaynak yaşı | 2 Cotton gözlemi |

Eksiklik desenleri de eşittir; yalnız ortak finite kesişim alınmadı.
Cotton takviminin3.519ardışık endpoint'i korunur; yeni beşli ortalama
gerçekten ardışık **Cotton gözlemleri** üzerindedir, beş DXY seansı veya
intraday realized variance diye adlandırılmaz.

## Saat ve kanıt sınırı

Kaynak barının availability saati **kaynak günü+1 gün00:00UTC varsayımıdır**;
karar origin+1gün00:15UTC'dir. Saklanan varsayımsal saatler değerlendirme
originlerinin hiçbirinde karar sonrasında değildir. Bu, Yahoo'nun tarihsel
teslim saati veya ilk-vintage kanıtı değildir. Kaynak dosyasının2026'ya uzanması
geçmiş feature'a gelecek fiyat eklemez: as-of sınırı ve causal rolling bunu
önler. Gelecekteki raw fiyat değişikliklerinin geçmiş çıktıyı değiştirmediği
regresyon testi vardır. Bu sınırlar kaynak/model kabulü yerine geçmez.

Mevcut risk recipe'si yalnız üç pamuk HAR girdisi ve origin EWMA state'ini
kullanır. DXY alanının feature tablosunda bulunması, risk modelinin onu
kullandığı anlamına gelmez. Önceki DXY fiyat-MAE/zamanlama deneylerinden bu
yeni risk girdisinin faydalı veya faydasız olduğu çıkarılamaz.

**PLAUSIBLE, henüz sınanmadı:** bu farklı piyasa durumu pamuk riskine ek
bilgi sağlayabilir. Tek sonraki test, aynı native risk tahminlerinde bu sabit
DXY beş-gözlem risk state'inin geçmişte normalize edilmiş QLIKE score'udur.
Lag, hedef, cohort, model, refit ve referans aynı kalmalı; sonra sonuçtan
yeni bir lag/window/işaret araması türetilmemelidir. Yeni eğitim başlatılmadı.

## Saklama ve yeniden oynatma

[Private kaynak sürümü](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/dxy-risk-source-audit-20261029)
9 dosya/13.188.224 mantıksal bayttır; exclusion0. ZIP SHA-256:
`02bac64091e3002d2bc55639594dc0fff11ab53613936aea2b796cf248e21271`.
Taze indirme ve restore sonrası saklanan script, yalnız geri kurulmuş
raw/history/cohort/ready dosyalarından aynı makbuz ve yeniden kurulan CSV'yi
bayt eşit üretti. Model veya market fit'i yoktur. Üç yeni source testi,
dört curvature testi ve Ruff geçti.
