# Named T+5 zayÄ±f-sinyal kontrolÃ¼ â€” mevcut tasarÄ±mÄ±n yÃ¼rÃ¼tÃ¼lmesi

PR44 yalnÄ±z girdileri doÄŸruladÄ±, Ã¶ÄŸrenici fit0. Bu teslimat aynÄ± tasarÄ±mÄ±
`Experiment.fit`/Ledger yoluna baÄŸlar; yeni model, grid veya piyasa fit'i deÄŸildir.
Eski readiness, kaynak/kapasite/loss deneyleri deÄŸiÅŸtirilmez ve tekrarlanmaz.

## Kilitli soru ve sÄ±nÄ±rlar

Mevcut sabit XGB, origin'de bilinen tek shallow-split Ã¶ÄŸreticinin pratik
bÃ¼yÃ¼klÃ¼kteki etkisini mevcut eksiklik, gÃ¼rÃ¼ltÃ¼ ve Ã¶rnek miktarÄ±nda Ã§Ä±karabiliyor mu?
ÃœÃ§ seed703/1703/2703 Ã— null/injected;32 alan, geÃ§miÅŸ3Ã—63 iÃ§ seÃ§im,
246 dÄ±ÅŸ origin/242 uygun/4 bilinmeyen,100 aÄŸaÃ§/depth2/eta.03/minchild20.
Toplam132 sentetik fit, piyasa fit0; bir sÃ¼reÃ§/iki thread,30 dakika oturum.
Fiyatlar tarihsel referans, etiketler semi-synthetic; tutarlÄ± traded futures
patikasÄ± veya baÄŸÄ±msÄ±z piyasa holdout'u iddiasÄ± yoktur.

Ã–ÄŸretici/threshold/beta/sigma ilk528 olgun satÄ±rdan sabitlenmiÅŸti; deÄŸiÅŸtirilmez.
Oracle yaklaÅŸÄ±k%10 beklenen kazanÃ§, Ã¶ÄŸrenci baÅŸarÄ±sÄ± deÄŸildir. Sentetik
gÃ¼rÃ¼ltÃ¼ beÅŸ gelecek gÃ¼nlÃ¼k innovation toplamÄ±dÄ±r; yalnÄ±z olgunluk kurallarÄ±na
uyan geÃ§miÅŸ etiketler eÄŸitime girebilir. Feature whitelist'e oracle/noise/
gelecek etiket eklenmez. Preprocessing ve hedef Ã¶lÃ§eÄŸi her fit'in geÃ§miÅŸinden gelir.

## Ã–n kayÄ±t ve doÄŸrulama

Kaynak/veri/ortam/design/rapor kimliÄŸi dondurulur; yeni namespace gerekir.
Mevcut readiness senaryolarÄ± bayt dÃ¼zeyinde yeniden Ã¼retilir; native Ã¶ÄŸrenici
ve feature/target kodu eski snapshot ile eÅŸleÅŸir. Taze GitHub Ã¶n kayÄ±t geri
kurmasÄ± olmadan `prepare` eÄŸitim deposu oluÅŸturamaz. Drive mirror kullanÄ±lmaz.

YÃ¼rÃ¼tme `python ml/full_year_cpu.py` Ã¼zerinden aynÄ± engine'e gider:
`prepare â†’ pilot-plan â†’ pilot â†’ report`. Profil
`named-weak-signal-control-t5-v1`; aÃ§Ä±k registration/decision-contract ve
ayrÄ± CPU ortamÄ± seÃ§ilir. BÃ¼tÃ§ede duruÅŸ checkpoint'i korur; baÅŸka kimliÄŸin
cache'i veya yarÄ±m makbuz completed sayÄ±lamaz. Ek Ã¶ÄŸrenme kontrolÃ¼ fit'i yoktur;
bu altÄ± senaryo zaten kontrolÃ¼n kendisidir.

Ham ve geÃ§miÅŸte seÃ§ilmiÅŸ tahminler, origin/target/eligibility, Ã¶lÃ§ek,
Ã¶ÄŸretici/noise ve seÃ§ilmiÅŸ aÄŸÄ±rlÄ±k korunur.132 native modelin tahminleri,
yalnÄ±z geÃ§miÅŸten yeniden hesaplanan preprocessing/scaler ile tekrar Ã§Ä±karÄ±lÄ±r.
BaÄŸÄ±msÄ±z doÄŸrulayÄ±cÄ± Gaussian quadrature ile beklenen MAE'yi ve scalar geÃ§miÅŸ
shrinkage seÃ§imini denetler; yeni fit yapmaz. Ham/selected yÃ¶n ve aktiflik,
gerÃ§ekleÅŸmiÅŸ MAE ve log/standardized MSE ikincil betimsel sonuÃ§lardÄ±r.

## Ã–nceden belirlenen karar

Birincil Ã¶lÃ§Ã¼t **ham Ã¶ÄŸrenicinin bilinen koÅŸullu daÄŸÄ±lÄ±mdaki beklenen dÄ±ÅŸ
fiyat-MAE kazancÄ±**; geÃ§miÅŸ shrinkage ve gerÃ§ekleÅŸmiÅŸ noise kaybÄ± ayrÄ± gÃ¶sterilir.

- ÃœÃ§ injected seed â‰¥%5 ve her null kazancÄ± â‰¤0: yalnÄ±z bu basit Ã¶ÄŸretici iÃ§in
  Ã¶ÄŸrenme yeterliÄŸi desteklenir; mevcut bilgi/temsil daha gÃ¼Ã§lÃ¼ ÅŸÃ¼phelidir.
- ÃœÃ§ injected seed <%5 ve null kontrolÃ¼ saÄŸlamsa: bu sabit dÃ¼zen yararlÄ±
  bilinen etkiyi yeterince Ã§Ä±karamÄ±yor; kaynakta sinyal yokluÄŸu Ã§Ä±karÄ±lamaz.
- KarÄ±ÅŸÄ±k seed veya baÅŸarÄ±sÄ±z null: belirsiz; otomatik reseed/grid yok.

ÃœÃ§ seed kalibre edilmiÅŸ istatistiksel gÃ¼Ã§ tahmini deÄŸildir. %5, T+1%53/T+5%55,
6/8 yÄ±l Ã¼rÃ¼n kapÄ±larÄ± deÄŸiÅŸmez; sentetik baÅŸarÄ± piyasa baÅŸarÄ±sÄ± yerine geÃ§mez.
SonuÃ§tan Ã¶nce yeni piyasa verisi/model/loss/grid veya canlÄ±126 skor aÃ§Ä±lmaz.

## Ön kayıt R2 — fit öncesi sınır düzeltmesi

R1 kaynak/ön kayıt değiştirilmeden korunur; fit0. Bir bilinmeyen origin'de
referans fiyatı da NaN: NaN–NaN karşılaştırması aynı eksikliği kabul eder,
fiyat/gerçekleşen hedef doldurulmaz. CSV okuyucu `null` kol adını eksik değer
saymamalıdır; yalnız boş alanlar NA kabul edilir. R2 bu rapor sınırlarını
regresyon testiyle korur. Öğretici/model/cohort/132-fit bütçesi değişmez.
