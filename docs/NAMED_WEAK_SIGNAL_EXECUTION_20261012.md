# Named T+5 zayıf-sinyal kontrolü — mevcut tasarımın yürütülmesi

PR44 yalnız girdileri doğruladı, öğrenici fit0. Bu teslimat aynı tasarımı
`Experiment.fit`/Ledger yoluna bağlar; yeni model, grid veya piyasa fit'i değildir.
Eski readiness, kaynak/kapasite/loss deneyleri değiştirilmez ve tekrarlanmaz.

## Kilitli soru ve sınırlar

Mevcut sabit XGB, origin'de bilinen tek shallow-split öğreticinin pratik
büyüklükteki etkisini mevcut eksiklik, gürültü ve örnek miktarında çıkarabiliyor mu?
Üç seed703/1703/2703 × null/injected;32 alan, geçmiş3×63 iç seçim,
246 dış origin/242 uygun/4 bilinmeyen,100 ağaç/depth2/eta.03/minchild20.
Toplam132 sentetik fit, piyasa fit0; bir süreç/iki thread,30 dakika oturum.
Fiyatlar tarihsel referans, etiketler semi-synthetic; tutarlı traded futures
patikası veya bağımsız piyasa holdout'u iddiası yoktur.

Öğretici/threshold/beta/sigma ilk528 olgun satırdan sabitlenmişti; değiştirilmez.
Oracle yaklaşık%10 beklenen kazanç, öğrenci başarısı değildir. Sentetik
gürültü beş gelecek günlük innovation toplamıdır; yalnız olgunluk kurallarına
uyan geçmiş etiketler eğitime girebilir. Feature whitelist'e oracle/noise/
gelecek etiket eklenmez. Preprocessing ve hedef ölçeği her fit'in geçmişinden gelir.

## Ön kayıt ve doğrulama

Kaynak/veri/ortam/design/rapor kimliği dondurulur; yeni namespace gerekir.
Mevcut readiness senaryoları bayt düzeyinde yeniden üretilir; native öğrenici
ve feature/target kodu eski snapshot ile eşleşir. Taze GitHub ön kayıt geri
kurması olmadan `prepare` eğitim deposu oluşturamaz. Drive mirror kullanılmaz.

Yürütme `python ml/full_year_cpu.py` üzerinden aynı engine'e gider:
`prepare → pilot-plan → pilot → report`. Profil
`named-weak-signal-control-t5-v1`; açık registration/decision-contract ve
ayrı CPU ortamı seçilir. Bütçede duruş checkpoint'i korur; başka kimliğin
cache'i veya yarım makbuz completed sayılamaz. Ek öğrenme kontrolü fit'i yoktur;
bu altı senaryo zaten kontrolün kendisidir.

Ham ve geçmişte seçilmiş tahminler, origin/target/eligibility, ölçek,
öğretici/noise ve seçilmiş ağırlık korunur.132 native modelin tahminleri,
yalnız geçmişten yeniden hesaplanan preprocessing/scaler ile tekrar çıkarılır.
Bağımsız doğrulayıcı Gaussian quadrature ile beklenen MAE'yi ve scalar geçmiş
shrinkage seçimini denetler; yeni fit yapmaz. Ham/selected yön ve aktiflik,
gerçekleşmiş MAE ve log/standardized MSE ikincil betimsel sonuçlardır.

## Önceden belirlenen karar

Birincil ölçüt **ham öğrenicinin bilinen koşullu dağılımdaki beklenen dış
fiyat-MAE kazancı**; geçmiş shrinkage ve gerçekleşmiş noise kaybı ayrı gösterilir.

- Üç injected seed ≥%5 ve her null kazancı ≤0: yalnız bu basit öğretici için
  öğrenme yeterliği desteklenir; mevcut bilgi/temsil daha güçlü şüphelidir.
- Üç injected seed <%5 ve null kontrolü sağlamsa: bu sabit düzen yararlı
  bilinen etkiyi yeterince çıkaramıyor; kaynakta sinyal yokluğu çıkarılamaz.
- Karışık seed veya başarısız null: belirsiz; otomatik reseed/grid yok.

Üç seed kalibre edilmiş istatistiksel güç tahmini değildir. %5, T+1%53/T+5%55,
6/8 yıl ürün kapıları değişmez; sentetik başarı piyasa başarısı yerine geçmez.
Sonuçtan önce yeni piyasa verisi/model/loss/grid veya canlı126 skor açılmaz.
