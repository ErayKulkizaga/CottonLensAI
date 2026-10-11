# Aynı named T+5 düzeninde bilinen zayıf etki: girdi hazırlığı

**Veri üreticisi ve bağımsız kontrol hazır; model fit0.** Bu, gerçek piyasa
başarısı veya kalibre edilmiş istatistiksel güç tahmini değildir. PR43'te
mevcut tahminlerin optimize edilen MSE'de de Naive'ın gerisinde kaldığı
doğrulandı. Önce aynı öğrenicinin pratik büyüklükte bilinen bir etkiyi
gürültü altında öğrenip öğrenemediği anlaşılmalıdır.

Sicil taraması (`xgboost/T+5/synthetic`, ayrıca `xgboost/weak`) ve kaynak
incelemesi eşdeğer kontrol göstermedi. Bu, aramanın mutlak eksiksizliği
iddiası değildir. Eski 3.380 Ridge zayıf-sinyal fit'i T+1/Texas üzerinde;
mevcut named XGB güçlü kontrolleri gürültüsüz `.02*x` veya `.02*(x²−1)`
etiketi kullanır. Yeni aday aynı deneyleri tekrar fit etmez.

## Tek kilitli karşılaştırma

`named-weak-signal-control-t5-v1`: aynı 32 özellik, named cohort, 22 iç/dış
iş, XGBoost depth2/100ağaç/eta.03/minchild20/seed42 ve geçmiş shrinkage.
Hiperparametre/model/feature araması yok. Üç gürültü seed'i
703/1703/2703; her birinde null ve injected, toplam **132 sentetik fit**
tavanı; gerçek piyasa fit0. Bir süreç/iki thread/30 dakika oturum.
Bu hazırlık aşamasında henüz hiçbir fit çalışmadı. Eğitim bağlantısı ve
ön kayıt son kaynak kimliğiyle ayrıca dondurulmadan çalıştırma başlamaz.

Öğretici, origin'de bilinen `named_contract_ret_1` alanının tek threshold
bölünmesidir; eksik kaynakta ortalama etki0. Threshold, noise sigma ve beta
yalnız ilk 528 olgun eğitim satırından belirlenir; son origin2022-03-15,
527 finite kaynak değeri. İç/dış outcome kalibrasyonda kullanılmaz.

- Threshold: 0,0017077478961123767.
- Noise sigma: 0,03527396091724038.
- Beta: 0,01680337858960806.

Koşullu lognormal fiyat-MAE formülü mevcut `weak_signal.py`'den kullanılır;
beta erken eğitimde **oracle %10** beklenen MAE avantajına sabitlenir.
Bu, değişmeyen %5 pratik hedefinin üzerinde öğrenim hatasına pay bırakır;
ürün başarı kapısı %10'a değiştirilmez. Dış tarihlerde oracle beklenen
avantajı %10,029333; bu bir modelin öğrenmiş performansı değildir.
Üç seed bunun bağımsız outcome ölçümü sayılmaz: aynı bilinen oracle
beklentisini, farklı eğitim gürültüsü altında öğrenci için sınayacaktır.

Null/injected aynı günlük normal innovations'ı alır. T+5 noise, beş
gelecek günlük innovation toplamının √5'e bölünmesidir; örtüşen hedefler
bağımsızmış gibi üretilmez. Model girdileri eski özelliklerdir; geleceğin
noise değeri bir feature olamaz. Etiketler yalnız eski target-date ve
availability olgunluk kurallarıyla fit'e girebilir. Bunlar semi-synthetic
label'lardır; tutarlı işlem gören futures fiyat patikası iddiası yoktur.

## Karar sınırı

Birincil soru, üç sabit seed'de ham öğrencinin bilinen koşullu dağılımdaki
beklenen dış fiyat-MAE avantajının ≥%5 olup olmadığıdır. Null beklentisi
pozitif kazanç vermemelidir. Geçmiş shrinkage ve gerçekleşmiş gürültülü
kayıp ayrıca raporlanır; birincil soruya göre sonradan değiştirilmez.

- Üç seed ≥%5: yalnız bu basit gürültülü öğreticiyi öğrenme yeterliği
  desteklenir; bütün karmaşık piyasa sinyallerine güç veya piyasa becerisi
  ispatı değildir.
- Üç seed <%5: bu sabit düzen bilinen etkiyi yeterince çıkaramadı; gerçek
  kaynakta sinyal yokluğu çıkarımı desteklenmez.
- Diğer durum: belirsiz; otomatik daha büyük model/grid/reseed yok.

## Gerçekten doğrulanan

Altı girdi ×3.520 tarih =21.120 scalar satır; değişmeyen gerçek history,
aynı origin/eligibility, kol başına246/242/4 ve değişmeyen hedef tarihleri.
60 ilgili regresyon/sentetik test ve Ruff geçti. Bağımsız quadrature erken
oracle %10 kalibrasyonunu ve dış oracle sonuçlarını doğruladı. Eski veri,
model, tahmin ve fit sicili değiştirilmez. İnceleme kaynakları:
`named_weak_signal.py::calibrate/build/expected`,
`test_named_weak_signal.py` ve checksum bağlı hazırlık arşivindeki
`prepare.py`, `verify.py`, `controls.py`.

Bu aşama yalnız veri/deney kurulabilirliğini tamamlar. Yeni gerçek veri
indirilmez; FAS kabulü, canlı 126-origin skor veya model yayını yapılmaz.
Saat/final-vintage varsayımları korunur. Tek sonraki adım, aynı 132-fit
tasarımını mevcut Experiment/ledger yoluna bağlayıp source/data/environment
kimliği ve rapor kurallarını taze GitHub ön kaydıyla doğrulamaktır.
# Delivery verification

Fresh GitHub download and archived-source replay reproduced the frozen readiness inputs, 21,120 independently checked scalar rows and all four tamper controls. The 176 earlier stages, 122 earlier proof files and trial bytes remain unchanged. This verifies inputs only: the new profile has zero fits and no market-skill result. The authorized private-repository/Drive preservation work precedes connecting the 132-fit synthetic control to the existing engine under a new final source identity.

