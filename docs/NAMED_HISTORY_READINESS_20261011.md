# Aynı kontrat geçmişi: veri ve zamanlama hazırlığı

PR40'ın tamamlanan named T+5 kapasite deneyi korunur. Bu aşama yeni
model eğitmez veya performans ölçmez. Soru: origin'de bilinen kontratın
geçmişini mevcut quote'larla, aynı cohort ve olgunluk kurallarında
kurabiliyor muyuz?

## Sabit tanım

`named_history.build` yalnız Cotton takvimi ve mevcut quote kayıtlarını
okur. Dört alan: aynı kontratın 1/5/10/20 Cotton gözlemlik log getirisi.
Origin'de bilinen ikinci kontrat her iki endpoint'te aynı adla izlenir;
değişen ikinci-sıra fiyat serisi kaydırılmaz. Sıfır kontrolü aynı dört
alanı ve NaN desenini tutup yalnız sonlu değerleri sıfırlar. Mevcut
train-only `Preprocessor` eksiklik bitlerini iki kolda aynı şekilde ekler.

Karar saati önceki araştırmadaki gibi kaynak tarihinden sonraki gün
00:15 UTC. Quote erişimi `max(report_date, displayed_date)+1 gün
00:00 UTC` **varsayımıdır**; gerçek UTC yayım veya first-vintage kanıtı
değildir. Current endpoint mevcut D0 kaynak seçimini/eskilik kuralını
korur. Geçmiş endpoint, referans tarihi geçmiş Cotton endpoint'ini
aşmayan ve **mevcut origin karar saatine kadar erişilebilir** en yeni
rapordur; referans eskiliği en fazla üç Cotton gözlemidir. Geçmiş barın
geçmiş karar saatinde de bilinmesi şart değildir: gecikmiş bar şimdi
gelmişse bugün kullanılabilir. Henüz gelmemiş eski-tarihli bar dışlanır.

Hedefler, named label/price metadata, değerlendirme sonuçları ve 2024+
bu yapıcının girdisi değildir. Hiçbir origin veya eğitim satırı özellik
eksik diye düşürülmez; eski 246/242/4 ayrımı aynen kalır.

## Doğrulanmış kapsam

- 3.520 tarih ve 14.080 endpoint kaydı; numeric/mask NaN desenleri aynı.
- Aynı 22 sabit eğitim işi; en az 528 olgun etiket, dört alanın tamamı
  mevcut en az 510 eğitim satırı. Kalan eksikler korunur, imputation
  yalnız gerçek fit aşamasının eğitim verisinde yapılacaktır.
- Üç iç blokta 63/63 ve dışta 246/246 origin'de dört getiri sonlu.
- Gerçek kaynaklarda ilk dış origin'den sonraki quote'ları değiştirmek
  geçmiş özellikleri/provenance'ı değiştirmedi. Tüm target/named label
  alanlarını bozmak da yeni özellikleri değiştirmedi.
- Ayrı brute as-of/bisect/math.log denetimi feature builder'ı çağırmadan
  bütün 14.080 endpoint'i ve 22 eğitim cohort'unu kontrol eder.
- 16 yeni sentetik test; ilgili protocol/curve/named testleriyle toplam
  91 geçti. Dört checksum'ı yeniden yazılmış bozulma kontrolü: numeric,
  kontrat, erişim saati, missing-mask eşleşmesi.

**VERIFIED veri hazırlığı; tahmin katkısı INCONCLUSIVE.** Saat/vintage
varsayımı, tek görülmüş 2023 yılı, T+5 overlap ve değerlendirilemeyen
6/8 yıl kapısı korunur. Canlı126 skorlanmaz. Model fit0; yeni veri indirme,
GPU, grid, model yayını veya PR merge yok.

## Korunan hazırlık düzeltmeleri

R1, geçmiş endpoint'teki karar saatini de şart koşuyordu; mevcut origin'de
bilinen geçmiş quote'ları gereksiz dışlıyordu. Model fit veya sonuç seçimi
öncesinde mevcut-origin erişimine geçildi. R1 dosyaları korunur. 1/5/10/20
alanlarında tüm geçmişte 7/12/12/13 hücre, 2023'te her birinde iki hücre
değişti; tamamen mevcut eğitim satırı sayısı 503→510. Bu performans
karşılaştırması değildir. R2 feature/endpoint byte'ları final hazırlıkla
aynıdır; final rapor yalnız portable input yollarını ve yardımcı kod
kimliğini taşır. Eksik output üst dizini nedeniyle duran başlatma da
model fit yapmadı; yardımcı kod üst dizini oluşturur ve var olan/yarım
result klasörünün üzerine yazmaz.

## Tek sonraki bilimsel adım

Yeni bir source-effect ön kaydı: aynı named hedef, train/inner/outer
tarihleri, sabit depth2/100-tree XGBoost, seed42, standardized log MSE,
cadence21 ve geçmiş shrinkage. İki kolun aynı 32 alanında bu dört getiri
numeric veya aynı-NaN sıfır kontrolü olacak. Böylece bilgi katkısı,
eksiklik veya input genişliği değişimiyle karışmaz. Eski 28-alan fit'leri
referans kanıt olarak kalır; yeni iki kolun yerine geçirilmez.

Bu yalnız **sonraki ön kayıt adayıdır**, çalıştırılmış deney değildir.
Sicil kontrolü, yeni profil/bütçe, fail-closed öğrenme kontrolleri ve
source/data/environment kimliği hazırlanıp taze GitHub restore
doğrulanmadan eğitim başlamaz. Başarı veya source'ta sinyal yokluğu
bu readiness sonucundan çıkarılmaz; eski gate'ler değiştirilmez.
