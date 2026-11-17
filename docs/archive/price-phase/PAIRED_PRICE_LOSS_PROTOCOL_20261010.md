# Tek kontrollü T+1 loss deneyi

Profil `paired-price-loss-control-v1`, deney `research-paired-price-loss-control-v1`.
Amaç, eski aramada birlikte değişen hedef/loss/parametrelerin arasından yalnız
loss katkısını ayırmaktır. MAE daha önce denenmiştir; bu yeni bir MAE grid'i değildir.

- Mevcut checksum bağlı legacy history/ready ve yeniden kurulan 24 outer
  makbuz kullanılır. Aynı 2016–2023 sekiz **126-origin dönemi**, toplam 1.008
  origin; bunlar tam yıl veya bağımsız holdout değildir. 2024+ kullanılmaz.
- İki kol aynı 24 özellik, training origin/etiketleri, train-only preprocessing,
  `price_delta` hedefi, parametreler ve seed başına eski ağaç sayılarını kullanır.
  Yalnız `reg:squarederror` / `reg:absoluteerror` değişir. İki kol da aynı CPU
  sürümünde fit edilir; eski GPU sonucu kontrol kolu yapılmaz.
- H5 etiketi cutoff'tan **önce** olgunlaşmış olmalı; ortak 120 gözlemlik warmup
  ve bütün eksiklikler korunur. Eski dönem ortası seçimleri Ocak'a taşınmaz.
  Mevcut kaynak/karar-saati varsayımı değiştirilmez veya PIT onayı sayılmaz.
- Eski parametre/iteration'lar farklı hedef/loss'lar altında seçilmişti; bazı
  dönemlerin ağaç sayısı çok düşüktür. Sonuç bu sabit tariflere koşulludur;
  bütün MAE modellerinin veya kaynak bilgisinin sınaması değildir.
- Yeni tuning, inner fit, early stopping, shrinkage veya OOS seçimi yoktur.
  Aynı üç seed'in log-getiri ortalaması iki kolda da kullanılır.

**Bütçe:** 48 piyasa fit'i, toplam 2.688 ağaç; 16 dönem/kol çıktısı, 2.016
ensemble ve 6.048 seed tahmin satırı. Ayrıca iki loss × bilinen-sinyal/negatif
blok-kaydırma/küçük örnek ezberleme = **6 sentetik fit**, piyasa kanıtından ayrı.
Mevcut %50/%20/%90 kontrol sınırları iki loss için ayrı uygulanır; başarısız
kontrol varken piyasa eğitimi başlamaz. Kullanılmayan GPU ailelerine bağımlılık yoktur.
Tek süreç/en fazla iki thread; 30 dakikalık oturum, doğrulanmış checkpoint'ten
devam. Başarısız girişim sessiz tekrar edilmez; bütçe artırılmaz.

Birincil ölçüt ortak origin'lerde absolute–squared paired fiyat-MAE katkısı;
Naive aynı origin'lerde sıfır-getiri baseline'dır. Dönem-koruyan recentered
blok20/60 bootstrap, 10.000 tekrar/seed42; dönemler, yön/aktif oran ve en büyük
%1 Naive hatası çıkarılmış duyarlılık birlikte raporlanır. Üç seed origin
sayısını üçe katlamaz; aralıklar tarihsel araştırma seçimlerini kapsamıyor.

Karar sırası: iki blokta kontrol katkısı alt sınırı pozitif ve ≥%5 Naive
kazancı, ≥%53 yön, ≥6/8 dönem → ileri doğrulama gerektiren sabit-tarif aday;
pozitif katkı fakat pratik kapılar başarısız → sınırlı katkı; iki kolda ve iki
blokta Naive kazancı üst sınırı %5'in altında → bu loss tariflerini büyütme;
diğer durumlar belirsiz. Naive korunur; otomatik arama veya model yayını yoktur.

Mevcut `Experiment`/`Ledger` ve CPU başlatıcı kullanılır:

```bash
python ml/full_year_cpu.py prepare --profile paired-price-loss-control-v1 --reference-root RESTORED_LINEAGE_ARCHIVE --cpu-environment EXISTING_CPU_ENV --drive-root NEW_RUN_ROOT
python ml/full_year_cpu.py pilot-plan --profile paired-price-loss-control-v1 --cpu-environment EXISTING_CPU_ENV --drive-root NEW_RUN_ROOT
```

Kod/testler tamamlandıktan sonra hazırlama kaynak/veri/ortam kimliğini dondurur.
Sıfır-fit ön kayıt GitHub kanıtına bağlanmadan `pilot` reddedilir. Sonrasında
`pilot → compare → report`; kimlik veya girdi değişirse eski cache kabul edilmez.
Eski sonuç/veri dosyaları ve canlı model değiştirilmez.
