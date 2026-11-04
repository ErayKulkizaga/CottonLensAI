# CottonLensAI uçtan uca denetim kaydı

## Başlangıç durumu

GitHub private. `main` başlangıçta `c38e151` idi ve denetim tabanı
`281dd8c` ile arasında 151 commit vardı. 65 açık PR'ın 35'i taslaktı ve
yalnız biri doğrudan `main` dalını hedefliyordu. Denetim dalı güncel
araştırma geçmişini taşıyor. Birleştirmeden önce aşağıdaki kapsamın
kanıtı tamamlanmalı ve yığılmış değişiklikler incelenmeli.

Yerel ana çalışma klasörü `470f1e5` tabanında. Git tarafından listelenen
31 değişmiş veya izlenmeyen yoldan 19'u, satır sonları eşitlendikten
sonra denetim dalındaki karşılığıyla aynıydı. İki eski belge
`docs/archive/` altında aynı içerikle bulunuyor. Dokuz farklı yolun farkı incelendi.
Backend API ve testi ilk denetim düzeltmesinden önce birebir eşleşiyordu.
CPU başlatıcısı, engine, ledger ve path pilot farkları yeni araştırma
profillerini ve ufuk kapsamını taşıyor. İki workbench testi arşiv yolu
ve gecikmeli Colab log kopyasını kapsıyor. Araştırma sözleşmesinin güncel
sürümü kullanıcının ispat projesi ve risk önceliği kararlarını içeriyor.
Bu karşılaştırmada aktarılması gereken ek bir yerel değişiklik saptanmadı.
Bir yerel yapılandırma dosyası inceleme ve yayın kapsamı dışında.
Ana çalışma klasörünün içeriği korundu.

## Doğrulanmış düzeltmeler

- Canlı senaryo, veritabanına B paketi içe aktarılmışken bellekteki A
  modeliyle çalışabiliyordu. Readiness yalnız A'nın daha önce
  içe aktarılmış olmasını arıyordu. İkisi de artık etkin importun
  manifestini gerektiriyor. Gerçek bir import varken model eksikse
  senaryo geliştirme formülüne düşmüyor.
- Aynı tarih tablosundaki A özellikleri, B'nin kendi penceresi kısa
  kaldığında eksik sırayı dolduruyordu. Senaryo penceresi artık çalışan
  modelle birlikte yüklenen paketten okunuyor. Eksik pencere 409 ile
  reddediliyor.
- Otomatik regresyon testi iki farklı yayın kimliği, gerçek Parquet
  içe aktarımı ve SQLite üzerinde bu iki hatayı yeniden üretti.
  Yeniden yükleme sonrası geçerli tahmin ve model yokken reddetme de
  sınandı. Başarılı kontroller backend Ruff ve 30 pytest.

- Aynı manifest korunurken bir Parquet özellik değeri değiştirilip
  paket yeniden checksum'lanınca eski içe aktarım makbuzu kullanılabiliyordu.
  İçe aktarım, readiness ve senaryo bu durumu kabul ediyordu.
  Makbuz artık doğrulanan dosya yolları ve hash'lerinin tamamına bağlı.
  Başlangıçta DB içe aktarımı belleğe yüklenen paket kimliğini de gerektiriyor.
  Eski manifest makbuzları geçmiş sorgular için korunur ve canlı
  kimliğe otomatik yükseltilmez. Test, üç yüzeyde reddi, değişmeyen
  pakette idempotansı ve önceki tahmin kayıtlarının korunmasını doğruladı.

Bunlar API çıkarım hataları. Tarihsel piyasa performansı üzerindeki
etkileri bu testlerle ölçülmedi.

### Yayın kurulumu ve geri alma

Kurulum `current` dizinini `.previous` konumuna taşıdıktan sonra
kesilirse, eski kod bir sonraki denemede elde kalan geçerli kopyayı
silebiliyordu. Ayrı bir süreç de ortak hazırlık dizinini devralabiliyordu.
Kurucu şimdi hedef başına işletim sistemi kilidi alıyor, elde kalan
yedeği checksum ile doğrulayıp önce geri getiriyor ve yeni hazırlık
kopyasını etkinleştirmeden önce yeniden doğruluyor. Başarısız ikinci
deneme, bozuk paket, bozuk hazırlık kopyası ve iki ayrı süreçle çakışma
otomatik test edildi. Windows backend paketinde 40 test ve Ruff geçti.
Bu doğrulama süreç kesintisi ve yeniden denemeyi kapsıyor.

### Grafik verisi ve paket sahipliği

İki paket aynı gün için 80 ve 84 bildirdiğinde, yeni tahmin 84 üzerinden
gösterilirken grafik ilk içe aktarımın 80 değerini kullanıyordu. Şema
0003, yeni piyasa gözlemlerini kendi doğrulanmış içe aktarım kimliğiyle
saklıyor. Grafik etkin paketin gözlemlerini okuyor ve eksik günlerini
önceki paketten doldurmuyor. Eski fiyat tablosu ve tahmin kayıtları
korunuyor. Tam içerik makbuzu olan eski içe aktarımlarda yalnız özgün
paket yeniden doğrulanarak ilgili grafik verisi tamamlanabiliyor.

Paket içinde yinelenen veya sonlu olmayan fiyatlar ve kanıt sözleşmesi
olan paketlerde canlı tahminin başlangıç fiyatıyla uyuşmayan Cotton
kapanışı içe aktarımı durduruyor. Eski manifestin `validated_holdout`
etiketi grafiğe taşınmıyor. SQLite şema geçişi, veri varken geri alma
reddi ve önceki tahminlerin korunması otomatik test edildi. Mevcut yerel
v1 paketinin 12.616 fiyatı ve 998 tahmini yalıtılmış bir veritabanında
iki içe aktarım çağrısıyla doğrulandı. Kaynak paket kimliği aynı kaldı.

### Piyasa verisi girişi

Yahoo yanıtının boş olmaması yeterli sayılıyordu. Tamamı eksik DXY
kapanışları ve DXY isteğine dönen CT=F etiketi, geçerli eski piyasa
önbelleğini değiştirebiliyordu. Her seri artık tarih, alan, mevcut
sembol etiketi ve kullanılabilir kapanış kontrollerini yazmadan önce
geçiyor. Sonlu negatif WTI geçmişi korunuyor. Testler, dört bozuk
yanıtın eski Parquet dosyasını bayt düzeyinde koruduğunu ve geçerli
yanıtın aynı veri şemasını ürettiğini doğruladı.

Bu testler sağlayıcının olası yanıtlarını sentetik olarak yeniden
üretti. Eski frozen deney verilerinde bu kusurların gerçekleştiği
henüz gösterilmedi.

### Hedef tarihleri ve eğitimde etiket olgunluğu

Ortak T+1/T+5 kohortu yalnız T+5 etiketini denetliyordu. Sonraki
kapanış eksik ama beşinci kapanış geçerliyse, T+1 hedefi eksik bir origin
ortak eğitim veya değerlendirmeye girebiliyordu. Ayrıca ters hedef
tarihleri reddedilmiyordu. Yeni ortak kohort iki sonlu getiriyi ve
`origin < T+1 < T+5` sırasını gerektiriyor. Açıkça tek ufuk isteyen
çağrı kendi ufkunu doğruluyor. Eksik özellik satırları korunuyor.

`research-full-year-v1-r2` için checksum'ı eşleşen ham Cotton
kapanışlarından 3.520 fiyat hedefi her ufukta yeniden kuruldu. Ham sıra
farkı hep bir seanstı. Kaydedilmiş risk hedefleri T+1'de 3.519 ve
T+5'te 3.515 satırda aynıydı. 16 yıllık/ufuk dosyası frozen origin
listeleriyle ve 4.012 saklı fiyat/risk hedefiyle eşleşti. 2.360
checksum-doğrulanmış ledger kaydının 5.244.446 eğitim etiketi kullanımı
tekrarlar dahil sayıldı. Her kayıtta son T+5 hedef tarihi ilk validation
veya tahmin tarihinden önceydi. En küçük aralık bir takvim günüydü.
48 XGBoost erken durdurma kaydının validation etiketleri de ilgili
iç skor bloğu başlamadan olgunlaşmıştı.

Yeni kontrol aynı 8 dönemi ve 2.006 origin'i, aynı split kimliğiyle
üretti. Bulunan sınır hatası bu tamamlanmış ana deneyin sonucunu
açıklamıyor. Bu denetim etiket ve fiyat dizisinin nedenselliğini
kapsar. Kaynakların gerçek tarihsel teslim saatleri ayrıca doğrulanmalı.

| Kanıt | SHA-256 / kimlik |
| --- | --- |
| Ham `market.parquet` | `23cfabb7ccb2553f0547bc3290ff4916ce91c788c98a94acdb2e5b9cd1f3151c` |
| Donmuş `history.parquet` | `2132162a410b4e12655d36a2ac3e9efd62a3fc0e49fdb08cb038d4cc328836d7` |
| `ready.json` | `7716394157e844073834afdfc0629b38b1febaf1fc8c2fef971945fa2e8bbf05` |
| `split_id` | `9d964b2c88a6d36b7e1fc8421efa1a3c86a3b0531b6cbcb072856918d014aa1d` |

### Özellikler, seçim ve rapor eşleşmesi

Aynı ana deneyin 24 özellik sütunu, checksum'ı doğrulanmış ham
seriden 3.520 origin üzerinde yeniden üretildi. 84.480 hücre
eksiklik deseni dahil eşleşti. DXY ve WTI için 3.519 kaynak tarihi
vardı ve tamamı önceki Cotton gözlemiyle sınırlıydı. İki seans
geciken DXY kaydı bir, WTI kaydı ikiydi. Bu ölçüm kodun kullandığı
bar tarihlerini doğrular. Gerçek tarihsel Yahoo teslim saati bu
bar tarihlerinden çıkarılamaz.

1.780 özgün ledger kaydından 16 yıl/ufuk seçimi ve OOS tahminleri
yeniden kuruldu. 96 fiyat adayı için üç geçmiş bloktaki MAE ve
küçültme seçimi, risk adayları için geçmiş QLIKE ve seçilen
tahminler saklı kayıtlarla eşleşti. T+1 ham fiyat-MAE farkı
Naive'a göre −%1,70065, küçültme sonrası −%0,26195. T+5 sırasıyla
−%1,97452 ve −%0,99644. Ağırlık T+1'de 4/8 ve T+5'te 3/8 yılda
sıfırdı. Küçültme ham tahminlerin kaybını azalttı. Bu tarife
referans üstünlüğü sağlamadı.

Genel full-year raporu 16 geçerli JSON kaydını yeterli sayıyordu.
Yeniden hash'lenen hatalı origin, fiyat, hedef tarihi veya gerçekleşen
hedefle rapor oluşturulabiliyordu. Rapor ve aralık düzeltmesi artık
kanonik yıl/ufuk dosyalarını ve checksum bağlı tarihçeyi doğruluyor.
Türetilen aralıklar özgün manifest ve çıktı hash'lerine bağlanıyor,
nokta tahminleri ve hedefler ayrıca karşılaştırılıyor. Özgün ve
düzeltilmiş 16'şar dosya kontrolden geçti. Geçici kopyada 10.000
tekrarla eski raporun bütün sayısal alanları yeniden üretildi. Yeni
istatistik kodu iki ek güç açıklaması alanı yayımlıyor.

### Sıra penceresi ve yayın saatinin ayrıştırılması

Araştırma modelinin sıra girdisi konuma göre dilimleniyordu.
Karışık bir tarihçe gelecekteki gözlemi pencereye alabiliyor,
silinmiş bir seans ise pencereyi sessizce sıkıştırabiliyordu.
Pencere artık pozitif tam sayı, tarihleri geçerli ve tekil,
sıra tarihçesi kronolojik olmalı. Kaynak seans numarası varsa
pencere ardışık kayıtları gerektiriyor. Tek adımda yalnız istenen
origin'ler dönüştürülür. Sıra dönüşümü son istenen origin'i aşmaz.
Eksik özellik içeren seansların sıra konumu korunuyor.

`research-v1` ve `research-v2-tf-placement` aynı hash'li 3.520
satırlık tarihçeyi kullanıyordu. Tarihler sıralı ve tekildi,
3.519 seans geçişinin tamamı ardışıktı. Bu iki tarihçede bulunan
sınır koşulu gerçekleşmedi. Ortak ön işleyici gelecek hedef
sütunlarını özellik olarak reddediyor. Hedef ölçekleyicisi boş
veya sonlu olmayan eğitim etiketiyle kurulamaz.

Genel yayın birleştirmesi, saat dilimsiz damgayı UTC kabul
edebiliyordu. Kesin yayın, kesin erişilebilirlik ve doğrulanmış
üst sınır için açık saat dilimi gerekiyor. Farklı ofsetlerin
UTC'ye çevrilmesi ve 00:15 sınırının mevcut eşitlik davranışı
regresyon testinde korundu. Dondurulmuş piyasa bazlı fiyat
deneyinde bu yayın birleştirme yolu kullanılmadı.

### Saklı fiyat modellerinden yeniden tahmin

`research-full-year-v1-r2` içindeki 1.300 fiyat kaydının tamamında
ledger ve model dosyalarının hash'leri, eğitim tablosunun kimliği,
origin sırası ve kesim tarihleri doğrulandı. Eksik model dosyası
çıkmadı. Medyan, ortalama, ölçek ve hedef dönüşümü yalnız kayıttaki
eğitim satırlarından bağımsız yeniden hesaplandı ve adaptörle
eşleşti. Yeniden eğitim yapmadan 29.192 tahmin tekrar üretildi.
Bu sayı doğrulama, erken durdurma ve tekrar origin kullanımlarını
da içerir.

| Aile | Model kaydı | En büyük mutlak log-getiri farkı |
| --- | ---: | ---: |
| Ridge | 626 | 8,196 × 10⁻⁸ |
| ElasticNet | 338 | 3,353 × 10⁻⁸ |
| XGBoost | 336 | 0 |

Örnek kayıtlar aile sırasıyla `00b6c42f…`, `006c7f6f…` ve
`00bf05bf…`. XGBoost dosyalarındaki ağaç sayısı kayıtlı iterasyon
sayısıyla aynıydı. Bu fiyat deneyinde ölçekleyici, model dosyası
ve dönüşüm hattı Naive farkını açıklamıyor. Kapsam üç CPU ailesidir.

### GitHub kapsamı

Bu kontrolde PR #75 dışındaki 65 açık PR'ın head commit'lerinin
tamamı denetim dalının atasıydı. Bu, dal kapsamını doğrular.
`main` birleştirmesi ve PR'ların son durumunun düzenlenmesi
ayrı teslimat adımlarıdır.

### Diğer fiyat deneyleri ve yayın öncesi fiyat kapısı

`research-availability-clock-pilot-v1` ve
`research-return-path-pilot-v1` için 32'şer yıllık çıktı,
aynı `research-full-year-v1-r2` hedefleriyle yeniden eşleştirildi.
Her çalışmada 8.024 satır vardı. Kaydedilen ham tahmin ve ağırlıktan
son tahmin tekrar üretildi. Naive MAE T+1 için 1,020952 ve T+5 için
2,357184 sent/lb. Hiçbir kol hiçbir yılda Naive'dan düşük MAE üretmedi.

| Deney ve kol | T+1 ham / seçilmiş Naive kazancı | T+5 ham / seçilmiş Naive kazancı |
| --- | ---: | ---: |
| Saat kontrolü | −%1,846 / −%0,328 | −%4,510 / −%0,800 |
| Saat müdahalesi | −%2,031 / −%0,154 | −%4,916 / −%1,340 |
| Yol kontrolü | −%1,746 / −%0,323 | −%4,233 / −%1,087 |
| Getiri yolu | −%4,424 / −%0,043 | −%5,971 / −%0,952 |

Genel nested karşılaştırma dondurulmuş origin sırasını
doğrulamıyordu. Yıl kazanımlarını tahmin vektöründen hesaplamak
yerine eski dosyanın MAE alanını kullanıyordu. Sentetik regresyonda
Naive ile eşit tahmin 8/8 kazanmış sayıldı. Yeni yol sıralı ve
örtüşmeyen katları, her origin için sonlu çıktıyı ve hedef
kronolojisini zorunlu tutuyor. Yıllar aynı hedeflere karşı
yeniden hesaplanıyor. Testte eşit tahmin 0/8 kazandı.

Schema-3 hazırlığı, iki ufukta sekiz katın eşlenmiş tahmin ve
kayıplarından yayın kararını dağıtım fitinden önce yeniden doğruluyor.
Yeni kilit eksik katları kabul etmiyor. Yeniden üretim, ilk refitten
önce ilgili ufkun özgün sonuç vektörlerini ve sonrasında yeni
vektörlerin aynı origin'lere ait olduğunu denetliyor. Eşikler aynı.
Bu turda piyasa fiti veya paket yayını yapılmadı.

### Taze tekrar ve yayın makbuzu

Genel schema-3 ihracı üç alanlı eski makbuzdan `passed` ve
`fresh_fits` işaretlerini okuyarak dağıtım hazırlığına geçebiliyordu.
Yeni `reproduction-t{h}-bound-v1.json` makbuzu kilitli kararın,
özgün ve tekrar tahminlerinin, kat kararının ve model dosyalarına
bağlı fit kayıtlarının kimliğini saklıyor. Tekrar tahmini, aynı
reçete, refit takvimi ve origin'lerdeki tekrar kayıtlarından
yeniden birleştiriliyor. Ensemble üyeleri ayrı doğrulanıyor.
İhraç makbuzu dağıtım verisi yüklenmeden önce yeniden denetliyor.

Eski makbuz ve yayınlar arşivde okunabilir. Bir eski tekrar
makbuzunu bağlamak, eksik bir fitin sessizce yeniden çalışmasına
yol açmıyor. Bozuk model baytı, yanlış fit anahtarı, değişmiş
tahmin, yanlış karar ve başka kilit otomatik testlerde reddedildi.

Ayrı CPU ana deneyinde mevcut `research-full-year-v1-r2`
tekrar kanıtı salt okunur kontrol edildi. 194 fiyat, 194 risk ve
192 eski kalibrasyon fit kaydı ile 968 native dosya checksum'la
eşleşti. 16 özgün–tekrar çıktı çifti aynı origin'lerdeydi.
Kayıtlı fiyat ve varyans farkının en büyüğü sıfırdı. Bu
tarihsel tekrar sonucu ileri dönem tahmin başarısı ölçmez.

### CPU tekrar makbuzu ve risk karşılaştırmaları

CPU tekrar raporu, aynı 16 çıktı çiftiyle model kaydı olmadan
`fresh_fits: true` üretebiliyordu. Yeni doğrulama checksum bağlı
tarihçeyi, yıl kararlarını, hedefleri ve EWMA referansını kontrol ediyor.
Her dönem fiyat ve risk tahmini, iki ayrı ledger anahtarı kümesindeki
native dosyaların hash'leri ve kayıtlı tahminlerden yeniden kuruluyor.
Yeni makbuz `reproduction/verification-bound-v1.json` ayrı yazılır.
Kayıtlı aralıkların eşleşmesi yalnız çıktı paritesidir.

Gerçek `research-full-year-v1-r2` arşivinde 16 çıktı çifti için
388 özgün ve 388 tekrar fit anahtarı eşleşti. Eski makbuz
`810dbfde78ebcb1df32b5a0c57abe3f4ef2a280b4ae8eefe215cf1d102eac978`
korundu. 8.024 özgün ve tekrar EWMA değeri yeniden hesaplandı ve
birebir eşleşti. Testte tek native dosya eksilince yeni doğrulama
durdu. Tarihsel kanıtlar veya canlı çalışma değiştirilmedi.

Üç saklı risk karşılaştırmasında aynı origin ve hedefler üzerinde
`log(v) + gerçek_varyans/v` kayıpları yeniden hesaplandı. Tabloda
pozitif fark EWMA'ya karşı aday lehinedir.

| Arşiv | Origin | Ortalama ham / seçilmiş QLIKE farkı |
| --- | ---: | ---: |
| `ct-ewma-har-control`, 2016–2023 | 2.006 | +0,01747463 / +0,00136570 |
| `ct-pre2010-risk-replication`, 2008–2009 aynı kontrat | 252 | −0,00186901 / yuvarlama düzeyinde sıfır |
| `ct-recent-risk-audit`, 2024–2026 incelenmiş dönem | 711 | Bu CSV'de ham yok / +0,00008992 |

Sekiz yıllık seçilmiş sonuç, en büyük %1 EWMA kaybı çıkarılınca
−0,00499849 oldu. Aynı-kontrat deneyinde 2008 ve 2009 ham katkıları
farklı işaretliydi. Yakın dönem seçilmiş küçük fark 2024'ten geldi.
Bu kapsamda vade geçişi tek başına başarısızlığı açıklamıyor.

## Kalan denetim kapsamı

| Alan | Sonraki doğrulama |
| --- | --- |
| Yayın kimliği ve içe aktarım | Kayıt ile yayın arasındaki geçiş ve PostgreSQL üzerinde şema uygulaması |
| Yerel ve GitHub eşlemesi | CI, birleştirme kararı ve main eşitliği |
| Veri girişi | Kaynak semantiği, vintage, seans ve saat, yarım cache ve eksiklik |
| Araştırma hattı | Diğer profillerin zaman, seçim ve native çıktı kapsamı |
| Metrikler ve karar | Diğer risk karşılaştırmaları, kaynak kanıtı ve ileri doğrulama sınırı |
| Canlı kayıt ve operasyon | Saat, kimlik zinciri, kilitli kohort, iş tekrarı ve geri kurma |
| Uygulama ve CI | Backend, frontend, temiz ortam ve release geri yükleme |

Önceki tahmin ve veri kanıtı bağımsız olarak korunur. Kapsam tamamlandıkça
bu tabloya test ve makbuz eklenir.
