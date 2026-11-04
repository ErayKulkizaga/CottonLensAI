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

## Kalan denetim kapsamı

| Alan | Sonraki doğrulama |
| --- | --- |
| Yayın kimliği ve içe aktarım | Revize fiyatların dashboard kapsamı ve kayıt ile yayın arasındaki geçiş |
| Yerel ve GitHub eşlemesi | PR bağımlılıkları, birleştirme planı ve main eşitliği |
| Veri girişi | Kaynak semantiği, vintage, seans ve saat, yarım cache ve eksiklik |
| Araştırma hattı | Hedef olgunluğu, geçmişe ait özellikler, split/purge, model seçimi ve native çıktılar |
| Metrikler ve karar | Ortak origin, referanslar, QLIKE, belirsizlik, release kriteri |
| Canlı kayıt ve operasyon | Saat, kimlik zinciri, kilitli kohort, iş tekrarı ve geri kurma |
| Uygulama ve CI | Backend, frontend, temiz ortam ve release geri yükleme |

Önceki tahmin ve veri kanıtı bağımsız olarak korunur. Kapsam tamamlandıkça
bu tabloya test ve makbuz eklenir.
