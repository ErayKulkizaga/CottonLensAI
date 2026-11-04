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
`docs/archive/` altında aynı içerikle bulunuyor. Dokuz yolun farkı
içerik düzeyinde incelenecek. Bir yerel yapılandırma dosyası inceleme ve
yayın kapsamı dışında. Ana çalışma klasörünün içeriği korundu.

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
- Otomatik regresyon testi iki farklı ZIP yayın kimliği, gerçek Parquet
  içe aktarımı ve SQLite üzerinde bu iki hatayı yeniden üretti.
  Yeniden yükleme sonrası geçerli tahmin ve model yokken reddetme de
  sınandı. Başarılı kontroller backend Ruff ve 30 pytest.

Bunlar API çıkarım hataları. Tarihsel piyasa performansı üzerindeki
etkileri bu testlerle ölçülmedi.

## Kalan denetim kapsamı

| Alan | Sonraki doğrulama |
| --- | --- |
| Yayın kimliği ve içe aktarım | Aynı sürüm ve manifest altında değişen payload, atomik aktivasyon, revize fiyatlar ve tekrar içe aktarım |
| Yerel ve GitHub eşlemesi | Dokuz farklı yerel yolun anlamsal farkı, PR bağımlılıkları ve birleştirme planı |
| Veri girişi | Kaynak semantiği, vintage, seans ve saat, bozuk/yarım cache, eksiklik |
| Araştırma hattı | Hedef olgunluğu, geçmişe ait özellikler, split/purge, model seçimi ve native çıktılar |
| Metrikler ve karar | Ortak origin, referanslar, QLIKE, belirsizlik, release kriteri |
| Canlı kayıt ve operasyon | Saat, kimlik zinciri, kilitli kohort, iş tekrarı ve geri kurma |
| Uygulama ve CI | Backend, frontend, temiz ortam ve release geri yükleme |

Önceki tahmin ve veri kanıtı bağımsız olarak korunur. Kapsam tamamlandıkça
bu tabloya test ve makbuz eklenir.
