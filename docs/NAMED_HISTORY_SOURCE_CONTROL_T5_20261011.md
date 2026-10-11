# Aynı kontrat geçmişinin bilgi katkısı — tek kontrollü T+5 deney

## Soru ve önceki kanıt

PR39 etiket kontrolü ve PR40 sabit kapasite kontrolü tamamlandı. Etiket
ve kapasite farkları pozitifti; sabit XGB'nin Naive kazancı −%0,493740,
iki blok aralığının üst sınırları %3,269655/%3,662587 idi. Bu sonuç yeni
deneyin baseline'ı değildir. PR41, aynı kontratın geçmişinin mevcut
28 alanda bulunmadığını ve saklanan quote'lardan dört getirinin
kurulabildiğini doğruladı; hiçbir piyasa katkısı ölçmedi.

**Birincil hipotez:** bilinen kontratın kendi geçmiş getirileri aynı
hedefte bilgi taşır; sabit model, aynı boyut ve eksiklik desenine sahip
kontrolden daha düşük ham fiyat-MAE üretir. Eski 28-alan modelini doğrudan
kontrol yapmak input genişliği/eksiklik etkisini karıştırır. Bunun yerine
yeni iki kol da aynı 32 alanı kullanır.

## Önceden sabitlenen tasarım

Profil `named-history-source-control-t5-v1`; deney
`research-named-history-source-control-t5-v1`.

| Unsur | Karar |
|---|---|
| Girdi | PR40 tamamlanmış frozen history/quote/ledger; PR41 taze restore kanıtı |
| Kol | `numeric`: eski 28 + aynı-kontrat 1/5/10/20 getirileri; `mask`: aynı 32 alanda yalnız sonlu yeni getiriler sıfır |
| Takvim ve eksik | 3.520 tarih korunur; iki kolun NaN bitleri aynı; satır filtreleme yok |
| Hedef | Aynı named T+5 standardized log return; fiyat ölçümü aynı kontratta |
| Model | Aynı CPU XGBoost, depth2, 100 ağaç, eta .03, min_child_weight20, alpha0, lambda1, seed42 |
| Eğitim | Aynı 22 iş/kol, en az 528 olgun etiket; expanding, cadence21 |
| İç seçim | Aynı geçmiş 3×63, fiyat-MAE, shrinkage [0,.25,.5,.75,1], küçük ağırlık tie-break |
| Dış örnek | Aynı görülmüş 2023: 246 makbuz, 242 skorlanabilir, dört açık eksik |
| Birincil | Ham mask-MAE eksi ham numeric-MAE; eşleşmiş aynı fiyat/origin/hedef |
| İkincil | Her kolun Naive kazancı, seçilmiş sonuç, yön ve aktiflik; en büyük %1 Naive hatası çıkarılmış duyarlılık |
| Belirsizlik | Aynı tam ordinal slotları koruyan paired bootstrap, 10.000/seed42, blok20 ve60; unknown ağırlığı0 |
| Bütçe | En fazla 44 piyasa + dört sentetik fit; tek süreç, en fazla iki thread, oturum30 dakika |

Farklı kontrat sırası kaydırılmaz. Geçmiş quote, mevcut origin karar
saatine kadar erişilebilir olmalıdır; yayımlanmamış eski-tarihli quote
dışlanır. Saat `max(report_date, displayed_date)+1 gün00:00 UTC`
varsayımı, karar ertesi gün00:15 UTC; ilk vintage/gerçek tarihsel UTC
kanıtlanmış değildir. Current D0 seçimi ve eskilik kuralları değişmez.
Yeni veri, 2024+ seçim, hedef/loss/parametre/lag araması yok.

## Çalıştırmadan önceki kontroller

1. Sicil kontrolü: named-label, capacity, curve ve readiness kanıtları
   karşılaştırılır; “eşleşme yok” yenilik kanıtı sayılmaz.
2. Yeni alanlarda bilinen doğrusal ve nonlinear sinyal, negatif shift ve
   küçük örneğe ezberleme kontrolleri. %50/%20/%90 eski sınırları korunur;
   başarısız/eksik kontrol varsa piyasa fit başlamaz. Sentetik başarı
   piyasa becerisi veya zayıf-sinyal gücü kanıtı değildir.
3. Kod/test/veri/ortam/protokol/rapor kimliği frozen; eski cache başka
   namespace'te kalır. Taze GitHub arşivinden prepare/replay doğrulanmadan
   piyasa fit yok. Fit/time bütçesinde checkpoint korunarak durulur.
4. `inputs()` tam `Experiment.history` içinden okur. Mask yalnız
   train/test projection'a uygulanamaz; bütün model girdi takvimi mask
   olmalıdır. Regresyon testi gerçek engine girdi yolunu denetler.
5. Kaynak ve olgun label saatleri korunur; native payload/adapter/
   tahmin/selection yeniden denetlenir. Rapor ve bağımsız scalar verifier
   fit öncesinde dondurulur. Eski kanıt dosyaları değişmez.

## Sonuç kararları

Kaynak katkısı ve pratik hedef ayrı karar alanlarıdır. Ham karşılaştırma
sonradan seçilmiş tahmine, alt döneme veya T+1'e çevrilmez.

- İki mask−numeric aralığının alt sınırı pozitif, numeric Naive kazancı
  en az %5 ve yön en az %55: yalnız araştırma adayı. Önce aynı tarifi
  kilitleyerek gerçek erişim/vintage ve bağımsız dönem kanıtı gerekir.
- İki alt sınır pozitif ama pratik nokta sınırları sağlanmıyor: bu sabit
  temsilin katkısı kaydedilir; başarı ilanı veya otomatik grid yok.
- Numeric–Naive iki aralığın üst sınırı %5'ten küçük: bu sabit kaynak/
  model düzeni pratik hedefi kurtarmıyor. Kaynakta bütün modellere karşı
  sıfır sinyal sonucu çıkarılmaz; benzer aramalar otomatik açılmaz.
- Diğer sonuç: belirsiz; eksik kanıt belirtilir, otomatik ek fit yok.

Tek görülmüş yıl 6/8 yıl kapısını değerlendiremez. CI'lar araştırma
geçmişindeki insan/model seçimlerini kapsamaz. Release/gate izinleri
false kalır; canlı126, üretim modeli ve eski kararlar değişmez.

## Durum

Yürütme kodu ve kontroller hazırlanıyor. Henüz frozen ön kayıt,
piyasa fit veya yeni performans sonucu yok. Ön kayıt hash'leri ve
teslim doğrulaması oluştuğunda ayrı ek kayıtla korunacak.

### Dondurulmuş ön kayıt ve teslim

Yukarıdaki hazırlık durumu öncesine aittir. Kod `a0892eb543b79d1c6d95e3abf9fc2afd644345de`
ile, ön kayıt `0489b4427aa0eac89f429d8121bd8f7fa895018b` ile yayımlandı.
[PR42](https://github.com/ErayKulkizaga/CottonLensAI/pull/42) ve
[Release](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/named-history-source-preregistration-20261011)
mevcuttur; ZIP ve manifest GitHub'dan taze indirilip frozen source ile
1.174 payload doğrulandı. Eğitim/skor hâlâ sıfırdır.

- ML source: `466e4d3f756178561aca5e1706384d0be7398a7cf01757bef34677777c1ed155`
- Ready: `c62f94bb9495605702580da6f747bbce698397aed4c64cae771e50d15b533f20`
- Design: `dbcc9de865336922d4cb451a8641189363aac002c6d2ff635518123be7999f57`
- ZIP: `0c9569b6cf5ab3bfe5045035712848a6222f32edcba127d3ab886348e95ec14a`

111 ilgili sentetik/mock test ve Ruff geçti. Eski sonuçlardan ayrı
işaretlenmiş rapor test fixture'ında 12 scalar aralık ve sekiz bozulma
kontrolü geçti; bu yeni piyasa kanıtı değildir. 173 eski sicil aşaması,
112 eski kanıt dosyası ve tüm trials içeriği korunur. Teslim kaydı:
`research/evidence/named-history-source-preregistration-delivery-20261011.json`.
Sonraki işlem taze ön kayıttan prepare/pilot-plan; ardından yalnız kilitli
44+4 fit bütçesi. Kontrol başarısızsa piyasa fit veya yeni tarif başlamaz.
