# Zayıf sinyal kontrolü — tamamlanan sonuç

**VERIFIED:** `research-weak-signal-control-v1-r2` tamamlandı: 3.380 sentetik
fit, 20 senaryo, 160 yıllık çıktı, senaryo başına aynı 2.006 origin ve toplam
40.120 OOS satırı. Yeni piyasa fit'i **0**. Ön kayıtlı karar
**RAW_ONLY_RECOVERS**; bu karar küçültmenin piyasa başarısızlığının ana nedeni
olduğunu kanıtlamaz. **Naive korunur; otomatik ek eğitim açılmaz.**

[Değişmeyen ön kayıt](WEAK_SIGNAL_PREREGISTRATION_20261009.md),
[sonuç/kimlik](../research/evidence/weak-signal-result-20261009.json),
[checksum manifesti](../research/evidence/weak-signal-result-release-20261009.json),
[model/tahmin ve doğrulama Release'i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/weak-signal-result-20261009).

## Gerçekte ne sınandı?

Tamamlanmış Texas `numeric_D0` tarifinin aynı 37 özelliği, eksiklikleri,
2016–2023 origin'leri, fiyatları ve expanding eğitim düzeni kullanıldı.
Ridge alpha1/window1, geçmiş 3×63 doğrulama, 21 gözlemde refit,
beş gözlemlik etiket olgunlaşması ve `[0,.25,.5,.75,1]` küçültme değişmedi.
On seed'de aynı gürültünün null/enjekte çifti üretildi. Etki yalnız 2010–2014
üzerinden kalibre edilmiş Texas–ulusal kondisyon farkına doğrusaldır.
Oracle model girdisi değildir; özgün piyasa hedefi değiştirilmedi.

Bu, fiyat ve özelliklerin kopyaları üzerinde **sentetik öğrenme tanısıdır**.
Tutarlı bir işlem gören fiyat yolu, piyasa becerisi, gerçek tarihsel
erişilebilirlik/vintage kabulü veya bağımsız holdout değildir. Normal gürültü
öğrenme için elverişlidir; gerçek getirideki asimetri/loss sorununu elemez.

## Sonuçlar

Kazanç aynı senaryonun Naive fiyat-MAE'sine göredir; pozitif daha iyidir.
Seed ortalamaları meta-analiz veya bağımsız piyasa örnekleri değildir.

| Enjekte koşulu | Ortalama MAE kazancı | Pozitif seed | Aynı-seed oracle katkısının medyan korunan oranı |
|---|---:|---:|---:|
| Oracle |%3,6054|10/10|%100|
| Ham Ridge |%1,9551|9/10|%59,8533|
| Geçmişte seçilmiş küçültme |%1,8752|10/10|%49,5611|

| Seed | Oracle kazancı | Ham kazanç | Seçilmiş kazanç |
|---|---:|---:|---:|
|1201|%4,5878|%3,3599|%2,3386|
|1202|%3,7215|%2,0306|%2,7670|
|1203|%4,1829|%2,5381|%1,7507|
|1204|%2,9509|%0,9950|%1,2461|
|1205|%3,7821|%2,2355|%2,1536|
|1206|%3,7137|%2,2505|%2,3255|
|1207|%3,5352|%2,3064|%2,0277|
|1208|%3,5963|%1,7732|%1,2934|
|1209|%3,5087|%2,1446|%1,6894|
|1210|%2,4753|−%0,0823|%1,1604|

Null koşulunda ortalama ham/seçilmiş kazanç **−%1,7070/−%0,1423**.
Ön kayıtlı pratik yanlış pozitif **0/10**; bu on seed bütün yanlış pozitif
olasılığını kalibre etmez. Oracle null'da Naive ile aynıdır.
Tüm seed/mode için yıllık, aktif/yön, en büyük %1 Naive hatası çıkarılmış
sonuç ve blok20/60,10.000 tekrarlı120 paired aralık Release'te saklanır.
Sıfır seçilmiş tahmin, aktif yön doğruluğuyla karıştırılmaz.

## Kararı red-team etme

**VERIFIED:** seçilmiş koruma oranı %50 eşiğinin yalnız **0,4389 yüzde puan**
altındadır. Ortalama seçilmiş kazanç hamdan **0,0799 yüzde puan** düşüktür;
küçültme altı seed'i kötüleştirir, dördünü iyileştirir ve tek ham-negatif
seed'i pozitife taşır. Enjekte80 yıl seçiminin yalnız4'ünde ağırlık0;
null80 seçiminin45'inde0'dır. Null'da küçültme on seed'in tamamında zararı
azaltır. [Yeniden hesaplanabilir betimleyici tanı](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/weak-signal-result-20261009)
paketindeki `mechanism_check.py` ve `mechanism-check.json` bunları kaydeder.

**STRONGLY SUPPORTED:** mevcut motor bu temsil edilebilir küçük etkiyi
tamamen yok etmiyor; küçültme de otomatik olarak tüm sinyali sıfırlamıyor.
Kategorik `RAW_ONLY_RECOVERS` sonucunu değiştirmiyoruz, fakat bir eşik
yakınlığını büyük mekanizma keşfi gibi sunmuyoruz. Daha önce gürültüsüz
kontrolün geçmesi küçük etkide duyarlılığı tek başına göstermiyordu;
şimdi bu belirli doğrusal örnek için kısmi duyarlılık doğrudan ölçüldü.

**NOT SUPPORTED:** bu sonuçtan "küçültmeyi kaldırınca gerçek piyasa düzelir",
"Texas verisinde sinyal yok" veya "daha büyük model şart" çıkarılamaz.
Gerçek Texas ham tahmini −%3,7557, seçilmişi −%0,0274 kazanç üretmişti;
küçültmeyi kaldırmak o kayıtlı sonucu kötüleştirir. Sentetikte oracle'ın
ulaştığı ortalama katkı bile piyasa %5 hedefinden küçüktür.

## Doğrulama, kimlik ve koruma

- 3.380 kayıtlı model payload'ı, yalnız eğitimde hesaplanan imputasyon/scaler
  ve hedef dönüşümü, train tarihleri ve `target_date_5 < cutoff` yeniden
  denetlendi. Maksimum kayıtlı log-getiri çıkarım farkı **3,7253e-8**.
- 160 geçmiş küçültme kararı, 60 senaryo/mode fiyat-MAE hesabı ve120 blok
  aralığı bağımsız yeniden hesaplandı; **sıfır yeni fit**.
- Bilimsel kaynak `f6028315cc38ff61ad81c9e9bc8b89bf9d11dd1070dd091b276aef33ed04192f`;
  kayıt `359a4c58ddb76d5fc79df23c2e8c7a47ab3d7cd45a8ea2e908ee14864f4fcb95`.
  Tahmin CSV SHA256 `88feb450587344322e7a070fb4e3adac4337e305e60fe98f5b689750291111be`.
- Tek CPU süreç/iki thread ve3380 fit bütçesi korundu; pilot tek30 dakikalık
  oturum içinde tamamlandı, başarısız fit girişimi0. GPU çalıştırılmadı.
- Ön kayıt kodunda18 yeni test,937 mevcut ML testi/3 skip ve Ruff geçti;
  GitHub backend/frontend/ML/Compose kontrolleri başarılı. Sonuç teslimatı
  bilimsel kaynak kodunu veya API/UI davranışını değiştirmez.
- İlk v1 sıfır-fit kaydı ve r2 ön kaydı değişmedi. 949 özgün girdi,6.646 eski
  Texas deney dosyası,42 eski kanıt,146 sicil nesnesi ve440 eski trial nesnesi
  korundu; ana dirty checkout ve gece görevi bu işte değiştirilmedi.

Arşivi yeni klasöre açıp kilitli CPU ortamında `python verify.py` çalıştırmak
20 senaryoyu,3380 payload'ı,160 seçimi ve120 aralığı eğitim yapmadan doğrular.
`python mechanism_check.py` betimleyici karşılaştırmayı tekrar üretir.
Güncel kaynakla eski checkpoint devam ettirilmez; pakette frozen source vardır.

## Sonraki tek karar

**Yeni piyasa grid'i açma; Naive'yi koru.** Önce mevcut sentetik **iç doğrulama
tahminlerinden**, yeni fit olmadan, küçültme seçiminin belirsizliğini incele:
3×63 geçmiş bloktaki ağırlık skor farkları, zaman bloklarına duyarlılık ve
seçim kararlılığı. Aynı20 senaryo/seed/model/etiket ve mevcut ağırlık kümesi
sabit kalır; OOS'tan ağırlık seçilmez ve eski sınıflandırma değiştirilmez.
Bu tek analiz, seçimde gözlenen küçük kaybın istikrarlı bir mekanizma mı
yoksa kısa doğrulama örneklerinin değişkenliği mi olduğunu ayırmak içindir.
Piyasa modelini güncellemez; yeni eğitim ve veri otomatik başlatılmaz.
