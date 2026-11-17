# Beşinci dış değerlendirmeden sonraki plan (17 Kasım 2026)

Değerlendirmenin özeti: araştırma disiplini güçlü; bilimsel kanıt orta düzeyde; ürün kanıtı zayıf. Önerilen yön: yeni deney yapmayı bırakmak, tahmin aralıklarını kalibre etmek, alan→üretim kopukluğunu açıklamak, canlı sistemi tamamlamak, sunumu sadeleştirmek ve kullanıcı testi yapmak.

`docs/archive/phase2-notes/RESEARCH_PLAN_20261116.md` dosyasının devamıdır. Mülakat yaklaşık 23 Kasım'da.

## 0. Eleştiri maddelerinin repoyla karşılaştırması
| Madde | Durum | Not |
|---|---|---|
| Birleşik kör test sonradan seçildi | Doğru, zaten açıklanıyor | Değişiklik yok |
| T50 bugün çalışmıyor (%43) | Doğru, zaten açıklanıyor | Değişiklik yok |
| Alan kazancı üretime geçmiyor | Doğru | Nedeni hiç incelenmedi → **B2** |
| %80 aralık gözlemlerin %64'ünü kapsıyor | **Durum daha kötü** | Monitörün aralık yöntemi (önceki sezonların örneklem dışı artıkları), gerçek zamanlı uygulandığında 2011–2025'te %80 aralıkta **%52**, %95 aralıkta **%79** kapsıyor (58 tahmin). Sebep: 2012 sonrasında hatalar büyüdü, eski artıklar aralığı dar tutuyor. Monitördeki `model_v1` aralıkları yalnızca geriye bakınca doğru. → **B1** |
| Sunumda 1.068 açılır alan | Doğru | Hepsi `cottonlens-sunum.html` içinde (İngilizce sayfada 42 var) → **C1** |
| Canlı sistemin kanıtı yok | Kısmen doğru | Monitör çalışıyor ve kaydı zincirli, ama zamanlanmış çalıştırma yalnızca `main` dalında çalışıyor. PR #78 birleşmediği için sistem henüz otomatik değil. → **A** |
| T49 bağımsız doğrulanmalı | Doğru | Yeterli verisi olan, hiç kullanılmamış bir kuru tarım pamuk bölgesi kalmadı. Bağımsız doğrulama ancak ileriye dönük kilitlerle yapılabilir. → **B3** |

## 1. Kararlar
- **Yeni hipotez testi yok.** T56 (alan sinyalinin piyasa testi) rafa kalkıyor. Sebep: üçüncü piyasa testi olurdu, testin gücü düşük ve eleştiri tam olarak deney şişmesini hedef alıyor.
- **Bundan sonraki işler kalibrasyon, teşhis ve ürün.** Yeni bir "USDA'yı geçtik" iddiası kurulmayacak.
- Önceki sonuçlar ve başarısızlıklar olduğu gibi kalıyor.

## 2. İş paketleri (öncelik sırasıyla)

### A. Canlı sistem (mülakattan önce)
1. **PR #78'in `main`'e birleştirilmesi.** Kullanıcı kararı. Birleşmeden zamanlanmış çalıştırma olmaz.
2. **Performans geçmişi sayfası.** `log.jsonl` dosyasından her rapor için şunları gösteren bir sayfa üretilecek:
   - USDA tahmini;
   - bizim tahminimiz ve aralığı;
   - doğrulama sonucu, yani tahminin geçip geçmediği ya da USDA rakamına dönülüp dönülmediği;
   - kesin değer açıklandığında hata.
   Sayfa iş akışı tarafından her çalıştırmada yeniden üretilecek.
3. **`latest.json`.** Son tahmin makinece okunabilir bir dosya olarak yayımlanacak. Bu, sunucu gerektirmeyen bir API görevi görür.
4. **Hata bildirimi.** İş akışı başarısız olursa ya da veri tazelik kontrolünden geçemezse (rapor gelmediyse, ürün durumu verisi eskiyse) otomatik olarak bir GitHub issue açılacak.
5. **10 Aralık 2026** ilk otomatik canlı çalıştırma olacak. Mülakatta sistem "kurulu, zamanlanmış, ilk otomatik kayıt 10 Aralık'ta" diye gösterilecek.

### B. Bilimsel düzeltmeler (mülakattan önce B1 ve B2)
1. **B1 · Aralık kalibrasyonu (ön kayıtlı kalibrasyon çalışması; hipotez testi değil).**
   - Teşhis yukarıdaki tabloda: gerçek zamanlı kapsama %52 / %79.
   - Önceden yazılacak üç aday yöntem:
     - (a) hata büyüklüğünü zayıf ürün oranına bağlayan ölçekli aralık (kuraklık yılında hata büyür);
     - (b) yalnızca son N sezonun artıklarını kullanan pencere;
     - (c) iki katı genişlikte basit muhafazakâr aralık, referans olarak.
   - Değerlendirme: 2011–2025 gerçek zamanlı kapsama. Seçim kuralı önceden yazılacak: %80 aralık için %72–88, %95 aralık için en az %90 kapsamayı sağlayan en dar yöntem.
   - Hiçbiri kuralı karşılamazsa monitör aralıkları "kalibre değil" etiketiyle gösterecek.
   - Kural karşılanırsa `model_v2.json` olarak dondurulacak; `model_v1` kayıtları da olduğu gibi kalacak.
2. **B2 · Alan→üretim kopukluğunun teşhisi (betimsel çalışma, yeni iddia yok).**
   - Hipotez: terk edilen tarlalar en düşük verimli tarlalardır. USDA alanı fazla tahmin ettiğinde verimi (hasat edilen akre başına) eksik tahmin eder ve iki hata birbirini kısmen dengeler. Bu yüzden yalnızca alanı düzeltmek üretim hatasını bir kısım iki kez sayar. T31 ("daha iyi alan, daha kötü üretim") de bunu ima ediyordu.
   - Yapılacak: Teksas 2008–2025 için alan bileşeni ile verim bileşeninin korelasyonu, ay ay. Ayrıca alan düzeltmesinin üretimdeki etkisinin ne kadarını verim bileşeninin geri aldığı hesaplanacak.
   - Sonuç makaleye ve sunuma "neden üretime geçmiyor" açıklaması olarak girecek.
   - Düzeltme önerisi (üretimi doğrudan zayıf ürün oranıyla düzeltmek) ayrı bir ön kayıt gerektirir. Bu, mülakattan sonraya kalıyor.
3. **B3 · Bağımsız ileri doğrulama.**
   - Kasım 2026 raporu ve ürün durumu verisiyle Oklahoma ve Kansas'ın 2026 hasat alanı da kilitlenecek. Teksas kilidinin yanına, keşifte kullanılmamış iki eyalet daha eklenmiş olur.
   - Puanlama Mayıs 2027'de.
   - T49'un bağımsız doğrulaması bundan sonra 2026, 2027 ve sonraki sezonların ileri kilitleriyle birikecek.

### C. Sunum (Codex)
1. **Ana sayfa sadeleşecek.** Şunlar ana sayfada kalacak:
   - üç ana bulgu (Teksas alanı %40; mekanizma: hata verimden alana kaydı, USDA üçte iki oranında tepki veriyor; kör revizyon yönü %67);
   - sınırlar (üretimde anlamlılık yok, piyasada değer yok, aralıklar kalibrasyon sürecinde);
   - canlı monitör paneli (A2).
2. **Deney arşivi ayrı bir sayfaya taşınacak.** 1.068 açılır alan ve model envanteri oraya gidecek.
3. Codex için `18_SUNUM_SADELESTIRME_CODEX.md` dosyası, A ve B tamamlandıktan sonra yazılacak.

### D. Kullanıcı testi (kullanıcının yapacağı)
- Hedef: 3 kişi (emtia analisti, tüccar ya da tekstil satın almacısı). Mülakat da bunlardan biri sayılabilir.
- Her biriyle 20 dakikalık görüşme. Üç soru sorulacak:
  - Bu rapor ayında hangi kararı veriyorsunuz?
  - Teksas alan tahmini o kararı değiştirir miydi?
  - Bir aralık size ne ifade ediyor, nasıl kullanırdınız?
- Notlar `docs/user_tests/` altına yazılacak. Ölçü: "kararı değiştirirdi" diyen kişi sayısı ve istenen özellikler.

## 3. Takvim
| Ne zaman | İş |
|---|---|
| 17–19 Kasım | B1 ön kayıt ve çalışma; B2 teşhis; B3 kilit; A2–A4 kodu |
| 19–20 Kasım | PR #78 birleştirme kararı (kullanıcı); ilk elle tetiklenen çalıştırma ve performans sayfası |
| 20–21 Kasım | Codex sunum sadeleştirmesi (C); mülakat provası |
| ~23 Kasım | Mülakat |
| 10 Aralık 2026 | İlk otomatik canlı kayıt; T50 yönlerinin puanlanması |
| Mayıs 2027 | ABD ve Teksas kilitlerinin, OK/KS alan kilitlerinin ve aralık kapsamasının ilk puanlaması |

## 4. Yapılmayacaklar
- Yeni ürün, bölge ya da dönem için yeni hipotez testi (T56 dahil).
- Sunucu ya da veritabanı gerektiren bir API. Statik `latest.json` yeterli.
- Mühürlü 2022 sonrası fiyat verisini açmak.

## 5. Durum (17 Kasım 2026, akşam)
| Paket | Durum |
|---|---|
| B1 · CAL1 | Tamamlandı. Yalnızca "2 kat" yöntemi kurala uydu (%83 / %93); geniş aralık. `model_v2` olarak donduruldu. |
| B2 · alan–verim dengelenmesi | Tamamlandı. 2012 sonrası korelasyon −0,57 (Aralık −0,92); alan kazancının %36'sı üretime geçiyor. |
| B3 · Oklahoma/Kansas kilidi | Tamamlandı. OK 243.609, KS 67.878 akre. |
| A2–A4 | Kod tamamlandı ve test edildi (16 test). Otomatik çalışma PR #78'in `main`'e birleşmesini bekliyor. |
| A1 · PR #78 birleştirme | Kullanıcı kararı bekleniyor. |
| C · sunum | Codex'e verilecek (17. dokümanın 7. bölümü). |
| D · kullanıcı testi | Kullanıcı yapacak. |
