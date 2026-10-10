# Gerçek eğri ve T+5 hedefi — sıfır-fit inceleme

**VERIFIED:** yeni eğitim olmadan, [tamamlanmış T+1 deneyinin](CONTRACT_CURVE_RESULT_20261010.md)
aynı 749 origin'inde T+1/T+5 hedefleri kaynak kontrat fiyatlarıyla ayrıştırıldı.
T+5 bir model sonucu değildir. Üç incelenmiş yıl bağımsız holdout değildir.

## Ne bulundu?

| Ölçüm | T+1 | T+5 |
|---|---:|---:|
| Asıl kohort / iki tarihinin tablosu bulunan | 749 /743 | 749 /743 |
| Kontrat çifti bilinmeyen; silinmedi | 6 | 6 |
| İlk kontratı değişen origin | 15 | 74 |
| Ayrı gözlenen kontrat değişimi | 15 | 15 |
| Geçiş origin'lerinin kapsanan Naive hata payı | %3,867570 | %10,154365 |
| Kapsanan Naive fiyat-MAE | 1,486635 | 3,431817 |
| Gelecek kontratını sonradan bilen gap-only ayrıştırma MAE | 1,462382 | 3,332571 |
| Bu **ex-post**, tahmin olmayan hata azalması | %1,631404 | %2,891947 |

T+5'te geçişteki geleceğin ilk kontratı 74/74 kez origin'in ikinci kontratı;
T+1'de 15/15. Bu, gelecek geçiş tarihinin origin'de bilindiğini kanıtlamaz.
Tablolardaki ilk-vade değişimleri doğrulanmış ICE roll veya işlem takvimi değildir.
Kapsanan T+5 yıl bazında ex-post azalmalar 2021/2022/2023: %4,109008 /
%3,137993 /%0,766364. Geçişte ortalama mutlak curve bileşeni %2,231772,
aynı hedef kontratının hareketi %2,572261: vade farkı tüm hareket değildir.

Tam 749 origin Naive MAE: T+1 1,4978238094314558, T+5 3,435113517241739.
Bilinmeyen altının toplam Naive hata payı %1,542068 /%0,896265. Bunlar
743'ün cebirsel metriklerine sessizce katılmadı; model kohortu değiştirilmedi.

## Mekanizma ve leakage sınırı

Origin'in ilk kontratı A, hedef tarihinin ilk kontratı B ise:

`log(F_B(t+h)/F_A(t)) = log(F_B(t)/F_A(t)) + log(F_B(t+h)/F_B(t))`.

Birinci terim mevcut eğri farkı, ikinci terim **aynı hedef kontratının** fiyat
hareketidir. A hedef gününde tablodan kaybolsa bile B origin'in on-vade
tablosunda bulunur; A'nın gelecekteki fiyatı uydurulmadı. Saklanan hedefteki
ayrıca çok küçük fark, kaynak fiyatının float32'ye yuvarlanmasından gelir.
Bu mekanizma bütün 1.486 kapsanan çiftte doğrulandı; yuvarlama çıkarılınca
azami log kimlik hatası 3,79e−16'nın altında. **Hedef düzeltilmedi.**

**VERIFIED:** sürekli-seri hedefi geçişte iki ayrı kontratı karşılaştırır.
Bu log değişim doğrudan bir kontratı elde tutmanın getirisi değildir.
**NOT SUPPORTED:** “roll, bütün başarısızlığın ana nedeni.” T+1'in önceki
±5-gözlem dışı 584 origin'inde ham numeric kollar hâlâ Naive altında; yeni
ayrıştırma da hareketin yalnız bir kısmını açıklar. %2,89 bir üst sınır veya
elde edilebilir kazanç değildir; vade farkı bazen hareketi kısmen iptal eder.
Bu hesap diğer modelleri veya tüm eğitim yıllarını elemez.

**Gelecekte hangi kontratın seçildiği ex-post bilgidir.** Yalnız tanı dosyasında
kalır; feature, kaynak saati, model/parametre seçimi, origin elemesi veya
gelecekteki roll takvimini origin'e taşıma için kullanılamaz. Final tabloların
referans günü, kanıtlanmış erişim zamanı/ilk vintage değildir. Gap-only satırlar
OOS tahmin veya ekonomik getiri olarak indekslenmez. Aralık/anlamlılık testi
yok: bir model performans deneyi yapılmadı.

## Sonraki tek bilimsel karar

[T+1 sonucunun ufuk incelemesi](CONTRACT_CURVE_RESULT_20261010.md) bu tanıyla
daraltılır; eski plan veya kayıt silinmez. Önce **tek T+5 duyarlılık ön kaydını**
hazırlamak: aynı dört mask/numeric D0/D1 kolu, 749 origin, 24 çekirdek özellik,
kaynak/null göstergeleri, alpha1 Ridge, expanding history, H5 olgunlaşması,
3×63 iç doğrulama, 21 refit ve geçmişten seçilen aynı shrinkage. Yalnız
model ufku T+5 olur; D0/D1 erişim ve ilk-vintage varsayımı aynen kalır.

Beklenen tavan 108 iç +144 dış =252 piyasa fit'i; en fazla1 ayrı sentetik
kontrol, 12 yıllık çıktı/2.996 tahmin. Bu **bütçe taslağıdır**, yeni kod/veri/
ortam/karar kimliği ve yürütme bütçesi dondurulmadan fit izni değildir.
Mevcut T+1 profili T+5'i reddetmeye devam eder; checkpoint yeniden kullanılmaz.

Birincil ölçüm bütün 749 origin'de seçilmiş numeric D0–mask D0 paired
fiyat-MAE, ayrıca Naive farkı; D1 ve ham sonuçlar ikincil. Yılları koruyan
blok20/60 ve T+5 overlap ön kayıtta tutulur. **74 origin, 15 bağımsız olay
bile değildir**; fiyat uçlarını paylaşmayan aralıkların betimleyici sayısı
125, bağımsız örnek sayısı değil. 5%/55%/6-of-8 gate'leri korunur; üç yıl
6/8'i ölçemez, 2/3 veya 3/3 yerine konmaz.

Tam kohort başarı kararı değiştirilmeden, bu 743 sabit tanı çiftinde seçilmiş
ve ham hata farkı geçiş/aynı-kontrat/bilinmeyen olarak **sonuç-sonrası açıklama**
raporlanmalıdır. Kaynak katkısı yalnız geçişlerde görünürse önce proxy mekanizması
açıklanır; gerçek aynı-kontrat hareketi/pozisyon becerisi iddia edilmez.
Bu alt gruplar model seçmez, eğitime girmez, yeni eşik veya bağımsız test olmaz.
Pozitif T+5, hedef-seride ufka bağlı katkıyı destekleyebilir; tek başına ekonomik
öngörü ispatı değildir. Negatif sonuç tüm eğride bilgi yokluğunu kanıtlamaz.

## Kanıt ve tekrar

[Sayısal kanıt](../research/evidence/curve-target-integrity-20261010.json),
[ayrı checksum Release'i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/curve-target-integrity-20261010).
Analiz/verifier yalnız pandas/numpy/standart kütüphane okur; model fit API'si yok.
Bağımsız Decimal yeniden hesaplaması 1.486 çift, 1.498 hedef tarihi, bütün
yıl/olay metrikleri ve altı bilinmeyen kaydı doğrular. Beş geçersiz hedef/
oracle/cohort/hash/overwrite kontrolü geçer. ML kaynağı, eski tahminler,
1.000 ham girdi, registry/trial/proof nesneleri korunur; yeni fit 0.

Arşiv açıldıktan sonra mevcut ayrı CPU ortamında:

```text
python analysis.py --experiment inputs --output replay
python verify.py --experiment inputs --audit replay --analyzer analysis.py --receipt replay-verification.json
python controls.py --experiment inputs --audit replay --analyzer analysis.py --verifier verify.py --receipt replay-controls.json
```

`replay` yeni dizin olmalıdır; mevcut veya yarım kalmış kanıt üzerine yazılmaz.

**Teslimat doğrulandı:** [makbuz](../research/evidence/curve-target-integrity-delivery-20261010.json).
GitHub'dan yeniden indirilen20 üye doğrulandı; rapor/1.498 tanı satırı,
completion, bağımsız Decimal makbuzu ve5 ret kontrolü byte olarak birebir
replay edildi. Arşiv1.967.150 byte, SHA256
`31ab3e517a6acaf932b5de43cdfb4a421056dfc3cb19ec51d144fba701af1dee`.
8 history testi/ML Ruff başarılı; yeni fit0. Önceki69 proof/159 çalışma/
513 tarif ve1.000 ham girdi korunur. PR #32 final-head sekiz CI başarılı;
PR #33'ün CI anlık durumu makbuzda ayrıdır, çalışan job geçmiş sayılmaz.
