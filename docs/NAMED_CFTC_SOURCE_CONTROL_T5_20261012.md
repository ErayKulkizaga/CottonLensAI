# Named T+5: korumalı CFTC içeriği kontrolü

## Soru ve kapsam

Sabit XGBoost/named-kontrat hedefinde CFTC'nin üç sayısal pozisyon alanı,
aynı kaynak yaşı ve eksiklik bilgisine ek tahmin katkısı sağlıyor mu?
[Önceki denetim](NAMED_CFTC_INFORMATION_AUDIT_20261012.md) eski CFTC
deneyinin farklı CT=F hedefini kullandığını doğruladı. O 1.855 fit yeniden
çalıştırılmaz. Tamamlanan 132-fit sentetik kontrol de tekrar edilmez.

Bu çalışma tek, daha önce incelenmiş 2023 kohortuna aittir. Bağımsız holdout,
sekiz yıl kapısı, üretim modeli veya kaynak kabulü değildir. Gerçek tarihsel
UTC yayını ve ilk vintage kanıtlanmadı. Korumalı saat altında içerik katkısını
sınar; daha erken erişim, bütün kaynak temsilleri veya bütün modeller hakkında
karar vermez. Naive'ı geçmek bir test veya süreç tamamlanmasıyla sağlanmış sayılmaz.

## Dondurulacak tarif

- Profil `named-cftc-source-control-t5-v1`; deney
  `research-named-cftc-source-control-t5-v1`.
- Private CFTC denetim Release'inden taze geri doğrulanmış gerçek named
  geçmişi ve mevcut `positions-2010-2023.csv`. Yeni veri indirme yok.
- Aynı 246 origin, 242 uygun hedef, dört bilinmeyen hedef korunur.
  Eğitim günleri etiket olgunluğu kuralından yeniden üretilir; sıkıştırılmış
  takvim veya eksik özellik nedeniyle origin düşürme yok.
- Aynı 32 alan + `cftc_age_L1`, `cftc_missing_L1`,
  `cftc_managed_net_share_L1`, `cftc_producer_net_share_L1`,
  `cftc_managed_share_change_L1`: iki kolda da 37 alan.
- Numeric kol özgün değerleri taşır. Mask kol yalnız son üç alanın sonlu
  değerlerini sıfırlar. NaN, yaş, missing, diğer bütün sütunlar aynıdır.
- CFTC report-date +7 takvim günü, sonraki Cotton gözlemi, L1 ve mevcut
  10-gözlem carry sınırı aynıdır. Legacy +3-gün alanları açılmaz; L2/L6 yok.
  Bilinen gecikme/revizyon maskeleri değişmez. Saat varsayım olarak kalır.
- Ayrı bir öğrenme motoru yok: mevcut `Experiment.fit`, ledger, preprocessing,
  standardized log-return, named T+5 hedefi, olgunlaşma ve refit düzeni.
- XGBoost depth2, eta.03, min_child_weight20, alpha0, lambda1, seed42,
  100 ağaç; window1, expanding history, cadence21. Hiperparametre araması yok.
- Aynı 3×63 geçmiş doğrulama; `[0,.25,.5,.75,1]` ağırlığı yalnız geçmişte,
  eşit blok-relative fiyat-MAE ortalaması ve en küçük ağırlıkla eşitlik kırma.
- 44 piyasa + dört güçlü sentetik mekanik kontrol; toplam en fazla 48 fit.
  Bir CPU süreç, en fazla iki thread, 30 dakika oturum. Hatalı denemeler de
  bütçeye girer; duraklamada mevcut checkpoint korunur, otomatik tarif değişmez.

## Eğitim kapısı ve doğrulama

Kod/testler bittikten sonra yeni source/data/environment/design kimliği
dondurulur. Eski modeller/data/evidence değişmez. Seçili girdiler private
GitHub denetim paketinin checksum envanterine bağlanır; gerçek named geçmişi
ile candidate projeksiyon tekrar hesaplanır. Sentetik öğretici hedefleri
piyasa deneyine aktarılmaz.

Ön kayıt, sabit raporlayıcı ve bağımsız doğrulayıcı GitHub'a teslim edilir;
taze indirme/restore makbuzu olmadan `prepare` ve `pilot` çalışmaz. Kaynak,
ortam, veri, tasarım veya yayın makbuzu değişirse cache reddedilir.
Her modelin scaler/target ve native tahmini geçmiş train satırlarıyla yeniden
doğrulanır. Yarım çıktı tamamlanmış deney olarak raporlanmaz.

57 ilgili regresyon testi ve Ruff geçti. Testler sentetik/veri bütünlüğü
kapsamındadır; piyasa becerisi kanıtı değildir. Backend/UI/artefact API şeması
değişmedi; GPU ve gerçek piyasa eğitimi henüz çalışmadı.

## Önceden belirlenen analiz ve karar

Birincil: **ham mask-MAE eksi numeric-MAE**, pozitif değer kaynak katkısıdır.
Ham ve küçültülmüş Naive kazancı, yön doğruluğu, aktif oranı ayrı verilir.
En büyük %1 Naive hatası çıkarılmış duyarlılık ikincildir. Her karşılaştırma
aynı origin, target-date, kontrat, fiyat, hedef ve uygunluk sırasını zorunlu
tutar; sessiz kesişim alınmaz.

Paired moving-block bootstrap: 10.000 tekrar, seed42, blok20 ve60. Dört
bilinmeyen hedef kaybı sıfır-bookkeeping weight ile tam 246 takvim slotunda
kalır; gerçek sıfır kayıp gibi ortalamaya girmez. Bu aralıklar geçmişteki
model/insan seçiminin çoklu karşılaştırma yükünü kapsamaz.

1. İki ham kaynak-katkısı aralığının alt sınırı pozitif, numeric ham Naive
   kazancı ≥%5 ve yön ≥%55: yalnız araştırma adayı; 6/8 ve ileri doğrulama
   sağlanmış sayılmaz.
2. Katkı aralıkları pozitif fakat pratik nokta eşikleri sağlanmıyor: katkı var,
   hedef çözülmedi; otomatik grid/kaynak ekleme yok.
3. Pozitif katkı gösterilmedi ve numeric ham Naive kazancının iki üst sınırı
   <%5: bu sabit saat/temsil/tarif mevcut pratik hedefi kurtarmıyor.
4. Diğer sonuçlar belirsizdir; kaynakta sinyal yokluğu sonucu çıkarılmaz.

Sonuç ilk karar tablosuna yerleştirilmeden yeni model/grid açılmaz. Üretim
modeli, live kayıtları, Drive özgünleri veya PR merge bu teslimata dahil değildir.

## Dondurulan ön kayıt

Kod commit'i `b9ea09c`; son ML source_id
`8eca71445366d207b74393acad43968ab5f8f5f6c6954ee21ce22dda24c3af38`
(267 ML dosyası), design_id
`b07ad81d9476d6599640fb7f7e2fc26db830d944560b7482660ba751c5d1ef26`.
Ready SHA `28189753d35ea4d3dca3ca8a50e8ec2c66d4bffa041585cd162aa49de889d7bd`.
[Ön kayıt makbuzu](../research/evidence/named-cftc-source-preregistration-20261012.json)
yeni ek aşamadır; eski181aşama ve527fit tarifi değişmez. Kayıtlı bütçe44+4;
paket hazırlanırken fit0. Tam tarifi kontrol eden `ml/history.py check`
RELATED_EVIDENCE verdi, aynı scope'ta tamamlanmış fit tarifi bulunmadı.
Bu sonuç tek başına yenilik kanıtı değildir; eski CFTC/hedef farkı yukarıda gerekçelidir.

## Tamamlanan sonuç — 44 piyasa + dört mekanik sentetik fit

Native44+4 model replay geçti. Bağımsız Decimal fiyat hesabı984satırı,
242eşleştirilmiş hedefi ve12tam-takvim bootstrap aralığını doğruladı;
doğrulama yeni fit üretmedi. Kaynak/devam cache kimliği değişmedi.

| Kol | Ham fiyat-MAE | Naive kazancı | Ham yön | Geçmiş shrinkage |
|---|---:|---:|---:|---:|
| Numeric CFTC | 1,998289576 | −%1,124208 | %52,892562 | 0 |
| Mask kontrol | 2,034042103 | −%2,933479 | %51,239669 | 0 |
| Named Naive | 1,976074380 | %0 | — | — |

Ham kaynak katkısı mask−numeric **+0,035752527cent/lb**, Naive-MAE'nin
**+1,809270yüzde puanı**. Blok20 aralığı **[−3,378239,+6,853990]**, blok60
**[−2,400693,+5,793537]** Naive-MAE yüzde puanıdır. Sıfır ikisinin içinde:
**INCONCLUSIVE** yalnız kaynak-katkısı işareti için.

Numeric ham Naive kazancı aralıkları blok20 **[−%4,773761,+%2,589670]**,
blok60 **[−%3,847223,+%2,105044]**. İki üst sınır da önceden belirlenen
%5'ten küçük: tarif kararı **FIXED_NAMED_CFTC_BELOW_PRACTICAL_GOAL**.
Bu, incelenmiş tek-yıl/final-vintage/saat-varsayımı içindeki sabit tarif
hakkındadır; bütün CFTC bilgisi veya bütün modeller için sinyal yokluğu değildir.

En büyük%1Naive hatası çıkarıldığında numeric kazancı −%1,052135, mask
−%2,860879; kaynak katkısı +1,760864Naive-MAE yüzde puanı. Başarısızlık
sadece birkaç aşırı hata ile açıklanmadı. İki kolun geçmiş seçimi0, küçültülmüş
kazanç0/aktiforan0. Bu `[0,0]` farkları kaynakta sinyal yokluğunu göstermez;
ham karşılaştırma ayrıca korunur.

Mekanik kontroller: lineer%92,479218, nonlinear%78,332078, ezberleme%99,959889
MAE iyileşmesi; negatif kontrol%2,197706. Bunlar yeni37alanın güçlü öğreticiyi
öğrenebildiğini gösterir; piyasa becerisi veya zayıf/karmaşık kaynak gücü değildir.

**Bir sonraki tek adım:** sonuç paketini private GitHub'dan taze geri kurup
native/tahmin/bağımsız aralıkları tekrar doğrulamak; sonra tamamlanan named
kontrollerin ortak kohortundaki hangi bilgi kaybını gerçekten ayırabildiğini
sıfır-fit sentezle kapatmak. Yeni model, lag, horizon veya kaynak grid'i yok.
Mevcut kanıtla daha erken CFTC saatini ya da yeni modelleri otomatik açmak
bilimsel olarak gerekçelendirilmedi. Tahmin başarısı henüz sağlanmadı.

## Kanıt teslimatı ve tekrar kapısı

Private sonuç Release'inden **542üye** taze geri indirildi;44+4native replay,
12scalar aralık ve sekiz coherent tamper reddi tekrar geçti. Yeni doğrulama
fit'i0. [Teslimat](../research/evidence/named-cftc-source-result-delivery-20261012.json).
48 modelin adapter/native/curve/ledger/veri/kod/tahmin bağları korunur;
tam Drive/model arşiv taşıması bitmiş sayılmaz. Eski181aşama/527tarif ve
eski kanıt dosyaları değişmedi. Kaynak-katkısı sınıfı INCONCLUSIVE, pratik
karar yukarıdaki eşiklere göre negatiftir; bu iki sonuç ayrı tutulur.

Ön kayıt helper'ındaki exact-repeat scope projection, `history_index`'in
identity_id/split_id içeren projection'ıyla eş değildi. İlgili geçmiş sorgusu
eski deneyleri gösteriyordu; no-match zaten yenilik kanıtı sayılmamıştı.
Dondurulmuş eski proposal değiştirilmedi. [Ek düzeltme](../research/evidence/named-cftc-source-history-scope-correction-20261012.json)
aynı hazır identity'den canonical scope üretir; yeniden fit/etiket/metrik
uyarlaması yok. Yeni canonical proposal ile komut artık **REPEAT_FIT_RECIPE**,
44tamamlanmış piyasa fit'i bulur. Gelecek tam-scope aramasında `history_index`
projection'ını kullan; model tarifi eşitliği ve partial-source araması birlikte
incelenir. Sentetik kontrol rollerinin yeni sicil projeksiyonu açıkça sentetiktir;
eski kayıtlar yeniden yazılmaz.
