# CFTC bilgisi: mevcut named hedefle aynı soru daha önce sınandı mı?

**VERIFIED — hayır.** CFTC pozisyon/OI oranları CT=F hedefinde denenmişti;
aynı-kontrat named T+5 hedefindeki son32-alan tarifte CFTC yoktur. Bu bir
unutulmuş kolon hatası değildir: eski ham-net alanlar açıkça audit-only'dir.
Bu kaynakta sinyal bulunduğu veya yeni deneyin başarılı olacağı sonucu çıkmaz.

## Gerçekten incelenen kanıt

- Eski `research-cftc-exploration-v1`: ready/source kimliği, dondurulmuş
  history,56 yıl/grup tahmin çıktısı,1.855 completed makbuz ve5.565 native
  payload hash'i doğrulandı. Yeni fit veya eski model inference yapılmadı.
-14 mevcut yıllık ZIP'ten730 Cotton033661/FutOnly raporu yeniden ayrıştırıldı;
  hesaplanan pozisyonlar, oranlar, değişim ve29 maskeli rapor tabloyla eşleşti.
-14.042 saklanan tahmin satırında tarih/gerçekleşen CT hedefi gruplar arasında
  eşit; fiyat-MAE scalar olarak yeniden hesaplandı. Eski comparison değerleriyle
  eşleşti; kayıt veya origin silinmedi.
- Named history SHA`da8a286df8e98b8a0932f088a9b384546d30a22ebac45db0bfe32a3a021d3465`.
  Mevcut eski source politikasıyla CFTC projection eklendi; bütün eski sütunlar
  birebir korundu. Bu yalnız kurulabilirlik, kaynak kabulü veya ön kayıt değildir.

## Eski deneyin gerçek kapsamı

CT=F log-return T+5,2016–2023/2.006 origin.24 temel alan; matched age/missing
kontrolü ve managed-net/OI,producer-net/OI,managed-share değişimi. Eski
Ridge/XGB geçmiş iç seçiminde kullanıldı;1.008 XGB/847 Ridge makbuzu doğrulandı.
All-maturity pozisyon/OI, tekil kontrat open-interest'ı değildir.

| Lag | Ham Naive kazancı | Seçilmiş Naive kazancı | Ham source katkısı, Naive-MAE yüzde-puanı | Seçilmiş source katkısı |
|---|---:|---:|---:|---:|
|1|−%1,915373|−%0,854363|+0,112085|+0,159496|
|2|−%1,777400|−%0,768402|+0,227706|+0,244899|
|6|−%2,613465|−%1,782930|−0,581111|−0,771451|

Matching-control eksi source fiyat hatası için saklanan blok20 aralıkları
lag1[−0,008539,+0,015168],lag2[−0,003127,+0,013481],lag6[−0,037796,+0,001551]
cent/lb: üçü de sıfırı içerir. İç katkı koşulu üçünde başarısızdı. Bu tarif
pratik hedefi sağlamadı; **INCONCLUSIVE** kaynak-geneli/named-hedef etkisi için.
Eski CT deneyi veya lag grid'i yeniden çalıştırılmaz.

2023 position_L1 seçimi Ridge/weight0; ham Naive kazancı−%1,168323.
Named ve CT'nin ortak246 origin'inde242 named hedef uygundur:242 getirinin
hiçbiri1e−12 toleransta aynı değil; medyan mutlak getiri farkı0,004633.
Bu eski deneyin mevcut named soru yerine kullanılamayacağını doğrular;
tek başına bu farkın CFTC'nin faydasını değiştireceğini kanıtlamaz.

## Timestamp ve eksiklik

`data.py::download_cftc` eski ham-net alanlarında report_date+3 takvim gününü
kullanır. Bu tarih gerçek yayın saati veya ilk sürüm kanıtı değildir.
`features.py::select_feature_rows` bunları model input'u olarak reddeder.
Eski1.855 fit'in hiçbir tarifinde bu dört ham legacy alan yoktur; son named
tarifte de yoktur. **NOT SUPPORTED:** bu alanlardaki muhtemel erken erişimin
mevcut model başarısızlığının ana nedeni olduğu açıklaması.

Gerçek CFTC araştırması ayrı korumalı yolu kullandı: report+7 takvim günü,
sonraki Cotton observation ve lag1/2/6, maksimum10 ek observation taşıma.
2013 kapanışı8,2018–19 kapanışı12,ION2023 yedi rapor ve iki revision-day maskeli.
Maskelenen yayın son güvenilir girdinin yaşını sıfırlamaz; origin silinmez.
Yıllık arşivler güncel tarihsel değerlerdir; ilk vintage/UTC publication **doğrulanmadı**.

| Named246-origin footprint | L1 | L2 | L6 |
|---|---:|---:|---:|
| En az bir position değeri eksik |44|44|44|
| Taşınan stale-olmayan referans dönemleri |45|45|45|
| Karar anında referans yaşı: min/medyan/max takvim günü |9/11/23|10/14/24|16/18/31|

Yaş **reference-period→decision** farkıdır, publication→decision farkı değildir.
246 günlük satır,246 yeni bağımsız CFTC haberi değildir. Kaynağın bu kadar
eski olması kısa-ufuk etkisini zayıflatabilir (**PLAUSIBLE**); katkıyı gerçekten
yok ettiği kanıtlanmadı. Maskeleri kaldırmak veya+3'ü gerçek erişim kabul etmek
leakage çözümü değildir. Ham legacy sütunlar doğrudan yeniden etkinleştirilmez.

## Karar ve tek sonraki bilimsel soru

**VERIFIED:** içerik/hash ve mevcut varsayım altında causal projection kurulabilir.
**NOT VERIFIED:** ilk sürüm, gerçek tarihsel UTC erişim ve named-hedef OOS katkısı.
Kaynak model/release kabulü kapalıdır; byte doğrulaması ekonomik sinyal değildir.
Dört metamorfik kontrol geçmiş feature'ların gelecekteki rapor/etiketten
etkilenmediğini, masked raporun yaşı tazelemediğini ve gelecekteki kaydın
backfill yapmadığını gösterir. Gerçek dünya erişim varsayımını kanıtlamaz.

Tek aday: aynı named T+5 cohort ve sabit XGB'de, **mevcut korumalı L1
pozisyon değerlerinin**, aynı genişlik/NaN desenindeki yalnız age/missing
kontrolüne katkısı. Kaynak clock/vintage varsayımı,32 eski alan, hedef,
olgunluk, expanding history, iç seçim ve cadence değişmez; yeni lag/model/grid yok.
Boyut artışıyla bilgi katkısı karıştırılmaması için iki kol aynı37 alanı kullanır;
kontrolde yalnız üç finite position değeri0 olur. Age/missing iki kolda aynıdır.
Bu henüz eğitim ön kaydı değildir; source/data/environment/rapor ve bütçe
kilitlenip GitHub geri kurması tamamlanmadan piyasa fit başlatılmaz.

Bu duyarlılık deneyi pozitif çıkarsa bile gerçek erişim/vintage ve bağımsız
dönem kanıtı gerekecek. Negatif sonuç yalnız bu korumalı temsil/tarifle ilgilidir.
Clock'u ve hedefi birlikte değiştirerek görülen kazanca neden atfedilmez.
T+1 sonucu çıkarılmaz; ürün kapıları değiştirilmez. Piyasa başarı hedefi açık.

## Koruma kapsamı

Bu aşamanın private paketi eski1.855 native modeli, scaler/adapter'ları,
makbuzları,56 tahmin çıktısını, kaynak/veri snapshot'larını ve candidate
projection'ı aynı baytlarla korur. Bu, tam proje/Drive taşıması değildir.
Eski training source_id9bff893f… makbuzlarda korunur; bu aşama orijinal
eski ML snapshot'ını bulmuş veya eğitimini yeniden üretmiş sayılmaz.
Paket güncel audit kaynağını ayrıca taşır; restore modeli çalıştırmadan
native byte hash'lerini, scalar sonuçları ve projection'ı yeniden doğrular.

## Tamamlanan taze teslimat ve ek kontroller

7.831 payload üyesi/422.154.747 özgün bayt private GitHub'dan geri kuruldu;
1.855 makbuz/5.565 native hash/14.042 scalar satır ve candidate projection
frozen audit kaynağında yeniden eşleşti. Model çalıştırma/eğitim0. Ayrıca
1.855 adapter'ın processor/target scaler'ı yalnız kayıtlı geçmiş eğitim
satırlarıyla aynı; bütün target_date5 değerleri ilk test tarihinden önce.
Bu, gerçek kaynak UTC/ilk-vintage doğrulaması yerine geçmez.

[CFTC özel duyuruları](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm)
10 Mart2023 için planlanan raporun16 Mart'ta çıkarıldığını ve26 Mart2019
Cotton pozisyonlarının29 Mart raporuna3 Nisan'da eklendiğini doğruluyor.
Bu yüzden reference-date+3'ü erişim kanıtı kabul etmek somut risk taşır;
mevcut protected maskeler ve legacy alan reddi korunur. Normal haftalarda
[CFTC genel takvimi](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm)
Cuma15:30 Eastern'dır; genel takvim her eski raporun timestamp/vintage kanıtı değildir.
Mevcut [ICE ürün saatleri](https://www.ice.com/products/254) New York14:20
kapanışını gösterir; bu güncel kural bütün eski seanslara otomatik taşınmaz.
Yayın-reaction penceresinin gecikmeyle kaybolması bir hipotezdir, ölçülmüş beceri değildir.

[Dönüş makbuzu](../../../research/evidence/named-cftc-information-audit-delivery-20261012.json).
Arşiv SHA782b0b355f3ac82a0e9a756928d2ec5395c5d36f566e3a9dfaac45f8412542d5;
86.612.746 sıkıştırılmış bayt. Eski kayıt/fit sicili değişmez; kaynak kabulü kapalı.
