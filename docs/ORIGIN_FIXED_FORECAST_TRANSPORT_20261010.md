# Sabit tahminlerin origin'de bilinen kontrata taşınması — 10 Ekim 2026

## Sonucu hesaplamadan dondurulan tasarım

`origin-fixed-forecast-transport-v1-r2`, yeni fit **0**. Önceki gerçek eğri
T+1/T+5'teki CT tahminlerinin tamamı saklanır. [PR37 kurulabilirlik
denetimi](FIXED_CONTRACT_FEASIBILITY_20261010.md) doğrudan hedef değiştirip
yeniden eğitim yapmanın aynı protokol olmayacağını gösterdi. Bu daha dar
soru, mevcut tahminlerin doğru kontrat hareketinde ne yaptığını ölçer.
**Aynı kontrat hedefinde eğitilmiş model testi değildir.**

Birincil soru: **ham T+5 numeric D0 tahmini, origin'de bilinen ikinci
kontratın Naive tahminini geçiyor mu?** İkincil: seçilmiş T+5, aynı hedefte
numeric–mask katkısı, ham/seçilmiş T+1. T+1 sonucu T+5 başarısızlığını
kurtarmak için kullanılmayacak; en iyi kol sonradan seçilmeyecek.

İlk taslak seçilmiş T+5'i birincil yaptı. Yeni hedef skoru hesaplanmadan
ham çıktı seçildi: eski ağırlıklar [0;0,25;0], iki yıl düz tahmin ve yaklaşık
üçte bir aktif kapsama demek. Gerçekleşen sıfır getiriler düz yön hit'ini
etkileyebilir; **koşulsuz %55 doğruluk imkânsızlığı iddia etmiyoruz**.
Seçilmiş skor, ham yön ve aktif oran ayrı raporlanır. İki eski taslak ve
bu anlamsal açıklamanın hash'leri saklanır; hedefe göre yeniden seçim yok.

## Tek dönüşüm

Karar anında D0'ın son varsayımsal erişilebilir raporundan **ikinci
kontrat adı** seçilir. Aynı isim hedef güne kadar sabit tutulur. Origin'in
referans günündeki fiyatı karar saatine kadar varsayımsal erişilebilir
olmalıdır; son eski quote ile doldurulmaz. Hedef gün, özgün Cotton T+1/T+5
tarihidir; geleceğin ilk kontratı kullanılmaz.

`forecast_named = origin_named_price × exp(frozen_CT_predicted_log_return)`

Çarpan beta=1 sabit; Naive=`origin_named_price`. Model, girdi, çıktı
ölçeği, geçmiş shrinkage ağırlığı, takvim ve hedef tarihi değiştirilmez.
Baseline ve aday **aynı adlandırılmış kontratın aynı hedef fiyatına**
karşılaştırılır; eski CT MAE ile farklı hedef MAE'si yarışmış sayılmaz.

İki ufukta aynı749 origin makbuzu korunur;739 uygun,10 açık
eksik/uygunsuz. Kaynakta olmayan hedef fiyatı ve geç/eksik origin fiyatı
Naive dahil hiçbir kol için doldurulmaz. Yeni veri indirme veya fit yok.

## Saat, istatistik ve karar sınırı

Karar: Cotton tarihi+1 gün00:15 UTC. Varsayımsal erişim:
`max(report_date, unverified publication calendar day)+1 gün00:00 UTC`;
eskilik≤3 Cotton gözlemi. UTC/ilk vintage kanıtı ve kaynak kabulü yok.
2021–2023 önceden incelenmiş araştırma geçmişidir; bağımsız holdout değil.

Yıl içinde blok20/60,10.000 tekrar,seed42; özgün252/251/246 ordinal
yuvaları korunur. Eksik yuvalar CSV'de bilinmeyen kalır. Bootstrap'ta
yalnız ağırlık0 ve katkı0 kullanılır, hata etiketi üretilmez; her draw
kendi uygun ağırlık toplamına bölünür. Recentered basic aralık ve sabit
gözlenen comparator MAE normalizasyonu; insan/model seçimi düzeltilmedi.
739 ve20/60 hiçbir şekilde takvim sıkıştırılarak hesaplanmaz. Her ufukta
aynı sekiz en büyük named-Naive hatası çıkarılan duyarlılık da raporlanır.

- Ham T+5 ≥%5 kazanç, ≥%55 yön,3/3 pozitif yıl ve iki alt sınır>0 ise
  aynı hedefte kilitli yeniden-eğitim takibi için adaydır; üretim başarısı
  değildir, eski6/8 şartının yerine geçmez.
- İki ham T+5 üst sınırı da %5'in altındaysa bu **dondurulmuş beta1 taşıma
  tarifi** pratik hedefi kurtarmıyor. Bütün yeni-hedef modelleri veya
  kaynaklarda bilgi yokluğu sonucuna genişletilmez.
- Diğer sonuç belirsiz; otomatik fit/grid başlamaz.

Sentetik aritmetik kontrolü tüm etiket biliniyorsa mevcut paired
bootstrap'a indirgemeyi, eksik yuvaları/ağırlıkları ve sabit paired
kayıpları doğruladı. Bu öğrenme kontrolü veya piyasa kanıtı değildir.
[Ön kayıt kimliği](../research/evidence/origin-fixed-forecast-transport-preregistration-20261010.json)
kod/girdi/contract hash'lerini sonuç hesaplanmadan bağlar. Basılı AMS
quote'u yürütülebilir settlement/fill ispatı sayılmaz; işlem yapılmaz.
