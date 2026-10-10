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

## Kayıtlı sonuç

**VERIFIED:** bu dondurulmuş beta1 taşıma tarifi pratik hedefi kurtarmadı.
Yeni fit0; aynı749 origin/ufuk,739 eşleştirilmiş uygun fiyat ve10 açık
eksik/uygunsuz kayıt korunur. Baseline ve aday aynı kontrat hedefindedir.

| Çıktı | Named Naive MAE | Model MAE | Naive kazancı | Yön | Aktif | Yıl kazanımı |
|---|---:|---:|---:|---:|---:|---:|
| **Birincil ham T+5 numeric D0** |3,055940|3,320714|−%8,664216|%46,9553|%100|0/3|
| Seçilmiş T+5 numeric D0 |3,055940|3,075036|−%0,624857|%14,8850|%33,6942|0/3|
| Ham T+5 mask D0 |3,055940|3,168876|−%3,695591|%48,7145|%100|1/3|
| Ham T+1 numeric D0 |1,332409|1,377037|−%3,349412|%48,1732|%100|0/3|
| Seçilmiş T+1 numeric D0 |1,332409|1,332409|%0|%0,1353|%0|0/3|

Birincil T+5 kazancının blok20/60 aralıkları **[−%15,259974;−%1,196524]** /
**[−%15,392809;−%0,875167]**. Yıllık kazanç −%0,511175 /−%9,603124 /
−%17,320578. En büyük sekiz named-Naive hatası çıkarıldığında −%9,442019;
sonuç yalnız birkaç uç hata tarafından taşınmıyor. Seçilmiş T+5 aralıkları
[−%1,971056;+%0,887594] /[−%1,925530;+%0,864890]. Shrinkage zararı
azaltıyor; kaldırılması işe yarar kaynak bilgisi ispatı değil.

Ham T+5 numeric–mask katkısı **−%4,791549**, blok20/60
[−%8,937417;+%0,227408] /[−%8,652294;−%0,265899]. Bu iki aralığın
işareti aynı kesinliği vermiyor; bütün eğri bilgisinde sıfır sinyal sonucu
çıkarılmaz. T+1 ham Naive aralıkları [−%6,599255;−%0,387430] /
[−%6,039934;−%0,539679]. Sıfır seçilmiş aralığı bilgi yokluğu kanıtı değil.

**STRONGLY SUPPORTED:** “mevcut tahminler iyi, yanlış proxy ölçümü onları
kötü gösteriyor” açıklaması bu sabit tahminlerde geçerli değil. Önceki
carry ayrıştırmasında küçük bir katkı proxy mekanizmasına bağlıydı;
burada gerçek origin-kontrat hareketi de Naive üstünlüğü vermedi.
Eski CT749 ve yeni named739 MAE'lerini birbirine yarışmış saymıyoruz.

**Henüz ayrılmayan iki açıklama:** CT üzerinde öğrenilen katsayıların
adlandırılmış kontrat hedefi için yanlış olması; veya mevcut bilgi/
Ridge temsilinin faydalı yön bilgisini üretememesi. Bu test sadece ilk
açıklamanın **ölçüm-only** sürümünü zayıflatır; eğitim-label sürümünü
sınamaz. Tüm modellerde/sources'da öğrenilebilirlik iddiası yok.

## Doğrulama ve tek sonraki bilimsel karar

Bağımsız datetime/Decimal/scalar kod5.992 satırı,5.936 fiyat dönüşümünü
ve32 ağırlıklı blok aralığını doğruladı. Dokuz ret testi: origin kontratı,
saat,fiyat ölçeği,hedef tarihi,eksik doldurma,origin silme,bozuk aralık,
değişmiş tahmin girdisi ve eski çıktı üzerine yazma. Ruff geçti.

İlk bağımsız doğrulama,T+1 mask blok20 paired farkının yüzde hesabında
−1,0712985827923478 ile−1,0712985827921568 farkında durdu. Fiyat
kayıplarının toplama sırası farkı yüzde normalizasyonuyla büyümüştü.
R1 kodu ve başarısızlık kaydı korundu. R2 aynı128×epsilon sınırıyla önce
bağımsız mutlak fiyat-kaybı biriminde denetler, yüzdeyi ters normalize
ederek de kontrol eder; fiyat/forecast kontrol sınırı değişmedi. **Analiz,
bootstrap,veri,rapor ve sayılar değiştirilmedi.** Bu birim düzeltmesi
olmaksızın ilk kayıt passed sayılmadı; arşiv iki verifier'ı içerir.

Sonraki **tek deney adayı**, aynı mature train satırları/özellikler/modelde
yalnız eğitim label'ını CT ile origin'de sabit kontrat arasında değiştiren
eşleştirilmiş T+5 kontrolüdür. Ortak named-label geçmişi2020'de başladığı
için önce tam500 train/3×63 bilinen inner origin kurulabilirliği ve aynı
engine desteği denetlenmeli; yanlışlıkla2010 proxy label'ı kullanılamaz.
Bir yıllık2023 tasarımı6/8 üretim kapısını karşılamaz. Bütçe/manifest ve
kontroller ayrı dondurulmadan yeni fit veya başka model/grid başlamaz.
Bu iki açıklamayı ayırmadan daha büyük model aramak bilimsel ilerleme değil.

[Ön kayıt](../research/evidence/origin-fixed-forecast-transport-preregistration-20261010.json),
[sonuç proof](../research/evidence/origin-fixed-forecast-transport-result-20261010.json),
[Release manifest](../research/evidence/origin-fixed-forecast-transport-result-release-20261010.json),
[payload](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/origin-fixed-forecast-transport-result-20261010).
