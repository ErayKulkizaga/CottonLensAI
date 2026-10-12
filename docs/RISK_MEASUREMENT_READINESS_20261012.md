# Risk ölçümü ve doğrulama kurulabilirliği

## Karar

**Mevcut 126-origin akışını bu küçük katkının doğrulama deneyi diye uzatmak için NO GO.**
Risk modelinin kaynak/hedef uyumunu araştırmak için **PIVOT**: aynı kontratın
günlük getiri riskini açık tanımla. Bu rapor yeni eğitim, dataset indirme veya
gerçek zamanlı yayın başlatmaz. Eski fiyat/risk hedefleri ve eşikler değişmez.

**VERIFIED:** mevcut veride varsayımlı, aynı-kontrat günlük kare-getiri etiketi
kurulabilir. Sekiz-yıl eğitim/değerlendirme protokolü kurulamaz; mevcut500train/
189validation şartını yalnız2023 karşılıyor. **NOT SUPPORTED:** gün içi veri
olmadığı için risk tahmininin imkânsız olduğu. Günlük kare getiri gürültülü bir
risk vekili olarak kalır; intraday realized variance ölçümü diye adlandırılmaz.

## Tamamlanmış deneyleri tekrar etmeyen soru

[PR50](https://github.com/ErayKulkizaga/CottonLensAI/pull/50) 482 eski HAR/GARCH
makbuzunu doğruladı. Bu inceleme aynı model/EWMA varyans tahminlerini koruyup
**yalnız gerçekleşen hedefin** CT veya aynı kontrat olması farkını ölçer.
Eski fiyat etiket/transport kontrolleri risk hedefini sınamış sayılmaz.
`python ml/history.py check --family har --horizon 1 --query named` kayıtlı eşdeğer
bulmadı; bu tek başına yenilik kanıtı değildir, ilgili eski fiyat kontrolleri
ayrıca incelendi. Yeni fit0;2016–2023 yeniden görülmüş araştırmadır.

## VERIFIED — fiyat alanları ve kontrat değişimi

2.006 T+1 origin'in her biri korunur. Sonraki kayıtlı open ile önceki close
arasındaki log değişim `g`, aynı bar open–close değişimi `u` ise:

`r² = g² + u² + 2gu`.

Toplamlar: g²0,16374719; u²0,48629849; çapraz terim−0,02617226;
r²0,62387343. Çapraz terim atılamaz. Bunlar fiyat-alanı kimliğidir;
işlem yapılabilir overnight getiri veya bağımsız varyans bileşenleri değildir.
1.738 hedef bar OHLC aralığına uyar;268 uymaz. Settlement/close ve high/low
anlamları doğrulanmadan bu268bar bozuk fiyat diye silinmez.

| Kaynakta gözlenen ön-vade durumu | Origin | CT model–EWMA QLIKE farkı toplamı |
|---|---:|---:|
|İsim değişmiş|20|+6,87136229|
|İsim değişmemiş|1.010|−1,84388190|
|İki tablodan biri yok; bilinmiyor|976|+15,90852250|

20 geçiş toplam pozitif CT katkısının%32,82'sini taşıyor. Bunlar resmi ICE
roll tarihi değildir; iki endpoint'te ilk kaynak fiyatı CT float32 fiyatıyla
eşleşir. Geçişte toplam mixed-front r²0,01907843, aynı gelecekteki kontratın
hareketi r²0,00908422; eğri gap²0,01536050 ve çapraz terim−0,00536630.
Geleceğin kontrat adı yalnız ex-post kimlik içindir; tahmine/feature'a girmez.

**Kanıtın sınırı:**976 bilinmeyen origin'in kontrat geçişi yokmuş gibi
sınıflandırılması yanlış olur. Bu örnek mevcut CT hedefinin bazı vade
farklarını risk diye ölçtüğünü gösterir; bütün başarısızlığı roll'e yüklemez.

## Aynı-kontrat hedefi eski tahmini kurtarıyor mu?

1.049 mevcut kaynak tablosu kullanıldı;52 eski tablo yeni indirilmedi.
Origin'in tam referans günündeki birinci/ikinci kontrat **ismi** hedefe kadar
sabit. Eksik origin/target fiyatı taşınmaz veya doldurulmaz. Kaynak takvimi
origin'den geçse label uygunsuz kalır. İlk/ikinci kolun bütün2.006 makbuzu
ve eksiklik nedenleri korunur. Birincil hedef ikinci kontrattır; iyi çıkan
kol sonradan seçilmez. T+5 bu analizde skorlanmadı.

| Hedef | Uygun origin | Aynı tarihlerde CT hedefi delta | Named hedef delta | CI20 | CI60 |
|---|---:|---:|---:|---|---|
|Birinci sabit kontrat|997|−0,00257063|−0,00244795|[-0,03113558;0,03063818]|[-0,02628914;0,02307741]|
|İkinci sabit kontrat|1.017|+0,00423642|+0,00704728|[-0,01411564;0,02634573]|[-0,01180252;0,02504135]|

Delta `QLIKE(EWMA)-QLIKE(model)`, yüzde değildir. Bootstrap10.000/seed42,
blok20/60, bütün2.006 tarih slotu/yıl sınırı korunur; unknown kayıtta
bookkeeping weight0, zaman sıkıştırılmaz. Eski nonnegative-loss helper'ına
iki kayıp aynı sabitle kaydırılır; paired fark/CI değişmez. Kaydırılmış
kayıpların yüzdesi yayımlanmaz.

**INCONCLUSIVE:** hedef değişimi pozitif ve sağlam risk üstünlüğü üretmedi.
Bu karşılaştırmada model ve EWMA hâlâ CT ile kurulmuştur; aynı-kontrat
etiketiyle eğitilmiş öğrenicinin veya doğru named EWMA'nın testi değildir.
Ölçüm düzeltmesinin tek başına eski modeli kurtardığı gösterilmedi.

## Veri miktarı ve güç sınırı

İkinci kontratta3×63 geçmiş validation'ın en erken train sayıları:
2021 [80,143,206],2022 [327,390,453],2023 [576,639,702]. Yalnız2023
500 şartını geçer.2018/2019'daki7/40 dış label iki tam yıl sayılmaz.
UTC yayın/ilk vintage doğrulanmış kaynak satırı0; conditional label,
historically-admitted veri değildir. Günlük fiyat tablosu intraday veri değildir.

Eski T+1 katkısı0,01043669 ve gözlenen hata ölçeği sabit tutulursa kaba
sqrt(N) hesabı blok20/60 için15.977/10.886 origin ölçeği verir.126 origin'de
koşullu80% MDE0,11752/0,09701 QLIKE birimidir. **Bunlar kalibre edilmiş güç,
zorunlu süre veya gelecekteki başarı garantisi değildir.** Etki gerçekten
sıfırsa daha çok gözlem üstünlük yaratmaz; rejim/noise değişirse hesap değişir.
Bu sonuçtan yeni başarı eşiği veya yüzde-gain çıkarılmaz.

## Tek sonraki bilimsel iş

Geniş model/grid veya aylık bekleyiş yerine, **sabit Log-HAR için eşleştirilmiş
CT-label/named-label kontrolünün yalnız ön kayıt kurulabilirliğini hazırlamak**.
Mevcut koşullu2023 verisi mekanizma sorusunu sınamaya adaydır; iki kolda
aynı train/validation/origin ve CT geçmiş özellikleri tutulur, yalnız risk
etiketi değişir. Aynı hedefe ait, geçmişten kurulmuş named EWMA referansı
olmadan named skill iddiası yoktur. GARCH future-variance target'ını doğrudan
kullanmadığı için etiket kontrolünde GARCH'ı değiştirmiş gibi gösterme.

Bu test henüz kodlanmış, ön kayıtlı veya eğitim onaylı değildir. Son kod/
veri/availability/baseline kimliği ve tam fit sayısı dondurulmadan fit kapalı.
Olumlu mekanizma bile tek incelenmiş yıl ile6/8 veya bağımsız ileri başarı
değildir. Formal doğrulama için eksik named-contract fiyat/erişim geçmişi
ayrıca gerekir; intraday satın almak otomatik ön koşul değildir.

## Doğrulama ve koruma

10 yeni synthetic araştırma testi+11 mevcut araştırma testi/Ruff geçti.
Bağımsız Decimal2.006 component,4.012 label makbuzu,20 geçiş kimliği,
dört full-ordinal weighted CI ve sekiz yıllık training-feasibility kontrolü
geçti. Dokuz girdi hash'i değişmedi. Old risk native/öğrenme kontrolleri,
ML source,530 fit tarifi ve eski kayıtlar yeniden çalıştırılmaz/değiştirilmez.
Yeni raw download, fit, Drive işlemi veya live model yayını yoktur.

[Dondurulmuş sonuç](../research/evidence/risk-measurement-readiness-20261012.json)
ve [teslimat makbuzu](../research/evidence/risk-measurement-readiness-delivery-20261012.json)
sicile eklenmiştir. [Private Release](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/risk-measurement-readiness-20261012)
28 dosya/0 dışlamayı taze indirme ve native restore ile doğruladı. Rapor,
iki CSV ve bağımsız doğrulama makbuzu byte-identical yeniden üretildi.
Eski 185 kayıt/165 kanıt hash'i, 530 fit tarifi, 29.916 tamamlanmış fit
kopyası ve ML kaynak kimliği korunur.
