# Eski loss araması gerçekten neyi sınadı?

**VERIFIED:** eski XGBoost T+1 seçimi 6.648 arşiv-bağlı fit makbuzundan,
123 dosyalık özgün kaynak ZIP'inden ve aynı history/ready'den yeniden kuruldu.
Her dönemde 128 seed42 aday × 3 iç blok, top5 × 3 seed onayı ve 3 outer
makbuzun özgün log-getiri ortalaması kullanıldı. Yeni fit **0**; 1.008 ortak
origin (8 × 126), üç seed'in 3.024 satırı ek bağımsız gözlem değildir.
[Protokol](LEGACY_LOSS_LINEAGE_PROTOCOL_20261010.md) ve
[PR #26](https://github.com/ErayKulkizaga/CottonLensAI/pull/26) ve
[checksum bağlı kanıt Release'i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/legacy-loss-lineage-20261010).

| Dönem | Seçilmiş hedef / loss | Geçmiş onay skoru¹ | OOS Naive kazancı |
|---|---|---:|---:|
|2016|scaled_log / absoluteerror|0,998315|+%0,12159|
|2017|scaled_log / squarederror|0,995235|−%2,49983|
|2018|price_delta / absoluteerror|0,995237|−%0,71889|
|2019|scaled_log / absoluteerror|0,999303|−%0,18709|
|2020|price_delta / squarederror|0,996791|−%1,08635|
|2021|scaled_log / absoluteerror|0,995422|−%0,26618|
|2022|price_delta / absoluteerror|0,993685|−%0,58983|
|2023|price_delta / absoluteerror|0,999860|−%0,20877|

¹ Üç normalized geçmiş blok skoru, üç seed üzerinden onaylanır; 1'in altı
validation'da Naive'den iyidir. Bu skor OOS MAE kazancı değildir.

**VERIFIED:** toplam Naive kazancı **−%0,611017**, yön **%46,9246**, kazanılan
dönem **1/8**. Yıl-koruyan paired blok20/60 %95 kazanç aralıkları sırasıyla
**[−%1,10211; −%0,12640] / [−%1,00967; −%0,21633]**. Bu eski seçilmiş fiyat
programı için pratik %5 hedefi **DECISIVE NEGATIVE**; güncel 2.006-origin
full-year veya Texas/NASS sonucu değildir. İncelenmiş tarih bağımsız holdout
olarak sunulmaz; bütün tarihsel araştırma seçimlerinin belirsizliği kapsanmaz.

## Yanlış teşhisi önleyen iki ayrım

**VERIFIED:** altı dönemde absolute loss, bunların üçünde `price_delta`
seçildi. Frozen `models.py:184` loss'u XGBoost objective'e geçirir;
`protocol.py:132–154` fiyat değişimini train-only mean/scale ile dönüştürür
ve ters çevirir. Bu kolda mutlak standardized-delta hatası, fiyat-MAE'nin
pozitif sabit scale'e bölünmüş halidir; dönüşüm mean kullanıyor diye MAE'nin
koşullu medyan hedefi ortalamaya dönüşmez. Float32 label yuvarlaması ayrı,
küçük sayısal etkidir. **“Fiyat-MAE'ye uygun loss hiç denenmedi” yanlış.**

**STRONGLY SUPPORTED:** bütün başarısızlığı yalnız MSE/MAE farkına bağlamak
mevcut kanıtla uyumlu değil. Doğrudan fiyat-MAE loss'u olan seçilmiş üç dönem
de Naive'den kötü. 128-aday seçim, validation'da sekiz dönemin sekizinde küçük
pozitif sonuç bulurken OOS'ta yalnız birini kazandı; seçim iyimserliği gerçek.
Bu, tüm bilgi kaynaklarında sıfır sinyal veya bütün koşullu medyan modellerinde
başarısızlık kanıtı değildir; ham eski programda shrinkage uygulanmamıştı.

**VERIFIED:** hedef dönüşümü, loss ve rastgele hiperparametreler birlikte
değişiyordu. Aynı parametrelerle iki loss'un ortak-origin kontrolü yok.
Dolayısıyla **kontrollü loss etkisi INCONCLUSIVE**. Absolute-loss içeren
başarısız bir arama, loss değişiminin tek başına etkisini ölçmüş sayılmaz.
T+5 makbuzları bu T+1 sonucuna dahil edilmedi; eksik T+5 araması negatif
piyasa kanıtı yapılmadı. Genel bir MAE/XGBoost grid'i yenilik diye tekrarlanmaz.

## Kanıt sınırı ve bütünlük

Özgün source `73a92265cbdeb2ecae26f589193086fbfac1367673d8985ae02a97a12741b6a1`,
ZIP SHA `8e134e2b72ae3d677076216590dac2dc745c3c400a1ac84d4c051533015f7a30`.
6.651 kullanılan girdi, 5 Ekim kamu arşivi envanteriyle eşleşir. Eğitim
tarihleri, H5 olgunlaşması, ortak warmup ve frame identity ayrıca doğrulandı.
Eksik karar dosyaları tamamlanmış gibi uydurulmadı; seçim kuralları geçmiş
makbuzlardan yeniden kuruldu ve gerçek outer tarif/iteration'larla eşleşti.

Model/adapter/curve dosyaları bu eski kopyada yok; 24 outer fit'in 72 beklenen
payload hash'i kamu arşivinde de bulunmadı. **Model çıkarımı yeniden doğrulandı
denmez.** Tahmin/etiket/kayıp ve kaynak/selection doğrulaması daha güçlüdür;
öğrenilmiş durumu yeniden yükleme sınırı ayrı korunur. Yeni audit bu kaybı
gidermek için eski deneyleri tekrar eğitmez.

Buradaki üreteçte 19 parametrede son-bit farkı ortaya çıktı; örneğin aday39
eta kaydı `0.025598866990931057`, yerel yeniden hesap `0.02559886699093106`.
İlk okuyucunun bu yüzden durması eski modelde hata kanıtı değildir. Exact
kayıtlı değerler ve kimlikler korunur; yuvarlama/cache eşitlemesi yapılmaz.
Bu boyutta fark, −%0,611 performansı açıklamaz. 11 dar kontrol ve Ruff geçti;
Python `-O` ile bilimsel kontrollerin kapanması reddedilir.

## Bağımsız doğrulama ve restore

24 özgün outer makbuzun üç-seed ortalamaları, 1.008 origin/hedef eşitliği,
train etiket olgunluğu, fiyat-MAE/yön/dönem metrikleri ve iki paired aralık
ayrı okuyucuyla denetlenir. ZIP/member checksum ve güvenli yol kontrolünden
sonra arşivi yeni dizine açın; mevcut CPU bağımlılık ortamında:

```bash
python verify.py
```

Komut yeni model eğitmeden makbuz/metrik kontrollerini yapar ve frozen
kaynakla dört çıktı dosyasını geçici dizinde birebir yeniden üretir. Paket
yalnız kullanılan 6.651 özgün girdiyi içerir; eski tüm araştırmanın tam model
yedeği değildir. `source/` güncel okuyucu, `inputs/` özgün kayıtlar ve
`analysis/` yeniden kurma çıktılarıdır. Yeni ortam kurulması gerekmemiştir.

## Sonraki tek ayırıcı test

Loss sorusu öncelikli kalırsa en küçük kontrol, **8 dönem × 3 seed × 2 loss =
48 CPU XGBoost fit'i** olur. Eski seçimin dönem-öncesi parametreleri ve
iteration sayıları dondurulur; iki kol da aynı `price_delta` hedefi, 24
özellik, eğitim tarihleri ve 1.008 origin kullanır. Yalnız squarederror /
absoluteerror değişir; iki kol da aynı CPU sürümünde yeniden fit edilir,
eski GPU çıktısı kontrol olarak kullanılmaz. Yeni grid, early-stop/seçim,
shrinkage veya 2024+ kullanımı yoktur. Eski yarım-yıl seçimlerini Ocak origin'lerine
uygulamak leakage yaratır; bu yüzden mevcut 2.006-origin kohortuna taşınmaz.

Birincil paired fiyat-MAE katkısı ve Naive karşılaştırmasıdır; aynı %5/%53/6–8
kapıları korunur. MAE lehine katkı olup %5 hedefi dışlanıyorsa loss küçük bir
etkidir; ana hedef çözülmez. İki loss da pratik hedefi kaçırırsa bu loss
açıklaması büyütülmez. Bu yeni test henüz başlatılmadı; önce ortak tarif,
kohort, runtime ve sınırlı bütçe için ayrı ön kayıt gerekir.
