# Fiyat-MAE çarpan kalibrasyonu — sonuç

**VERIFIED:** sabit Texas/NASS `numeric_D0` Ridge tahminlerine yalnız olgun
model-eğitim etiketlerinden 169 skaler düzeltme uygulandı. Yeni model fit'i **0**;
aynı 2.006 T+1 origin / 2016–2023, 37 özellik, aynı model/scaler/etiketler.
[Protokol](PRICE_MAE_CALIBRATION_PROTOCOL_20261010.md),
[PR #25](https://github.com/ErayKulkizaga/CottonLensAI/pull/25) ve
[yeniden üretilebilir kanıt Release'i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/price-mae-calibration-20261010).

| Kol | Fiyat MAE | Naive kazancı | Kazanılan yıl | Aktif tahmin oranı | Aktif yön |
|---|---:|---:|---:|---:|---:|
| Naive | 1,02095224 | %0 | — | %0 | — |
| Eski ham | 1,05929665 | −%3,75575 | 0/8 | %100 | %49,8504 |
| Kalibre ham | 1,05910367 | −%3,73685 | 0/8 | %100 | %49,7507 |
| Eski seçilmiş | 1,02123243 | −%0,02744 | 1/8 | %37,3878 | %48,9333 |
| Kalibre seçilmiş | 1,02121852 | −%0,02608 | 1/8 | %37,3878 | %48,5333 |

Kalibre seçilmiş tüm-origin yön **%18,3948**; sıfır tahminler nedeniyle bu
aktif yönle aynı ölçü değildir. Kalibrasyon hiçbir yılın geçmişte seçilen
ağırlığını değiştirmedi: 2016–2019 ve 2021 sıfır; 2020/2022/2023 0,25.

Birincil seçilmiş adayın eski kontrole katkısı yalnız **%0,001362**.
Yıllar içinde paired moving-block 10.000 tekrar / seed42 aralıkları:

| Karşılaştırma / %95 aralık | Blok20 | Blok60 |
|---|---:|---:|
| Kalibre seçilmiş–eski seçilmiş, kontrol MAE'sinin yüzdesi | [−%0,008165; +%0,010593] | [−%0,007880; +%0,010457] |
| Kalibre seçilmiş–Naive, Naive MAE'sinin yüzdesi | [−%0,297280; +%0,231340] | [−%0,236592; +%0,185847] |

Kalibre seçilmiş yıllık Naive kazançları 2016–2023:
`0, 0, 0, 0, −0,680489, 0, +0,203915, −0,142772` (%).
En büyük 21 Naive hatası çıkarılınca kazanç **−%0,05687**; ham kalibre
sonuç **−%4,13670**. Negatif sonuç yalnız birkaç büyük hataya bağlı değil.

## Mekanizma ve karar

**VERIFIED:** exact eğitim-fiyat-MAE minimizer'ının getiri düzeltmesi
`log(f)` aralığı [−0,00048081; +0,00013821]; mutlak düzeltme medyanı
0,00016370 (yaklaşık 1,64 baz puan). Eğitim MAE'sindeki düşüş medyanı
0,00006902 fiyat birimi. Eğitim kaybının iyileşmesi OOS üstünlük sağlamadı.

**STRONGLY SUPPORTED:** bu sabit tahminlerde global çarpan yanlılığı,
tekrarlanan başarısızlığın ana açıklaması değildir. Ham tahmin 8/8 yılda
Naive'den kötü kalıyor; küçültme zararı azaltıyor. Sıfır ağırlığın bütün
kaynakta sıfır sinyal anlamına geldiği veya küçültmeyi kaldırmanın sorunu
çözeceği çıkarılmaz.

Kilitli karar **CALIBRATION_DOES_NOT_RESCUE_FIXED_RECIPE**: kontrol karşısında
pozitif katkı gösterilemedi ve iki blokta Naive kazancının üst sınırı %5'in
çok altında. **Naive korunur; bu global kalibrasyon tarifine yeni grid veya
model fit'i eklenmez.** %5 MAE / %53 yön / 6/8 yıl kapıları değiştirilmedi.

**NOT SUPPORTED:** bu negatiften tüm koşullu medyan modellerinin başarısız
olacağı, bütün bölgesel kaynakların yararsızlığı veya tarihsel erişim/vintage
onayı çıkarılamaz. Faktör, eğitim artığından hesaplanan tek global parametredir;
cross-fit veya tam koşullu quantile model değildir. Tekrar kullanılmış yıllar
bağımsız holdout değil; aralıklar bütün araştırma seçimlerini kapsamıyor.
Bu sonuç T+5 deneyi değildir. Üretim yayını ve tarihsel kaynak kabulü kapalıdır.

## Doğrulama ve yeniden üretim

169 düzeltme bağımsız konveks subgradient optimum sertifikasıyla; eğitim
kohortları/olgunlaşma, 169 kayıtlı model çıkarımı, 8 geçmiş seçim, 2.006
origin/hedef/fiyat eşitliği, yıllık/ham/seçilmiş/uç-hata metrikleri ve 12 paired
aralık ayrıca doğrulandı. Frozen kaynakla geçici dizinde rapor ve CSV birebir
yeniden üretildi. İlk bağımsız okuyucu float32 getiriyi `exp` öncesi float64'e
çevirmediği için kontrol durdu; okuyucu düzeltildi. Deney kaynakları, faktörler
ve tahminler değiştirilmedi; bu piyasa pipeline'ında hata bulgusu değildir.

Kaynak commit `66f9fa6685518681a5d276d6777505e3baeda47b`; ML source
`b20fec5aae9c3bab3b0112679c32c1cb4bde588577cb43608cd42cb53e4438a0`.
Kullanılan 694 girdi özgün NASS Release envanteriyle eşleşir; eski arşivin
3.945 üyesi başlangıçta doğrulandı. Dokuz dar test ve aynı kaynak commit'inin
push/PR CI'sı geçti. Yeni model makbuzu oluşturulmaz; skaler güncellemeler
ayrı envanterlenir. Eski sicil/trials/kanıtlar korunur.

Release ZIP'i mevcut CPU bağımlılık ortamında ayrı dizine açın:

```bash
python verify.py
```

Komut model eğitmeden bütün skaler/çıkarım/metrik kontrollerini ve rapor
replay'ini yapar. ZIP/member checksum doğrulaması önce tamamlanmalıdır.

**Tek sonraki inceleme:** yeni eğitim önermek yerine, mevcut MAE/absolute-loss
tariflerinin tamamlanmış ortak-origin çıktıları gerçekten var mı ve MSE
kontrolüyle hangi loss hipotezini sınamışlar, sicil ve payload'lardan belirle.
Koşullu temsil/loss etkisi ile zayıf piyasa bilgisi hâlâ ayrılmış değildir;
eski kapsam doğrulanmadan yeni quantile/absolute-loss deneyi başlatılmaz.
