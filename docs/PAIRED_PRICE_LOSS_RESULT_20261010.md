# Sabit fiyat-delta loss karşılaştırması — sonuç

**VERIFIED:** aynı 24 özellik/price_delta/training kohortu/parametre/seed/ağaç
sayısında yalnız MSE–MAE değişti. 48 piyasa + ayrı 6 sentetik fit tamamlandı;
sonuç doğrulaması yeni fit **0**. 1.008 ortak T+1 origin, 8 × 126-origin dönem,
2.016 ensemble /6.048 seed satırı. Tam yıl veya bağımsız holdout değildir.
[Protokol](PAIRED_PRICE_LOSS_PROTOCOL_20261010.md), [ön kayıt PR #27](https://github.com/ErayKulkizaga/CottonLensAI/pull/27)
ve [sonuç Release'i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/paired-price-loss-result-20261010).

| Kol | Fiyat MAE, cent/lb | Naive kazancı | Yön | Dönem kazanımı |
|---|---:|---:|---:|---:|
| Naive | 0,99319444 | %0 | — | — |
| MSE | 1,01281107 | −%1,975104 | %46,9246 | 1/8 |
| MAE | 0,99804419 | −%0,488298 | %48,4127 | 3/8 |

İki modelde aktif oran %100; shrinkage veya sonradan OOS seçim yok.
**MAE–MSE katkısı +%1,458009**, MSE kontrol MAE'sine göre. Naive kazancındaki
1,486806 **yüzde puanlık** fark bu göreli katkıyla aynı sayı değildir.

| Paired %95 kazanç aralığı | Blok20 | Blok60 |
|---|---:|---:|
| MAE–MSE; kontrol MAE'sine göre | [+%0,636274; +%2,306992] | [+%0,991865; +%1,939195] |
| MAE–Naive; Naive MAE'sine göre | [−%0,975066; −%0,015690] | [−%0,850977; −%0,130343] |
| MSE–Naive; Naive MAE'sine göre | [−%2,930460; −%0,980510] | [−%2,612588; −%1,318557] |

Dönemler korunarak paired moving-block, recentered basic interval;
10.000 tekrar /seed42. Bu aralıklar eski tarif/ağaç seçiminin ve bütün araştırma
geçmişinin belirsizliğini kapsayan bağımsız doğrulama değildir.

MAE yıllık dönem kazançları 2016–2023 (%):
`+0,422077, −2,361570, −0,492681, −0,842011, +0,173589, +0,036235, −0,776554, −0,161879`.
En büyük 11 (%1 yuvarlanmış) Naive hatası çıkarılınca MSE/MAE kazançları
**−%2,033698 /−%0,466780**. Başarısızlık yalnız birkaç büyük gözleme bağlı değil.

## Bilimsel karar ve teşhis

Kilitli karar: **LIMITED_LOSS_CONTRIBUTION_PRACTICAL_GOAL_UNMET**.
Birincil loss katkısı sorusu **DECISIVE POSITIVE, bu sabit tariflerle sınırlı**.
%5 Naive kazancı /%53 yön /6/8 dönem pratik sorusu **DECISIVE NEGATIVE**.
Olumlu loss katkısı, başarılı tahmin modeli veya yayın izni anlamına gelmez.
Naive korunur; bu sonucu daha büyük MAE grid'ine dönüştürmeyin.

**STRONGLY SUPPORTED:** loss seçimi fazla hatanın bir bölümünü yaratıyor,
ancak tekrarlanan Naive başarısızlığını tek başına açıklamıyor. Eski aramaların
karma loss/hedef seçimi bu soruyu ayıramıyordu; bu kontrollü sonuç ayırdı.
MAE kolunun ham çıktısı da kötü olduğundan burada shrinkage'in sinyali
sıfırlaması açıklaması geçerli değil. Eğitimden hesaplanan dönüşümler ve
kaydedilmiş çıkarımlar eşleşti; bir fiyat dönüşümü/ortak-origin hatası bulunmadı.

**NOT SUPPORTED:** bütün Cotton bilgisinde sinyal yokluğu, bütün nonlinear/MAE
modellerinin başarısızlığı veya daha büyük modelin çözüm olacağı. Eski parametre
ve iteration'lar farklı hedef/loss'lar altında seçilmişti; bazı ağaç sayıları
çok küçük. Sonuç bunlara koşullu. T+5, full-year 2.006-origin Ridge, tarihsel
PIT kabulü veya gerçek kontratta ileri beceri sınaması değildir.

## Doğrulama ve yeniden üretim

Ön kayıt/veri/kod/ortam, 24 eski olgun training kohortu ve 48 yeni makbuz
kimliği; train-only imputer/scaler/target; H1/H5 olgunlaşması, kayıtlı 54
XGBoost modelinin native çıkarımı ve altı loss-kontrol eşiği doğrulandı.
16 üç-seed ensemble, 1.008 ortak origin/hedef, fiyat-MAE/yön/dönem/uç-hata
hesapları ve altı aralık bağımsız hesaplandı. Frozen engine ile geçici dizinde
rapor ve iki CSV birebir yeniden üretildi; fitting çağrıları yasaklandı.
İlk çalışma ağacındaki 830 deney dosyası değişmedi. Arşiv yalnız yetkili
completed/payload/receipt çıktıları tutar; tekrarlı local-work kopyaları dahil değil.

Eğitim kaynak commit'i `c1a962b714fecca83a21e52667f573a8b8221236`;
ML source `d3ca00e498639dc266f34dafb60f2f24dbba5e66ee0deeda0d9c4c0e5b7f7313`;
identity `cd9512fc07b0011d427265e0d75b85381e46a08c1eac96774606d4dba2a8c953`.
Ön kayıt ZIP'i ve eski sicil/kanıtlar değiştirilmedi; tamamlanmış piyasa ve
sentetik fit makbuzları ayrı türlerle ek indekslenir.

ZIP/member checksum kontrolünden sonra mevcut ayrı CPU ortamıyla:

```bash
python verify.py
```

Komut veri indirmez veya eğitim yapmaz. Önceki cache veya çalışma ağacı
gerektirmeden arşivdeki kaynak, girdiler, modeller, aralıklar ve raporu denetler.
Yeni sonuç kaydı/release teslimat makbuzu bu belgeye bağlıdır; tamamlanmış
fit'ler yeniden çalıştırılmaz, eski ön kayıt sonuçla değiştirilmez.

**Sonraki bilimsel karar:** yeni model açmadan, mevcut sicilde tam kapsamı
kontrol edilerek en fazla 2–3 bilgi sorusunun sıfır-fit kabul elemesini yapmak.
Kullanılabilir bilgi azlığı ile temsil/model çıkarım sınırı hâlâ ayrışmadı;
MAE veya genel kalibrasyon aramasını tekrarlamak bu ayrımı çözmez.
