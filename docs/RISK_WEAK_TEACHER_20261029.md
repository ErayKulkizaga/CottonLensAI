# T+1 risk: zayıf ve gürültülü bilinen-sinyal kontrolü

Bu çalışma piyasa modeli başarısı değildir. Amaç, mevcut dört parametreli
EWMA-relative QLIKE HAR ve geçmiş doğrulama düzeninin küçük, bilinen bir
etkiyi öğrenip koruyabilmesini sınamaktır. Önceki risk sentetikleri deterministik
ortalama kurtarma; gürültülü unit test ise eğitim objective kontrolüdür.
Eski 132-fit fiyat/XGB çalışması farklı hedef, kayıp ve learner kullanır.

## Dondurulan tasarım

- Resmî AMS kaynağının aynı sekiz yılı, 174 işi, 2.006 ordinal origin ve
  1.989 scoreable etiketi; 500 minimum train, beş seans olgunluk, 21 refit,
  189/3×63 geçmiş validation ve aynı shrinkage ağırlıkları korunur.
- Model aynı dört parametreli QLIKE-HAR'dır. Sabit null/weak koşulları ve
  seed42/1701/2703: en çok **1.044 sentetik, sıfır piyasa fit'i**. Tek CPU
  süreç/en fazla iki thread, 30 dakikalık checkpoint oturumu; grid yoktur.
- İlk olgun train'in 619 origin'i (2012-08-31–2015-03-18, cutoff2015-03-26)
  `actual/EWMA` inovasyon havuzunu ve relative-HAR5 normalizasyonunu belirler.
  Gelecekteki etiketler bu kalibrasyona katılmaz. Havuz ortalaması1, standart
  sapması1,68895, maksimumu12,83097'dir. İnovasyonlar IID yeniden örneklenir.
- Null ortalama origin EWMA'sıdır. Weak ortalama
  `EWMA*exp(0.15*standardized(relativeHAR5) − past_log_normalizer)` olur.
  Beta0,15 ilk tasarımda sabitlendi; seed/beta araması yapılmaz. Ortalama
  mevcut model sınıfında tam temsil edilebilir. Latent mean/inovasyon sütunları
  model girdisinden çıkarılır. İki koşul aynı seed'in aynı inovasyonlarını kullanır.
- İlk sıfır-fit kontrolünde oracle beklenen QLIKE katkısı0,01043714 bulundu;
  bu fit sonucuna göre seçilmiş bir etki değildir. Null için beklenen katkı0'dır.
- Birincil kurtarma koşulu: üç weak seed'in her birinde ham beklenen katkı>0
  ve ortalama oracle katkısı kurtarma oranı≥%50. Bu yalnız kontrol ölçütüdür;
  eski fiyat/risk başarı kapılarının yerine geçmez. Ham, geçmişte seçilen,
  gerçekleşmiş ve bilinen ortalamaya göre beklenen katkılar ayrı raporlanır.

`E[loss(EWMA)−loss(p)] = log(EWMA/p)+mean/EWMA−mean/p` doğrudan hesaplanır.
Null'da bunun pozitif olması hesap hatasıdır. Oracle mean üst sınırı sağlar.
Yıl sınırlarını aşmayan 20/60 blok ve10.000/seed42 yordamı korunur; sentetik
aralıklar bağımsız piyasa doğrulaması veya kalibre edilmiş gerçek güç değildir.

Bu **sabit tasarımlı** kontroldür: gerçek geçmiş feature matrisini ve origin
EWMA state'ini tutar, yalnız label üretir. Sentetik getirilerden fiyat/HAR/EWMA
yeniden üreten bir piyasa simülasyonu değildir. IID inovasyonlar gerçek
serial bağımlılığı ve sonraki rejimlerin bütün kuyruklarını temsil etmez.
Sonuç bütün gerçek sinyallerin öğrenilebilirliğine genellenemez.

## Kimlik ve koruma

Son source:`dd95399ca9cd5b3c64ffcfc0c37b5ec9c648e612dbe53834fc85cb9c46232833`.
Design:`baec563d2d52ef143cc4ed619dd6493d4d5c0cc062a05bf3c1dbddbfb064d40b`.
[Son ön kayıt](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/risk-weak-teacher-prereg-r3-20261029)
304 dosyayı taze GitHub restore ve frozen-source `verify` ile eğitimden önce
doğruladı. ZIP SHA-256:
`956462e46d37697d9b5372eeea7ff527bf6038bf54a5975c58462926096d9542`.

V1 hazırlığı transient writer-lock dosyasını hash'e kattığı için doğrulamada
durdu; V2 ise mevcut indexer'ın sentetik role sınıflandırmasına uygun açık
prefix taşımadığı için fit öncesinde superseded oldu. İkisi de korunur ve
hiçbir fit/performance sonucu değildir. Regresyon testleri iki kusuru korur.
V3 açık immutable input listesi, `synthetic-` role ve canonical data/split
kimliği kullanır. 22 ilgili test ve Ruff geçti. Eski piyasa fiyatları,
modeller, kanıtlar ve kullanıcı tarafından kapatılmış collector korunur.

CLI: `python -B -m research.risk_weak_teacher prepare|verify|pilot|compare`.
Pilot yalnız source/ready/design/budget ve taze geri kurmaya bağlı authorization
ile çalışır. Başarısız native fit otomatik tekrarlanmaz. Raporlama fit çağırmaz.

## Sonuç: kontrol geçti, piyasa üstünlüğü gösterilmedi

1.044 sentetik fit tamamlandı; başarısız attempt0, yeni piyasa fit0.
Bilinen ortalamanın oracle katkısı her seed için0,01043714'tür.

| Seed | Ham beklenen katkı | Oracle kurtarma | Seçilmiş beklenen katkı | Ham gerçekleşen katkı |
|---|---:|---:|---:|---:|
| 42 | +0,00907474 | %86,95 | +0,00722190 | +0,01233628 |
| 1701 | +0,00802447 | %76,88 | +0,00650367 | +0,01587172 |
| 2703 | +0,00682121 | %65,36 | +0,00579039 | +0,01301907 |

Ortalama kurtarma%76,40; üç seed'in her birinde pozitif ham beklenen katkı
vardır. Beklenen pozitif yıllar hamda8/8, seçilmişte7/8,6/8,6/8'dir.
Null koşulunda her seed'in ham ve seçilmiş beklenen katkısı negatiftir.
Geçmiş shrinkage pozitif etkiyi azaltır; kontrolün tamamını sıfıra indirmez.

Seed1701'de ham gerçekleşen katkı pozitif olsa da CI20
[−0,00005753;+0,03264990] ve CI60[−0,00068626;+0,03219981] sıfırı içerir.
Bu senaryoda etki tasarımdan bilinmektedir: aralıkta sıfır bulunması, etkinin
olmadığını veya modelin öğrenemediğini göstermez. Buna karşılık diğer seed'ler
ve bu bağımsız inovasyon varsayımı, gerçek piyasanın güç garantisi değildir.

**VERIFIED:** mevcut risk learner/selection düzeni bu küçük, tam temsil
edilebilir etkiyi bu üç sabit tasarımda öğrenebiliyor. “Küçük bir ilişkiyi bile
öğrenemeyen bozuk model” genel açıklaması bu kapsamda desteklenmiyor.
**NOT SUPPORTED:** buradan gerçek girdilerde sinyal yokluğu, bütün nonlinear
temsillerin yetersizliği veya piyasa tahmin başarısı çıkarılamaz. Gerçek piyasa
çalışmalarının EWMA üstünlüğü hâlâ doğrulanmış değildir.

Bağımsız denetim1.044native model,71.604Decimal kayıp scalar'ı,72blok aralığı
ve48geçmiş ağırlık seçimini doğruladı. İlk bağımsız CSV okumasında literal
`null` koşulu pandas'ın varsayılan NA listesine takıldı; okuyucu artık yalnız
boş alanı missing kabul eder. Yeni regresyon testi var; CSV/model/fitler
değiştirilmedi veya yeniden çalıştırılmadı. 22ilişkili+3verifier testi ve Ruff
geçti. Sonuç modeli canlıya yayımlamaz; kapalı collector kapalı kalır.
