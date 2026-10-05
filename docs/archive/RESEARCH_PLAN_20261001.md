# CottonLens: sonuçlara dayalı araştırma planı — 1 Ekim 2026

Durum: uygulanacak plan; bu belge yeni kodun, verinin veya modelin hazır olduğunu göstermez. Önceki genişleme planlarının uygulama sırasını günceller. Eski sonuçlar, eşikler, kaynak paketleri ve veri sözleşmeleri korunur. Gerçek eğitim yalnız Colab'da yapılır.

Uygulama kaydı: salt okunur tahmin/kaynak incelemesi ve mevcut motora bağlı, süre sınırı
olan T+5 pilotu eklendi. İlk inceleme gerçek kayıtlarla çalıştırıldı. Harici veri kabulü,
yeni gerçek pilot eğitimi ve sonraki model/release aşamaları henüz tamamlanmadı.

## 1. Karar ve mevcut kanıt

Öncelik, T+5 için yeni bilgi içeren küçük karşılaştırmalarla fiyat hatasını azaltmak; T+1'i daha düşük bütçeyle izlemektir. Başlangıç araştırma bütçesinin yaklaşık %75'i T+5'e ayrılır. Bu, hedef veya kabul eşiği değişikliği değildir. Daha fazla hesaplamanın başarı garantisi olduğu varsayılmaz; her tur tamamlanmış deney, karşılaştırma ve sonraki karar üretir.

Doğrulanmış özet: `output/reports/research-market-v1-ablation-review-20261001.json`. Kaynak kimliği `9e071564c70150ac1f25e6e506daac768ae58ad556ddca38c93c4b16cf73e0c4`; rapor SHA-256 `3000f31f6285e6ff161b2cdb3dd5ddd5aac43ff8209e404d3fa81ff54e929b15`.

- 1.008 dış origin, 128 dış değerlendirme birimi ve 20.360 tamamlanmış alt eğitim kaydı var. Kayıtlı hesaplama süresi 44.085 saniye, yaklaşık 12,25 saat; bu ölçüm GPU'nun aktif kaldığı süre veya kullanıcının beklediği toplam süre değildir.
- T+1'de sekiz feature/aile birleşiminin tamamı fiyat-MAE bakımından Naive'ın gerisinde.
- T+5'te dış sonuçta en iyi görünen `expanded_availability / XGBoost`: %0,763 MAE iyileşmesi, %57,54 yön doğruluğu ve 5/8 blok kazanımı. MAE kazancının %95 aralığı yaklaşık −%1,416 ile +%2,184. Fiyat başarı gate'i geçilmedi; bu model doğrulanmış kazanan veya deployment adayı ilan edilmez.
- Dış sonuçlar daha önce görüldü. Araştırma teşhisinde kullanılabilir; bağımsız holdout değildir. Rapor kayıtları doğrulandı, bütün model payload'ları doğrulanmış değildir.
- Yeni harici veri grupları henüz katı araştırma protokolüne kabul edilmedi. AMS, FAS ve diğer kaynakların indirilmiş olması, eğitime hazır oldukları anlamına gelmiyor.

Hedef ölçekleme, squared/absolute loss, geniş XGBoost/CatBoost ayarları ve üç seed mevcut kodda var. Bunlar yeni katkı gibi yeniden sunulmayacak. Düşük düzenlileştirme hipotezi veya sentetik öğrenme kontrolleri, ilgili kod/veri değişmedikçe tekrar çalıştırılmayacak. Eski A/ablation turu yeniden açılmayacak.

## 2. İlk teslim: kayıp nedenini ve veri durumunu görünür yap

Tahmini efor: 0,5–1 iş günü. Yeni gerçek fit yok.

Mevcut tahmin ve küçük karar kayıtlarından horizon/yıl bazında şu rapor çıkarılır: Naive ve train median-return MAE; geçmiş çoğunluk-yön doğruluğu; model tahminlerinin yayılımı, flat/yukarı/aşağı oranı; büyük hareketlerde ve sakin dönemlerde hata; fiyat seviyesi, eksik veri ve model yanlılığıyla ilişki. Eldeki eğitim/validation eğrileri kullanılır, eksik eğri uydurulmaz. Belirli dönemlerin çıkarılması ana skoru değiştiremez.

Etiket hizası, hedef olgunlaşması, fiyat-getiri dönüşümü ve CT=F proxy'sindeki şüpheli sıçramalar kontrol edilir. Bağımsız kontrat verisi olmadan bir sıçrama kesin olarak roll hatası diye adlandırılmaz ve keyfî düzeltilmez. Doğrulanmış bir hata varsa etkilenen deneyler geçersiz işaretlenir; yeni arama önce bu hata düzeltilene kadar durur.

Tek kaynak tablosu: gerçek satır sayısı, tarih kapsamı, birim, rapor dönemi, yayın kanıtı, revizyon durumu, ortak origin sayısı ve kullanılabilir fold sayısı. FAS'taki iki raporun sayısal eşleşmesi bütün yılların doğrulandığı anlamına gelmez. Bu tablo veri işinin kalanını somutlaştırır.

Entegrasyon: mevcut `research/engine.py`, `research/ledger.py`, `research/diagnostics.py` ve `sources/` sınırları kullanılır. Yeni bir paralel pipeline kurulmaz. Tam rapor için 20 bin model dosyası yeniden taranmaz.

## 3. Veri işini sınırlı paketlere böl

İlk paket yalnız AMS pamuk spot ve FAS Export Sales'tir. İkisi de hazır değilse tüm veri ekosistemini bitirmeyi beklemek yerine doğrulanabilen altküme dondurulur. Bir kaynağın yayın kanıtını araştırmaya başlangıçta en fazla yarım iş günü ayrılır; sonuç yoksa somut eksiklik kaydıyla beklemeye alınır. Kaynağın hazır sayılması için deadline yeterli değildir.

### Kabul ve zamanlama

Bir kaydın belirli sürümünün tahmin anına kadar kamuya açık olduğuna dair kanıt gerekir. İlk yayın saniyesini bulmak her durumda şart değildir: doğrulanmış sürüm ve yayın tarihi/zaman dilimiyle kurulmuş muhafazakâr bir erişilebilirlik üst sınırı da kullanılabilir. Yalnız genel takvim, embargo, dosya adı, bugünkü indirme zamanı veya keyfî gecikme geçmiş erişilebilirliği kanıtlamaz. Revize edilmiş güncel seri geçmiş ilk yayınmış gibi kullanılamaz.

Mevcut katı kural korunur: kanıtı eksik kayıt model eğitimine, seçime ve gate hesabına girmez. Bu kayıtlar içerik, kapsam ve parser testlerinde kullanılabilir. Geçmiş sürümü belirsiz verilerle ayrı keşif eğitimi yapılması, ileride açıkça tartışılabilecek bir protokol değişikliğidir; bu plan bunu sessizce etkinleştirmez.

### İlk feature grupları

| Grup | İçerik | Kritik kontrol |
|---|---|---|
| AMS | Spot seviyesi, geçmiş değişimler, vadeli proxy ile birim uyumlu fiyat farkı, veri yaşı/eksiklik | Spot kalite/teslim kapsamı farklıdır; fark risksiz arbitraj spread'i diye sunulmaz |
| FAS | Net satış, sevkiyat, iptal, outstanding sales; tanımlar tutarlıysa ülke bileşimi | Dönem sonu ve yayın günü ayrılır; marketing year, running bale ve revizyonlar korunur |
| Mevcut piyasa | Önceki paketler ve mevcut eksiklik göstergeleri | Hazır ve tamamlanmış deneyler tekrar üretilmez |

FAS mevsimsel normalizasyonu yalnız geçmiş training verisinden öğrenilir. Beklenti verisi olmadan rapor değişimi “beklenti sürprizi” değildir. Düşük frekanslı değerler yalnız bilindikleri andan sonra taşınır; yaşları ayrıca kaydedilir.

AMS'nin 2020–2023 kapsamı tam sekiz dış bloğu desteklemeyebilir. Her kaynak için training süresi ve üç iç blok gereksinimi uygulanarak uygun dış bloklar fit'ten önce hesaplanır. Kısa cohort ayrı kimlik ve ayrı Naive karşılaştırması taşır; 6/8 gate'ini geçmiş gibi raporlanamaz. Kaynak eklendiğinde açılan/kapanan tarihler ana sıralamaya sessizce karıştırılmaz.

Sonraki veri sırası: erişim ve kullanım koşulları uygun mısır/soya ile resmî FX; ardından doğrulanabilir WASDE/NASS sürümleri; iklim, uydu ve metin bundan sonra. Hepsi ilk teslimin bağımlılığı değildir. NASA/uydu yeniden işlenmiş geçmiş ürünleri için sürüm sorunu ayrıca çözülür. Ücretli kaynak alınmaz.

## 4. İlk yeni bilgi deneyi: küçük, eşit, sonuçlandırılabilir

Yalnız veri kabulünü geçen paketler için aynı origin'lerde `base`, `base + AMS`, `base + FAS` karşılaştırılır. İlk adımda birleşik paket ve bütün çapraz kombinasyonlar açılmaz.

- XGBoost ve CatBoost; aile başına önceden sabitlenmiş dört tarif; seed 42.
- İlk pilot T+5; T+1 aynı çerçevede daha küçük takip bütçesiyle çalışır.
- Her uygun dış blokta üç geçmiş iç validation bloğu, geçmişte early stopping ve kilitli iterasyonla refit korunur. Kaynak tarih aralığı uymuyorsa blok uydurulmaz.
- Üç paket, iki aile, iki horizon ve en fazla dört dış blok için çekirdek bütçe: `3 × 2 × 2 × 4 × 4 × 3 × 2 = 1.152` alt fit. T+5 tek başına en fazla 576. Baseline, dış refit ve olası ek doğrulama işleri çalıştırmadan önce ayrı sayılır. Bu sayılar GPU süresi değildir.
- İlk tam adaydan fit, veri hazırlama, metric ve Drive süreleri ölçülür. Colab için önce 60–120 dakikalık çalışma dilimi kullanılır; yeni iş açmadan süre sınırı denetlenir. Devam eden tek fit sınırı aşabilir. Tamamlanmış işler korunur; kesilen son fit yeniden gerekebilir.

Önceliklendirme: aynı geçmiş iç bloklarda en az %0,5 göreli MAE iyileşmesi ve karşılaştırılabilir blokların çoğunluğunda katkı. Sekiz blok varsa en az beşinin iç sonuçlarında katkı aranır. Kısa cohort sonuçları yalnız ön keşiftir. Grup seçimi her dış fold için yalnız o fold öncesindeki iç sonuçlardan yapılır; bütün dış sonuçlara bakıp geçmiş fold'ların feature seçimleri değiştirilmez.

Pozitif grubun devam testi birleşik paket ve grubu çıkarma ablation'ıdır. Geçerli yayın zamanına ek 1/5/10 Cotton gözlemi gecikme hassasiyeti önceden tanımlanır; en iyi gecikme seçilmez. Bu stres testi eksik yayın veya revizyon kanıtının yerini tutmaz. Katkı yalnız veri yaşı/eksiklikten geliyorsa rapor bunu açıkça söyler.

## 5. Model araştırmasını bilgiye göre büyüt

### Tabular ve Naive'a kontrollü yaklaşma

İlk öncelik mevcut XGBoost/CatBoost'tur. Orta büyüklükte tabular verilerde ağaçların güçlü olduğuna ilişkin araştırma bu tercihi destekler; Cotton başarısını kanıtlamaz [1].

Katkı gösteren veri üzerinde 32 aday, ilerleme varsa 128 aday açılır. Mevcut loss/hedef dönüşümlerinin sonuçları önce incelenir. Finalist en fazla beş tarif seed 17/42/101 ile doğrulanır. Aday seçimi iç ortalama göreli fiyat-MAE, eşitlikte düşük karmaşıklıktır. Her dış fold için geçmişe bağlı seçim korunur.

Yeni düşük maliyetli hipotez: getiriyi `w × model_getirisi` olarak Naive'a yaklaştırmak. `w ∈ {0, 0.25, 0.5, 0.75, 1}` yalnız geçmiş iç out-of-fold tahminlerden seçilir. Bu, gürültülü aşırı tahminin zararını sınırlayabilir; tek başına %5 kazanç yaratacağı varsayılmaz. Final modelin fiyat yönü ayrıca yeniden hesaplanır.

Optuna mevcut motora bağlı kalır. İlk iki iç validation sonucu oluşmadan blok performansıyla pruning yapılmaz. Pruned/yarım deneme aday olamaz. Paralel Optuna yürütmesinin arama sırasını birebir üretmekle kilitli tarifi yeniden üretmek ayrıdır; bütün parametreler ve kararlar kaydedilir [5].

### Sequence: bir kontrollü rakip, sonra genişleme

Mevcut MLP/LSTM/TCN kodunun gerçek Colab sonuçları önce doğrulanır. Ek ucuz kontrol DLinear; harici feature'lar katkı gösterirse tek yeni büyük aday TimeXer'dır. TimeXer exogenous değişkenler için tasarlanmıştır [2]; DLinear araştırması da karmaşıklığın mutlaka daha iyi sonuç vermediğini hatırlatır [3]. Bu makalelerin sonuçları günlük Cotton T+1/T+5 için başarı kanıtı değildir. TFT ve TimeXer aynı anda yeni bağımlılık yükü yaratmaz; bu turda TimeXer önceliklidir.

Aile başına önce 8 tarif, sonra katkı veren en fazla iki ailede 24, gerekçesi varsa 48. Pencere 20/60/120, genişlik 32/64/128 ve mevcut learning-rate/dropout aralıkları korunur; en fazla 300 epoch ve 25 patience. Eksik feature satırı atılıp sequence zamanı sıkıştırılmaz. Yeni ailede environment/export smoke testi geniş aramadan önce yapılır. Modern pretrained time-series modelleri eğitim tarihleri doğrulanmadan tarihsel kanıta alınmaz.

### Sonraki kontrollü iyileştirmeler

Güçlü adayda history 3/5 yıl/tam; cadence önce 21/126, katkı gerekçesi varsa 5/63. Seçilmiş tek model ailesinde ölçülür; bütün parametrelerle çaprazlanmaz. Ayrı yön sınıflandırıcısı fiyat-MAE başarısı yerine geçmez. En fazla üç modelin negatif olmayan ensemble ağırlıkları geçmiş OOF tahminlerden öğrenilir; Naive da referans bileşen olabilir.

%80/%90 aralıklar geçmişte olgunlaşmış tahmin hatalarıyla kalibre edilir. Coverage, genişlik ve interval score birlikte raporlanır; zaman serisinin durağan olmadığı koşullarda otomatik kapsama garantisi iddia edilmez.

İki ardışık turda %0,5 iç-validation ilerleme yoksa aynı veri/arama alanı büyütülmez. Hata teşhisine veya sıradaki bilgi kaynağına geçilir. Çok sayıda backtest'ten şans eseri iyi sonuç seçmek bilinen bir risktir [4].

## 6. GPU, notebook ve devam etme sözleşmesi

L4 ilk pilot için başlangıç seçeneğidir; ölçülmüş fiyat/performans optimumu iddia edilmez. Aynı küçük pilotun süre/bellek ölçümü uygun olduğunda A100'le karşılaştırılır. A100 özellikle sequence batch throughput'u yararlıysa kullanılır. Büyük GPU aynı veriyle doğruluğu kendiliğinden artırmaz. Python metric, küçük veri veya Drive darboğazı ayrıca profillenir.

Yalnız iki aktif notebook kalır:

1. `ml/notebooks/cottonlens_data_workbench.ipynb`: kaynak durumu, normalize etme, kanıt/coverage kontrolü, snapshot dondurma.
2. `ml/notebooks/cottonlens_research_workbench.ipynb`: status, diagnose, feature pilot, search, compare, lock, reproduce, export.

Varsayılan aşama salt okunur status ve hesaplanan iş bütçesidir. “Tümünü çalıştır” bitmiş ablation'ı veya bütün araştırma turlarını açmaz. Kullanıcı yalnız tur ve azami çalışma süresini seçer. Ekranda cached, running, yerelde tamamlanan, Drive'a doğrulanan, başarısız ve kalan işler ayrılır; aile/horizon/fold/aday/seed görünür. ETA ölçümlerden aralık olarak verilir.

Veri ve çalışan modeller yerel Colab diskinde; tamamlanmış payload'lar checksum doğrulamasıyla Drive'da. Sahipli kilit, kesinti ve bozuk checkpoint testleri uzun turdan önce geçer. MLflow ledger'ın yeniden üretilebilir görünümüdür. Yeni profil ve namespace kullanılır; eski kaynak/deney kimliği üzerine yazılmaz. Backend DB/runtime ve preservation backup korunur.

## 7. Kabul ve release

CT=F günlük fiyat proxy'si, T+1/T+5 kayıtlı gözlem hedefleri, beş gözlem purge ve etiket olgunlaşması korunur. Training dışına scaler, imputasyon, hedef dönüşümü veya feature seçimi fit edilmez.

- Tam sekiz blokta en az %5 Naive fiyat-MAE kazancı, T+1 %53 / T+5 %55 yön doğruluğu, en az 6/8 blok kazanımı. Eski dört bloklu raporda 3/4 değişmez.
- Median-return ve geçmiş çoğunluk-yön ayrıca raporlanır. Farklı cohort skorları tek sıralama oluşturmaz.
- Fold sınırlarını koruyan 10.000 paired-bootstrap tekrarı; blok 20, hassasiyet 10/40. Sık tekrarlanmış araştırmadan sonra aralıklar bağımsız doğrulama gibi sunulmaz.
- 2024+ yalnız seen historical audit; model/feature/cadence seçimine yön vermez.
- Kilitlenen adayın sonraki ilk 126 olgunlaşmış origin tahminleri değiştirilemez kaydedilir. Bu gerçek ileri dönem kanıtı aylar gerektirir; mühendislik teslimiyle karıştırılmaz.

Release için gerçek yeniden eğitimle reproduction ve CPU parity ≤1e−6 log-return, checksum, exporter/importer/runtime sözleşmesi ve geçici DB'de API testleri gerekir. Cache okumak reproduction değildir. Yeni feature sırası açıkça sürümlenir; eski 24 kolon sözleşmesi sessiz değişmez. Eğitim kütüphaneleri backend'e taşınmaz. Gate geçmeyen model araştırma olarak kalır; Naive birincil referansı korunur.

## 8. Teslim sırası ve bitti tanımı

| Sıra | Teslim | Mühendislik hedefi / bağımlılık |
|---|---|---|
| 1 | Mevcut tahminlerden hata teşhisi, kaynak kapsam tablosu, dondurulmuş pilot tanımı | 0,5–1 gün; yeni fit yok |
| 2 | İlk uygun AMS/FAS altkümesi, as-of join, ortak cohort ve dar testler | 1–3 gün; yayın kanıtına bağlı, eksik kanıt blocker olarak görünür |
| 3 | Tek Research Workbench'te sınırlı yeni veri pilotu ve karar raporu | İlk karşılaştırma için toplam 3–5 iş günü hedefi; uygun veri yoksa hazır olduğu söylenmez |
| 4 | Katkı gösteren grupta tabular/sequence rakipler, seed ve belirsizlik raporu | İlk aday karşılaştırması için 7–10 iş günü hedefi; GPU süresi pilot sonrası |
| 5 | Kilit, reproduce/export, demo ve ileri dönem kayıt | Yalnız kabul ve parity kontrolleri sonrası |

Zorunlu dar testler: gelecekteki veriyi değiştirmek geçmiş feature/split'i değiştirmemeli; yayın ve revizyon erişimi doğru olmalı; hedef dönüşümü geri çevrilmeli; aynı origin kıyası korunmalı; tamamlanmış fit tekrar çalışmamalı; bozuk/yarım kayıt reddedilmeli; belirsiz kaynak export'a ulaşmamalı. Gerçek GPU ve runtime parity Colab/release aşamasında doğrulanır, yerel test sonucu yerine geçmez.

Her turun tek raporu: hipotez, kullanılan veri ve evidence seviyesi, gerçek fit sayısı, süreler, bütün adaylar, baselines, belirsizlik, başarısız hipotez ve sonraki karar. Kalıcı doğrulanmış dersler mevcut Obsidian proje notuna kısa eklenir. Cotcast ile ortak protokol olmadan “Cotcast'i geçtik” denmez; bağımsız incelenebilir sistem ve gerçek sonuçlar sunulur.

## Kaynaklar ve yorum sınırı

1. [Grinsztajn et al., tree models on tabular data](https://arxiv.org/abs/2207.08815) — tabular aile önceliğine destek, Cotton sonucu değil.
2. [TimeXer: exogenous information](https://arxiv.org/abs/2402.19072) — tek yeni sequence rakibinin mimari gerekçesi.
3. [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504) — basit lineer sequence kontrolünün gerekçesi; çalışmanın uzun horizon sonuçları burada doğrudan genellenmez.
4. [The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf) — çoklu deneme ve seçilmiş tarihsel sonuç riski.
5. [Optuna FAQ](https://optuna.readthedocs.io/en/latest/faq.html) — paralel optimizasyonda yeniden üretim sınırlamaları.
6. [USDA FAS Open Data](https://apps.fas.usda.gov/opendatawebV2/) — güncel veri/release bilgisi, tek başına tarihsel ilk-vintage kanıtı değildir.

Bu belge yalnız planlama teslimidir. Yeni model eğitimi veya yeni veri grubunun kabulü yapılmamıştır.
