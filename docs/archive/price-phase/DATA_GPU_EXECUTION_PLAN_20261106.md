# CottonLensAI — Veri doğruluğu, sızıntı önleme ve A100 araştırma planı

## 1. Amaç, kararlar ve tamamlanma tanımı

**Amaç:** Önce mevcut veri ve yöntem hatalarını kanıtlarıyla düzeltmek; ardından doğrulanmış girdilerle A100 üzerinde sistematik araştırma yapmak.

Bu planın üç ayrı başarı koşulu vardır:

1. **Mühendislik doğruluğu:** Bilinen hatalar giderilmiş, tekrarını yakalayan testler mevcut.
2. **Veri kabulü:** Kullanılacak her girdinin anlamı, kontratı, zamanı ve kaynak bağımlılıkları deneyin sözleşmesini karşılıyor.
3. **Bilimsel başarı:** Model, kilitli değerlendirmede Naive karşısında gereken üstünlüğü gösteriyor.

Birinci ve ikinci koşulların sağlanması üçüncüyü garanti etmez. Mutlak hatasızlık iddiası yerine, kritik belirsizlikte duran ve doğrulanabilir kanıt üreten bir sistem kurulacak.

### Kesinleşen kararlar

- Kullanıcının belirttiği **A100 erişimi için saat veya parasal bütçe tavanı yok**. Önceki 30 dakika/2 GPU-saat sınırları kaldırıldı.
- Her araştırma kampanyasının yine de **sonlu aday listesi, toplam iş sayısı ve bilimsel durma koşulu** olacak. Kaynak bolluğu açık uçlu arama anlamına gelmeyecek.
- Ücretsiz ve mevcut kaynaklar veri açığını kapatmazsa, **örnek veri kabul testiyle birlikte ücretli kaynak teklifi** hazırlanacak. Satın alma ayrı karar.
- Gerçek GPU eğitimi Colab’da; yerel makine geliştirme, denetim ve hafif testler için kullanılacak.
- Eski sonuçlar, başarısızlıklar ve kaynak sürümleri korunacak. Yeni düzeltmeler eski kanıtların üzerine yazılmayacak.
- Mevcut üretim API’si, aktif inference modeli ve kapalı toplama görevleri bu planla otomatik değişmeyecek.
- İncelenmiş 2016–2023 ve 2024+ dönemleri bağımsız holdout olarak yeniden adlandırılmayacak.

**Son gözden geçirmede eklenenler:** eğitim öncesi zorunlu kabul makbuzu, tarihsel erişilebilirlikle nihai hedef doğruluğunun ayrılması, kontrol kodunun bağımsız denetimi, veri onarımından etkilenen işlerin transitif takibi ve geniş aramanın tamamını kapsayan seçim yanlılığı kontrolleri.

## 2. Aşama A–C: Veriyi doğrula, onar ve eğitime kabul et

### A. Mevcut durumu yeniden doğrula

Önceki kök neden raporu dahil hiçbir özet kesin doğru kabul edilmeyecek.

- Güncel kod, sicil, deney tarifleri ve kaynak manifestleri karşılaştırılacak.
- Belgelerdeki önemli sayılar ilgili hesap veya kayıtla eşleştirilecek.
- Özellikle 217 kontrat kimliği sorunu, 139 HAR farkı ve etkilenen eğitim/test işleri ayrı ayrı doğrulanacak; farklı kapsamlı sayılar toplanmayacak.
- Eski LSTM ortak-origin düzeltmesinin mevcut raporlara yansıyıp yansımadığı kontrol edilecek.
- Daha önce giderilmiş hatalar yeniden geliştirilmek yerine mevcut regresyon testleriyle doğrulanacak.
- Yereldeki rapor ve envanter dosyaları mevcut çalışma değişiklikleri korunarak teslimata dahil edilecek.

**Çıktı:** Her sorun için kimlik, kanıt, etkilenen kapsam, düzeltme, test ve durum içeren tek hata sicili.

Durumlar: `OPEN_DEFECT`, `EVIDENCE_GAP`, `FIXED_VERIFIED`, `NOT_REPRODUCED`, `SUPERSEDED`. Kanıt eksikliği “düzeltildi” olarak kapatılmayacak.

### B. Veri sözleşmesini kilitle

Her veri akışı için şu bilgiler zorunlu olacak:

- Sağlayıcı, ürün/sembol, kontrat kimliği, teslim ayı ve birim.
- Fiyat alanının anlamı: settlement, close, last veya başka bir alan.
- İşlem tarihi, seans takvimi ve saat dilimi.
- Yayın zamanı, ilk gözlenme zamanı ve revizyon/vintage kimliği; bilinmiyorsa açıkça bilinmiyor.
- Ham belge/yanıt hash’i, parser sürümü ve kaynak konumu.
- Bu kaydın hangi feature, hedef veya başlangıç durumuna girdiği.

**Zaman açısından farklı roller ayrılacak:**

- Model girdisi, karar anında erişilebilir olmalı.
- Eğitim etiketi, ilgili fit başlamadan erişilebilir ve olgunlaşmış olmalı.
- Değerlendirme hedefi doğal olarak daha sonra oluşur; hangi nihai veya ilk yayımlanmış hedef sürümüyle skorlanacağı önceden kilitlenir.
- Sonradan bulunan tarihli fiyat tanığı fiyatın doğruluğunu destekleyebilir; geçmişte erişilebilirliğini otomatik kanıtlamaz.

CT=F serisi ile aynı-kontrat hedefi ayrı sözleşmeler olacak. Aynı-kontrat hedefe geçiş eski deneyin sessiz düzeltmesi sayılmayacak; ayrı veri/deney kimliği gerektirecek. Kontrat seçimi origin’de bilinen kuralla yapılacak.

### C. Katmanlı kabul ve karantina

| Katman | Zorunlu kontrol | Hata durumunda |
|---|---|---|
| Kaynak | Sembol, rapor türü, belge iç tarihi, yanıt formatı | Yanıt reddedilir; eski cache korunur |
| Bütünlük | Hash, boyut, sürüm, yarım dosya ve tekrar | Yeni sürüm veya karantina |
| Şema | Tip, birim, sonlu değerler, duplicate ve sıra | Hatalı kayıt işaretlenir |
| Piyasa anlamı | Kontrat yaşam döngüsü, alan semantiği, listeleme kapsamı | Tahminle kimlik düzeltmesi yapılmaz |
| Takvim | Gerçek seans, tatil, DST ve ufuk tarihi | Eksik seans sıkıştırılmaz |
| Mutabakat | Kaynaklar arasındaki fiyat/kontrat/tarih farkı | Çelişki çözülmeden kabul verilmez |
| Erişilebilirlik | Karar/fit anında bilinen sürüm | Bilinmeyen bilgi doğrulanmış sayılmaz |
| Bağımlılık | Türetilmiş hücrelerin bütün kaynakları | Belirsizlik aşağıdaki hesaplara taşınır |

Ek kurallar:

- High–low dışındaki settlement otomatik kırpılmayacak.
- Düşük hacim veya büyük hata nedeniyle gün silinmeyecek.
- Eksik quote, yokluğu doğrulanmış kontratla karıştırılmayacak.
- Aynı yayıncıdan iki belge bağımsız iki sağlayıcı sayılmayacak.
- Kaynaklar çoğunluk oylamasıyla “doğru” ilan edilmeyecek.
- Kaynağın tarihsel kapsamı açıklanamadığında ileri veya geri doldurma yapılmayacak.
- EWMA gibi durum taşıyan hesaplarda başlangıçtan gelen belirsizlik izlenecek; gizli reset uygulanmayacak.

Mevcut readiness ve reconciliation modülleri genişletilecek. Özellikle “tam tablo”, “kontrat yok” ve “şüpheli dönemin dışı temizdir” varsayımları ayrıca sınanacak.

### Onarım yöntemi

1. Mevcut arşiv ve kaynak tanıkları kullanılarak hücre düzeyinde kapsam matrisi çıkarılacak.
2. Eksik kanıt için yalnız gereken tarihler/alanlar hedefli olarak araştırılacak.
3. Düzeltmeler eski/yeni değer, kanıt ve gerekçeyle ek katmanda tutulacak.
4. Feature, hedef, scaler, seçim ve fit bağımlılıkları üzerinden etki yayılımı hesaplanacak.
5. Türetilmiş veriler doğrulanmış ham katmandan yeniden üretilecek.
6. Veri düzeltmesi model skorları görülmeden kilitlenecek.
7. Kritik boşluk kapanmıyorsa ücretli kaynak için örnek kabul testi ve teklif hazırlanacak.

**Geçiş kapısı G1:** Deneyin gerekli tüm veri bağımlılıkları kabul edilmiş olacak. Kapsam eksikse piyasa eğitimi bloke kalacak. Daha dar veri evreni kullanılacaksa kapsam ve dışlanan tarihler sonuçlara bakılmadan yeni ön kayıtla tanımlanacak.

## 3. Aşama D–E: Sızıntıyı engelleyen yürütme ve bağımsız değerlendirme

### D. Eğitim öncesi zorunlu kabul makbuzu

Araştırma katmanına sürümlü bir kabul kaydı eklenecek. Şunlara bağlanacak:

- Veri ve kaynak sürümleri.
- Kod, parser ve takvim kimlikleri.
- Feature/target tanımları.
- Origin’ler ve split.
- Zaman ve bağımlılık kontrolleri.
- Kritik açık hata listesi.
- Doğrulayıcı sürümü ve sonuçları.

Yeni piyasa eğitim yolu, ilgili makbuz olmadan çalışmayacak. Makbuz hazırlanmış olsa bile veri veya tarif değişirse geçersiz olacak. Eski deneylerin okunması ve sıfır-fit replay çalışmaları korunacak.

Başarısızlık “uyarı verip devam et” biçiminde olmayacak. Kabul ve eğitim aynı immutable girdilere bağlanacak; kontrol sonrası dosya değişikliği de yakalanacak.

### Sızıntı ve hata enjeksiyonu testleri

| Test | Beklenen davranış |
|---|---|
| Gelecek satırları değiştirme | Önceki origin’lerin girdileri ve sabit tarifle tahminleri değişmez |
| Prefix testi | O tarihte kesilmiş veri ile tam verinin aynı geçmiş bölümü eşleşir |
| Geleceğe bakan join, centered rolling, backfill | Kontrol başarısız olur |
| Tüm veride scaler/imputer/feature selection | Fit sınırı ihlali yakalanır |
| Yanlış early-stopping/tuning bloğu | Değerlendirme etiketi erişimi reddedilir |
| Olgunlaşmamış T+1/T+5 etiketi | Eğitim işi başlamaz |
| Eksik seans veya ters tarih sırası | Pencere sessizce daraltılmaz |
| Duplicate veya yanlış kontrat/simge | Kaynak kabulü başarısız olur |
| Revizyonu geçmişe taşıma | Point-in-time kontrolü başarısız olur |
| Eski cache’i yeni veriyle kullanma | Kimlik uyuşmazlığı nedeniyle reddedilir |
| Değiştirilmiş kabul makbuzu veya payload | Bütünlük kontrolü başarısız olur |

Geçmiş tahminlerin değişmezliği test edilirken **tarif ve o tarihteki seçim durumu da sabit tutulacak**. Gelecekte seçilmiş farklı bir modelin geçmişe uygulanmasıyla karıştırılmayacak.

T+1/T+5 purge, etiket zaman aralıklarından kontrol edilecek. Mevcut sabit protokol gevşetilmeyecek; yeni bir protokol ayrı kimlikle tanımlanacak.

Her kritik hata için:

- Geçerli örnek geçmeli.
- Tek başına enjekte edilen hata reddedilmeli.
- Hata giderildikten sonra test tekrar geçmeli.
- İlgisiz hata nedeniyle yanlışlıkla başarısız olmuş test, başarılı koruma sayılmamalı.

### E. Bağımsız değerlendirme

- Model ve Naive tam olarak aynı hedef/origin/kontrat kümesinde karşılaştırılacak.
- Ortak küme oluşturulurken eksilen tahminler gizlenmeyecek; zorunlu kapsam eksikse birincil değerlendirme tamamlanmış sayılmayacak.
- Fiyat dönüşümü, MAE, yön ve yıllık sonuçlar ikinci bir hesaplayıcıyla denetlenecek.
- İkinci hesaplayıcı ana sistemin aynı kayıp/hedef yardımcı fonksiyonlarını kullanmayacak.
- Naive ve sıfır düzeltme eşitliği, birim dönüşümü ve sıfır gerçekleşen hareketler test edilecek.
- Ham/seçilmiş sonuç, aktif oran ve aktif yön doğruluğu ayrı verilecek.
- Bootstrap blokları, yıl sınırları, tekrar sayısı ve seed sonuçtan önce kilitlenecek.
- Restore edilmiş native model ve preprocessing, saklanan tahminlerle karşılaştırılacak.

**Geçiş kapısı G2:** Kritik hata enjeksiyonlarının tamamı yakalanmış, geçerli örnekler geçmiş ve bağımsız değerlendirme belirlenen toleransta uyuşmuş olacak.

“Tüm testler geçti” yerine hangi hata sınıflarının denetlendiği ve hangi bilinmeyenlerin kaldığı raporlanacak.

## 4. Aşama F–G: A100 altyapısı ve kontrollü geniş araştırma

### F. A100 çalışma altyapısı

Veri onarımı sürerken sentetik altyapı testleri hazırlanabilir. Gerçek piyasa eğitimi G1 ve G2’yi bekleyecek.

- Mevcut Colab kurulumu, lock dosyaları, çalışma gözetimi ve ledger kullanılacak.
- Gerçek donanım, VRAM, CUDA/sürücü ve paket sürümleri oturum başında kaydedilecek.
- A100 talebi karşılanmazsa başka cihazda sessiz devam edilmeyecek.
- CPU/GPU seçimi algoritma bazında yapılacak; küçük CPU işleri gerektiğinde Colab CPU’sunda yürütülecek.
- GPU kullanımı, veri hazırlama, aktarım, eğitim, checkpoint ve arşivleme süreleri ayrı ölçülecek.
- Mevcut sürümle uyumsuzluk görülmeden toplu paket yükseltmesi yapılmayacak.
- FP32 temel çalışma olacak. Mixed precision yalnız sayısal ve karar kararlılığı kontrolünü geçerse ayrı yürütme profili olarak açılacak.
- CPU ve GPU eğitimlerinin birebir aynı model olması beklenmeyecek; her backend kendi kayıtlı modelini yeniden üretebilmeli.

### Kesinti ve paralellik

Mevcut ledger tek yazıcı varsayımı taşıyor. Bu yüzden tek ledger’a çok süreç yazdırılmayacak.

- İlk yürütme tek koordinatör ve tek GPU worker ile başlayacak.
- Paralellik gerekiyorsa her worker ayrı immutable sonuç alanına yazacak; koordinatör doğrulanmış makbuzları birleştirecek.
- Aynı işin iki worker tarafından tamamlanması, bozuk checkpoint, OOM ve oturum kesilmesi sınanacak.
- Restart, tamamlanmış işleri tekrar fit etmeyecek.
- Optimizer ve rastgelelik durumu olmadan tam devam mümkün değilse yeniden başlayan attempt açıkça kaydedilecek.
- OOM sonrası batch boyutu veya model kendiliğinden değiştirilmeyecek; yeni tarif olarak kaydedilecek.
- Colab yerel diski ve Drive kalıcı doğrulanmış arşiv sayılmayacak.
- Private GitHub arşivinden taze restore kontrolü yapılacak; lisanslı veri için yeniden dağıtım hakkı ayrıca dikkate alınacak.

**Geçiş kapısı G3:** A100 üzerinde gerçek cihaz kullanımı, model kaydetme/yükleme, kesinti sonrası devam, native replay ve arşiv geri yükleme doğrulanmış olacak.

### G. Araştırma kampanyası

A100 erişimi, aynı tarihsel skora sınırsız uyum sağlamak için kullanılmayacak.

İlk kampanyadan önce tek bir makine okunur manifest üretilecek:

- Ekonomik/istatistiksel hipotez ve önceki sicilden farkı.
- Veri, hedef ve feature kümesi.
- Model aileleri ve sonlu hiperparametre uzayı.
- Seed listesi ve seed’lerin seçimde nasıl ele alınacağı.
- İç/dış zaman bölmeleri.
- Tuning, pruning, early stopping ve eşitlik çözümü.
- Birincil adayın seçim kuralı.
- Toplam aday, refit ve sentetik kontrol sayısı.
- Başarısızlık, tekrar ve kampanyayı durdurma kuralları.

**Manifestte açık alan varsa kampanya başlayamaz.** Bu plan mevcut veri kabulü belli olmadan yeni bir fiyat modeli veya feature aralığını keyfî seçmiyor; manifestin hazırlanması veri denetiminin çıktısına bağlı zorunlu teslimattır.

Araştırma sırası:

1. Mevcut kontrollerin kapsamadığı öğrenme soruları belirlenir.
2. Gereken pozitif/negatif, zayıf-sinyal ve rejim kontrolleri çalıştırılır.
3. En fazla üç farklı hipotez için sonlu kampanya hazırlanır.
4. İç doğrulamada adaylar taranır.
5. Her dış fold için seçim yalnız o fold’dan önceki geçmişten yapılır.
6. Kampanyanın sonuçları kilitli değerlendiriciyle hesaplanır.
7. Yalnız önceden tanımlı devam koşulları varsa sonraki kampanya açılır.

XGBoost GPU, mevcut LSTM/TCN yolları ve uygun CPU referansları altyapı adaylarıdır. Yeni aile eklemek için veri/hipotez gerekçesi gerekir. Aynı tarifte binlerce gereksiz seed veya geniş grid otomatik açılmayacak.

### Geniş aramanın istatistiksel koruması

- Aday sayısı arttığında şans eseri kazanan seçme riski açıkça hesaba katılacak.
- Tek kazananın CI’ı bütün model aramasının doğrulaması sayılmayacak.
- Null kontroller yalnız tek modeli değil, gerekli olduğunda seçim prosedürünü de kapsayacak.
- Pruning yalnız iç geçmiş veriyi kullanacak.
- En iyi seed seçilmeyecek; seed birleştirme kuralı önceden sabit olacak.
- Çoklu adaylara ilişkin doğrulayıcı iddialarda test ailesi ve düzeltme yöntemi ön kayda yazılacak.
- Dış sonuçtan sonra değiştirilen tarif yeni geliştirme çalışması sayılacak.
- A100 üzerinde binlerce iş çalışabilir; bağımsız doğrulama ihtiyacı bundan etkilenmez.

### Bilimsel durma kuralları

| Durum | Eylem |
|---|---|
| Veri, sızıntı veya değerlendirme kusuru | Etkilenen işler durur; sonuçlar karantinaya alınır |
| Native replay veya checksum uyuşmazlığı | Arşiv/yürütme sorunu çözülmeden devam edilmez |
| Geçerli kampanya başarı ölçütlerini sağlamıyor | Sabit kampanya kapanır; aynı arama kendiliğinden büyütülmez |
| Tarihsel aday koşulları sağlıyor | Bağımsız ileri doğrulama ön kaydı hazırlanır |
| Tarihsel veri kanıtı temin edilemiyor | Piyasa eğitimi bloke; kaynak teklifi veya ayrı ileri veri tasarımı sunulur |

Mevcut fiyat hedefleri korunur: en az %5 MAE kazancı, ufka uygun %53/%55 yön, sekiz yıllık protokolde 6/8 yıl ve kilitli belirsizlik koşulları. Farklı kapsamdaki bir çalışma bu kapıları sağlamış gibi sunulmaz.

## 5. Teslimatlar, tekrar kontrol politikası ve kapanış

### Teslim sırası

1. **Hata ve kapsam sicili:** Önceki iddiaların yeniden doğrulanması.
2. **Veri sözleşmesi ve kapsam matrisi:** Kaynak, zaman ve bağımlılık boşlukları.
3. **Kanıtlı onarım paketi:** Eski/yeni farklar ve etkilenmiş işlerin listesi.
4. **Zorunlu kabul kapısı:** Eğitim yoluna bağlanmış, değişiklikte geçersizleşen makbuz.
5. **Sızıntı ve bozulma testleri:** Her kritik hatanın gerçekten yakalandığının kanıtı.
6. **Bağımsız değerlendirici:** Ortak-origin hesap ve native replay.
7. **A100 yürütme paketi:** Çalışan notebook/CLI, checkpoint, resume ve restore.
8. **Kilitli araştırma manifesti ve sonuçları:** Başarılar, başarısızlıklar, pruning ve kesintiler dahil.
9. **GitHub teslimatı:** İncelenmiş kod/belge değişiklikleri, CI ve checksum bağlı uygun kanıt arşivleri.
10. **İleri doğrulama kararı:** Yalnız yeterli aday varsa ayrı protokol; otomatik model yayını yok.

### Tekrar kontrolün uygulanışı

“Tekrar tekrar kontrol”, aynı testi değişmeyen dosyalarda sınırsız çalıştırmak anlamına gelmeyecek.

Kontroller şu noktalarda tekrar edilecek:

- İlk kaynak kabulünde.
- Veri/parser/takvim/kod değiştiğinde etkilenen bağımlılıklarda.
- Kampanya başlamadan.
- Checkpoint’ten devam ederken.
- Sonuç üretirken.
- Uzak arşivden taze geri yüklerken.

Tam backend/frontend testleri ilgili değişiklikler ve nihai entegrasyonda; ML/research testleri değişen kapsamda çalışacak. CI gerçek piyasa modeli eğitmeyecek.

### Kapanış şartı

Bir iş yalnız şu kanıtlarla kapatılacak:

- Sorun yeniden gösterilmiş veya şüphenin kapsamı sınırlandırılmış.
- Düzeltme ve etkilediği alanlar belirlenmiş.
- Önce hata veren regresyon testi artık geçiyor.
- Bozuk örnek hâlâ reddediliyor.
- Eski kanıtlar korunuyor.
- Gerekli yerel ve uzak geri yükleme doğrulamaları geçiyor.

**Veri açığı kapanmadan araştırma hazır; tarihsel sonuç olmadan model başarılı; ileri kanıt olmadan bağımsız üstünlük doğrulanmış ilan edilmeyecek.** Planın sonunda çözülebilen kusurlar kapatılacak, çözülemeyenler somut kanıt gereksinimleriyle bloke kalacak ve A100 araştırması yalnız geçerli aşamalarda ilerleyecek.
