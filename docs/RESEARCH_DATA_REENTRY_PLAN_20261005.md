# Veri bütünlüğü, kaynakların yeniden değerlendirilmesi ve sonraki araştırma planı

5 Ekim 2026 — **inceleme sonucunda hazırlanmış uygulama taslağı**. Bu belge yeni
eğitim veya veri kabulü yapıldığını göstermez. Ürün amacı ve mevcut kararlar
[STATUS](STATUS.md) ve [işlem sözleşmesinde](TRADING_RESEARCH_CONTRACT_20261005.md)
kalır. İncelenen ek: `CottonLens_Model_Dogrulugu_Plan (1).md`; belge kimliği ve
bu incelemenin sayısal kanıtı [doğrulama kaydında](../research/evidence/plan-review-20261005.json).

## Karar

Ekli planı olduğu gibi uygulamayacağız. Ana eksik, bütün modelleri yeniden
yarıştırmak değil; **eldeki ham bilgiden modele nelerin gerçekten ulaştığını,
hangi sürümün ne zaman kullanılabildiğini ve tahminin hangi uygulanabilir
işleme karşılık geldiğini** tamamlamak. Kaynağı araştırma dışında tutmak,
kaynağın işe yaramadığını kanıtlamaz. Buna karşılık salt kolon eklemek veya
`model_eligible=true` yazmak da kaynak kabulü değildir.

10 gün bir çalışma bütçesi olabilir; güçlü model veya bağımsız canlı performans
kanıtı için süre garantisi olamaz. Önceki fiyat hedeflerinin eşikleri korunur:
%5 MAE, T+1 %53 / T+5 %55 yön, sekiz yılda 6/8 kazanım. Ekteki T+1 %55 ek hedefi
eski deneyleri yeniden sınıflandırmak için kullanılmaz. %0,55 MAPE izlenebilir;
evrensel başarı ölçüsü veya yüzde doğruluk değildir.

Seçilen ürün amacı vadeli piyasada yön ve pozisyondur. Birincil ürün ölçüsü
işlem sözleşmesindeki maliyet sonrası sonuçtur; MAE/MAPE ve yön yardımcıdır.
Fiyat programı ile işlem araştırmasının hedefleri ve kayıtları karıştırılmaz.

## 1. Önceden yapılanlar: yeniden başlatılmayacak işler

**VERIFIED:** Sicil doğrulandı: 122 deney/kontrol/tarihçe kaydı, 430 tarif,
24.575 benzersiz tamamlanmış fit makbuzu. Sekiz ilgili profilde 60 grubun **432
yıllık tahmin dosyası** checksum ile kontrol edildi; ham ve küçültülmüş fiyat-MAE
yeniden hesaplandı. Aynı profilin gruplarında tarih/fiyat/T+5 hedef vektörleri
eşit. Sonuçlar sicille `1e-9` yüzde puan toleransında aynı.

Aşağıdaki aralıklar lag1/2/6 kaynak kollarının **Naive'a göre seçilmiş fiyat-MAE
kazancıdır**; en iyi lag önerisi veya kaynaklar arası eşit-cohort sıralaması değildir.

| Deney profili | OOS kapsamı | Saklanan sonuç | Bu plan için anlamı |
|---|---|---|---|
| `fas-exploration-v1` | T+5; 2016–2023; 2.006 origin | −%1,5665 … −%0,5860 | Ulusal satış/sevkiyat ve haftalık değişimleri denenmiş. Ülke dağılımı test edilmiş sayılmaz. |
| `nass-exploration-v1` | T+5; aynı 2.006 | −%1,0569 … −%0,6922 | Ulusal kondisyon denenmiş; Texas büyüme evreleri denenmiş sayılmaz. |
| `wasde-exploration-v1` | T+5; 2019–2023; 1.254 | −%0,7457 … −%0,2965 | Üç dünya bilançosu özelliği denenmiş; ülke revizyonları ayrı hipotezdir. |
| `cftc-exploration-v1` | T+5; 2.006 | −%1,7829 … −%0,7684 | Managed/producer net-OI ve managed değişimi denenmiş. |
| `weather-exploration-v1` | T+5; 2.006 | −%1,8860 … −%1,0117 | Üç Southern High Plains noktası; bütün Cotton Belt veya hava tahminleri değil. |
| `fx-exploration-v1` | T+5; 2.006 | −%0,5659 … −%0,5284 | BRL/CNY/INR çaprazları zaten denenmiş. |
| `fundamental-joint-pilot-v1` | T+5; 2.006 | joint_L1 +%0,2171, 3/8 yıl; diğer joint lag'leri negatif | FAS+NASS+hava birlikte de çalıştırılmış. Eski “pending” notuna göre tekrar açılmaz. |
| `wasde-text-pilot-v1` | T+5; 1.254 | −%0,7558 … %0; lag2 tamamen Naive | Sabit TF-IDF/SVD/Ridge yeniden koşulmaz; sıfır ağırlık metinde sinyal yokluğunu kanıtlamaz. |
| `availability-clock-pilot-v1` | T+1/T+5; 2.006/kol | müdahale −%0,1538 / −%1,3396; 0/8 yıl | DXY/WTI saat varsayımı deneyi tamamlandı; aynı tarif tekrarlanmaz. |

İlk sekiz satır bu incelemede tahminlerden yeniden hesaplandı. Son satırın
bağımsız kontrolleri [mevcut sonuç belgesinde](AVAILABILITY_CLOCK_RESULT_20261005.md).
Kaynak deneyleri varsayımlı zaman/sürüm statüsündedir; T+5 sonuçları T+1'i veya
maliyet sonrası işlem hedefini elemez. Ham sonuçlar da kayıtta bulunur.

Ridge, ElasticNet, XGBoost, CatBoost ve LSTM için geçmiş kayıt var. LSTM'nin eski
498/439 origin kıyası düzeltilmiştir; tam tahminleri kayıp olduğu için aynı
cohort'ta yeni tabloya eksiksizmiş gibi konulamaz. LightGBM için sicilde kayıt
bulunmadı; bu, şimdi çalıştırılması gerektiğini veya hiçbir yerde denenmediğini
göstermez. TCN öğrenme kontrolü ayrı başarısızlık kaydıdır.

## 2. Dataset ve özellik kapsamı: yeniden dahil etme kararları

Altı mevcut derlenmiş kaynak tablosunun satır sayısı ve SHA256'sı manifestleriyle
doğrulandı. **Byte bütünlüğü doğrulaması, yayın/vintage doğrulaması değildir.**

| Kaynak | Gerçekte mevcut/denenmiş kapsam | Eksik veya dışarıda kalan | Düzeltme ve kabul koşulu |
|---|---|---|---|
| **WASDE — ilk genişletme adayı** | 95 derlenmiş satır; `stock_use`, `production_use`, `stock_use_change`. 96 benzersiz ham XML checksum'ı ayrıca doğrulandı; hepsinde World/US/China/India/Brazil tabloları var. | US üretim/ihracat/stok, Çin tüketimi, Hindistan/Brezilya üretimi ve ayrı revizyonları model girdisi olmamış. | Mevcut parser/ham XML'den yeni sürümlü bölgesel tablo üret. Önce seviyeler + aynı pazarlama yılına ait revizyonlar; ülke/ölçü birimi/rapor ayı anahtarları zorunlu. Rapor PDF/TXT hücreleriyle uzlaştır; ilk sürüm veya o sürümün erişilebilir üst sınırını belgeleyerek kabul et. Kanıt eksikse yalnız açıkça etiketlenmiş duyarlılık paketi. |
| **FAS Export Sales — ikinci aday** | 752 haftalık ulusal toplam; satış, sevkiyat, değişimleri. 15 yıllık checksum doğrulanmış ham dosyada 27.887 satır; `countryCode`, `outstandingSales`, `currentMYTotalCommitment` alanları mevcut. | Çin/Vietnam/Türkiye/Pakistan dağılımı, bekleyen satış, 4 haftalık momentum, yıl karşılaştırması kullanılmamış. | Ülke kataloğu ve All Upland birimiyle yeni country-week tablo; pazarlama yılı sınırında mükerrer toplama yok. Negatif net satış iptal olabilir, sıfırlanmaz. Ülke payını net satışın sıfır/negatif toplamına körlemesine bölme: ilk temsil sevkiyat veya bekleyen satış payı ve net satış seviyesi. API'nin bugünkü revize sürümü tarihsel ilk sürüm sayılmaz; rapor/vintage uzlaştırması gerekir. |
| **NASS Crop Progress/Condition** | 311 ulusal kondisyon haftası. Mevcut 2023 rapor örneğinde Texas ve Cotton Planted tabloları da bulunuyor; tam bölgesel tarih henüz denetlenmedi. | Texas ve diğer eyaletlerin planted/squaring/bolls opening/setting/harvested alanları, 5 yıllık ortalamadan fark. | Var olan rapor parser'ını eyalet/evre/hafta boyutuna genişlet. Ulusal değeri Texas diye kullanma. Beş yıllık ortalama raporda yayımlandığı haliyle veya yalnız geçmişten kurulmalı. Sezon dışı değerleri sonraki sezona taşıma; `(D)/(NA)`/eksik sıfır değildir. Her eyalet/alan için ayrı coverage ve kaynak kanıtı. |
| **CFTC** | 730 haftalık kayıt; futures-only `033661`, bütün vadelerin toplamı. Bilinen belirsizlikler için 29 rapor maskeli. Ham long/short/OI var. | 4 haftalık değişim, 52 haftalık sıralama/z-score ve extreme göstergeleri; gerçek yayın olaylarına göre hizalama. | Önce rapor tarihi ile yayın olayını ayır; tatil, kapanma ve düzeltme takvimini bağla. Sırf daha çok satır için maskeleri kaldırma. Belirli raporun doğru sürümüne kanıt varsa yeni pakette kurtar. Rolling istatistikler yalnız geçmiş raporlardan; bütün-vade OI, kontrat bazlı OI yerine kullanılamaz. |
| **Hava** | 15.339 nokta-gün; Lubbock/Hale/Gaines, POWER yağış/Tmax; eşit ağırlık, 7/30 gün ve sabit sezon etkileşimi. | South Texas/Delta/Georgia; 14 gün, anomali, GDD, kuraklık ve gerçek ekim evresi etkileşimleri. | Önce kaynak sürümü/latency sorunu. POWER geçmişi güncel yeniden analizdir; NASA ilk verilerin sonradan değiştiğini açıklıyor. İlk sürüm arşivi veya ileriye dönük zamanında kayıt gerekir. Bölgesel ağırlıklar geçmişte bilinen ekim/üretim verisiyle sabitlenmeli. Yalnız Tmax ile standart GDD hesaplanmaz; Tmin/ortalama ve agronomik tanım gerekir. Eğitim öncesi bütün tarihten klimatoloji kurulmaz. |
| **DXY/WTI + FX** | Çekirdekte DXY/WTI; BRL/CNY/INR için 3.595 kaynak günü ve ayrı pilot mevcut. | İşlem anındaki erişilebilir sürüm; önerilen US rates ayrı bilgi kaynağı. | Mevcut FX/döviz çevrimini tekrar ekleme. ECB aynı-gün currency/EUR ÷ USD/EUR yönü korunur. US rates ancak vadesi, fiyat/yield birimi ve sürüm saati belirlenirse ayrı aday; henüz kabul edilmiş tarih yok. |
| **Vadeli yapı — işlem hedefi için ön koşul** | CT=F OHLCV var; mevcut publication loader `contract_curve` türünü destekliyor. Sicilde doğrulanmış tarihsel eğri deneyi bulunmadı. | Tekil kontrat, yakın/sonraki vade eşzamanlı fiyatları, kontrat OI, roll ve gerçekten uygulanabilir giriş/çıkış. | ICE/sağlayıcıdan tarihsel kapsam ve kullanım hakkı incelemesi. Sembol+vade+seans+bar saati+settlement/işlem fiyatı ayrımı zorunlu. Önce küçük geçiş örneği doğrulanır; ancak sonra tarih genişler. CT=F'yi kaydırarak next-month veya spread üretilmez. Ücretsiz tam geçmiş erişimi henüz doğrulanmadı. |
| **Polyester/freight** | Sicilde doğrulanmış paket bulunmadı. | Ürün, ülke, sıklık, birim, lisans ve yayın/vintage. | Ön araştırma adayı; isim benzerliğiyle proxy alınmaz. Kaynak kimliği ve ekonomik mekanizma bulunamazsa ilk deney dışında gerekçeli bekler. |

WASDE: `sources/wasde.py:cotton_rows` ülke alanlarını zaten ayrıştırıyor;
`research/wasde_exploration.py:balance` bunları **World ile sınırlıyor**.
Dolayısıyla burada gereken ilk iş yeni veri aboneliği değil, mevcut kanıttan
daha kapsamlı fakat kontrollü bir temsil kurmak. Bu bir performans garantisi değildir.
Ocak 2019 WASDE yayımlanmamıştı; “eksik ayı düzeltmek” için sahte rapor üretilmez.
ABD stocks-to-use paydasında iç kullanım + ihracat, dünya oranında dünya iç
kullanımı tanımı açıkça ayrılır; aynı ekonomik anlama sahipmiş gibi birleştirilmez.
Raporlar arası revizyon piyasa beklentisine göre “sürpriz” değildir; arşivlenmiş
beklenti verisi olmadan bu adla özellik üretilmez. Aylık raporun günlük tekrarları
bağımsız bilgi örneği sayılmaz; rapor/olay sayısı ayrıca gösterilir.

Tam yayın saniyesini bulamamak her dosyayı sonsuza kadar dışlamayı gerektirmez.
Mevcut `sources/public.py:reviewed_upper_bound` belirli sürümün **en geç ne zaman
erişilebilir olduğuna dair kanıtı** destekliyor. Bu yolu kullanacağız. Bugünün
indirme saati 2016'daki erişilebilirliği kanıtlamaz; genel yayın takvimi de belirli
dosyanın sürümünü doğrulamaz. Kabul araştırma/üretim statüsü ve kullanım hakkıyla
ayrı kaydedilecek; eski manifestlerin bayrakları değiştirilmeyecek.

Resmî dayanaklar: [NASA sürüm değişimi](https://power.larc.nasa.gov/docs/tutorials/service-data-request/api/),
[CFTC yayın/düzeltme istisnaları](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm),
[USDA değişiklikler](https://www.usda.gov/historical-changes-revisions),
[WASDE Ocak 2019 açıklaması](https://esmis.nal.usda.gov/sites/default/release-files/3t945q76s/2801rc34n/1z40nn26q/wasde0125.pdf),
[ICE ürün/veri sayfası](https://www.ice.com/products/254/Cotton-No-2-Futures/data?marketId=5460932).
Erişim tarihi 5 Ekim 2026; bu sayfalar tek başına bütün tarihsel satırları kabul ettirmez.

## 3. Veri bozulmasını engelleyen uygulama sözleşmesi

1. **Önce kimlik:** mevcut Release, registry/trials, ham dosya, tablo, hazır deney,
   origin/target ve özellik sırası hash'leri sabitlenir. Eksik eski model payload'ı
   “sağlam cache” sayılmaz. Çatışma çıktığında hash güncelleyip hatayı gizlemek yok.
2. **Yeni sürüm:** ham → normalleştirilmiş → erişilebilirlik kanıtı → özellik →
   deney zinciri yeni namespace'te oluşturulur. Özgün dosya/ledger/Release
   değiştirilmez. Çıktı geçici alanda doğrulanır, tamamlanma işareti en son atomik
   yayımlanır; çökme veya yarım indirme tamamlanmış paket oluşturamaz.
3. **Satır kimliği:** source/series/region/contract, observation/reference period,
   report date, marketing year, unit, value/qualifier, vintage/source hash,
   published_at (biliniyorsa), available_at/availability_basis ve evidence,
   retrieved_at ayrı tutulur. Düzeltme yeni olaydır; önceki tarih üzerine yazılmaz.
4. **Causal join:** tek UTC karar saati; `available_at <= decision_at`. Yeni
   sözleşmedeki 00:15 UTC saatine göre açık politika kullanılır. Kontratın borsa
   seans etiketi ve UTC günü eşit varsayılmaz. Geleceği değiştirme testi geçmiş
   özelliklerini değiştirmemeli; train-only dönüşümler ve olgun hedef şartı korunur.
5. **Kayıp ve kapsam:** kaynak eksiği origin silmez; yaş/eksiklik bilgisiyle kalır.
   Hedef veya uygulanabilir fiyat yoksa ayrı engel/kapsam kaydı; sessiz kesişim,
   backfill veya yapay işlem fiyatı yok. Yeni cohort gerekiyorsa bütün referanslar
   aynı cohort'ta hesaplanır; eski 2.006 origin sonuçlarıyla karıştırılmaz.
6. **Geri dönüş:** yalnız yeni sürüm işaretçisi değişir; bozuk yeni paket karantinaya
   alınır. Eski snapshot ve Naive geri dönüş referansı korunur. Public kopyadaki
   secret temizliğiyle değişmiş byte'lar eski cache checksum'ı yerine konulmaz.

**VERIFIED — entegrasyondan önce düzeltilecek iki nokta:**

- `research/protocol.py:attach_releases` bugün karar saatini ertesi gün **00:00**
  yapıyor ve eşit timestamp'i dışlıyor (`allow_exact_matches=False`). Bu, yeni
  00:15 / `<=` sözleşmesiyle uyumlu değil. Varsayılan eski politika korunarak yeni
  paket için açık karar-saati parametresi ve sınır testleri gerekir. İncelenen
  kaynak pilotları kendi varsayımlı hizalamalarını kullanıyor; bu tespit onların
  başarısızlığının kanıtlanmış sebebi olarak sunulmaz.
- `research/publications.py:main`, `attach_package` yerine doğrudan
  `attach_releases` çağırıyor; `max_age_days` ve otomatik yaş/eksiklik davranışı
  ortak yardımcıyla aynı değil. Yeni kabul yolundan önce CLI ve engine aynı
  uygulanmış tazelik politikasında birleştirilmeli; aynı girdide eşitlik testi konmalı.

Zorunlu kontroller: değiştirilmiş checksum/şema/kimlikte durma; çift kayıt ve
çelişen revizyonda durma; aynı/sonraki saat, DST/tatil, yıl/sezon geçişi;
WASDE aynı-crop-year farkı; FAS ülke toplamı/birim/negatif satış; NASS nitelikli
eksik; hava zaman standardı ve tam geçmiş gereksinimi; kontrat değiştirmeden PnL;
train-only scaler/imputer/encoder; cache kimliği ve yarım çıktı; aynı origin/target;
bağımsız metrik hesabı; önce/sonra özgün checksum eşitliği ve küçük restore denemesi.

## 4. Dört iş paketi ve durma koşulları

### A — Mevcut tahminin işlem karşılığını ve veri kabul sınırını belirle

İlk iş, mevcut işlem sözleşmesindeki **eğitimsiz T+1 analizidir**: saklanan kontrol
ve zamanlama müdahalesi tahminlerinin `sign(selected_return)` pozisyonunu, aynı
origin'lerde sonraki kayıtlı Cotton open→close hareketiyle eşleştir. Ham işaret
ayrı tanısaldır. Flat/long/short referansları, aktif oran, yıl sonuçları ve başa
baş çift yön maliyet raporlanır. CT=F sonucu proxy kalır; open'ın karar sonrasında
uygulanabilir olduğu ve kontrat kimliği kanıtlanmadan net işlem üstünlüğü denmez.

Aynı paket, yukarıdaki kaynak matrisini admission aday kayıtlarına dönüştürür;
uygulanabilir kontrat/seans verisinin küçük örneği ve 00:15 saat testleri çıkarılır.
Beklenen emek **1–2 çalışma günü**, sıfır fit. Kanıt bulunamazsa açık blocker;
başka bir model grid'iyle devam edilmez. Kaynak içerik incelemesi yine ilerleyebilir.

### B — Önce WASDE, ardından FAS için yeni veri paketi

Önce mevcut ham dosyalardan bölgesel WASDE tablo ve revizyonları kur. Sonraki aday
FAS ülke/commitment tablosu. İlk pakette bütün kaynaklar aynı anda birleştirilmez.
Her pakette kaynak/kolon kapsamı, veri kaybı, revision/availability kanıtı, orijinal
hash koruması ve kullanım statüsü raporlanır. Salt parser başarısı eğitim kabulü
sayılmaz. Texas evreleri ve CFTC maskeleri aynı matriste ayrı bağımlılıklar olarak
kalır; hava bölge genişletmesi vintage sorunu çözülmeden kapsamlı veri indirmesine dönüşmez.

Beklenen emek mevcut dosyalar için **2–4 gün**; yayın kanıtı/sağlayıcı beklemesi
bu tahmine dahil değildir. Sıfır gerçek model fit'i. Yeni veri edinimi gerekirse
önce küçük kapsam ve ücretsiz kullanım doğrulanır; ücretli servis otomatik açılmaz.

### C — Tek ayrıştırıcı pilot; başarıya göre daha fazla model değil

A ve B uygun olduğunda ilk yeni hipotez: **WASDE ülke revizyonları, eski üç dünya
özelliğinin ötesinde T+1 işlem kararına katkı veriyor mu?** Bu hipotez sicildeki
eski T+5 dünya oranı veya metin deneyinin tekrarı değildir.

- İki kol: aynı mevcut bilgi + dünya özellikleri + aynı rapor yaşı/eksiklik
  kontrolleri; müdahalede bunlara yalnız önceden seçilmiş ülke revizyonları eklenir.
- Aynı yayımlanmış sürümler, karar saati, kontrat, origin, giriş/çıkış, hedef,
  preprocessing, seçim ve refit. Saat düzeltmesi sadece müdahaleye uygulanmaz.
- İlk tarif: Ridge alpha=1/seed42/window1; geçmişe bağlı standartlaştırılmış
  **uygulanabilir giriş→çıkış log-return** hedefi. T+1 birincil; eski kapanış
  hedefiyle aynı deneymiş gibi gösterilmez. T+5 sonraki ayrı karardır.
- Aynı geçmiş validasyonda parametre ve varsa flat eşiği kuralları ilk hesaptan
  önce kilitlenir. İlk mekanizma testinde eşik taraması yapılmaz; sabit işaret
  kuralı ve sabit kontrat birimi kullanılır. Fiyat-MAE için seçilen shrinkage'ın
  işlem faydasını optimize ettiği varsayılmaz.
- Birincil ölçüm: tüm uygun karar günlerinde müdahale−kontrol net kontrat-birim
  PnL farkı. Flat ve sabit yönlerle de karşılaştır. Yıllık sonuç, drawdown,
  turnover, aktif oran, maliyet başa baş noktası; fiyat hatası yardımcıdır.
- Yılları aşmayan paired bootstrap: 10.000 tekrar, seed42, blok20 ve60. Maliyet
  ve risk/istikrar eşiği ilk sonuçtan önce sözleşmeye yazılır. Tarihsel inceleme
  tekrarlarının tamamını kapsayan bağımsız anlamlılık iddiası yok.
- Kontrat/giriş/maliyet kanıtı yoksa **bu işlem pilotu başlamaz**. İstenirse ayrı
  açıkça varsayımlı fiyat/proxy çalışması kaydedilir; onun başarısı işlem GO'su değildir.
- Aynı 2.006 origin/sekiz yıl/21-refit düzeni korunabilirse iki kol, tek ufuk için
  mevcut mekanik 338 fit ölçeğindedir. Yeni veri cohort'u netleşmeden bu kesin
  bütçe sayılmaz; engine planı tam fit sayısını çıkarıp dondurmalı. Tek CPU süreç,
  en fazla iki thread, ilk oturum 30 dakika/checkpoint duraklatmalı; otomatik grid yok.

Pozitif etki hem maliyet sonrası pratik eşiği hem istikrarı karşılıyorsa yalnız
kilitli ileri doğrulamaya geç. Aralık pratik etkiyi dışlıyorsa bu temsil kapatılır.
Belirsizse veri/bağımsız örnek sayısı sınırı açıklanır; aynı yıllarda yeni model
yarışması açılmaz. Bu aşamayı hazırlama/test emeği **2–3 gün**; koşul sağlanmazsa
fit bütçesi sıfır kalır.

### D — Kilitli ileri kayıt ve kanıt paketi

`research/prospective.py` ve `research/live.py` mevcut; yeni CSV tabanlı paralel
canlı sistem kurulmaz. Naive/EWMA kayıtları ve kaçırılmış günler önce denetlenir.
Zamanında oluşturulan tahmin, input manifesti ve kod/model kimliği değişmez;
gerçekleşmeler ve skorlar ayrı eklenir. Kaçırılmış tahmin geçmişe doldurulmaz.
İleri kayıt süreci güvenli veriyle aday model beklenmeden sürdürülebilir; bu planda
yeni servis/zamanlayıcı çalıştırılmış değildir.

2016–2023 ve görülmüş 2024+ tarihler yeni locked holdout olamaz. Daha önce gerçekten
incelenmemiş bir tarih kesimi ancak erişim/inceleme geçmişiyle gösterilebilirse
kullanılır; aksi halde holdout kilit sonrasındaki gelecektir. Ensemble, P(up) ve
interval ancak hedefe uygun geçmiş-only kalibrasyon, Brier/log-loss, kalibrasyon
eğrisi ve kapsama/genişlik doğrulamasıyla eklenebilir. Point forecast ağırlıklı
ortalaması kendiliğinden olasılık veya %80 interval üretmez.

Kurulum/rapor emeği **1–2 gün**, ileri performans kanıtının süresi ayrıca gerçek
takvim ve bağımsız olay sayısına bağlıdır. Böylece ilk teknik teslimat yaklaşık
6–11 çalışma günü ölçeğindedir; kaynak beklemeleri ve başarılı model garantisi hariç.

## 5. Ekli 10 günlük plandaki her başlığın karşılığı

| Ekli başlık | Düzeltilmiş uygulama |
|---|---|
| Gün1 baseline | Saklanan ortak-origin sonuçlarını doğrula; her modeli yeniden eğitme. Eksik LSTM tahminini eksik göster. MAE/RMSE/MAPE, üç durumlu flat ve aktif yön semantiği açık olsun; balanced accuracy tanımı sabitlensin. |
| Gün2 PIT | Mevcut package/upper-bound altyapısını kullan; saat ve CLI tazelik farklarını düzelt. Beş metadata alanı tek başına yeterli değil. |
| Gün3 USDA | B paketinde önce WASDE bölgesel, sonra FAS ülke bilgisi; eski toplamlarla aynı deney sayma. |
| Gün4 CFTC/crop/weather | Kaynak matrisiyle koşullu yeniden kabul; hepsinin bir günde tamamlandığı iddiası yok. |
| Gün5 eğri/makro/store | Kontrat verisi A'nın ön koşulu; FX zaten denenmiş. 50–100 özelliğe ulaşmak bir amaç değil; hipoteze gereken küçük temsil seçilir. |
| Gün6 yedi model yarışı | Çıkarıldı. Eski denemeler sicilden okunur; C'de yalnız tek mekanizma testi. |
| Gün7 ablation | Aynı origin ve aynı yaş/eksiklik kontrolüne karşı eşlenmiş katkı; sırayla ekleyip en iyi test sonucunu seçme. Tek negatif sonuçla bütün veri kaynağını silme. |
| Gün8 ensemble/belirsizlik | Koşullu sonraki iş; örnek ağırlıklar kullanılmaz. Öğrenme/kalibrasyon ayrımı ve dağılım hedefi gerekir. |
| Gün9 WF/holdout | Mevcut engine, olgun etiket, purge ve train-only dönüşümler korunur; incelenmiş tarihe yeni holdout etiketi verilmez. |
| Gün10 rapor/live | Mevcut sicil/ledger/Release/prospective yolunda manifest, tahmin, maliyet ve karar kaydı. Yeni üst dizinler veya ikinci eğitim motoru yok. |

## Başlangıç komutları ve teslimat sınırı

```bash
python ml/history.py validate
python ml/history.py check --feature wasde --horizon 1
python ml/history.py check --feature fas --horizon 1
python ml/history.py check --family ridge --horizon 1
python ml/history.py check --proposal output/proposal.json --json
```

`proposal.json` tam tarif ve frozen `scope_id` içerir; yer tutucu ile fit başlamaz.
Serbest metin eşleşmesi ilgili kanıtı buldurur, tek başına “denendi” kararı vermez.
Yeni veri/karar zamanı/cohort değişince yeni kimlik gerekir; eski cache taşınmaz.

Bu incelemede hazırlananlar: plan, 432 çıktı/altı tablo için doğrulama kaydı ve
ham WASDE/FAS alan mevcudiyeti incelemesi. Kaynak kabulü, model eğitimi, ham veri
değişikliği veya canlı yayın yapılmadı. Sonraki somut teslimat **A paketi ve ilk
WASDE kabul dosyasıdır**; bütün kaynaklar/model aileleri aynı anda açılmayacak.
