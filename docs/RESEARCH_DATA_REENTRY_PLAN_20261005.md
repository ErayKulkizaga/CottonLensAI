# Veri bütünlüğü, kaynakların yeniden değerlendirilmesi ve sonraki araştırma planı

5 Ekim 2026 — **inceleme sonucunda hazırlanmış uygulama taslağı**. Bu belge yeni
eğitim veya veri kabulü yapıldığını göstermez. Ürün amacı ve mevcut kararlar
[STATUS](STATUS.md) ve [işlem sözleşmesinde](TRADING_RESEARCH_CONTRACT_20261005.md)
kalır. İncelenen ek: `CottonLens_Model_Dogrulugu_Plan (1).md`; belge kimliği ve
bu incelemenin sayısal kanıtı [doğrulama kaydında](../research/evidence/plan-review-20261005.json).

**8 Ekim güncellemesi:** ilk teslimat, fiyat/seans denetimi, WASDE sayısal
mutabakatı ve [bölgesel T+1 deneyi](WASDE_REGIONAL_T1_RESULT_20261008.md) tamamlandı.
424 piyasa fit'i, aynı 1.254 origin'de seçilmiş D0/D1 Naive kazancı
−%0,5229 / −%0,0107: bu tarif %5 hedefini kurtarmadı. Eski veriler değişmedi.
WASDE kapsamını/modelini büyütmek otomatik sonraki iş değildir.

**9 Ekim bağımsız veri işi:** [Texas/NASS denetimi](NASS_REGIONAL_AUDIT_20261009.md)
311 mevcut raporda kondisyon ve 636 gelişim tablosunu ayırdı; iki revizyon ayrı
korundu. Ekim 46 raporla sınırlı, tam sezon varsayımı yapılamaz. Veri karantinada;
[30 Mayıs tek raporlu kontrol](NASS_CLOCK_CASE_20261009.md) tamamlandı: 22 hücre
eşleşti; tarihsel erişim saati doğrulanmadı. Sınırlı CDX araması tekrar edilmeyecek.
[Texas T+1 ön kaydı](NASS_REGIONAL_T1_PREREGISTRATION_20261009.md) tamamlandı:
ortak 2.006 origin ve 676 piyasa + 1 sentetik fit bütçesi; ulusal kondisyon ortak.
[Yürütme/sonuç](NASS_REGIONAL_T1_RESULT_20261009.md) tamamlandı: 676 piyasa+1
sentetik fit; seçilmiş D0/D1 Naive kazancı −%0,0274/−%0,2556. Sabit tarifin %5
hedefi iki gecikme/blokta dışlandı; Naive korunur, yeni grid yok. Vintage/saat
kabulü kapalı; FAS engeli/gece görevi değişmedi. Sonraki operasyon kontrolü
mevcut ileri görevde ilk zamanında yayının makbuz zinciridir.

**FAS işi — dış kanıt bekleyen ülke bilgisinin kabul denetimi; sıfır fit.**
9 Ekim: [değer/sürüm/saat koruması ve ortak-origin adapter'i](FAS_REVIEWED_ALIGNMENT_20261009.md)
sentetik paketle tamamlandı. Gerçek country-feature snapshot'ları ve tarihsel
erişim kabulü hazırlanmadı; tek ön kayıt/eğitim aşaması hâlâ kapalı.
[Karantina ülke hazırlığı](FAS_COUNTRY_PREPARATION_20261008.md) ayrıca tamamlandı:
752 hafta × dört kod, seviyeler/paylar/dört haftalık akışlar. Negatif stok
bileşeninin bulunduğu haftada pay üretilmez; kaynak sürümü/saat kabulü ve
eğitim adapter'i açılmadı. Dış kanıt beklerken aynı PDF/form araması tekrarlanmaz.
[Yeni okunabilir kanıt](../research/evidence/fas-country-field-audit-20261008.json):
15 yıllık kaynakta 27.887 ham satır; aynı pazarlama yılı kuralıyla 26.313 satır,
752 hafta ve 64 ülke kodu. Ulusal satış/sevkiyat toplamları eski tabloyla tam
eşleşir; commitment = outstanding + accumulated eşitliğinde sıfır fark vardır.
18 girdinin önce/sonra hash'i eşit. 3.121 negatif net satış satırı korunur;
65 haftada ulusal net satış toplamı sıfır/negatif olduğundan bu payda ile ülke
payı oluşturulmaz. Eksik ülke satırları sıfır sayılmaz.
Commitment, outstanding ve accumulated toplamına eşit olduğundan bunlar üç
bağımsız bilgi kaynağı diye sayılmaz; yeni temsil bu cebirsel tekrarı gözetir.

**Ülke raporu kontrolü tamamlandı, kabul tamamlanmadı:** [iki haftalık mutabakat](FAS_COUNTRY_REPORT_REVIEW_20261008.md) dört ülke/üç stok alanında 22/24 eşleşme buldu. 28 Mayıs Çin/Pakistan accumulatedExports farkı −55/−53 balya; ulusal toplam eşitliği bunları gizliyor. Değer veya ±50 sınırı değiştirilmedi. Haftalık akışlar, tam tarihsel katalog ve sürüme bağlı erişim doğrulanmadı. [Alt sınıf kontrolü](FAS_MAY28_VERSION_AUDIT_20261008.md) bu iki farkın yuvarlama varsayımıyla uyumunu gösterdi, nedeni kanıtlamadı. Şimdi tek ayırıcı test Çin'in 28 Mayıs özgün tam-balya accumulated hücresidir; yalnız güncel API/aynı PDF aramasını büyütmek yeterli değil. Kanıt olmadan aşağıdaki ön kayıt/eğitim aşamalarına geçilmez.

[Tarihsel form erişim denetimi](FAS_HISTORICAL_REFERENCE_20261008.md) dört kod için 2020 tanığı sağladı; özgün Çin sayısal hücresi **dış kanıt bekliyor**. Aynı formları/boş düzeltme sorgusunu tekrarlamak veya anahtar almak bu ilk sürüm boşluğunu çözmez. Aşağıdaki eğitim sırası bu nedenle açılmadı.

Sonraki üç teslimatın sırası ve durma koşulu:

1. **FAS kabul kanıtı PR'ı:** checksum bağlı ülke kod kataloğu; mevcut iki
   rapor örneğinde ülke bazlı outstanding/commitment/sevkiyat değerleri ve toplam
   kapsamı; hafta sonu, pazarlama yılı ve birimler; kullanılan belirli sürümün
   en geç erişim kanıtı. Bir ülkenin satırının yokluğu sıfır kabul edilmez.
   Katalog veya rapor/sürüm kanıtı yoksa ilgili alan bilinmeyen/karantinada kalır;
   model eğitimi açılmaz. Önce mevcut dosyalar kullanılır; ağ/veri edinimi gerekirse
   kapsam ayrıca kaydedilir. Tahmini emek 1–2 çalışma günü; kaynak beklemesi hariç.
   İki rapor örneği bütün 752 haftanın doğrulaması değildir; kabul edilen kapsam
   kanıtın gerçekten kapsadığı ülke/alan/sürüm/tarihlerle sınırlandırılır.
2. **Yalnız kabul yeterliyse tek ön kayıt PR'ı:** ulusal toplamların ötesinde
   ülke dağılımı/bekleyen satışların katkısını sınayan tek hipotez. Ülke/alan
   sayısı kapsama ve ekonomik gerekçeye göre, OOS sonuç görülmeden sınırlandırılır.
   Model, kaynak ve hedef aynı anda değiştirilmez; aynı origin, hedef, karar saati,
   eski ulusal bilgi kontrolü ve aynı eksiklik kontrolleri korunur. Sicil kontrolü,
   birincil ölçü, pratik etki sınırı ve tam fit bütçesi hesaplanıp kilitlenmeden
   yürütme olmaz. Bu belge yeni deney kimliği veya eğitim izni oluşturmaz.
3. **Sonuç PR'ı:** yalnız kilitlenen pilot; ham/seçilmiş sonuçlar, yıllar,
   paired blok20/60 aralıkları ve bağımsız hesap. Negatif veya belirsiz sonucu
   daha büyük grid'e dönüştürme yok. Tarihsel saat kanıtı yetersizse ancak ayrı
   varsayım duyarlılığı olarak açıkça kararlaştırılabilir; gerçek PIT kanıtı denmez.

Her PR ayrı branch, dar diff, uygun test/CI ve checksum doğrulamasıyla GitHub'a
gider. Kod/küçük kanıt Git'te, büyük ham veri/checkpoint Release'te kalır.
Eski sicil/kanıt değiştirilmez; tamamlanan denetim ek kayıt olur. Gerçek zamanlı
aday üstünlüğü bulunmadan eski tarihleri yeni holdout diye yeniden kullanmayız.

**Kapsam açıklaması:** kullanıcı projeyi araştırma/ispat olarak tanımladı; gerçek
alım/satım yapılmayacak. Aşağıdaki broker/giriş/maliyet koşulları yalnız gerçek
veya net işlem kazancı iddiasına aittir, tahmin araştırmasını bloke etmez.
[Fiyat/seans denetimi](PRICE_SEMANTICS_AUDIT_20261005.md) yeni kanıtı ve sınırları kaydeder.

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

Seçilen araştırma amacı vadeli piyasada yön ve pozisyon fikrini incelemektir;
gerçek işlem yapılmaz. Güncel sözleşmede birincil kanıt tahmin katkısıdır;
PnL yardımcı simülasyondur. Net işlem iddiası ayrıca fiyat/kontrat/maliyet kanıtı
ister. Fiyat programı ile işlem simülasyonunun hedef ve kayıtları karıştırılmaz.

## 1. Önceden yapılanlar: yeniden başlatılmayacak işler

**5 Ekim inceleme snapshot'ı — VERIFIED:** Sicil doğrulandı: 122 deney/kontrol/tarihçe kaydı, 430 tarif,
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

**5 Ekim'de saptanıp ilk teslimatta düzeltilen iki nokta:**
Yeni sürümlü 00:15 UTC politika ve CLI/engine tazelik eşitliği uygulanmıştır;
[teslimat kanıtı](TRADING_WASDE_DELIVERY_20261005.md). Aşağıdaki iki paragraf
bulgunun tarihsel halidir; aynı düzeltme yeniden yapılmaz.

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

### A — Tamamlanan tahmin/işlem karşılığı denetimi

İlk teslimatta tamamlanan iş, mevcut sözleşmedeki **eğitimsiz T+1 analizidir**: saklanan kontrol
ve zamanlama müdahalesi tahminlerinin `sign(selected_return)` pozisyonunu, aynı
origin'lerde sonraki kayıtlı Cotton open→close hareketiyle eşleştir. Ham işaret
ayrı tanısaldır. Flat/long/short referansları, aktif oran, yıl sonuçları ve başa
baş çift yön maliyet raporlanır. CT=F sonucu proxy kalır; open'ın karar sonrasında
uygulanabilir olduğu ve kontrat kimliği kanıtlanmadan net işlem üstünlüğü denmez.

Sonuç ve sınırlar [ilk teslimatta](TRADING_WASDE_DELIVERY_20261005.md) ve
[fiyat/seans denetiminde](PRICE_SEMANTICS_AUDIT_20261005.md) kayıtlıdır. Aynı
proxy analizi tekrar yapılmaz. Gerçek kontrat/uygulanabilir fiyat kanıtı hâlâ
net işlem iddiasının ayrı koşuludur; FAS kaynak içerik denetimini engellemez.

### B — Önce WASDE, ardından FAS için yeni veri paketi

WASDE tablo, revizyon, sayısal doğrulama ve sınırlı T+1 pilotu tamamlandı;
[negatif sonuç](WASDE_REGIONAL_T1_RESULT_20261008.md) korunur. FAS ülke/commitment
adayında sıfır-fit alan denetimi ve karantina ülke aritmetiği tamamlandı; kabul
ve Cotton'a hizalanmış eğitim paketi henüz tamamlanmadı. İlk pakette bütün
kaynaklar aynı anda birleştirilmez.
Her pakette kaynak/kolon kapsamı, veri kaybı, revision/availability kanıtı, orijinal
hash koruması ve kullanım statüsü raporlanır. Salt parser başarısı eğitim kabulü
sayılmaz. Texas evreleri ve CFTC maskeleri aynı matriste ayrı bağımlılıklar olarak
kalır; hava bölge genişletmesi vintage sorunu çözülmeden kapsamlı veri indirmesine dönüşmez.

Beklenen emek mevcut dosyalar için **2–4 gün**; yayın kanıtı/sağlayıcı beklemesi
bu tahmine dahil değildir. Sıfır gerçek model fit'i. Yeni veri edinimi gerekirse
önce küçük kapsam ve ücretsiz kullanım doğrulanır; ücretli servis otomatik açılmaz.

### C — Tamamlanan WASDE pilotunun kapsamı ve kapanış

İlk taslaktaki giriş→çıkış/net PnL deneyi yürütülmüş sayılmaz. Kullanıcının
araştırma/ispat açıklamasına göre ayrı [ön kayıt](WASDE_REGIONAL_T1_PREREGISTRATION_20261008.md)
ile close→next close T+1 hedefinde D0/D1, numeric/mask dört kol çalıştırıldı.
Tarif, 424 fit ve karar sınırları ilk fit'ten önce kilitlendi. Bu çalışma eski
üç dünya özelliğine karşı yalnız revizyon ekleme deneyi olarak sunulmaz.

[Sonuç](WASDE_REGIONAL_T1_RESULT_20261008.md): %5 pratik hedef bu sabit tarifte
desteklenmedi; Naive korunur. Sonuç net işlem becerisi, bütün WASDE temsillerinde
sinyal yokluğu veya gerçek erişim/vintage onayı değildir. Yukarıdaki FAS kabul
denetimi dışında yeni eğitim, model araması veya otomatik hedef değişimi yoktur.

### D — Kilitli ileri kayıt ve kanıt paketi

9 Ekim görev düzeltmesi dağıtıldı: [sabit kaynak/yayın penceresi](FORWARD_RUNTIME_WINDOW_20261009.md),
gündüz gerçek çağrı exit 0; gece yayını ve tahmin becerisi henüz doğrulanmadı.
Mevcut görev/tetikleyiciler korunur; 03:35 Türkiye saatinde bu sohbetin devam
kontrolü aktiftir. Genel sistem güç ayarları ve araştırma karar saati değişmedi.

9 Ekim sıfır-fit denetimi tamamlandı: altı missing origin, sıfır yayımlanmış
tahmin; kesim-öncesi yerel makbuz yok. [Kanıt ve durum komutu](FORWARD_EVIDENCE_AUDIT_20261009.md).
Sonraki operasyon kontrolü mevcut görevde tek gerçek 00:05→00:20 UTC zinciridir;
altyapı kurulmuş olması ileri performans birikimi sayılmaz.

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
