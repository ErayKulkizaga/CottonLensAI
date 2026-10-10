# Güncel kararlar — 10 Ekim 2026

**Gerçek eğri T+1 testi tamamlandı:** [kayıtlı sonuç](CONTRACT_CURVE_RESULT_20261010.md).
252 piyasa +1 ayrı sentetik fit, 749 ortak origin/kol; 12/12 geçmiş ağırlık0,
seçilmiş bütün kollar Naive. Ham numeric D0/D1 Naive kazancı −%2,711585/
−%2,282736; ham kaynak katkısı −%1,325738/−%1,011608 ve küçük pozitif
etkiyi dışlamayan aralıklar. Sıfır paired aralığı kaynakta bilgi yokluğu değildir.
253 native çıkarım, 12 seçim, 16 aralık ve exact rapor/CSV replay geçti.
±5 vade-değişim gözleminden uzak584 origin'de de ham kollar negatif.
Saat/vintage varsayımı ve 6/8 gate sınırı korunuyor; Naive değişmez.
Tek sonraki iş ufuk farkını izole eden T+5 ön kayıt incelemesi; yeni fit/grid yok.

## Önceki kararlar — değişmeyen kanıt zinciri

Aşağıdaki ön kayıt ve araştırma kararları tarihsel aşamalardır; son durum üsttedir.

**Tamamlanan deneyin ön kayıt kararı:** [gerçek vade farkı T+1](CONTRACT_CURVE_PROTOCOL_20261010.md).
Kullanıcının devrettiği kararla saat/ilk vintage varsayımlı duyarlılık seçildi;
tarihsel kaynak kabulü açılmadı. Geçmiş iç blok kaynak uygunluğu 2021–2023'te
749 ortak origin bırakıyor; bütün Cotton history korunuyor. Dört eşleşmiş
kol, sabit Ridge alpha1, 252 piyasa +en fazla1 ayrı sentetik fit; yeni grid
yok. Üç yıl 6/8 kapısını ölçemez veya değiştiremez. Kod/test/veri/ortam ve
karar sözleşmesi GitHub'da sıfır-fit ön kayda bağlandıktan sonra çalıştırılır.

**Sıfır-fit bilgi elemesi tamamlandı:** [mevcut gerçek vade fiyatları](INFORMATION_ADMISSION_20261010.md).
1.000 eski girdi rehash; 997 AMS tablosunda 9.970 gerçek kontrat fiyatı,
2016–2019 kapsamı yok. Eski spot-basis deneyi bu farkı sınamadı. UTC/ilk sürüm
kabulü hâlâ yok, model-eligible 0; yeni fit/veri edinimi 0. Sıradaki tek iş
bu alan için kaynak/clock/cohort kabul kararı ve tek eşleştirilmiş ablation
ön kayıt önerisi. Yeni model/grid veya FAS kabulü kendiliğinden başlamaz.

**Yeni sonuç doğrulandı:** [sabit price_delta loss karşılaştırması](PAIRED_PRICE_LOSS_RESULT_20261010.md).
48 piyasa + ayrı 6 sentetik fit, 1.008 ortak T+1 origin. MAE–MSE katkısı
%1,458009; blok20/60 alt sınırları +%0,636274/+%0,991865. Birincil katkı
pozitif, fakat MAE Naive kazancı −%0,488298, yön %48,4127, dönem 3/8;
pratik hedef karşılanmadı. 54 kayıtlı çıkarım, altı aralık ve rapor/CSV birebir
replay doğrulandı; GitHub'dan yeni indirilen 579 üyeli sonuç arşivinde aynı
54 çıkarım/altı aralık/exact replay tekrar geçti. [Teslimat makbuzu](../research/evidence/paired-price-loss-delivery-20261010.json).
Naive korunur;
bu sabit tarif genel MAE/grid veya yeni model aramasına dönüştürülmez.
Sonraki bilimsel karar, teslimattan sonra en fazla 2–3 bilgi sorusunun mevcut
sicil/kaynaklarla sıfır-fit kabul elemesi; yeni fit/veri kabul izni yoktur.

**Karar eki aşamasındaki önceki iş sırası; yukarıdaki sonuç durumu günceldir:**
[10 Ekim karar ekine](RESEARCH_PRIORITY_ADDENDUM_20261010.md)
göre PR #27'nin mevcut kayıtlı deneyini **sıfır yeni fit ile sonuçlandırmak**.
48 piyasa + ayrı 6 sentetik fit yerelde tamamlandı; bu incelemede 54 kaydedilmiş
model çıkarımı, training-only dönüşümler/H5 olgunlaşması ve ortak-origin
fiyat-MAE/yön hesabı doğrulandı. MAE kolu −%0,488298 Naive kazancı, %48,4127
yön, 3/8 dönem; MSE −%1,975104. Loss katkısı pratik Naive üstünlüğü değildir.
Bağımsız altı blok aralığı, portable rapor/CSV replay ve sonuç arşivi teslimatı
henüz kapanmadı; yeni bilimsel sınıf/başarı ilan edilmiyor. #25 ve #26 tamamlandı;
yeniden koşulmaz. Bu docs-only ek sicil/trials veya eski kanıtları değiştirmez.

**Yeni operasyon engeli:** [gece yayın penceresi denetimi](FORWARD_WINDOW_AUDIT_20261010.md).
00:05/00:20 UTC makbuzları yok; yayın 0, eski altı missing ve 126-origin kilidi
korunuyor. Sabit 226 ML dosyası ve 115 store dosyası doğrulandı. OS uyku olayı
22:51–06:59 UTC aralığını gösteriyor; daha sonraki 07:05 görev isteği
`0x800710E0` ile reddedilmiş. Gece tetikleyicisinin kesin hata nedeni bilinmiyor;
Operational günlük kapalı. Görev/otomasyon/güç ayarları değişmedi, yeni fit/skor
yok. Sonraki operasyon işi mevcut interactive görev reddini incelemek;
zamanında yayın veya öğrenilmiş model becerisi iddia edilmez.

**PR #27 ön kayıt aşamasındaki tarihsel durum; yukarıdaki yerel yürütme durumu günceldir:**
[eşleştirilmiş fiyat-loss protokolü](PAIRED_PRICE_LOSS_PROTOCOL_20261010.md),
[PR #27](https://github.com/ErayKulkizaga/CottonLensAI/pull/27).
İki CPU XGBoost kolunda aynı price_delta, 24 özellik, olgun kohort ve eski
seed/ağaç sayıları; yalnız MSE/MAE değişir. Bütçe 48 piyasa fit'i / 2.688 ağaç,
ayrı 6 sentetik kontrol; 1.008 ortak origin. Yeni arama veya küçültme yok.
19 profil testi, 74 ortak kontrol, geniş ML 1.004 passed/3 skip ve Ruff geçti;
kod commit'inin push/PR CI'sı başarılı. Hazırlama/pilot-plan sıfır fit doğrulandı.
Sıfır-fit kimlik/veri/ortam kanıtı kamu Release'ine bağlanır; eşleşmeden pilot
başlamaz. Eski düşük iteration'lara koşullu sonuç; genel MAE veya bütün
kaynaklarda sinyal yokluğu sınaması değildir. Naive ve gece görevi korunur.

**Son sıfır-fit yeniden kurma:** [eski loss seçiminin gerçek kapsamı](LEGACY_LOSS_LINEAGE_RESULT_20261010.md).
6.648 makbuz / 128 aday / 8 geçmiş seçim; 1.008 ortak T+1 origin.
Seçilmiş eski program −%0,611017 Naive kazancı, %46,9246 yön, 1/8 dönem;
iki paired aralığın üst sınırı negatif. Altı dönemde MAE, üçünde doğrudan
price_delta+MAE seçilmişti. “MAE hiç denenmedi” yanlış; aynı parametreli
loss ablation'ı ise yok. Öğrenilmiş model payload'ları eksik olduğundan
çıkarım replay'i iddia edilmez; kaynak, kohort, seçim ve tahmin/kayıp kuruldu.
Genel MAE grid'i tekrarlanmaz. Loss'u ayırmak gerekirse yalnız ayrı ön kayıtlı
48-fit eşleştirilmiş kontrol; eski Haziran seçimleri Ocak origin'lerine taşınmaz.
Yeni fit 0; mevcut Naive ve gece görevi korunur.

**Son tamamlanan piyasa postprocessing testi:** [fiyat-MAE kalibrasyonu](PRICE_MAE_CALIBRATION_RESULT_20261010.md).
169 olgun-geçmiş skaler güncelleme, yeni model fit'i 0, aynı 2.006 T+1 origin.
Kalibre ham/seçilmiş Naive kazancı −%3,73685/−%0,02608; seçilmiş kontrole
katkı yalnız %0,001362 ve iki paired aralık sıfırı içeriyor. Naive'ye göre
üst sınırlar %0,23134/%0,18585, %5 hedefinin altında. 169 optimum/maturity,
169 model çıkarımı, 8 seçim ve 12 aralık bağımsız doğrulandı; rapor/CSV replay
birebir. **Global bias kalibrasyonu tarifini büyütme; Naive korunur.**
Tam koşullu quantile/loss etkisi veya tüm kaynaklarda bilgi yokluğu sınanmadı.
MAE/absolute-loss kapsamı incelemesi yukarıda tamamlandı; yeni eğitim
kendiliğinden başlamaz.

**Yeni sıfır-fit sonuç:** [iç doğrulama seçim kararlılığı](SELECTION_STABILITY_RESULT_20261010.md).
160 eski karar / 1.440 iç makbuz / 30.240 geçmiş tahmin yeniden kuruldu.
Enjekte koşulunda leave-one-block-out 59/80 kararı değiştirir; blok20 kayıtlı
ağırlığı medyan %56,19 sıklıkla seçer. Blok60'ta %98,20, fakat 63 satırda
yalnız dört başlangıç olduğundan bu güç kanıtı değildir. Kaynak/veri ve eski
ağırlıklar değişmedi; 5.960 kullanılan girdi orijinal envanterle eşleşti.
Naive korunur; eski küçültme kaldırılmaz. Bu tanının sonraki çarpan kalibrasyonu
yukarıda tamamlandı. Yeni veri, model fit'i veya canlı model yayını yapılmadı.

**Son kontrol:** [zayıf sinyal sonucu](WEAK_SIGNAL_RESULT_20261009.md).
3.380 sentetik fit / 160 çıktı / 40.120 satır tamamlandı; yeni piyasa fit'i 0.
Ham/seçilmiş ortalama MAE kazancı %1,9551/%1,8752; oracle %3,6054.
Kilitli karar RAW_ONLY_RECOVERS: medyan oracle koruması %59,8533/%49,5611.
Seçilmiş sonuç %50 eşiğini yalnız 0,4389 yüzde puan kaçırır; 10/10 seed pozitiftir.
Null'da pratik yanlış pozitif 0/10, küçültme on seed'de de zararı azaltır.
Bu, küçültmenin piyasa başarısızlığını açıkladığı veya piyasa sinyali bulunduğu
kanıtı değildir. 3.380 çıkarım / 160 geçmiş seçim / 120 aralık bağımsız replay edildi.
949 girdi, 6.646 eski deney dosyası, 42 eski kanıt, 146 sicil / 440 trial korunur.
Kaynak/ön kayıt değişmedi; 18 yeni kontrol, yerel 937 ML testi / 3 skip;
ön kayıt GitHub CI'da 947 ML testi / 12 skip ve diğer üç job doğrulandı.
**Naive korunur; ardından yapılan seçim kararlılığı analizi yukarıda tamamlandı.**
Ön kayıt ve ilk sıfır-fit v1 arşivi değiştirilmez.

**Son araştırma işi:** [Texas/NASS T+1 sonucu](NASS_REGIONAL_T1_RESULT_20261009.md).
676 piyasa + ayrı 1 sentetik fit, 32 çıktı, 8.024 tahmin tamamlandı; aynı 2.006
origin/kol. Seçilmiş sayısal D0/D1 Naive kazancı −%0,0274/−%0,2556; en yüksek
%95 üst sınır %0,2274, %5 hedefinin altında. Texas'ın kontrol üzerine küçük
katkısı belirsiz; kaynakta evrensel sinyal yokluğu çıkarılmaz. **Naive korunur;
aynı tarifte yeni grid/fit açılmaz.** 676 çıkarım, 32 seçim, 24 aralık bağımsız
replay edildi; 937 ML testi/3 skip. Eski ön kayıt ve 949 girdi değişmedi.
Saat/vintage varsayımsal, tarihsel kabul kapalı; gece görevi değişmedi.
Sonraki operasyon kontrolü mevcut görevde ilk zamanında yayının makbuz zinciri.

**Son veri işi:** [NASS 30 Mayıs 2023 saat kontrolü](NASS_CLOCK_CASE_20261009.md).
22 hücre (21 sayı + 1 NA) TXT/PDF'lerde eşleşti; Texas PDF'si yalnız planted'ı
ayrıca doğrular. Schedule/PDF/HTTP metadata tarihsel erişim tanığı yapılmadı.
Üç sınırlı CDX sorgusu timeout/503; tekrar arama yok. Sıfır fit/kabul, 949 eski
girdi değişmedi. Ayrı varsayım-duyarlılığı ön kaydı yukarıda kilitlendi;
yürütme kodu ve kimlik doğrulaması tamamlanmadan eğitim başlamaz.

**Önceki veri işi:** [Texas/NASS bölgesel rapor denetimi](NASS_REGIONAL_AUDIT_20261009.md).
311 mevcut rapor/949 girdi doğrulandı; Texas kondisyonu ve 636 gelişim tablosu
karantina paneline ayrıldı. İki gerçek önceki-hafta revizyonu özgün kayıtları
değiştirmeden korunur. Ekim yalnız 46 raporda var; tam sezon kabul edilmez.
Yeni fit 0, tarihsel erişim kabulü kapalı; tek raporlu saat kontrolü yukarıda
tamamlandı. Gece görevine ve otomasyonuna dokunulmadı.

**Çalışan kaynak:** [İleri yayın görevi düzeltmesi](FORWARD_RUNTIME_WINDOW_20261009.md).
Mevcut görev 226 dosyalık frozen source `44f3d16f…` kullanır; gerçek gündüz
çağrısı exit 0 ve quote makbuzu doğrulandı. Yayın penceresi ağ/skor/mirror'dan
ayrıldı; yavaş yazım sonrası saat tekrar denetlenir. Eski altı origin missing,
yeni zamanında yayın **0**, yeni fit **0**. WakeToRun açık; sistem güç politikası
değişmedi. 03:35 Türkiye saatli devam kontrolü aktif; gece yayını henüz kanıtlanmadı.

**Son operasyon kanıtı:** [İleri kayıt denetimi](FORWARD_EVIDENCE_AUDIT_20261009.md)
08:09 UTC snapshot'ında altı origin'in altısı missing, yayımlanmış tahmin **0**.
Her origin'de kesim-öncesi yerel makbuz yok. Durum komutu artık kaynak/girdi/saat
ve kilitli çıktıyı doğrular; salt dosya sayısı kanıt sayılmaz. Eski kayıtlar
değişmedi, yeni fit 0; o denetim anında görev ana checkout'u kullanıyordu. Sonraki tek operasyon
kontrolü gerçek 00:05 yakalama →00:20 yayın zinciridir; saat/backfill değiştirilmez.

**Son yazılım kontrolü:** [FAS sürüm/değer/saat sözleşmesi ve ortak-origin
hizalaması](FAS_REVIEWED_ALIGNMENT_20261009.md) sentetik paketlerle doğrulandı.
Yeni piyasa fit'i **0**. İki kol aynı eksiklik göstergelerini taşır; seçilen
kaynak sürümü/saatinin izi korunur. Gerçek tarihsel kaynak kabulü kapalı;
erişimi kanıtlanmış snapshot ve sayısal inceleme sonrası tek ön kayıt kalır.

**Son veri hazırlığı:** [FAS ülke karantina paneli](FAS_COUNTRY_PREPARATION_20261008.md)
3.008 satır/752 hafta; dört kodda eksik satır yok, 413 negatif satış korunur.
29 Temmuz 2021 kod2230 bekleyen satış −76: ham değer korunur, o haftanın stok
payları tanımsız bırakılır. Yeni fit **0**, saat politikası unset ve kabul kapalı.
CPU eğitim motoru çalışıyor; bu ülke pilotunun engeli FAS tarihsel kaynak kabulüdür. Özgün
tek hücre bütün tarihçeyi kabul ettirmez; sonrasında adapter/test ve ön kayıt gerekir.

Bu dosya ve [işlem araştırması sözleşmesi](TRADING_RESEARCH_CONTRACT_20261005.md) güncel karar kaynaklarıdır. `archive/` belgeleri tarihsel tanıklık/kanıttır; eski “sonraki deney” talimatları etkin değildir.

- Son çalışma: **bölgesel WASDE T+1 deneyi tamamlandı:** 424 piyasa fit'i + ayrı 1 sentetik kontrol, dört kol × 1.254 origin, 5.016 tahmin. Seçilmiş sayısal D0/D1 Naive kazancı −%0,5229 / −%0,0107; iki blokta da üst sınır %5'in altında. Ham sayısal tahmin −%10,2134 / −%9,3435, 0/5 yıl kazanıyor. Kilitli karar: **FIXED_RECIPE_BELOW_PRACTICAL_GOAL; Naive korunur, aynı program büyütülmez.** 424 model çıkarımı ve 20 geçmiş seçim bağımsız doğrulandı; 2.682 eski dosya değişmedi. Kaynakta evrensel sinyal yokluğu veya gerçek saat onayı çıkarılmaz. [Sonuç, mekanizma ve sınırlar](WASDE_REGIONAL_T1_RESULT_20261008.md).
- Tamamlanan [ön kayıt](WASDE_REGIONAL_T1_PREREGISTRATION_20261008.md) değişmedi. Yeni kaynak/model denemesinden önce sicil kontrolü ve bu negatif sonucun kapsamı incelenir; sonraki fit veya canlı yayın otomatik başlatılmaz.
- Son sıfır-fit iş: [tarihsel FAS kod tanığı](FAS_HISTORICAL_REFERENCE_20261008.md). 17 Nisan / 1 Temmuz 2020 arşiv formlarında dört ülke kodu ve 1404 etiketi eşleşti; HTML byte digest’leri CDX ile aynı. Sürekli/tüm tarihsel katalog, birim ve ilk yayın sayısal hücresi doğrulanmadı. Tek 4 Haziran düzeltme görünümü yalnız başlık verdi; bu revizyon yokluğu değildir. Özgün Çin hücresi **BLOCKED_EXTERNAL_EVIDENCE**; aynı form/PDF araması tekrarlanmaz, sıfır fit ve kabul kapalı. [Yayıncıya hazır tek-hücre talebi](FAS_HISTORICAL_REFERENCE_20261008.md#karar-ve-tek-eksik-girdi); mesaj gönderilmedi.
- Önceki sıfır-fit iş: [28 Mayıs sürüm/yuvarlama kontrolü](FAS_MAY28_VERSION_AUDIT_20261008.md). Çin/Pakistan değerleri ayrı alt sınıfların varsayımsal yuvarlama aralıklarıyla uyumlu; hesaplama yöntemi/ilk sürüm nedeni kanıtlanmadı. Mevcut ±50 doğrudan kontrol başarısız ve eğitim kabulü kapalı kalır. Eski USDA adresleri şu anda 404; bu geçmişte yayın yokluğu değildir. Sonraki tek ayırıcı test Çin'in 28 Mayıs özgün as-issued tam-balya hücresidir; güncel API bunu tek başına çözmez. 8 girdi, 24 eski kanıt, 134 önceki kayıt ve trials korunur; sıfır fit.
- Önceki sıfır-fit iş: [FAS ülke raporu kontrolü](FAS_COUNTRY_REPORT_REVIEW_20261008.md). İki haftada dört ülke × üç stok alanından 22/24 eşleşti; 28 Mayıs 2020 Çin/Pakistan accumulatedExports farkları −55/−53 balya, sabit ±50 sınırının dışında. İkinci PDF okuyucusu doğruladı; nedeni henüz belirlenmedi, veri ve tolerans değiştirilmedi. 4 Haziran GovDelivery kopyasında 12/12 eşleşme üçüncü bağımsız hafta veya ilk sürüm/saat kanıtı değildir. Kaynak kabulü kapalı; sonraki tek iş 28 Mayıs özgün yayın sürümü karşılaştırması. Sıfır fit, 10 eski girdi ve 21 eski kanıt hash'i aynı.
- Önceki sıfır-fit iş: [FAS ülke alanı denetimi](../research/evidence/fas-country-field-audit-20261008.json). 15 kaynaktaki 27.887 ham satırdan 26.313 ülke/hafta satırı ve 752 hafta; 64 kod, ulusal akış toplamları eski tabloyla eşit, commitment özdeşliğinde sıfır fark. 18 girdinin hash'i değişmedi; 3.121 negatif satış ve eksik ülke satırları korunur. 65 sıfır/negatif ulusal net satış haftası ülke payı için uygun payda değildir. Ülke adları, ülke bazlı rapor değerleri ve tarihsel saat/vintage doğrulanmadı; eğitim kabulü yok. Sonraki iş ve üç PR sınırı [güncellenen planda](RESEARCH_DATA_REENTRY_PLAN_20261005.md).
- Önceki tek raporlu saat kontrolü: 11/14 Aralık 2018 repostunda Cotton değerleri aynı; ilk karar kesiminden önce erişim kanıtı kurulamadı. [Kanıt](WASDE_CLOCK_CASE_20261008.md).
- Önceki resmî “as reported” mutabakatı: 95 rapor, 950 temel hücre, oran/revizyonlarla 2.470 kontrol; sıfır fark, 117 bilinmeyen revizyon korunur. CSV geçmişinin teslim zamanı rapor saatinden farklıdır; `ReleaseTime=12:00` erişim kanıtı sayılmadı. [Kanıt](WASDE_AS_REPORTED_20261008.md); [PDF/XML ve kabul koruması](WASDE_REGIONAL_VERIFICATION_20261008.md).

- Kullanıcı açıklaması: **araştırma/ispat projesi**, gerçek alım/satım yok. ICE Cotton No. 2 araştırma referansı; tahmin katkısı birincil, yön/pozisyon ve PnL yardımcı simülasyon. Broker/komisyon bilgisi tahmin araştırmasını engellemez. Eski fiyat başarı eşikleri korunur.
- Fiyat/seans denetimi tamamlandı: 268/2.006 hedef `close` high–low dışında. Tarihi doğrulanan 997 USDA raporunun 996'sında ilk vadeli fiyat Close ile eşleşiyor; kapsamdaki 136 aralık-dışı Close'un tamamı eşleşiyor. 20 kontrat değişimi görüldü. Aynı-kontrat yollarında ham müdahale MAE kazancı T+1 −%1,52, T+5 −%3,22; yalnız roll temizlemek kayıtlı başarısızlığı açıklamıyor. Hiçbir fiyat değiştirilmedi. [Kanıt, saat karşılığı ve sonraki tek kontrol](PRICE_SEMANTICS_AUDIT_20261005.md).
- İlk uygulama teslimatı: sürümlü 00:15 UTC yayın hizalaması ve CLI/engine tazelik eşitliği düzeltildi; kayıtlı 2.006 T+1 origin'de sıfır-fit işlem proxy analizi tamamlandı. Seçilmiş müdahalenin 251 işlemi yalnız 2017'de; kontrol katkısı belirsiz. Ham sonuç ayrı tanıdır, strateji seçimi değildir. 95 rapor/26 alanlı bölgesel WASDE aday paketine geri alındı; erişilebilirlik/vintage doğrulanmadan eğitime kabul edilmedi. 690 eski dosyanın hash'i değişmedi. [Sonuçlar, kanıt ve tek sonraki iş](TRADING_WASDE_DELIVERY_20261005.md).
- Mevcut fiyat programında T+1 ve T+5: **Naive korunur**. %5 fiyat-MAE, %53/%55 yön ve 6/8 tam-yıl kazanımı eşikleri değiştirilmedi.
- Önceki tamamlanan zamanlama çalışması `research-availability-clock-pilot-v1`: 676 fit, 32 yıllık çıktı, 8.024 tahmin satırı. Daha güncel DXY/WTI bilgisi pratik hedefi kurtarmadı. [Sonuç ve sınırlar](AVAILABILITY_CLOCK_RESULT_20261005.md).
- Zamanlama varsayımına bağlı bu çalışma üretim erişilebilirliği kanıtı değildir. Yeni kodla eski cache devam ettirilmez.
- Legacy LSTM ortak origin'lerde ölçülmemişti; aynı pencere toplu T+1 kazancı −%1,6961. Tam tahmin yok; yeniden üretim/paired CI uydurulmaz.
- TCN bilinen-sinyal kontrolü başarısızken genel durum geçiyordu. Yeni aile bazlı politika ve eski cache yeniden değerlendirmesi uygulanmıştır; TCN yeniden eğitilmedi.
- Kaynak T+5 testlerinden T+1 çıkarımı yapılmaz. Ham, küçültülmüş ve aktif tahmin metrikleri ayrıdır. Sıfır ağırlık kaynakta sıfır sinyal kanıtı değildir.
- 2016–2023 ve görülmüş 2024+ geliştirme/inceleme tarihidir; bağımsız holdout olarak kullanılmaz.

## Devam etmeden önce

1. `python ml/history.py check` ile aile, ufuk ve kaynak/tarif geçmişini kontrol et; [sicil kullanımını](../research/README.md) oku.
2. Önceki sonuç/tahmin kapsamını ve dondurulmuş kimliği incele. Tekrar gerekiyorsa gerekçeyi kaydet; daha büyük grid/compute kendiliğinden gerekçe değildir.
3. Tahmin araştırmasında fiyat alanı/hedef ve karar saati varsayımları açık olmalı. Gerçek işlem/net PnL iddiasında kontrat, giriş/çıkış ve maliyet ayrıca doğrulanmalı; CT=F günlük barı gerçekleşme kanıtı değildir.

Yeni eğitim, yeni veri veya otomatik canlı model yayını bu depo düzenlemesinin parçası değildir. [Tarihsel belge indeksi](archive/README.md) ve [GitHub kanıt Release'i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/evidence-20261005) geçmişi korur.
