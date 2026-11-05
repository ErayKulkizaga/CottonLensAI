# FAS kaynak kabul koruması ve ortak-origin hizalaması

**VERIFIED — yazılım sözleşmesi:** FAS'ın mevcut `export_sales` yayın paketi
yoluna değer/sürüm/saat denetimi eklendi. Aynı kaynak paketi `compile_review`
ve `load_package` sırasında tekrar denetlenir. İkinci eğitim motoru yoktur;
`fas_country_alignment.attach_matched` mevcut `attach_package` /
`attach_releases` mekanizmasını kullanır. Yeni piyasa fit'i **0**.

Önceki genel FAS kabul yolu checksum bağlı yayın/vintage dosyalarını istiyordu,
fakat bu dosyaların **aynı özellik değerlerine ve aynı erişim saatine** ilişkin
olduğunu FAS'a özel bir sözleşmeyle kontrol etmiyordu. WASDE'de bu kontrol vardı.
Bu eksik kontrolün tamamlanmış ulusal FAS deneylerini etkilediği gösterilmedi:
onlar ayrı `fas_exploration` yolunda varsayımsal gecikme, ineligible kaynak ve
release kapalı olarak yürütüldü; bu teslimat onları yeniden sınıflandırmaz.

## Kabul sözleşmesi

`fas-specific-version-review-v1` inceleme kaydı şu kimlikleri bağlar:

- Aynı kaynak dosyası/hash'i, vintage kimliği ve sıralı özellik listesi.
- Sayısal hücreler ve açık `null` değerlerinin kanonik float/null hash'i.
- Gözlem döneminin sonu, erişim saati ve exact/upper-bound ayrımı.
- **1404 All Upland / running bales** birimi ve checksum bağlı dayanak dosyaları.

Genel yayın takvimi, PDF oluşturma saati ve bugünkü indirmeyi geçmişe tarihleme
kabul edilen dayanak türleri değildir. Belirli sürüme bağlı incelenmiş erişim
üst sınırı kullanılabilir; bu durumda gerçek yayın saati bilinmeyen kalır.
Upper-bound kaydı için mevcut `reviewed_upper_bound` kontrolü de korunur.
FAS paketi açık **cotton-next-day-0015-v1** ve sınırlı `max_age_days` ister;
eski gece yarısı varsayılanına sessizce dönmez.

**Sınır:** kod inceleme beyanının tutarlılığını denetler; yayıncı/arşiv kanıtının
doğruluğunu tek başına kanıtlamaz. Bir boolean veya kendi yazdığımız JSON dış
kanıt yerine geçmez. Özellikle dört haftalık türevin bütün bileşenleri aynı
erişilebilir sürüm kapsamında incelenmelidir. Gerçek FAS tarihçesi kabul edilmedi.

## Ortak-origin davranışı

- Karar anına eşit kayıt alınır; bir saniye sonraki sürüm alınmaz.
- Yeni sürüm geldiğinde eski origin'lere geri yazılmaz.
- Cotton origin/Close/hedef kolonları korunur; hafta sonu veya eksik kaynak
  nedeniyle sahte origin eklenmez, mevcut origin silinmez.
- Ulusal kontrol ve ülke adayı aynı kaynak yaşını ve **her özellik için aynı
  eksiklik göstergesini** alır. Grupların tek sayısal farkı ülke özellikleridir.
- `null` sıfır değildir; sıfır ve negatif net satış sayısal değer olarak kalır.
  Süresi geçmiş hücreler eksikleşir. İmputer/scaler burada fit edilmez.
- Her origin'de seçilen `export_sales_vintage_id`, `source_sha256`,
  `observed_through` ve `available_at` saklanır; bu provenance kolonları model
  gruplarına eklenmez. Süresi geçmiş seçimin izi, hücreleri masked olsa da korunur.

Sentetik gelecek-sürüm testi, yeni null desteğinde tümü eksik bir sütunun
sonraki sayısal sürümle dtype değiştirebildiğini yakaladı. Derleme/yüklemede
FAS özellikleri sabit `float64` yapıldı; test dtype dahil eşitlik ister.
Bu, yeni hazırlık yolunda yakalanıp düzeltilmiş bir sorun; eski model
başarısızlığının nedeni olarak sunulmaz.

## Uyumluluk ve kalan iş

Backend/model artifact/API şeması değişmedi. Mevcut karantina paneli ve ulusal
keşif deneyi değiştirilmedi. İnceleme kaydı bulunmayan eski **strict FAS yayın
paketi** artık kabul edilmez; bunu atlamak için eski flag'ler açılmaz.
Diğer kaynakların finite-hücre politikası ve varsayılan hizalaması korunur.

Hizalama ve eksiklik kontrolü sentetik paketle hazırdır. Gerçek çalıştırma için
hâlâ erişimi kanıtlanmış sürümlerden ülke özellik snapshot'ları, sayısal/sürüm
incelemesi ve tek kilitli deney ön kaydı gerekir. Mevcut latest-API paneline
saat atayıp eğitim açılmaz. Önceki **BLOCKED_EXTERNAL_EVIDENCE** kararı sürer;
aynı PDF/form araması tekrarlanmaz.

Doğrulama **yalnız sentetik**: saat sınırı, geç sürüm, upper-bound, null/zero,
değişen değer/clock/period/unit, checksum yeniden hesaplanmış sahte payload,
boş paket, örtüşen gruplar, origin sırası ve provenance çakışması sınanır.
Piyasa becerisi veya gerçek timestamp doğruluğu sonucu çıkarılmaz.
[Kanıt kaydı](../research/evidence/fas-reviewed-alignment-20261009.json)
test/restore kapsamını saklar; bu kaydın sicil sınıfı piyasa sorusu için
**INCONCLUSIVE**, türü **no-fit sözleşme doğrulaması**dır.

Yerel doğrulama: **117 odaklı test**, kapsamlı ML **836 geçti / 3 atlandı**,
Ruff geçti. İki sentetik saat paketinde dörder origin birebir korundu;
gerçek karantina paneline sahte sentetik kabul beyanı ekleyen probe reddedildi.
55 eski dosyanın hash'i, 138 önceki sicil kaydı ve ana kullanıcı checkout'unun
Git durumu korundu. Model eğitimleri / GPU / yeni piyasa verisi indirme yoktur.

[35 üyeli Release](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/fas-reviewed-alignment-20261009)
ve [checksum envanteri](../research/evidence/fas-reviewed-alignment-release-20261009.json)
224 ML kaynak dosyasını, sentetik paketleri ve reddedilen karantina probe'unu
saklar. Ayrı dizine restore edilen frozen kodla iki DataFrame/grup birebir
eşleşti ve karantina yeniden reddedildi. Örnek fiyat/hedefler sentetiktir.
