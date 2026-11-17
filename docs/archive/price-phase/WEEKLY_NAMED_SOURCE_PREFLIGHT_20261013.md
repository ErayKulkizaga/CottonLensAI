# Eski named settlement kaynağı: sıfır-fit kurulabilirlik

USDA AMS'nin resmi [Historical Weekly Reports dizini](https://apps.ams.usda.gov/Cotton/Historical%20Weekly%20Reports/)
altı seçili sezonda 305 haftalık PDF bağlantısı içeriyor. Önceden belirlenmiş
altı örnek (2010 Şubat/Kasım, 2013/2014/2015/2016 Şubat) 30 işlem günü ve
180 ayrı teslimat-ayı settlement hücresi sağladı. Bunlar grafik veya spot
fiyat değildir; tabloda açıkça ICE futures settlement, cents/lb yazar.

`research/weekly_named_source.py` basılı yayın tarihini ilk sayfadan,
altı kontrat başlığını ve günlük satırları tek settlement sayfasından çıkarır.
Eksik/bozuk hücre, karışık başlık, tekrar ve gelecekteki tarih hata verir;
sayısal fallback yoktur. Dokuz sentetik parser testi ve Ruff geçti.
Bağımsız pdfplumber metni 180 settlement hücresinin sırasını doğruladı;
2010/2013 tabloları Poppler render üzerinden görsel olarak da incelendi.

## Saat ve kapsam sınırı

Örneklerde işlem tarihi raporun basılı tarihinden 1–7 takvim günü önce.
İşlem tarihi, basılı yayın tarihi, şimdiki retrieval UTC ve dosya hash'i
ayrı tutulur. İlk-vintage/gerçek publication UTC ve günlük erişim **doğrulanmadı**.
Pazartesi fiyatını cuma raporundan okuyup pazartesi origin'ine vermek yasaktır.
Report-clock değerlendirmesi yalnız o origin'den önce yayımlandığı varsayılan
raporun bilgilerini kullanabilir; bu da ayrı, açık tarihsel saat varsayımıdır.
Exchange-close clock eski raporların kanıtladığı bir erişim saati değildir.

ESMIS'in weekly PDF/TXT dizini en eski listelenen kayıtlarında 2018 Kasım/Aralık
gösteriyor. 2017-10 filtresi release satırı döndürmedi; sayfanın “Latest Release”
PDF'i bu ayın verisi sayılmadı. Cornell eski publication adresi ESMIS'e yönlenir.
Bu sınırlı erişim kontrolü 2016–2018 verilerinin dünyada yokluğunu ispatlamaz.

Kaynak scope'u günlük on-kontrat excerpt'ine eşit değildir: altı settlement
kolonu ve haftalık yayın gecikmesi vardır. Eski dataset, origin, baseline,
500 olgun eğitim/3×63 validation, 2016–2023 ve 6/8 yıl kapıları değişmez.
Yeni kaynak yalnız checksum-bound readiness denetimine girebilir. Henüz
model girdisi kabulü, ölçülmüş EWMA üstünlüğü veya Colab geçişi yoktur.

## Sonraki sınırlı toplama

Fiyat/geçmiş kıtlığını ölçmek için yalnız bu altı sezonun gözlenen 305 PDF
bağlantısı, en çok 5 MB/dosya ve toplam 250 MB sınırıyla toplanabilir.
0 TL/0 fit; tek proses. Altı örnek byte/hash eşliğiyle yeniden kullanılır.
Ham PDF, retrieval makbuzu ve parser hataları korunur. Eksik haftalar ya da
tatil satırları doldurulmaz. Sonraki çıktı sadece kapsam/süreklilik raporudur;
aynı HAR fit'lerini tekrar açmak veya yılları seçerek kapıyı geçirmek değildir.

Yerel kanıt: `output/weekly-named-source-preflight-20261013/v1/`
ve ayrı metadata scope'u `gap-v1/`.

## Tamamlanan kapsam ve saat denetimi

305/305 PDF başarıyla indirildi: 143.785.872 byte, 0 capture failure.
V1 153 raporu kabul etti; 152 biçim reddi değişmeden saklandı.
Ayrı V2 tam ay adlarını, futures dışı endeks işaretlerini ve açık tatil
satırlarını ele aldı: 250 rapor, 1.196 farklı işlem tarihi, 36 unavailable
satır. 55 rapor hâlâ dar parser tarafından reddedilir; bu bütün içeriklerinin
kullanılamaz olduğunun kanıtı değildir. Yeni dosya indirme veya fit yapılmadı.

Orijinal ve Correction `CN20140606WWCMR` dosyalarında 2014-06-04 Oct-14
settlement 79,96/76,96 farklıdır. İki vintage korunur; tüm gün unknown
bırakılır. Correction filename'i first-vintage seçim kuralı değildir.
Bunun için sıra bağımsız regresyon testi eklendi; toplam 17 test/Ruff geçti.

Cotton takviminde 1.100 yeni tarih kalır. 2010/11/12/13/14/15/16 dağılımı
171/91/83/197/210/223/125. Basılı rapor tarihi+1day00:00UTC erişim ve
origin+1day00:15UTC karar **varsayımıyla**, yalnız 5 yeni tarihin güncel
quote'u origin'de erişilebilir olur. Diğerleri daha sonra yayımlanmıştır.
Bu 5 sayısı bile gerçek yayın UTC/vintage doğrulaması değildir.
Günlük exchange availability varsayımıyla 1.100 tarihi kullanmak farklı,
kanıtlanmamış bir clock varsayımı olur; sessizce açılmaz.

2016 değerlendirmesinde yalnız 125/250 observed quote; 2017 0/251,
2018 9/251, 2019 43/252, 2020 246/253, 2021 251/252, 2022 250/251,
2023 250/251. 2018+ değerleri eski günlük excerpt panelidir. Tüm 3.520
Cotton calendar slot korunur. Bu kaynak eski günlük origin/ikinci kontrat
protokolüne değiştirilmeden takılamaz; 500/189 ve 6/8 kapılarını düşürerek
kurulabilir sayılmaz. Kaynak keşfi ilerlemedir; ölçülmüş EWMA üstünlüğü,
causal feature kabulü veya Colab geçişi değildir.

## Kalıcı kanıt

Private `weekly-named-source-preflight-20261013` GitHub Release taze indirildi;
831 dosya hash ile geri kuruldu. Bağımsız 305PDF/1.500 hücre/3.520 takvim
doğrulaması tekrar geçti; verification JSON byte eşit. Yeni fit0, eski197
aşama ve541 fit tarifi değişmedi. Kanıt: `research/evidence/weekly-named-source-preflight-20261013.json`.
