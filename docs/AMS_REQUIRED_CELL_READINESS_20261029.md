# Gerekli hücreler ve tam günlük kaynak — 29 Ekim 2026

**VERIFIED:** 2021-02-09 annual dosyası ilk kontrat olarak Mayıs 2021'i,
ikinci kontrat olarak Temmuz'u içeriyor. Aynı tarihli iki günlük kaynak
Mart/Mayıs sırasını gösteriyor. Bu, bilinen iki bozuk annual yılın **dışında**
hedef kontratının da değiştiği başka bir kusurdur. Native sonuçlar korunur;
henüz düzeltilmiş forecast/refit veya piyasa başarısı yoktur.

## Yeni doğrudan kanıt

USDA arşiv rehberi MP_CN002'yi settlement içeren tam günlük rapor olarak
tanımlar. ESMIS `daily-spot-quotations` içindeki CNDDSQ.PDF ilk sayfası on
kontrat için Today/Previous/Change ve ICE settlement dipnotu taşır.
Basılı tarih, sayfa tarih alanı ve dosya kimliği ayrı doğrulanır. Sayfadaki
12:00Z alanı **gerçek tarihsel yayın saati kabul edilmez**.

Sabit16tarihli manuel denetim14istek/2.563.195yeni yanıt byte'ı kullandı;
eski adaylar/404 yeniden indirilmedi. Dört tarih doğrulandı. Başarısız ve
yanlış basılı tarihli kaynaklar saklandı, doldurulmadı.

| Tarih | Tam günlük kaynağın gösterdiği |
|---|---|
| 2019-08-05 | Oct19–Jul21 on kontrat; eski annual etiketleri farklı |
| 2019-11-18 | Dec19–Oct21 on kontrat; eski annual etiketleri farklı |
| 2020-12-04 | MP_CN001 ile on fiyat/isim aynı; annual'da beş uzak fiyat +0,01 |
| 2021-02-09 | MP_CN001 ile on fiyat/isim aynı; annual'da ön Mart21 yok, fazladan Mar23 var |

İki PDF okuyucusu (pypdf/pdfplumber)40fiyat üçlüsünü/120scalar'ı bağımsız
doğruladı; Today−Previous=Change eşitliği de kontrol edildi. Previous kolonu
keyfî bir önceki güne atanmadı. Bu iki örtüşen örnek, MP_CN001'in genel
olarak intraday fiyat olduğu açıklamasını **desteklemiyor**; farkı rapor
türüne yüklemek yerine gerçek hücre/sıra farkı kaydedilir. Aynı yayıncıdaki
iki rapor bağımsız fiyat-vendor doğrulaması veya first-vintage kanıtı değildir.

## Onarımın doğrulanmış kapsamı

Dört tam günlük tablo ayrı kanıt katmanına eklendi; mevcut günlük/weekly
katmanla1.335tarih var. İki eski şüpheli blokta24yerine22tarih çözümsüzdür.
2020-12-04 ve2021-02-09 çatışmaları için yeni doğrudan tanık vardır;
2018-11-26 ve diğer çelişkiler gizlenmez. Özgün ham dosyalar değişmedi.

Tam23Cotton gözlemli943pencerede2.829Decimal HAR scalar'ı doğrulandı.
139eski değerlendirme HAR girdisi farklı;174ortak işin106train/28test girdisi
etkileniyor. Bunlar gerçek girdideki alt-sınır hatalardır; düzeltmenin
tahmin başarısına katkısı ölçülmüş değildir.

## Neden yalnız22günü tamamlamak yeterli olmayabilir?

`ams_named_readiness.attach` EWMA'yı ilk gözlenen kare getiriden başlatıp
her adjacent-Cotton-date çiftinde λ=0,94 ile taşır. HAR son22gözlenen çifti
kullanır. Weekly kaynakta görünmeyen uzak kontrat, "quote yok" demek değildir.
Bilinmeyen olası güncelleme, yeterli sonraki bilinen çiftle HAR'dan çıkar;
EWMA'da ise azalan bir ağırlıkla kalır. Sessiz reset veya tarih sıkıştırması
yapılmadı. Şüpheli blok dışında prefix tanığı, eski tam annual tablonun
uzak kolonlarını gereksiz yere daraltmaz.

2.006slot/2.824gerekli origin/174iş üzerinde katı provenance hesabı:
1.955hedef çifti,1.794HAR,667EWMA ve662tam girdi kurulabilir. Bu hesap,
bilinen bozuk blokların dışındaki annual veriyi **koşullu** kullanır;
bütün annual kaynağın bağımsız doğrulandığını söylemez.

2.053olası bilinmeyen EWMA hücresi **muhafazakâr üst sınırdır**. Uzak bir
kontrat o eski tarihte henüz listelenmemiş olabilir. Bu sayı2.053gerçek
veri hatası, o kontratların varlığı veya başarısız tahmin sayısı değildir.
λ^k yalnız belirsiz başlangıç farkının sonraki çarpanıdır; başlangıç farkı
sınırlandırılmadan hata sınırı veya "ihmal edilebilir" hükmü çıkarılmaz.
Bağımsız set/bisect hesabı tüm2.824origin/174işi ve üst-sınır hücrelerini
aynı açık kaynak kapsamları altında yeniden doğruladı.

## Teslimat ve karar

36ilgili test/Ruff geçti. Yeni piyasa/sentetik fit0; eski554tarif ve tüm
native modeller korunur. UI/API/canlı kaynak ve kapalı saatlik görev değişmedi.
Private `ams-required-cell-readiness-20261029` Release gerekli normalized
girdileri, yeni tam PDF'leri, başarısız denemeleri ve donmuş kaynakları taşır.
Eski kullanılmayan ham kaynaklar önceki checksum-bound Release'te korunur.
GitHub'dan taze401dosya/0istisna hash ile geri kuruldu. Donmuş kaynakla
11rapor/CSV/overlay/makbuz byte eşit yeniden üretildi; dört ham PDF production
parser ile de tekrar eşleşti. Replay0fit/ağ erişimi0. Arşiv SHA
`07eaca18da1b617a8ddf823ef614be8013da12889aac517a65e1cabe61719025`;
kanıt `research/evidence/ams-required-cell-readiness-20261029.json`.

**Karar:** Bu named kaynak henüz yeni fit'e kabul edilmez. Eksik uzak
geçmişi yok sayan bir EWMA üretip güçlü referansın kurtarıldığını söylemek
yanlış olur. Önce tek erken tarih için (2018-08-01) özgün on-kontrat günlük
tanık veya gerekli kontratların tarihli listing kanıtı bulunmalıdır. Daha
çok altı-kontrat PDF toplamak bu soruyu tek başına çözmez. Yeni grid yoktur;
CT/yalnız2023çalışmaları bu kaynak kusuruyla otomatik geçersizleşmez.
