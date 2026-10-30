# Günlük konsolsuz risk kaydı ve ilk eğitim durumları

Kullanıcı 30 Ekim 2026'da yalnız günlük, konsol açmayan ayrı Windows görevini
açıkça onayladı. Eski saatlik `CottonLens-Observed-Archive-v2` kapalı kalır.
Yeni görev adı `CottonLens-Risk-Daily-v1`; iki günlük pencere 00:05 ve
00:20 UTC, Türkiye saatiyle 03:05 ve 03:20'dir. Saatlik tekrar, oturum
açma tetikleyicisi, kaçan işi sonradan çalıştırma veya uyandırma yoktur.

## Somut eğitim sonucu

Private arşivden doğrulanmış v4 protokolü ayrı çalışma alanına kopyalandı;
eski kaynak/veri/model/126-baseline kohortu değişmedi. Yeni kohort gerçek
saatle **2026-10-30 14:19:03.625811 UTC**'de kilitlendi. İlk origin 30 Ekim;
karar 31 Ekim 00:15 UTC, hedef sonraki resmi Cotton seansı **2 Kasım**.

**10 piyasa fiti tamamlandı:** dokuz geçmiş-iç doğrulama modeli ve bir
dış model durumu. Minimum iç train 3.914, dış train 4.108 olgun satır;
warmup119 ve `target_date_5 < cutoff` değişmedi. Ayrı CPU ortamı, tek süreç,
iki thread, 30 dakika sınırı kullanıldı. Yeni sentetik fit/GPU/grid yoktur.
Native öğrenme kontrolü byte/ortam/200-tahmin eşliğiyle yeniden doğrulandı.

Kaydedilen 189 tahmin geçmişteki küçültme seçimi içindir. Dış checkpoint'in
tek tahmini `IN_SAMPLE_STATE_CHECK_ONLY` durum kontrolüdür. Seçilen ağırlık
1,0'dır. **Henüz ileri tahmin veya ileri başarı skoru yoktur.** Eğitim
tamamlanması, EWMA veya fiyat-Naive üstünlüğü değildir. Toplam31fit sınırının
10'u kullanıldı; aynı roller yeniden eğitilmez.

Salt okunur bağımsız denetim 10 modelin hash/kimlik/plan, olgunluk, beş
katsayı, gerçek training loss/gradient ve kayıtlı tahminlerini yeniden kurdu.
Decimal fiyat/getiri/HAR/EWMA kontrolü geçti; native dosyalar değişmedi.

## Çalıştırma ve konsol güvencesi

Yeni çalıştırıcı `research/ct_forward_daily.py`, değişmeyen frozen core'u
kullanır. Windows görevinin doğrudan executable'ı **pythonw.exe**'dir;
PowerShell veya cmd alt süreci açılmaz. PE başlığındaki GUI subsystem2
doğrulanır; konsol executable'ını yeniden adlandırmak kabul edilmez. Görev
Task Scheduler içinde görünürdür; `Hidden` ayarı konsol gizlemek için
kullanılmaz. stdout/stderr ve tam traceback dosyaya yazılır.

00:05 aşaması yalnız uygun Cotton seansında tek ham istek yapar. Kaynak
makbuzu, resmi takvim, bilinen fiyat/tarih örtüşmesi ve 00:15 kesimi korunur.
Gerekirse yalnız kilitli yıllık/21-slot refit yapılır. 00:20 aşaması yeni
fit yapmaz; yalnız gerçekten zamanında bilinen kaynak/model/seçimle risk
tahmini kaydeder. Kesim kaçarsa geriye tahmin doldurulmaz. Her çalışmada
ayrı execution_source_id/core_source_id ve tahmin/makbuz bağı tutulur.

Kaynak kodu working-tree değişikliklerinden bağımsız ayrı hash dizininde
dondurulur. Görev mevcut sınırlı kullanıcı/interactive oturumuyla çalışır;
**bilgisayar uyanık ve kullanıcı oturumu açık olmalıdır**. Kilit ekranı
oturum kapatma değildir. Kapalı/uyuyan makineden çalışma garantisi yoktur.
Geç kalan çalıştırma araştırma saatini kaydırmaz.

İki günlük aşamanın ardından yeni çalışma dosyaları private GitHub
Releases'e eklenir. `research/ct_forward_backup.py` önce deponun private
olduğunu ve sabit CLI/kod hash'lerini doğrular; her parçayı taze indirip
tüm üyelerin checksum'larını kontrol etmeden yedek sicilini ilerletmez.
Önceki bulut nesneleri ve kaynak dosyalar silinmez. Drive'a yazılmaz.
Yedekleme yayın makbuzundan sonra yapılır; bilimsel yayın saatini geriye
çekmez. Hata ayrı makbuz ve sıfır olmayan süreç sonucuyla korunur.
GitHub CLI alt süreci de `CREATE_NO_WINDOW` ile konsolsuz çalışır.

Resmi takvim şu an 1 Ocak 2027'ye kadar kanıtlıdır; kapsam dışındaki
origin veya sonraki hedef için kaynak alma/eğitim/yayın durur. 2027 uzatması
ayrı kaynak makbuzu ve Cotton-sütunu incelemesiyle eklenmelidir. Model
tarifi veya değerlendirme dönemi bu gerekçeyle değiştirilmez.

## Dönem sonu raporu ve bilimsel sınır

Yeni `research.ct_forward_report` komutu yalnız son hedef olgunlaştıktan
sonra açılır. 252 slot korunur. Kaynak zinciri ham byte'lardan; hedef,
EWMA ve HAR Decimal ile; tahmin native katsayılardan; train/gradient,
geçmiş seçim ve yayın saatleri kayıtlardan yeniden denetlenir. Fit,
optimizer, yeni veri indirme veya ara skor yoktur.

```powershell
python -m research.ct_forward_report --root STUDY --repo FROZEN_ANALYSIS_SOURCE --output NEW_REPORT
```

**V4'teki tüm252tam-yayın şartı yeni bir operasyon kapısıdır; önceki
kullanıcı planındaki 6/8 tarihsel risk kapısı değildir.** Tek eksik yayın
bu ön kayıtta doğrulayıcı başarıyı engeller. Böyle bir sonuç modelde sinyal
yok demek değildir: `INCONCLUSIVE_INCOMPLETE_PUBLICATION`, yalnız dönem
sonu tanımlayıcı eşlenmiş sonuç ve `CI=null` üretilir. Sıfır yayın için
ortalama da `null` olur; sahte `[0,0]` üstünlük/sinyalsizlik kanıtı yoktur.
Önceden kilitlenen şart, sonuç görülerek sessizce gevşetilmez.

Tamlık şartının etkisi yalnız varsayım örneğidir: bağımsız günlük kaçırma
olasılığı %1 **varsayılırsa**, 252 kaydın tamamının bulunma olasılığı
`0,99^252 ≈ %7,945` olur. Bu ölçülmüş uptime, model gücü veya mevcut
görevin gerçek başarı olasılığı değildir; günler bağımlı olabilir.

Rapor testlerindeki pozitif/negatif örnekler yalnız sentetik/analitik
fixture'dır, eğitilmiş model veya piyasa sonucu değildir. Kaynak gelecekte
geldiği halde modelin önce tamamlanmış gösterilmesi ve tutarlı hash'lerle
gizlenmiş geçersiz optimizer da denetimde reddedildi. Yeni risk başarısı
iddiası yalnız gerçek ileri kayıtlar ve dönem sonu raporuyla yapılabilir.
