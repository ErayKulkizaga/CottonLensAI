# Texas/NASS T+1: artımlı bilgi deneyi ön kaydı

**VERIFIED:** tasarım ve girdiler donduruldu; **0 piyasa / 0 sentetik fit**.
Bu bir tahmin sonucu veya tarihsel kaynak kabulü değildir. [Kanıt](../research/evidence/nass-regional-preregistration-20261009.json)
ve [yeniden üretim paketi](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/nass-regional-preregistration-20261009).

## Sınanan tek hipotez

Texas kondisyonunun ulusal kondisyondan farkı ve rapordaki gelişim–ortalama
farkları, sabit Cotton çekirdeği **ve ulusal kondisyonun üzerine** T+1 fiyat
tahmininde katkı sağlar mı? Her iki kolda ulusal bilgi, rapor yaşı ve bütün
eksiklik göstergeleri aynıdır. Böylece sadece kaynak takvimini veya ulusal
kondisyonu eklemek Texas etkisi diye sunulmaz.

Sicil kontrolü önce yapıldı. Eski `nass-exploration-v1` ulusal kondisyonu
T+5'te test etti: lag1/2/6 Naive kazançları −%0,6922 / −%1,0569 / −%0,8788.
Bu Texas/gelişim/T+1 testi değildir. NASS T+1 sorgusunda kayıtlı piyasa sonucu
yoktu; bu durum evrende hiç denenmediğini kanıtlamaz. Yeni dört tam tarifin
frozen scope'unda tamamlanmış fit eşleşmesi yok. DXY/WTI ve bölgesel WASDE
negatif sonuçları incelendi; model/grid değiştirmek için kullanılmadı.

## Sabit tasarım

| Unsur | Kilitlenen karar |
|---|---|
| Profil / deney | `nass-regional-t1-pilot-v1` / `research-nass-regional-t1-pilot-v1` |
| Girdiler | Eski full-year-v1-r2 history/ready; 311 raporlu Texas karantina paneli ve iki Git kayıtlı içerik/saat denetimi |
| Ortak değerlendirme | 2016–2023: **250,251,251,252,253,252,251,246**; **2.006 origin/kol**; 2024+ dışarıda |
| Kollar | `mask_D0`, `numeric_D0`, `mask_D1`, `numeric_D1` |
| Model / hedef | Ridge alpha1, seed42, pencere1; yalnız T+1 standardized log-return ve mevcut fiyat dönüşümü |
| Eğitim | Genişleyen geçmiş; ortak120 gözlem warmup; yalnız `target_date_5 < cutoff` olgun etiketleri |
| Seçim / refit | Sabit model; geçmiş3×63 blokta ayrı shrinkage `[0,.25,.5,.75,1]`, eşitlikte küçük ağırlık; her21 kayıtlı Cotton gözleminde refit |
| Ön işleme | Yalnız eğitimden median/mean/std ve mevcut otomatik eksiklik göstergeleri |
| Birincil ölçü | Seçilmiş `numeric_D0`–`mask_D0` paired fiyat-MAE; ayrıca ortak origin'lerde Naive |
| İkincil duyarlılık | D1; OOS sonucuna göre iyi gecikme seçilmez |
| Piyasa bütçesi | **288 iç +388 dış =676 fit**, **32 yıllık çıktı**, **8.024 tahmin satırı** |
| Ayrı öğrenme kontrolü | **1 sentetik Ridge fit**, `all-executed-families-v2`; known-signal MAE kazancı ≥%50; başarısızsa piyasa fit'i başlamaz |
| Kaynak sınırı | Bir süreç, en çok iki thread, mevcut ayrı CPU ortamı;30 dakikalık oturum/checkpoint, bütçe aşımı yok |

Naive sıfır getiri ile hesaplanır; model fit'i değildir. Bu ön kayıt engine'e
yürütülebilir profil eklemez. Bütçe **676 piyasa +1 sentetik**, gerçekleşen fit **0**.

## Modele hangi bilgi girecek?

Her kol **37 mantıksal /74 dönüştürülmüş** özellik içerir:

- Mevcut24 Cotton/cross-market/takvim çekirdek özelliği, aynı frozen değerlerle.
- Aynı iki rapor yaşı/erişilememe özelliği.
- Her iki kola aynı üç ulusal kondisyon alanı: good+excellent, poor+very poor
  ve yalnız ardışık aynı-yıl haftalarda good+excellent değişimi.
- Sekiz bölgesel alan: Texas−ulusal good+excellent ve poor+very poor farkı;
  ilk farkın ardışık haftalık değişimi; planted, squaring, setting bolls,
  bolls opening, harvested için **current−raporda yayımlanan2018–2022 vb.
  beş-yıl ortalaması**. Yüzde puanlar100'e bölünür.

Kontrol, bölgesel sayısal hücre biliniyorsa sabit0, bilinmiyorsa NaN taşır.
Bu karşı-olgusal sütundur; kaynak değerine sıfır yazılmaz. İki kolun otomatik
eksiklik göstergeleri birebir aynı kalır. Gelecekteki ortalamalardan klimatoloji,
geriye doldurma veya kaynak eksikliğinden origin silme yoktur.

Önceki haftanın sonradan revize sütunu kullanılmaz: haftalık değişimler, iki
raporun **kendi current değerleri** ile hesaplanır. Yeni raporda evre tablosu
yoksa eski evre taşınmaz. İlk hafta, aralıklı hafta veya yeni yıl deltası NaN'dır.

## Erişim varsayımı ve etkili bilgi miktarı

D0: seçilmiş dosya/değer sürümünün, basılı release gününü izleyen
**00:00 UTC** itibarıyla erişilebilir olduğu **varsayılır**. Karar Cotton
kaynak tarihini izleyen **00:15 UTC**. D1, ilk uygun kararın bir ek kayıtlı
Cotton gözlemi sonrasıdır. Azami yaş iki gecikme için aynı D0 başlangıcından
**10 gözlem**; D1 son kullanma sınırını uzatmaz. Yeni yılda eski yıl taşınmaz.
Release günü, gözlem haftası, dosya hash'i, karar saati ve
`assumed_available_at` provenance olarak korunur; model özellik listesine girmez.

**Tarihsel erişim ve ilk vintage doğrulanmış değildir.** Varsayım yanlışsa
gerçek erişim/vintage leakage'i mümkündür; sentetik gelecek testleri bunu
tarihsel olarak çözmez. `available_at` için sahte onay üretilmez, gerçek
kaynak kabulü/runtime açılmaz. Bu çalışma yalnız bu varsayıma bağlı duyarlılıktır.

| OOS kapsamı | D0 | D1 |
|---|---:|---:|
| Ortak origin | 2.006 | 2.006 |
| Kullanılabilir sezon raporu yok | 1.080 | 1.088 |
| Texas kondisyon farkı bilinen origin | 926 | 918 |
| Kondisyon farkı haftalık değişimi bilinen | 889 | 881 |
| Planted farkı bilinen | 151 | 151 |
| Squaring farkı bilinen | 432 | 432 |
| Setting bolls farkı bilinen | 449 | 449 |
| Bolls opening farkı bilinen | 543 | 536 |
| Harvested farkı bilinen | 366 | 358 |

D0 OOS'ta **180 ayrı kullanılabilir rapor vintage'ı**; ilk iç eğitimde **106**.
2.006 günlük origin,2.006 bağımsız temel bilgi değildir. Arşiv tam sezon değildir:
planted bütün311 raporun yalnız46'sında vardır. Sayısal katkı yılın tamamında
ölçülür; sezon/yaş tanıları önceden tanımlı yardımcı sonuçtur, iyi alt dönem
sonradan birincil hedef yapılamaz. Küçük/kararsız etki kaynağın tüm temsillerde
işe yaramadığını kanıtlamaz.

## Sonuçtan önce kilitlenen karar

Yıl içinde paired blok bootstrap: **10.000 tekrar, seed42, blok20 ve60**.
Origin, target tarihi ve gerçekleşen değer eşitliği zorunludur. Ham ve
küçültülmüş sonuçlar, aktif/tüm-origin yön, yıllar, en büyük ceil(%1) Naive
hataları çıkarımı ve source-age0–4 tanısı ayrıca verilir. Sıfır ağırlık ve
`[0,0]` aralığı kaynakta sıfır sinyal kanıtı değildir.

1. Kimlik/hizalama/payload/olgunluk kontrolü başarısızsa bilimsel karar yok.
2. Her iki gecikmede kontrol kazancının iki blok alt sınırı pozitif ve D0'ın
   Naive nokta kazancı ≥%5, yön ≥%53, yıl kazanımı ≥6/8 ise **varsayıma bağlı
   aday**; gerçek erişim/ileri doğrulama sonraki iş olur, model yayımlanmaz.
3. Her iki gecikmenin Naive kazancında iki blok üst sınırı <%5 ise bu sabit
   tarif pratik hedefi kurtarmıyor; Naive korunur, grid büyütülmez.
4. İki gecikmede kontrol katkısı desteklenir ama pratik eşikler sağlanmazsa
   sınırlı katkı kaydedilir; hedef çözülmüş sayılmaz.
5. Diğer durumlar belirsiz/gecikmeye duyarlı; iyi gecikme seçilmez, otomatik
   ek fit başlatılmaz.

Görülmüş2016–2023 ve bu aralıklar bağımsız doğrulama değildir. %5, %53/%55,
6/8 koşulları değiştirilmedi. Karar sözleşmesi ayrı dosya/hash ile bağlanır.

## Kimlik ve tek sonraki iş

Kaynak **`9cdf008af87dccbb973359e0f5848c4c90ff55ec747626c713d47d94d0eb5e3c`**,
**231 ML dosyası**. Ön kayıt
**`aa4abe3e2b66d706cc38ed6cd53f91b3d194b5ac7212996ab8e311f3bdae24dd`**;
karar sözleşmesi SHA256
**`d0bfff3318c7c0f179cd1255ef6c0e525b3bd3bf90b16065575c4542d4c7289e`**.

21 yeni sentetik regresyon testi; yakın parser/sicil kontrolleriyle **62 geçti**,
Ruff geçti. Bağımsız betik bütün3.520 satır × iki gecikme ×11 alanı **77.440**
kez yeniden hesapladı;96 eski history sütunu aynıdır.244 frozen payload,7 pinned
girdi ve231 kaynak hash'i doğrulanır.949 eski NASS girdi,143 sicil kaydı,
38 eski kanıt JSON'u +marker, trials ve13 ileri dosya korunur. GPU doğrulaması yok.
Gece deployed kaynak/otomasyon ve dirty ana checkout değişmedi.

Release'i ayrı klasöre açın; dış `checksums.json` ve frozen `complete.json`
doğrulansın. Mevcut Python3.12 CPU bağımlılık ortamında ağsız:

```bash
python verify.py registration
```

Yeni namespace için yalnız ön kayıt komutu:

```powershell
$env:PYTHONPATH='ml/src;ml'
python -m cottonlens_ml.research.nass_regional_pilot --repo . --output NEW_NAMESPACE --reference FROZEN_FULL_YEAR_R2 --panel PINNED_TEXAS_PANEL
```

**Sonraki tek iş:** bu değişmez tasarım/sözleşmeyi mevcut Experiment/Ledger CPU
yoluna bağlamak; yeni yürütme namespace/source kimliğini doğrulayıp yalnız
**676 piyasa +1 sentetik** fit'i çalıştırmak. Ön kayıtv1 veya eski cache değişmez.
Yeni veri indirme, GPU, model/grid araması, T+5'e sonradan geçiş veya gece
görevini güncelleme bu deneye eklenmez. Somut piyasa sonucu yürütme sonrasında
elde edilecek; bu ön kayıt başarı metriği değildir.
