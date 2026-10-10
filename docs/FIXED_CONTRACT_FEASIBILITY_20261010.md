# Origin'de sabitlenen kontrat hedefi — 10 Ekim 2026

**VERIFIED:** mevcut arşiv, eski full-year protokolünü aynı origin ve iç
bloklarla adlandırılmış kontrat hedefine doğrudan aktarmaya yetmiyor.
Bu bir tahmin başarısızlığı deneyi değildir; yeni fit **0**, performans
skoru yok. Önceki deneyler, hedefler, modeller ve sabit kapılar korunur.

## Dondurulmuş soru ve veri

`fixed-contract-feasibility-v1`, [PR36](https://github.com/ErayKulkizaga/CottonLensAI/pull/36)
sonrasında, sonuca bakmadan donduruldu. Aynı 3.520 Cotton tarihinin ve
749 değerlendirme origin'inin tamamı saklanır. Her tarihte D0'ın karar
anında varsayımsal olarak erişilebilir son raporundaki ilk veya ikinci
kontrat adı seçilir, bu **isim** hedef güne kadar sabit tutulur. Geleceğin
ilk kontratı seçilmez; hedef tarihleri özgün Cotton T+1/T+5 tarihleridir.

Karar anı kaynak tarihini izleyen gün 00:15 UTC; varsayılan rapor erişimi
`max(reference/report date, unverified publication calendar day)+1 day
00:00 UTC`. En çok üç Cotton gözlemi eskilik sınırı korunur. Bunlar tarihsel
UTC yayını/ilk vintage ispatı değildir; historically-admitted **0**.

İki farklı hedef tanımı ayrıdır:

- **Contemporaneous:** seçilmiş kontratın origin referans günü fiyatından
  aynı kontratın hedef günü fiyatına. Origin fiyatının karar anında
  varsayımsal erişimi ayrıca şarttır. Eksik/geç fiyat doldurulmaz.
- **Available quote:** açık kaynak tarihli son erişilebilir fiyattan hedef
  güne. Bu ayrı bir last-known-price tahmin tanımıdır; origin kapanışının
  doldurulması veya gerçek elde tutma getirisi değildir.

## Kapsam: sayıların tamamı 749 özgün origin üzerinden

| Kontrat seçimi | Hedef tanımı | T+1 uygun | T+5 uygun |
|---|---|---:|---:|
| İlk | Contemporaneous |724|665|
| İlk | Available quote |731|672|
| İkinci | Contemporaneous |739|739|
| İkinci | Available quote |746|746|

İlk kontrat T+5'te **74** kez hedef tablosunda yoktur; bunu kontratın
kesin sona erdiği veya işlem yapılamadığı şeklinde yorumlamıyoruz.
İkinci contemporaneous T+5'in 10 eksik/uygunsuz origin'i: dört geç
origin quote'u, üç eksik origin referans quote'u, üç eksik hedef quote'u.
Yıllık uygunluk 248/252, 249/251, 242/246. Hiçbir origin kaydı silinmedi.

## Neden aynı protokol kullanılamıyor?

Adlandırılmış kontrat etiketleri **2020-01-02**'de başlıyor. Önceki
2010–2019 proxy etiketleri aynı kontrat hedefinin eğitim etiketleri
değildir. Source-aware label maturity, hedef quote'un varsayımsal erişimini
de de kontrol eder; yalnız Cotton hedef tarihinin geçmiş olması yetmez.

Özgün 120 gözlem warmup, T+5 ortak olgunlaşma, 3×63 iç origin, her iç
kesimde **500** olgun eğitim etiketi ve tam dış cohort şartları aynen
uygulandı. İki kontrat × iki tanım × üç yıl: **12/12 doğrudan aktarım
kontrolü başarısız**. Örneğin ikinci contemporaneous ortak T+1/T+5 olgun
eğitim sayıları 2021'de [46,109,171], 2022'de [277,340,403], 2023'te
[523,586,646]. 2023 sayı eşiğini geçse de seçim anında bilinen iç hedefler
[63,61,63] ve dış cohort 242/246; değişmeden aktarım hâlâ geçmiyor.

500 şartı evrensel bir öğrenilebilirlik teoremi değildir. Buradan hiçbir
modelin öğrenemeyeceği veya kaynakta bilgi olmadığı sonucu çıkmaz. Ancak
eşik azaltmak, uygun yılları seçmek veya eksik origin'leri sessizce almak
**yeni deney tasarımıdır**; eski protokolmüş gibi sunulamaz.

## Kanıt ve koruma

3.520 D0 kaynak seçimi, 14.080 aday satırı, 7.655 Decimal log-label hesabı,
96 eğitim/label-availability kesimi ve 12 aktarım kararı bağımsız scalar
doğrulandı. Sınırdaki 12 hedef tarihi de korunur. Sekiz ret kontrolü:
kontrat değişimi, origin saati, label değeri, eksik label doldurma, eğitim
sayısı, origin silme, kaynak hash'i ve eski çıktı üzerine yazma. Ruff geçti.

ML source `20d0c1869afc35b9c83b34cfd3fc8de8ec90a56ebd64c1cf3dcb3b085685d7b3`
değişmedi. Contract SHA256
`58d5a2c2b575c19f1301c9e58a46acf34d247559eb3ddf9f83ee90f6d93d496c`.
Eski 163 çalışma/518 fit tarifi/81 proof ve 1.000 ham kaynak checksum'ı
korunur. Tahmin becerisi sınıfı **INCONCLUSIVE**; kurulabilirlik bulgusu
doğrulanmıştır. Üç yıllık varsayımlı çalışma 6/8 yıl kapısını değiştirmez.

Makine kanıtı: [proof](../research/evidence/fixed-contract-feasibility-20261010.json),
[Release manifest](../research/evidence/fixed-contract-feasibility-release-20261010.json),
[checksum-bound payload](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/fixed-contract-feasibility-20261010).
Bağımsız replay ve ret testleri payload'dadır; backend/UI/GPU değişmedi.

## Tek sonraki bilimsel soru

**Mevcut dondurulmuş CT tahminleri, origin'de bilinen ikinci kontratın
hareketine taşınınca da Naive'den kötü mü?** Ayrı sıfır-fit ön kayıtla
aynı origin/hedef/tahmin faktörleri korunmalı; baseline ve aday aynı
kontrat fiyatına karşılaştırılmalı. 10 eksik kayıt açık tutulmalı;
geleceğin kontratı veya eski fiyatın doldurulması kullanılmamalı.
Bu, aynı hedefte yeniden eğitilmiş modelin beceri testi değildir.
Last-known-price kullanılırsa güncel Cotton hareketini zaten bilen bir
nowcast baseline olmadan eski quote Naive'sini geçmek ileri tahmin sayılmaz.
Yeni model/grid/veri veya otomatik yeniden eğitim bu sonuçla açılmaz.

## Ayrı teslimat doğrulaması

21 üye /1.847.474 byte; ZIP SHA256
`e2ab2d87148be258dfbf5a15ac64b970897d5375c9abbd3ce4b8116a3f398d6e`.
Yerel temiz dizin ve GitHub'dan yeniden indirilen arşivde rapor,14.080
satır,tamamlama kaydı,bağımsız makbuz ve sekiz ret kontrolü byte olarak
aynı üretildi; fit0. İlk yerel helper'ın60 saniyelik toplam kontrol
sınırında kalan dizini korundu;180 saniyelik helper ile yeni namespace
tamamlandı. Bilimsel script/veri/tolerans değişmedi; eski yarım çalışma
tamamlanmış sayılmadı. Arşiv RESULT.md bu teslimat ekinden önce donduruldu.

[PR37](https://github.com/ErayKulkizaga/CottonLensAI/pull/37),
[teslimat kanıtı](../research/evidence/fixed-contract-feasibility-delivery-20261010.json).
Sicil164 çalışma/518 tarif; önceki nesneler ve ham checksum'lar korundu.
Teslimat makbuzunun gözlemlediği sonuç commit'inde12 CI kontrolünün
dokuzu başarılı,üç ML kontrolü sürüyordu; sonraki commit CI'sı ayrıdır.
Obsidian kanonik proje notuna yeni hedef/eksik-label/nowcast sınırı işlendi.
