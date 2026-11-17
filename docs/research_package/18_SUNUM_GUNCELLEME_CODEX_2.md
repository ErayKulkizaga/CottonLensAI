# 18 · Sunum güncelleme özeti, ikinci tur (Codex için, 17 Kasım 2026)

Bu dosya `docs/portfolio/cottonlens-sunum.html` (CSS ve JS aynı klasörde) güncellenirken kullanılacak. 17. dokümandan, yani kullanıcının "Başarıya ulaşıldı mı?" sorusundan sonra yapılan her şeyi kapsar. İki dosya birlikte okunmalı; çakışan bir nokta olursa bu dosya geçerlidir.

- Repo `ErayKulkizaga/CottonLensAI`, dal `main`. Bağlantılar için sabit commit: **`030555f887badbbef154cefc3982ab95276a9f0a`**.
- Bütün sayılar `research/evidence/` altındaki kanıt dosyalarından ya da `docs/prereg/` altındaki ön kayıtların sonuç bölümlerinden gelir.

## 1. Kısa durum: başarıya ulaşıldı mı?

**Kısmen.** USDA'yı bir değişkende kanıtlı biçimde geçtik ve hatasının nedenini bulduk. Üretimde ve piyasada geçemedik. Bu tur yeni bir başarı iddiası eklemedi; mevcut iddiaları daha dürüst ve daha denetlenebilir hâle getirdi.

| Ne | Durum |
|---|---|
| Teksas hasat alanı, USDA hatası %11,4 → %7,0 (p = 0,0005) | **Kanıtlı.** Artık tek komutla yeniden üretiliyor. |
| Kör sezonlar 88/128 (p = 0,007); revizyon yönü %67 (p = 0,012) | Kanıtlı; sınırları önceki gibi. |
| Bugünkü üretim tahmininde anlamlı kazanç | **Yok.** Nedeni bu turda bulundu (bölüm 3.2). |
| Piyasada para kazandırma | **Yok.** İkinci piyasa testi (T55) de geçmedi. |
| Tahmin aralıkları | Gerçek zamanlı kalibre edildi; dürüst ama geniş (±%16). |
| Canlı sistem | `main`'de otomatik çalışıyor; alarm sistemi denetlendi ve düzeltildi. |

**Sayfada kullanılacak ana cümle (değişmedi):**
> "USDA'nın pamuk tahminlerinin nerede ve neden yanıldığını bulduk: 2012'den sonra Teksas'ta hata verimden hasat alanına kaydı. O değişkende USDA'yı istatistiksel olarak anlamlı biçimde geçen gerçek zamanlı bir düzeltme kurduk. Üretimdeki kazanç tutarlı ama henüz anlamlı değil, çünkü alan ve verim hataları birbirini dengeliyor. İleriye dönük test git'te tarihli olarak kilitli."

## 2. Yapılanlar (sırasıyla)

| Adım | Ne yapıldı | Sonuç | Kanıt |
|---|---|---|---|
| T55 (ön kayıtlı) | Piyasa testi, sinyalin çalıştığı 2000–2009 döneminde (Yahoo `CT=F` günlük kapanışları) | 30 olay; sinyal USDA'nın revizyonunu %67 doğru bildi, ama rapor günü getirisi ortalama +%0,47, isabet %53, p = 0,082. **GEÇMEDİ.** Kazancın yarısı tek bir yıldan (2003) geliyor. | `docs/prereg/T55_MARKET_2000_2009_PREREG_20261117.md`, `research/evidence/t55-market-2000-2009-20261117.json` |
| Beşinci dış değerlendirme | Yeni hipotez testi yapılmaması kararlaştırıldı; plan kalibrasyon, teşhis ve ürün işlerine odaklandı | Önceden önerilen T56 piyasa testi rafa kalktı | `docs/PLAN_REVIEW5_20261117.md` |
| CAL1 (ön kayıtlı) | Tahmin aralıklarının gerçek zamanlı kalibrasyonu | Eski yöntem gerçek zamanlı uygulandığında %80 aralık %52, %95 aralık %79 kapsıyordu. Üç aday yöntemden yalnızca "2 kat" kurala uydu: **%83 / %93**. Aralık geniş: %80'de ±%16. | `docs/prereg/CAL1_INTERVAL_CALIBRATION_PREREG_20261117.md`, `research/evidence/cal1-intervals-20261117.json` |
| B2 (betimsel) | Alan kazancı neden üretime geçmiyor | 2012 sonrasında Teksas'ta alan ve verim hataları zıt yönlü: korelasyon −0,57, Aralık'ta −0,92; 2012 öncesinde +0,21. Alan kazancının yalnızca **%36**'sı üretime geçiyor (5,4 puandan 2,0 puana). | `research/evidence/b2-area-yield-offset-20261117.json` |
| B3 · yeni ileri kilitler | Keşifte kullanılmamış iki eyaletin 2026 hasat alanı | Oklahoma **243.609** akre (USDA 320.000, −%23,9); Kansas **67.878** akre (USDA 77.000, −%11,8). Mayıs 2027'de puanlanacak. | `research/evidence/t49-forward-lock-okks-2026-20261117.json`, `docs/FORWARD_LOCK_2026.md` |
| Monitör `model_v2` | CAL1 aralıkları monitöre eklendi | Kasım kaydı `model_v1` ile yapılmıştı, değiştirilmedi; Aralık'tan itibaren `model_v2` kullanılıyor | `research/monitor/model_v2.json` |
| Canlı sistem | Performans geçmişi sayfası, `latest.json`, hata bildirimi (GitHub issue) | PR #78 `main`'e birleşti; iş akışı `main`'de elle çalıştırıldı ve başarılı oldu | `docs/monitor/index.html`, `docs/monitor/latest.json`, `.github/workflows/usda-monitor.yml` |
| Repo düzeni (PR #79) | İki çalışma klasörü tek klasörde birleşti; dokümanlar yeniden düzenlendi | Ön kayıtlar `docs/prereg/` altına, fiyat aşaması belgeleri `docs/archive/price-phase/` altına, eski rekolte notları `docs/archive/phase2-notes/` altına taşındı. README, belge dizini ve AGENTS.md yeniden yazıldı. Hiçbir şey silinmedi, arşivlendi. | `README.md`, `docs/README.md` |
| Denetim düzeltmeleri (PR #80) | Bütün repo denetlendi; 8 bulgu düzeltildi | (1) Ağustos'ta yanlış alarm verilmesi önlendi. (2) QuickStats kesintilerinin sessizce "veri yok" sayılması engellendi. (3) Aynı anda iki çalıştırma önlendi. (4) Elle girilen ay doğrulanıyor. (5) Monitörün kontrol ve geri dönüş testleri eklendi. (6) Yeniden üretim artık asıl iddiaları da kapsıyor. (7) Test açıklaması düzeltildi. (8) Gereksiz bot commit'leri kalktı. | `research/monitor/monitor.py`, `research/reproduce.py` |
| Doğrulama | Bütün kontroller yeniden çalıştırıldı | **409 test geçti** (1 atlandı); `reproduce.py` **10/10** eşleşti (T51 Teksas alanı, T50 ve birleşik kör test dahil); CI'daki 8 kontrol yeşil | — |

## 3. Sunuma girmesi gereken yeni içerik

### 3.1 Tahmin aralıkları kartı (CAL1)
- Başlık önerisi: "Belirsizliği olduğundan küçük göstermiyoruz".
- Metin: "Aralıklarımızı yalnızca o yıla kadarki veriyle, yani gerçek zamanlı test ettik. Eski %80 aralık gerçek değerlerin yalnızca %52'sini kapsıyordu. Önceden yazılmış kurala uyan tek yöntemle kapsama %83 / %93'e çıktı. Bedeli: aralık geniş, yaklaşık ±%16. Teksas 2026 için %80 aralığı 2,62–4,06 milyon akre ve USDA'nın 3,90 milyonu da bu aralığın içinde."
- Önceki "%80 aralık %64 kapsıyor" ifadesini bu kartla değiştir.

### 3.2 "Neden üretime geçmiyor?" kartı (B2)
- Metin: "Terk edilen tarlalar zaten en düşük verimli tarlalar. USDA hasat alanını fazla tahmin ettiğinde, ortalama verimi de düşük tahmin ediyor; iki hata birbirini kısmen götürüyor. 2012 sonrasında alan ve verim hatalarının korelasyonu −0,57 (Aralık'ta −0,92). Bu yüzden alan düzeltmesinin kazancının yalnızca %36'sı üretime geçiyor."
- Bu kart T51 ve T53 kartlarının hemen yanına ya da altına girmeli, çünkü onların "neden geçmedi" sorusunu cevaplıyor.

### 3.3 T55 kartı (T52 kartının yanına)
- "T55 · PİYASA TESTİ 2000–2009: Sinyal USDA'nın revizyonunu %67 doğru bildi. Rapor günü fiyat getirisi ortalama +%0,47, isabet %53, p = 0,082; ön kayıtlı eşiği geçmedi ve kazancın yarısı tek bir yıldan geliyor. Okuma: revizyon öngörülebilir, ama piyasa bunu büyük ölçüde zaten fiyatlıyor."

### 3.4 İleri kilitler bölümü
- Mevcut ABD ve Teksas kilitlerinin yanına Oklahoma (243.609 akre) ve Kansas (67.878 akre) eklenmeli, "keşifte kullanılmamış eyaletler" notuyla.
- Teksas alan kilidinin aralığı **2,62–4,06 milyon akre** olarak güncellenmeli (eski 2,92–3,64 milyon geçmişe bakarak hesaplanmıştı).
- Takvim: **10 Aralık 2026**'da T50 revizyon yönleri (ABD aşağı, Teksas aşağı) ve monitörün ilk otomatik `model_v2` kaydı; **Mayıs 2027**'de dört alan kilidi ve ABD üretimi.

### 3.5 Monitör bölümü
- "Rapor ayın 14'ünde hâlâ kaydedilmemişse ya da yeni kayıt USDA rakamına dönmüşse iş akışı otomatik olarak GitHub issue açıyor. Aynı anda yalnızca bir çalıştırma olabiliyor. Kayıtlar hash zinciriyle bağlı; geçmiş bir tahmin sonradan değiştirilirse zincir kırılır."
- Performans sayfasına bağlantı ver: `docs/monitor/index.html`. Makinece okunabilir son kayıt: `docs/monitor/latest.json`.
- "İlk otomatik kayıt 10 Aralık 2026" ifadesi doğru.

### 3.6 Denetlenebilirlik satırı
- "Tek komutla yeniden üretim: 10 ana sonucun 10'u birebir eşleşiyor; Teksas alan sonucu, revizyon yönü ve birleşik kör test dahil. 409 otomatik test geçiyor."
- Dürüst not: birleşik kör test adımı iki dönemin commit'lenmiş sonuçlarını birleştiriyor; dönemlerin kendisini yeniden çalıştırmıyor.

### 3.7 İsteğe bağlı: fiyat tahmini kıyası
- "Pamuk vadelisinde 'fiyat değişmez' tahmininin hatası 2010–2021'de 1 gün için %1,23, 5 gün için %2,94. Fiyat aşamasında en iyi modelimiz bu tahmini yalnızca %0,22 geçti." Bu sayı fiyat tahmini iddialarını değerlendirmek için iyi bir referans.
- Başka şirketlerin adını ya da rakamlarını sayfaya koyma.

## 4. Bağlantılar: yollar değişti

Sayfadaki bağlantılar `08ce79f…` commit'ine sabit. O commit'te çalışmaya devam ediyorlar, ama yeni içerik yeni yollarda. Bütün bağlantıları **`030555f887badbbef154cefc3982ab95276a9f0a`** commit'ine çevir ve şu yol değişikliklerini uygula:

| Eski yol | Yeni yol |
|---|---|
| `docs/T47_…` – `docs/T55_…` (bütün `*_PREREG_*.md`) | `docs/prereg/<aynı dosya adı>` |
| `docs/CAL1_INTERVAL_CALIBRATION_PREREG_20261117.md` | `docs/prereg/CAL1_INTERVAL_CALIBRATION_PREREG_20261117.md` |
| `docs/RESEARCH_PLAN_20261116.md`, `docs/TRIED_AND_LEARNED_20261116.md` | `docs/archive/phase2-notes/<aynı ad>` |
| Fiyat aşaması belgeleri (`docs/STATUS.md`, `docs/AMS_…`, `docs/CT_…` vb.) | `docs/archive/price-phase/<aynı ad>` |
| `docs/CROP_NOWCAST_TRIALS.md`, `docs/FORWARD_LOCK_2026.md`, `docs/paper/…`, `docs/research_package/…`, `docs/adr/…`, `research/…` | Değişmedi |

Yeni bağlantılar:
- `docs/prereg/CAL1_INTERVAL_CALIBRATION_PREREG_20261117.md`
- `docs/prereg/T55_MARKET_2000_2009_PREREG_20261117.md`
- `research/evidence/cal1-intervals-20261117.json`
- `research/evidence/b2-area-yield-offset-20261117.json`
- `research/evidence/t49-forward-lock-okks-2026-20261117.json`
- `research/evidence/t55-market-2000-2009-20261117.json`
- `docs/monitor/index.html`
- `docs/PLAN_REVIEW5_20261117.md`

## 5. Sadeleştirme (beşinci değerlendirmenin ana eleştirisi)
- Sayfada şu an **1.022 açılır alan (`<details>`)** var; değerlendirme bunu "ürün değil, arşiv gibi" diye eleştirdi.
- Ana sayfada yalnızca şunlar kalmalı:
  - üç ana bulgu: Teksas alanı %40; mekanizma (hata verimden alana kaydı, USDA üçte iki oranında tepki veriyor); kör test ve revizyon yönü;
  - sınırlar kutusu: üretimde anlamlı değil (nedeni B2), piyasada değer yok (T52, T55), aralıklar geniş;
  - canlı monitör paneli: `latest.json`'daki son kayıt ve kilitler;
  - denetlenebilirlik satırı (3.6).
- Model envanteri, deney listesi ve açılır alanlar ayrı bir "Arşiv" sayfasına ya da sayfanın sonundaki tek bir katlanmış bölüme taşınmalı.
- Deney aralığını "T01–T55" yap.

## 6. Kullanılmaması gereken ifadeler
- "USDA'yı geçtik" (koşulsuz), "USDA'dan daha iyi tahmin ediyoruz".
- "Piyasada para kazandırır" ya da "alım satım sinyali".
- "Tahmin aralıklarımız dar ve kesin". Doğrusu: kalibre ama geniş.
- "Bağımsız olarak doğrulandı". Oklahoma/Kansas p = 0,056; yeni eyalet kilitleri henüz puanlanmadı.
- "Testlerin bir kısmı özel veri yüzünden başarısız". Bu yanlıştı; doğrusu 409 testin hepsinin geçtiği.

## 7. Codex'e verilecek istem
> `docs/research_package/18_SUNUM_GUNCELLEME_CODEX_2.md` dosyasını oku (gerekirse 17. dokümana da bak) ve `docs/portfolio/cottonlens-sunum.html`, `.css` ve `.js` dosyalarını güncelle: bölüm 3'teki yeni içeriği ekle, bölüm 4'teki bağlantıları yeni commit'e ve yollara çevir, bölüm 5'teki sadeleştirmeyi uygula. Her sayıyı adı geçen kanıt dosyasıyla doğrula ve bölüm 6'daki ifadeleri kullanma. Görsel dili koru; masaüstü, 390 px telefon genişliği, açık ve koyu temada yatay kaydırma olmadığını kontrol et. Yalnızca `docs/portfolio/` altındaki sunum dosyalarını değiştir; commit yapma.
