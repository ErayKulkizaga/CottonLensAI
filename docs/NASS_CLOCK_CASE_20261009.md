# NASS 30 Mayıs 2023: içerik doğru, tarihsel erişim saati açık

**VERIFIED:** Mevcut TXT, ulusal resmî PDF ve Texas PDF'sinde kontrol edilen
**22 hücre** eşleşti: **21 sayısal + 1 `(NA)`**. **INCONCLUSIVE:** kullanılan
değerlerin ilk karar kesimi **31 Mayıs 2023 00:15 UTC** öncesinde erişilebilir
olduğunu kanıtlayan, sürüme bağlı tarihsel tanık bulunamadı. Kanıt eksikliği,
raporun o anda yayımlanmadığı anlamına gelmez.

[Makine kanıtı](../research/evidence/nass-clock-case-20261009.json) ve
[çevrimdışı yeniden üretim paketi](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/nass-clock-case-20261009).
Bu tek raporluk kontrol tamamlandı: **0 fit, 0 kabul edilen satır**. Gece
görevi/otomasyon, canlı runtime ve eski sonuçlar değişmedi.

## İçerik doğrulaması

Gözlem haftası **28 Mayıs 2023**, release günü **30 Mayıs 2023**.
Ulusal PDF'nin basılı 4–5. sayfaları ve Texas PDF'nin 2. sayfası iki bağımsız
okuyucuyla (`pypdf 6.10.0`, `pdfplumber 0.11.9`) çıkarıldı; ayrıca görsel olarak
incelendi. Bu iki okuyucu aynı yayıncıdan gelen dosyaları inceler; bağımsız
ekonomik veri kaynakları sayılmazlar.

| Satır / sütun sırası | Mevcut TXT | Ulusal PDF | Texas PDF |
|---|---|---|---|
| Texas planted: önceki yıl / önceki hafta / bu hafta / 2018–2022 ortalaması | 58 / 35 / 50 / 54 | 58 / 35 / 50 / 54 | 50 / 35 / 58 / 54; sıra bu hafta / önceki hafta / önceki yıl / ortalama |
| Texas squaring: aynı ulusal sıra | 11 / `(NA)` / 5 / 10 | 11 / `(NA)` / 5 / 10 | Bu tablo yok |
| Texas kondisyon: very poor / poor / fair / good / excellent | 2 / 19 / 51 / 24 / 4 | 2 / 19 / 51 / 24 / 4 | Cotton kondisyon tablosu yok |
| Ulusal kondisyon: aynı sıra | 1 / 12 / 39 / 41 / 7 | 1 / 12 / 39 / 41 / 7 | Kapsam dışı |

Texas PDF'si **yalnız planted'ın dört hücresini** ayrıca doğrular; bütün
kondisyon/gelişim panelini doğrulamaz. `(NA)` sıfır yapılmadı. Derlenmiş eski
ulusal satır, manifest ve 2023 yıllık denetim kimlikleri de yeniden doğrulandı.

## Saat alanları birbirinden ayrıldı

| Alan | Gözlenen değer | Ne kanıtlar? |
|---|---|---|
| NASS takvimi | 30 Mayıs, 16:00 ET = **20:00 UTC** | Planlanan saat; fiilî erişim değil |
| ESMIS sayfası tarih alanı | **2023-05-30T12:00:00Z** | Tarih etiketi; takvimden sekiz saat önce olması teslim saati diye kullanılamaz |
| Ulusal PDF CreationDate / ModDate | 30 Mayıs **18:46:18 UTC** | Dosya metadata'sı; sunulduğu saat değil |
| Ulusal PDF güncel HTTP Last-Modified | 30 Mayıs **20:00:59 UTC** | 2026 yanıtındaki sunucu metadata'sı; tek başına tarihsel erişim tanığı değil |
| Texas PDF CreationDate / ModDate | 30 Mayıs **21:00:33 / 21:03:03 UTC** | Dosyanın yazdığı `−05:00` offset aynen çevrildi; ET takvimiyle karıştırılmadı |
| Texas PDF güncel HTTP Last-Modified | **15 Ağustos 2024 17:51:46 UTC** | 2023 ilk yayınını belirlemiyor |
| Mevcut TXT ingestion | **29 Eylül 2026 20:26:36 UTC** | Bizim kopyayı edinmemiz; 2023 erişimi değil |
| İki PDF ingestion | **9 Ekim 2026 12:03:26–29 UTC** | Bu incelemede HTTP 200 ile alındılar |

Metadata ve içerik, raporun 30 Mayıs üretimiyle uyumludur. Bunun kullanılan
**ilk vintage** ve kesim-öncesi teslim olduğunu söylemek için yeterli değildir.
`available_at=null`, `first_version_verified=false`, `model_eligible=false`,
`release_allowed=false` korunur. `public.py` kabul kuralı gevşetilmedi.

Üç CDX sorgusu **20230530–20230602**, status200, en fazla20 kayıt ile sınırlıydı:
Cornell TXT ve yayın sayfası **ReadTimeout**, Texas PDF **HTTP503 / Internet
Archive: Temporarily Offline**. URL, parametre, sorgu zamanı ve hata/ham yanıt
pakette korunur. Cornell TXT eski host eşleştirmesi **varsayımsal** URL'dir;
doğrulanmış ilk yayın adresi denmedi. Timeout/503 boş indeks veya geçmişte yayın
yokluğu sayılmadı. **Sorgular tekrar edilmeyecek; kapsam genişletilmeyecek.**

Mevcut düzeltme mutabakatında 30 Mayıs için eşleşen notice yoktur. Bu, revizyon
yokluğu veya ilk sürüm kanıtı değildir. Yalnız bu olayın içerik uyumu doğrulandı;
311 haftanın tüm saatleri veya tüm sezon kabul edilmedi.

## Karar ve tek sonraki iş

Saat boşluğu, önceki ulusal NASS T+5 negatif sonucunun gösterilmiş nedeni
değildir: bu yeni Texas paneli o modele verilmemiştir. Ulusal T+5 çalışması
Texas kondisyon/gelişim bilgisini veya T+1'i elemez; içerik eşitliği de artımlı
tahmin katkısı göstermez.

**Sonraki tek iş:** mevcut Texas karantina panelinin artımlı T+1 katkısı için
**bir erişilebilirlik-varsayımı duyarlılık pilotunun ön kaydını** hazırlamak.
Önce sicilde ulusal NASS, zamanlama ve WASDE sonuçlarının kapsamı kontrol edilir.
Kullanılacak alanlar/eksiklik, saat ve gecikme varsayımları, aynı origin/target,
eğitim-içi preprocessing, olgunlaşma/purge, sabit model/seçim, ham/küçültülmüş
sonuçlar ve tam fit bütçesi hesaplanıp kilitlenmeden eğitim başlamaz. Eksik
erken ekim sezonunu tamamlanmış saymak veya null'ı sıfırla doldurmak yasaktır.
Gerçek PIT kabulü bu ön kayıtla açılmaz; sonuç üretim erişilebilirliği veya
bağımsız holdout kanıtı olarak sunulmaz. Yeni grid/model/otomatik kaynak araması yok.

## Yeniden üretim ve koruma

Release'i ayrı dizine açın. Python3.12+, mevcut `pypdf6.10.0` ve
`pdfplumber0.11.9` ortamında, **ağsız**:

```bash
python replay_case.py .
python test_replay.py
```

Çıktı `replay-result.json` ile JSON olarak eşit olmalı; dış `checksums.json`,
17 girdi için `inputs.json`, betikler ve frozen ML ZIP iç manifesti doğrulanır.
`capture_http.py` yalnız gerçekleşmiş sorguların tarifidir; yeniden çalıştırmak
eski yanıtı yeniden üretmez. Betikler ML source kimliğine sessizce dahil değildir:
ayrı `audit_code_id` ve üç betiğin SHA256'sı Git kanıtında kaydedilmiştir.

Sekiz çevrimdışı kontrol geçti: eksik/simge, çift satır/tablo, sütun sayısı,
geçersiz yüzde, crop sınırı, restore/hash bozulması ve HTTP metadata'sından
erişim onayı üretilmemesi. Ruff geçti. ML/backend/frontend kodu değişmedi;
GPU doğrulaması/eğitimi yapılmadı. ML source `ec003b687a…`, **229 dosya** aynı.

Önceki **949 NASS girdi**, **142 sicil kaydı**, **37 kanıt JSON'u + `.gitkeep`**,
`trials.json`, 13 ileri kilit/girdi/origin dosyası ve dirty ana checkout korunur.
Yeni sicil kaydı `nass-clock-case-v1` **INCONCLUSIVE** kaynak/saat incelemesidir;
metric ve tahmin alanı boş, piyasa fit'i sıfırdır. Paket anahtar, private task
XML'i, canlı quote verisi veya makineye özel yollar içermez.
