# Range-EWMA B kolu: ileri doğrulama planı — 13 Kasım 2026

Durum: **PLAN — kilitlenmedi, toplama başlamadı.** Bu belge, A kolunun
(`range-ewma-confirmatory-t1-v1`, CONFIRMED, `8701743`) ön kayıtta tanımladığı
ileri doğrulama için uygulanabilir bir tasarım önerir. Aşağıdaki **Karar
gerektirenler** bölümü onaylanmadan kod kurulmaz, görev açılmaz, kilit atılmaz.
EWMA referansı ve fiyat kapıları değişmez.

## Neden ileri doğrulama gerekli?

A kolu tarihsel proxy veride yapıldı. İleri test şu riskleri ortadan kaldırır:

- Yahoo verisinin sonradan revize edilmiş olması.
- Geçmiş barların karar anında farklı görünmüş olması.
- Keşif sürecinin seçim etkisi.

İleri testte her tahmin, hedef seans kapanmadan önce zaman damgalı olarak
dışarıya (private GitHub) yazılır.

## Dondurulmuş içerik (A kolundan aynen)

- **Tahminci:** `research/range_ewma.py` (`e8255b5d…`): 0,5·EWMA(r²) +
  0,5·EWMA(k·Parkinson), α=0,06, yalnız-geçmiş k (≥250 geçerli bar), geçersiz
  barda aynı günün r²'si.
- **Kayıp ve kural:** QLIKE, havuzlanmış eşit ağırlık, ortak tarih bloklu
  CI20/CI60 (10.000 örnek, seed 42).
  - CONFIRMED için ortalama > 0, iki CI alt sınırı > 0 ve piyasaların en az
    ⌈2/3⌉'ünde pozitif ortalama gerekir.
  - Kırpmalar ve T+5 raporlanır; karar kapısı değildir.
- **Piyasalar (8):** A kolunda kabul edilen ZW, KE, ZL, SB, KC, CC, HE ve CT=F.
  CT ileri dönemde keşif verisi değildir, bu nedenle havuza girer. CT ayrıca tek
  başına raporlanır; tek başına gücü düşük olduğu için CT tek başına bir karar
  kapısı değildir.

## Güç: süre seçimi

Havuzlanmış kuralın geçme olasılığı. Hesap, 2011–2026 tarihsel kazanç
serilerinden 150 rastgele ortak pencere ve pencere başına 600 bootstrap örneğiyle
yapıldı (`power.py`, planlama amaçlı):

| Süre (seans) | ~Takvim | 8 piyasa (HE dahil) | 10% eksik gün | 7 piyasa (HE hariç) | 10% eksik gün |
|---:|---|---:|---:|---:|---:|
| 126 | 6 ay | 0,43 | 0,40 | 0,55 | 0,48 |
| 252 | 1 yıl | 0,77 | 0,75 | 0,76 | 0,67 |
| 504 | 2 yıl | 1,00 | 1,00 | 0,89 | 0,95 |
| 756 | 3 yıl | 1,00 | 1,00 | 1,00 | 1,00 |

Piyasa başı ortalama kazanç (2011–2026): ZW 0,011; KE 0,018; ZL 0,013;
SB 0,027; KC 0,012; CC 0,019; CT 0,022; HE 0,356.

**Öneri: 504 seans (~2 yıl, bitiş ~Kasım 2028).** Bu süre, HE katkısı olmadan
da ~0,9 güç verir. Bu sayılar iyimserdir: ileri etkinin tarihsel ortalamaya
eşit olduğunu varsayar ve pencereler örtüşür. Etki yarıya inerse güç belirgin
düşer; bu senaryo ayrıca hesaplanmadı.

Bu tablo planlama içindir; A kolunun tarihsel kazanç serilerinden rastgele ortak
pencereler alınarak hesaplanmıştır. İleri etki büyüklüğünün tarihsel ortalamayla
aynı olacağı varsayılamaz. Sonuçlar HE'nin büyük katkısına duyarlıdır.

## Saat ve veri kuralları

- **Günlük koşu:** Pazartesi–Cuma 22:30 UTC, tek tekrar 22:50 UTC
  (TSİ 01:30 / 01:50). Gerekçe: Bütün piyasalar kapanmış olmalı (en geç
  ~19:30 UTC) ve Yahoo'ya uzlaşma gelmesi için zaman kalmalı. Koşu, CBOT gece
  seansı (yaz saatinde ~00:00 UTC) açılmadan önce bitmeli. Mevcut CT 00:15 UTC
  saati bu nedenle panel için kullanılmaz.
  **Kilitten önce doğrulanacak:** CME, ICE US ve ICE seans/uzlaşma saatleri ile
  yaz/kış saati geçişleri. Kontrol yalnız resmi borsa sayfalarından yapılır.
- **Origin:** O gün koşuda gözlenen, tarihi koşunun UTC tarihine eşit veya ondan
  küçük olan son tamamlanmış bar. Koşunun UTC tarihinden sonraki tarihli barlar
  (yarım bar) atılır ve kaydedilir. Bir piyasada o gün yeni bar yoksa o piyasa
  için origin yoktur (tatil veya veri gecikmesi).
- **Tahmin girdisi:** O koşudaki tam geçmiş indirmesi. Ham dosyanın SHA-256'sı,
  tahmin, k, iki EWMA durumu ve yeni bar tahmin payload'ına yazılır. Önceki
  barların revizyonu durumu değiştirebilir. Bu değişiklik makbuzda iki ham hash
  arasındaki farkla kaydedilir, sonradan düzeltilmez.
- **Hedef:** Origin barı ile o piyasanın bir sonraki barı.
  - Birincil: Hedef tarihinden sonraki ilk 5 seans içindeki son koşunun (olgunluk
    anlık görüntüsü) kapanışları.
  - Duyarlılık: İlk gözlenen kapanışlar.
- **Zaman damgası:** Her koşunun tahmin paketi o gece private GitHub Release
  asset'i olarak yüklenir. Hedef seans kapanmadan önce yüklenmeyen tahmin
  **geçersizdir**, skorlanmaz ve doldurulmaz. Yükleme, taze indirme ile hash
  doğrulamasından geçer. Bu, mevcut `ct_forward_backup.py` düzenini izler.
- **Eksik gün:** Bilgisayar kapalıysa veya kaynak hatalıysa origin eksik kalır.
  Eksiklik oranı piyasa başına raporlanır. %25'i aşan bir piyasa havuzda kalır
  ama "eksik" diye işaretlenir. Kural sonradan değiştirilmez.
- **Ara skor yok:** Bitiş tarihinden önce kayıp hesaplanmaz veya gösterilmez.
  Yalnız operasyon sağlığı (koşu sayısı, yükleme başarısı, eksik origin) izlenir.

## Uygulama (onaydan sonra)

1. `research/range_ewma_forward.py`: koşu, origin seçimi, payload, yükleme ve
   bitişte skor. A modülünü import eder, formülü kopyalamaz. Sentetik testler
   yazılır; ağ ve saat bağımlılıkları mock edilir.
2. Ayrı hash dizininde dondurulmuş runner. `pythonw.exe` doğrudan çalıştırılır,
   konsol açılmaz. Mevcut `ct_forward_daily.py` güvenlik desenleri korunur.
3. Yeni Windows görevi (önerilen ad `CottonLens-RangeEWMA-Daily-v1`). Saatlik,
   oturum açma veya kaçanı sonradan çalıştırma tetikleyicisi yoktur. Eski
   görevler kapalı kalır.
4. Kilit: ilk koşudan önce plan, kod ve görev XML'inin hash'leri private
   Release'e yüklenir. İlk origin, kilitten sonraki ilk tam koşudur.

## Karar gerektirenler

1. **Süre / bitiş tarihi:** Güç tablosuna göre seçilir ve kilitte sabitlenir.
2. **Toplama mekanizması:** yeni günlük Windows görevi (bilgisayar 22:30 UTC'de
   açık ve oturum açık olmalı) mi, yoksa başka bir yöntem mi?
3. **GitHub:** Dalın push edilmesi, ön kayıt / A sonucu ve günlük tahminler için
   private Release yüklemeleri.
4. **Sicil:** A sonucunun ve B kilidinin `registry.json`/`STATUS` kaydı.
