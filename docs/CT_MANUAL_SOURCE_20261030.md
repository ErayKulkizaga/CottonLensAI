# Manuel ileri risk kaydı: kaynak kabulü

Profil `ct-manual-source-admission-v1`. **Sıfır fit, sıfır tahmin; ileri
kohort veya model yayını yok.** [Tarihsel own-level aday](CT_LEVEL_DXY_CONTROL_20261030.md)
ve başarı kapıları değişmez. Saatlik Windows görevi ve üç tetikleyicisi
kapalıdır; yeni otomasyon kurulmaz.

## Doğrudan gösterilen sorun

30 Ekim 2026 09:58:30 UTC'de gözlenen ham Yahoo CT=F yanıtında 610 tarih
vardır. 503 tamamlanmış fiyat barı, 106 tamamen boş OHLCV yer tutucusu ve
henüz kapanmamış bir 30 Ekim barı ayrı korunur. Boş satırlar Cotton seansı
sayılmaz, fiyat doldurulmaz. Açık bar geçmiş özelliklerine alınmaz.
İncelenen resmi IFUS **Cotton** takviminde 2026 için beklenen 208 tamamlanmış
seansın tamamı vardır; ABD federal/Canola/IFEU takvimi kullanılmaz.

Eski 4.205 satırlık seed ile ortak 476 fiyatın 475'i tam eşittir. Tek fark
eski dosyanın son tarihi 22 Eylül 2026'dadır: eski close **82,8799972534**,
yeni close **79,6900024414**. Ham OHLCV de farklıdır. Eski toplamanın özgün
HTTP makbuzu bulunamadığından nedenin açık bar, sağlayıcı revizyonu, kontrat
alias'ı veya başka bir kusur olduğu **kanıtlanmış değildir**. Sonradan alınan
yanıt, tarihsel doğru settlement veya ilk vintage kanıtı değildir.

Varsayılan `append_snapshot` bu revizyonu gerçekten reddetti. Sessizce eski
son fiyatı tutup 27 yeni seansı eklemek son EWMA'yı **0,0002961505521** yapar;
açıkça tanımlanmış yeni seed sürümünde **0,0002527680607** olur: fark **%17,162964**.
Bu yalnız girdi-durumu hassasiyetidir; eski modelin hatası, tahmin kazancı
veya piyasa başarısı değildir. 2026 son satırı 2016–2023 fit/sonuç kapsamı
dışındadır; eski tarihsel tahminler değiştirilmez ve bu farkla açıklanmaz.

## Uygulama ve sınır

`research.ct_manual_source` yalnız saklanmış girdilerle çalışır. HTTP durum,
ham byte hash, saat sırası, CT=F/günlük/NY metadata, tüm OHLCV uzunlukları,
tekil tarihler, pozitif fiyat, tamamlanma sınırı ve takvim boşluklarını
kontrol eder. Regular close 14:20 **America/New_York** ile değerlendirilir;
DST sabit UTC farkıyla taklit edilmez. Erken tatil kapanışları için bu
muhafazakâr bir üst sınırdır, yayın/settlement timestamp'i değildir.
Takvim kapsamı 1 Ocak 2027'de biter; sonrasına tatil tahmin edilmez.

Varsayılan çalışma `REJECTED_REVISION` raporu üretir. Ayrı
`ct_manual_seed_decision.json`, yalnız bu iki özgün dosyanın hash'lerine ve
tek terminal değişikliğe bağlı **PROPOSED_PRE_COHORT_SEED_VERSION** kararıdır.
Eski seed'in üzerine yazmaz; 4.232 satırlık ayrı öneri üretir. Ek geçmiş
revizyonu, örtüşmede tarih farkı veya kilitlenmiş/tahmin üretilmiş durum
bu yoldan geçirilemez. Öneri **ileri eğitim/tahmin yetkisi değildir**.
Yeni protokol kilidinden sonra bu başlangıç istisnası kullanılamaz.

Yahoo metadata `NYM`, `ALTSYMBOL`, `Cotton Mar 26` yazıyor. Bu, belirli ICE
kontratını kanıtlamaz; tek başına sayısal fiyatların yanlışlığını da kanıtlamaz.
Çalışma sağlayıcı-proxy tahmini olarak nitelenir. Close'un traded high/low
dışında olması işaretlenir; settlement farkı olasılığı yüzünden sessizce
silinmez veya düzeltilmez. Tarihsel yayın/vintage sınırı açık kalır.

## Kontroller ve yeniden üretim

41 dar test/Ruff geçti. Ayrı doğrulayıcı 610 ham satırı, 503 tamamlanmış
fiyatı, 208 takvim seansını, 476 ortak fiyatı, değişmeyen ilk 3.520 tarihsel
fiyatı ve 4.231 skaler EWMA güncellemesini denetledi. Fit, optimizer ve ağ
girişleri kapalı çalıştırıldı; eski seed/raw/trials byte hash'leri değişmedi.

İlk bağımsız okuyucu CSV'nin boş close hücrelerini string dtype'a taşıdığı
için doğru quote eşitliği kontrolünü geçemedi. Başarısız `strict-v1` kaynak/
çıktısı korunur; boş hücreleri açık NA olarak okuyan v2 ve gerçek rapor
regresyonu geçti. Bu başarısız ilk sürüm kabul edilmiş kanıt sayılmaz.

```powershell
python -m research.ct_manual_source --repo . --seed SEED.parquet --source-dir SOURCES --output NEW_STRICT
python -m research.ct_manual_source --repo . --seed SEED.parquet --source-dir SOURCES --initial-version-decision research/ct_manual_seed_decision.json --output NEW_PROPOSAL
python -m research.verify_ct_manual_source --execution NEW_PROPOSAL
```

Sonraki tek bilimsel adım: taze GitHub geri kurması doğrulanmış bu öneriden
sonra, aynı own-level learner/EWMA için **manuel ileri protokolün** örneklem,
ilk uygun origin, makbuz/cut-off, olgun etiket, refit ve eksik kayıt kurallarını
kilitlemek. 00:15 UTC kesimi / 00:30 öncesi yayın penceresi kaydırılmaz;
eski 126-baseline kohortu kullanılmaz, kaçan origin'e tahmin doldurulmaz.
Yeni model/grid veya gerçekleşmemiş ileri başarı ilan edilmez.
