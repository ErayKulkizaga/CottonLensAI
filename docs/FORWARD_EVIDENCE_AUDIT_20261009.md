# İleri kayıt denetimi — 9 Ekim 2026

**VERIFIED:** 08:09:18 UTC'de dondurulan yerel arşivde altı origin'in
tamamı `missing`; yayımlanmış Naive/EWMA tahmini **0**. Bu, modelin başarısız
olduğunu değil, bağımsız ileri tahmin kanıtının henüz birikmediğini gösterir.
[Makine kanıtı](../research/evidence/forward-evidence-audit-20261009.json)
94 dosyanın checksum'ını, kaynak yakalama saatlerini ve denetim kod kimliğini tutar.
Canlı görev denetim sırasında yeni eksik origin ekledi; son sayı altıdır.

| Origin | Karar kesimi UTC | Barı içeren ilk yerel makbuz UTC |
|---|---|---|
| 2026-10-01 | 2026-10-02 00:15 | 2026-10-02 17:36:12 |
| 2026-10-02 | 2026-10-03 00:15 | 2026-10-03 09:05:07 |
| 2026-10-05 | 2026-10-06 00:15 | 2026-10-08 11:05:08 |
| 2026-10-06 | 2026-10-07 00:15 | 2026-10-08 11:05:08 |
| 2026-10-07 | 2026-10-08 00:15 | 2026-10-08 11:05:08 |
| 2026-10-08 | 2026-10-09 00:15 | 2026-10-09 08:05:05 |

Altı origin'de de arşivlenmiş kesim-öncesi uygun snapshot sayısı sıfır.
60 kaydedilmiş çalıştırmanın başlangıç saati 00:15–00:30 UTC arasında değil.
Bu, bütün görev çağrılarının kaydı değildir; bilgisayar kapalıydı veya sağlayıcı
veriyi geç verdi diye kesin neden çıkarılamaz. Veri indirme, eğitim, hedef
skoru veya eski tahminleri yeniden üretme yapılmadı.

## Düzeltilen somut hata

`research/live.py --status` önceden yalnız lock varlığını ve JSON dosyası
sayısını veriyordu. Kayıtların doğruluğunu veya gerçekten yayımlanmış tahmin
sayısını denetlemiyordu. Şimdi mevcut `research/prospective.py` üzerinden:

- Lock politikası, kayıt zinciri ve girdi checksum'ı doğrulanır.
- Seçilen kaynak makbuzu bulunur; kaynak checksum'ı ve girdiyle aynı geçmiş
  olduğu doğrulanır. Makbuzun bar tarihleri tamamlanmış günlük bar olmalıdır.
- Yayımlanmış kayıt için yakalama ≤00:15; kayıt 00:15 dahil, 00:30 hariç;
  Naive ve EWMA çıktıları kilitli tarifle aynı olmalıdır.
- Eksik kayıtlar, açık yayın pencereleri ve henüz kaydedilmemiş kaynak origin'leri
  ayrılır. Hafta sonu/seans takvimi uydurulmaz; eksikler tamamlanmaz.
- Bozuk kanıt veya aktif/kesilmiş writer varsa exit 2; durum komutu ağ,
  writer, eğitim ve performans hesaplama yoluna girmez.

```bash
# ML bağımlılıkları mevcut ayrı CPU ortamında; PYTHONPATH=ml/src
python -m cottonlens_ml.research.live --store EXISTING_STORE --status
python ml/history.py check --query forward-evidence
```

`EXISTING_STORE`, `forward/` ve `market/` içeren mevcut arşiv köküdür.
Olmayan dizinde lock veya dosya oluşturulmaz. Büyük arşivde bu komut bütün
makbuzları okur; ağ servisi veya ucuz sağlık ping'i değildir.
Normal collector ve eski aday-model prospective yolu bu PR ile değiştirilmedi.
Runtime/API artifact şeması değişmedi; yalnız araştırma CLI durum çıktısı genişledi.

## Koruma ve sınırlar

94 dosya değişmeden ayrı yerel snapshot'a kopyalandı; denetim orada tekrarlandı.
Ham kotasyonlar ve makine yolları public kanıta konmadı. Kaynak makbuzlarında
yeniden dağıtım incelemesi tamamlanmamış; hash envanteri ham veri yerine geçmez.
Eski bilimsel kanıtlar ve sicil kayıtları korunur; yeni kayıt piyasa deneyi değildir.
Lock `29cb4f5d…` ilk 126 origin'i **eksikler dahil** kilitler: 120 slot kalır;
bundan sonraki tüm yayınlar zamanında olsa bile kapsam en çok 120/126 olur.
Performans mevcut ayrı score yolunda tüm hedefler olgunlaşana kadar kapalıdır.

Windows görevi hâlâ korunmuş ana checkout'u kullanıyor. Bu PR görev/host kurmaz,
gece boyunca çalışma garantisi vermez veya yeni kodu o checkout'a dağıtmaz.
Sentetik kontroller gerçek ileri performans veya tarihsel FAS kabulü değildir.
Bu operasyon sorunu eski OOS model başarısızlıklarının nedeni olarak sunulmaz.

## Tek sonraki doğrulama

Bir sonraki gerçek Cotton barı için mevcut görevde **00:05 UTC yakalama →
00:20 UTC yayın** zincirinin gerçekten çalıştığı makbuzla gösterilmeli.
Başarı: barı içeren makbuz ≤00:15, yayımlanan kayıt <00:30, aynı kaynak/input
checksum'ları; başarısızlık: eksik kalır, neden kaydedilir. Saat değiştirme,
geçmişi doldurma veya yeni model fit'i bu kontrolün çözümü değildir.
Yeni host/zamanlayıcı veya cohort değişikliği ayrı açık karar gerektirir.
FAS tarihsel sürüm engeli ayrıca devam eder; bu denetim kaynak kabulü sağlamaz.
