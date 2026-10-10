# İleri yayın: gece penceresi denetimi — 10 Ekim 2026

**VERIFIED:** 00:05 UTC yakalama /00:20 UTC yayın penceresinde çalıştırma
ve kesim-öncesi CT=F makbuzu bulunmuyor. Zamanında yayımlanmış tahmin **0**;
altı eski missing origin ve ilk 126 origin kilidi değişmedi.
[Checksum bağlı yeni kanıt](../research/evidence/forward-window-audit-20261010.json).

- Sabit `44f3d16f90af44d9ec0ecbc058055c89a2779e5b3e7ad2006eed19428c0ed965`
  kaynak kimliği ve 226 ML dosyası doğrulandı. Snapshot'ın mevcut CPU ortamında
  `research.live --status` çağrısı exit 0; hash zinciri, seçilen kaynak/girdi
  eşitliği ve Naive/EWMA kilit tarifi geçerli. Yayımlanmış çıktı olmadığından
  gerçek tahmin çıkarımı veya model becerisi doğrulandığı söylenmez.
- Son başarılı çalıştırma 9 Ekim **22:05:04,511 UTC**. Kaynak kimliği ve
  `forward_record_ids` altı mevcut kayda bağlı; 75 run makbuzu kimliği kontrol edildi.
- Windows Power-Troubleshooter Event 1, **22:51:28,057–06:59:28,071 UTC**
  uyku/uyanma aralığını bildiriyor; bu aralık gece penceresini kapsıyor.
- Daha sonraki **07:05:05 UTC** görev isteği `0x800710E0` /Win32 4320 ile
  reddedilmiş: Windows mesajı “İşletmen veya yönetici isteği reddetti.”
  Bu kod gece tetikleyicilerinin hata kodu olarak sunulmaz. TaskScheduler
  Operational günlüğü kapalı; reddin kesin nedeni eldeki kanıtla belirlenemiyor.
- Görev `Ready`, `Interactive`, `WakeToRun=true`, `StartWhenAvailable=false`.
  Bunlar çalışan konfigürasyon gözlemidir; kullanıcı oturumu veya genel güç
  politikasının hatanın kesin nedeni olduğu kanıtlanmadı.

## Korunanlar ve sınırlar

115 kaynak/girdi/forward/run dosyası denetim öncesi-sonrası aynı; eski 13
forward dosyası önceki kamu kanıtıyla eşleşiyor. Önceki 154 sicil kaydı ve
56 kanıt değişmedi; 460 trial kaydı korunuyor. Primary checkout değişmedi.
Aktif writer lock yok; hiçbir lock silinmedi. Yeni piyasa/sentetik fit **0**.

Arşivlenmiş tamamlanmış bar girdilerinde 9 Ekim origin'i henüz yok. Geçmişe
doldurulmadı, yedinci missing kayıt uydurulmadı. Kesim saati, dönemler,
eşikler, kaynak kabulü, görev/otomasyon ve güç ayarları değiştirilmedi.
Ham kotasyonlar, tam run gövdeleri ve makine/hesap bilgili özgün OS XML'i
kamu kanıtına eklenmedi; güvenli projeksiyon özgün event hash'ini taşıyor.
Yerel saatler dış zaman tasdiki değildir.

**Sonraki operasyon işi:** mevcut görevin interactive oturum/çalıştırma
reddini incelemek. Sadece yeni gecenin gelmesini beklemek zamanında yayın
garantisi vermez; güvenlik veya güç politikasını sessizce değiştirmeyin.
Bu denetim araştırma/eğitim izni veya öğrenilmiş model başarısı değildir.
Tarihsel FAS specific-version erişim/değer kabulü hâlâ kapalı; eski PDF/API
denetimleri tekrar açılmadı. Performans skoru hesaplanmadı.
