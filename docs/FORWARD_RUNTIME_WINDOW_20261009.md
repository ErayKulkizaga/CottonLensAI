# İleri kayıt: sabit kaynak ve yayın penceresi — 9 Ekim 2026

**VERIFIED:** Mevcut Windows görevi test edilmiş, değişmez ML snapshot'ına
bağlandı. Gerçek gündüz çağrısı exit 0; CT=F makbuzu 09:09:56 UTC'de alındı.
Son bar 8 Ekim. Altı eski origin hâlâ missing; yeni zamanında yayın **0**.
Bu çalışma operasyon düzeltmesidir, piyasa becerisi veya yeni model eğitimi değildir.
[Checksum bağlı kanıt](../research/evidence/forward-runtime-window-20261009.json).

## Değişiklik ve doğrulama

- 00:15 dahil /00:30 hariç yayın çağrısı yalnız önceden yakalanmış girdileri
  kullanır. O çağrıda Yahoo/USDA indirme, skor hesaplama veya mirror yapılmaz.
  Normal toplama bu pencere dışında devam eder; motor çoğaltılmadı.
- Başlangıç saatini kullanmak, yavaş Parquet yazımı 00:30'u aşınca yanlış
  `published` üretebiliyordu. Sentetik regresyonla yeniden üretildi; gerçek
  çalışmada yazımdan sonra saat tekrar okunur, geciken kayıt missing kalır.
  Altı eski kayıtta yayımlanmış tahmin yoktu; eski piyasa sonuçlarını bu hataya
  bağlamıyoruz. Atomik rename'in küçük dosya sistemi gecikmesi ve dış zaman
  tasdiki yokluğu ayrıca sınır olarak kalır.
- Launcher her çağrıda frozen manifesti ve bütün ML dosya hash'lerini doğrular.
  Run makbuzu gerçek `execution_source_id` ve `forward_record_ids` tutar.
  Snapshot değişirse veri toplama başlamaz; aynı namespace güncellenmez.
- Görev güncellemesi aynı kullanıcının SID'sini doğrular: Windows hesap adını
  SID olarak saklayabilir. İsim metni karşılaştırması gerçek görevde güvenli
  biçimde reddetti; SID çözümlemesi düzeltildi ve yabancı hesap reddi test edildi.
- Özgün görev XML'i özel yedekte saklandı. Üç tetikleyici ve principal aynıdır;
  `WakeToRun=true`. Çalışan görev/writer güncellenmez; başarısız persistence
  eski XML'e döner. Veri dizini, eski kayıtlar ve dirty primary checkout korunur.

ML kaynak kimliği:
`44f3d16f90af44d9ec0ecbc058055c89a2779e5b3e7ad2006eed19428c0ed965`
(226 dosya). Eski kayıtlara bu kimlik sonradan yazılmadı.
Bu sürüm araştırma görevine geri döndürülebilir şekilde dağıtıldı; PR yığını
henüz birleştirilmedi. Öğrenilmiş model veya API/runtime artifact yayını yok.

Salt okunur kontrol mevcut CPU ortamı ve snapshot `ml/src` PYTHONPATH'iyle:

```bash
python -m cottonlens_ml.research.live --store EXISTING_STORE --status
python ml/history.py check --query forward-runtime
```

Görev güncellemesi mevcut `ml/scripts/install-live-task.ps1` üzerinden açık
`-UpdateExisting -ExpectedSourceId ID -BackupPath NEW_PRIVATE_XML -WakeToRun`
parametreleriyle yapılır. Python/repo/store/Drive/secrets yolları korunarak açıkça
verilir. XML yedeği mirror dışında olmalıdır; orijinal üzerine yazılmaz.
Yeni görev oluşturma varsayılanı ve eski launcher kullanımı uyumlu kalır.

## Gerçek takvim engeli ve sonraki kontrol

Bir sonraki barın 00:05 UTC makbuzu ≤00:15, yayını <00:30 olmalıdır.
Türkiye saatiyle 03:35'te aynı sohbet için devam kontrolü aktiftir; değişmeyen
durumda bildirim tekrarlanmaz. Makine kapalı/oturum kapalı ise çalışmaz;
`WakeToRun`, işletim sistemi uyandırma politikasını aşmaz. AC yalnız önemli
uyandırmalar, DC uyandırmalar kapalıydı; genel güç politikası değiştirilmedi.
Gece çalışma veya ilk gerçek yayın henüz doğrulanmış değildir.

13 eski forward dosyası, 36 önceki kanıt dosyası, 140 önceki sicil kaydı ve
trials korunur. Ham kotasyon, özel XML, makine yolları ve tam run gövdesi açık
pakete konmaz; güvenli projeksiyon özgün run'un hash'ini ayrıca taşır.
FAS tarihsel sürüm/erişim kabulü kapalıdır. Bu düzeltme yeni fit izni veya
Naive'yi yenme kanıtı değildir; başka grid/deney otomatik başlatılmaz.
