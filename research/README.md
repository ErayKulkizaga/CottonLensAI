# Deney geçmişi ve tekrar kontrolü

## Sicil

- `registry.json`: deney/grup/ufuk bazında mevcut OOS kanıtı, origin sayısı, ham/seçilmiş MAE, kaynak/veri kimliği, yıllık kapsam, tahmin dosyalarının SHA-256 değerleri ve kanıt sınıfı. Eski özetler ayrıca **testimony** olarak işaretlidir.
- `trials.json`: gerçekten tamamlanmış, checksum ve dondurulmuş kimliği doğrulanmış ledger **fit makbuzlarının** benzersiz tarifleri. Tek fit ≠ tamamlanmış OOS deney. Model payload'larının yeniden hash'lenmesi bu indeksleme adımının iddiası değildir.
- `evidence/`: legacy LSTM, TCN, zamanlama deneyi ve bağımsız metrik doğrulaması; eski dosyaları değiştiren düzeltmeler değildir.
- `legacy-history.json`: önceki araştırma özetinin değiştirilmeden korunmuş kopyası.
- `organization.json`: taşınan belgeler/notebook'lar, eski yollar, orijinal hash'ler ve nedenleri.
- `archive-summary.json`: Release varlıkları, arşiv manifesti checksum'ı, kapsam ve açıkça dışlanan/gizli bilgileri ayıklanan dosyalar.

Son eklenen kayıtlar `clock-t1-open-close-proxy-v1` (2.006 origin, sıfır yeni fit,
gerçek işlem becerisi için inconclusive) ve `wasde-regional-candidate-v1`
(95 rapor/26 alan, eğitime kabul edilmemiş veri hazırlığı). İlki yeni model eğitimi,
ikincisi tamamlanmış piyasa deneyi değildir. [Sonuç ve yeni Release](../docs/TRADING_WASDE_DELIVERY_20261005.md).

`INCONCLUSIVE`, bir tarifin hiç denenmediği anlamına gelmez. Tamamlanmamış çıktı, sentetik kontrol veya sıfıra küçültülmüş tahmin tüm kaynakta sinyal yokluğunu göstermez. Sicil evrende denenmiş bütün fikirleri kapsamaz; yalnız mevcut kanıtı kapsar.

## Proje komutları

Depo kökünde, Python 3.12+; ek paket kurulmaz:

```bash
python ml/history.py validate
python ml/history.py check --family xgboost --horizon 5
python ml/history.py check --feature nass --horizon 5
python ml/history.py check --query availability
python ml/history.py check --query trading --horizon 1
python ml/history.py check --query wasde-regional
python ml/history.py check --query wasde-regional-numeric-verification
python ml/history.py check --query semantics
python ml/history.py check --query reconciliation
python ml/history.py check --profile availability-clock-pilot-v1 --json
python ml/history.py list
```

Son eklenen iki kayıt `cotton-ohlc-semantics-audit-v1` ve
`ams-cotton-price-reconciliation-v1`: sıfır fit kaynak denetimleridir; yeni
piyasa başarı/başarısızlık deneyi değildir. 998 eski AMS belge hash'i,
997 doğrulanmış rapor tarihi, 996 Close eşleşmesi ve 20 kontrat değişimi
korunur. Tarihi okunamayan bir rapor ve eşleşmeyen bir fiyat açıkça ayrıdır.
[Rapor ve Release](../docs/PRICE_SEMANTICS_AUDIT_20261005.md).

`wasde-regional-numeric-verification-v1` de sıfır-fit veri doğrulamasıdır.
95 raporun sayısal mutabakatı, erişilebilirlik/vintage onayı ve T+1/T+5
piyasa katkısı birbirinden ayrılır. Eski aday kaydı değişmez; bu ek kayıt
ve [doğrulama raporu](../docs/WASDE_REGIONAL_VERIFICATION_20261008.md)
hangi kontrolün gerçekten tamamlandığını gösterir.

Kesin tarif kontrolü: `trials.json` içinden `scope_id` ve `recipes[0]` alınarak `{ "scope_id": "...", "recipe": {...} }` biçiminde bir öneri JSON'u oluşturun:

```bash
python ml/history.py check --proposal output/proposal.json --json
```

`REPEAT_FIT_RECIPE` (exit **3**) aynı tam tarif ve frozen kapsamda tamamlanmış fit makbuzu bulunduğunu gösterir; başarıyı veya tüm deneyin bitmesini göstermez. `RELATED_EVIDENCE` ve `NO_REGISTERED_MATCH` exit **0**; bunlar eğitim izni değildir. Eksik/bozuk sicil veya kanıt checksum'ı exit **2**, işlem durur. Kaynak/veri/politika/dönem farkı otomatik eşdeğer sayılmaz. Kasıtlı tekrarın bilimsel gerekçesini yeni manifestte kaydedin.

Yeni makbuzları read-only kaynaktan indeksleme, mevcut sicilin üzerine yazmadan:

```bash
# PYTHONPATH=ml/src (Windows: ml/src)
python -m cottonlens_ml.research.history_index --source-root output/recovered --output output/new-trials.json
```

İndeks çıktısını mevcut sicille inceleyerek birleştirin ve yeni deneyin prediction kapsamını `registry.json` içine kaydedin. Sadece yeni profil kodunun bulunması “denendi” kaydı yaratmaz.

## Release yedeği

[GitHub evidence-20261005](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/evidence-20261005), bilimsel dosyaları içerik hash'iyle tekilleştirir. Manifest her orijinal göreli yolu, orijinal/public SHA-256 değerini ve varlık/iç yolunu korur. Parçalar bağımsız ZIP'tir; büyük veri Git ağacına eklenmez.

Mevcut ham kaynaklar, model/checkpoint'ler, ledger'lar, tahminler, raporlar, kaynak snapshot'ları ve kesintiye uğramış çalıştırmalar korunur. Yazılım ortamı/cache, kişisel ayarlar, UI ekran görüntüleri ve içeriği güvenle yayınlanamayan veritabanları dışlanır; dışlamalar indekste görünür. Geçmişte kaybolmuş LSTM tahminleri oluşturulmaz. Yerel orijinaller silinmez/değişmez. Ayıklanmış dosya eski hash doğrulamasını geçmez ve eski deney cache'i olarak kullanılmamalıdır.

İndirme/restore komutları ana README'de. Restore araçları model/pickle çalıştırmaz. Başarısız checksum, tehlikeli yol veya mevcut hedef dosya üzerine yazma girişimi durur. Restore edilen eski bilimsel snapshot yeni kodla otomatik devam ettirilmez.

Eski `research-v2-tf-placement` için 7.331 fit ve kısmi on-call için 102 fit makbuzunun model payload kopyaları yerelde yok: toplam 22.299 dosya referansı. Makbuzlar/tahmin kanıtı korunur; bunlar yeniden kullanılabilir tam checkpoint gibi gösterilmez. Yeni zamanlama deneyinin 6.157 dosyası checksum doğrulamasıyla geri kurulmuştur. Kaybolan eski dosyalar yeni eğitimle yeniden üretilmedi.
