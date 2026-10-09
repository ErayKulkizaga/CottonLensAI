# CottonLensAI

Pamuk piyasası için **araştırma/ispat projesi** ve açıklanabilir tahmin uygulaması.
ICE Cotton No. 2 referansında tahmin katkısı ve yön/pozisyon simülasyonu incelenir;
gerçek alım/satım yapılmaz. Mevcut T+1/T+5 fiyat tahmininde Naive korunuyor. Son
[Texas/NASS T+1 deneyi](docs/NASS_REGIONAL_T1_RESULT_20261009.md) tamamlandı:
676 piyasa fit'i, 2.006 ortak origin/kol; seçilmiş D0/D1 Naive kazancı
−%0,0274/−%0,2556. Sabit tarif %5 hedefini kurtarmadı; yeni grid açılmıyor.

[Zayıf sinyal kontrolü](docs/WEAK_SIGNAL_RESULT_20261009.md) tamamlandı:
3.380 sentetik fit, 20 senaryo, 40.120 OOS satırı bağımsız doğrulandı.
Ham/seçilmiş ortalama kazanç %1,955/%1,875; oracle %3,605. Kilitli karar
`RAW_ONLY_RECOVERS`, fakat seçilmiş sonuç %50 koruma eşiğini yalnız 0,439 yüzde
puan kaçırdı. Bu piyasa başarısı veya küçültmenin ana kusur olduğunun kanıtı
değildir. [Seçim kararlılığı analizi](docs/SELECTION_STABILITY_RESULT_20261010.md)
de tamamlandı: enjekte koşulunda bir doğrulama bloğu çıkarılınca 59/80 karar
değişiyor. Bu, küçültmeyi kaldırmayı veya piyasa başarısı iddiasını desteklemez.
Sonraki ayırıcı aday geçmişten fiyat-MAE kalibrasyonu; otomatik yeni eğitim yok.
İlk sıfır-fit kayıt ve r2 ön kaydı korunuyor.

## Nereden başlanır?

1. [Güncel durum ve kararlar](docs/STATUS.md).
2. [İspat amacı ve yardımcı simülasyonun sınırları](docs/TRADING_RESEARCH_CONTRACT_20261005.md).
3. **Yeni deney önermeden önce** [deney sicilini](research/README.md) kontrol edin.

```bash
python ml/history.py validate
python ml/history.py check --family ridge --horizon 1
python ml/history.py check --feature nass --horizon 5
python ml/history.py check --query availability
```

Komut bağımlılık kurmadan çalışır ve dosya değiştirmez. `RELATED_EVIDENCE`, geçmişte ilgili çalışma bulunduğu anlamına gelir. Tam tarif ve dondurulmuş kapsam için `--proposal` kullanılabilir. Eşleşme bulunmaması bir fikrin hiç denenmediğini kanıtlamaz. Eksik tahminler ve yarım deneyler negatif kanıt sayılmaz.

## Depo düzeni

```text
backend/                 FastAPI, CPU inference, artifact sözleşmeleri ve testleri
frontend/                Angular uygulaması
ml/src/cottonlens_ml/     Veri, özellik, araştırma ve eski eğitim motorları
ml/tests/                Sentetik, zamanlama, kimlik ve sözleşme testleri
ml/notebooks/            İki güncel data/research workbench
ml/notebooks/archive/    Eski deneylerin notebook'ları; yeni çalışma girişleri değil
ml/scripts/              Mevcut kaynak toplama ve operasyon araçları
research/                Deney/fit sicili, düzeltme kanıtları, arşiv indeksi
research/evidence/       Küçük, checksum bağlı doğrulama ve sonuç kayıtları
docs/                    Güncel durum, karar ve kullanım belgeleri
docs/archive/            Tarihsel planlar ve yorumlar; güncel talimat değil
runtime/                 Yerel artifact/import alanı; büyük dosyalar Git'te tutulmaz
.github/                 Kod ve sözleşme CI kontrolleri
```

Python giriş araçları `ml/` altında kalır: `history.py`, `full_year_cpu.py`, `source_bundle.py`, `colab_setup.py` ve mevcut veri inceleme araçları. Kaynak/veri kimlikleri nedeniyle eski tarifler yeni kodla sessizce yeniden başlatılmaz.

## Ölçülmüş son durum

[Texas/NASS T+1 ön kaydı](docs/NASS_REGIONAL_T1_PREREGISTRATION_20261009.md):
aynı 2.006 origin'de ulusal kondisyonun üzerine Texas katkısını sınayacak
**676 piyasa + 1 sentetik** fit bütçesi kilitlenmişti ve deney tamamlandı.
[Sonuç](docs/NASS_REGIONAL_T1_RESULT_20261009.md) bu sabit tarif için negatiftir;
erişim ve vintage varsayımsaldır. Ön kayıt tarihsel belge olarak korunur.

[NASS tek raporlu saat kontrolü](docs/NASS_CLOCK_CASE_20261009.md): 22 hücre
TXT/PDF'lerde eşleşti; metadata tarihsel erişim kanıtı sayılmadı. Sıfır fit,
kabul kapalı. Sonraki iş tek varsayım-duyarlılığı ön kaydı; yeni model/grid yok.

[Texas/NASS denetimi](docs/NASS_REGIONAL_AUDIT_20261009.md), mevcut 311 rapordan
Texas kondisyonunu ve 636 gelişim tablosunu ayırdı; iki gerçek geçmiş-hafta
revizyonu ayrı korundu. 949 girdi değişmedi. Ekim yalnız 46 raporda; tam sezon
ve tarihsel erişim doğrulanmış değil. Sıfır fit, panel karantinada; önceki ulusal
T+5 sonucu bu bölgesel bilgiyi test etmiş sayılmaz.

[İleri yayın görevi](docs/FORWARD_RUNTIME_WINDOW_20261009.md), checksum bağlı
sabit kodla gerçek gündüz toplamasında exit 0 verdi. Yayın penceresinde ağ/skor
işi yapılmaz; yavaş yazımın son saat kontrolü düzeltildi. Zamanında tahmin henüz
0; bu model başarısı değildir. Gece kontrolü için 03:35 Türkiye saatli devam aktif.

[İleri kayıt denetimi](docs/FORWARD_EVIDENCE_AUDIT_20261009.md): 9 Ekim 08:09 UTC
snapshot'ında altı origin'in altısı missing, yayımlanmış tahmin **0**. Kaynak
makbuzu/girdi/saat doğrulayan `research.live --status` tamamlandı; model eğitimi
ve performans skoru yapılmadı. Görev kurulumu ileri tahmin kanıtı değildir.

[Zamanlama deneyi](docs/AVAILABILITY_CLOCK_RESULT_20261005.md), 2016–2023'te her kol/ufuk için aynı 2.006 origin'de 676 küçük CPU fit tamamladı. DXY/WTI daha yeni barları fiilen kullandı; geçmiş Yahoo teslim saatleri **varsayımdır**.

| Sonuç | T+1 Naive'ye fiyat-MAE kazancı | T+5 kazancı |
|---|---:|---:|
| Kontrol, seçilmiş küçültme | −%0,3285 | −%0,8000 |
| Zamanlama müdahalesi, seçilmiş küçültme | −%0,1538 | −%1,3396 |

Bu tarif pratik hedefi geçmedi. Sonuç bütün modellerin veya bütün temel kaynakların sinyalsiz olduğunu kanıtlamaz. Tekrar incelenmiş 2016–2023 ve daha önce görülmüş 2024+ bağımsız holdout değildir.

Eski LSTM'nin 498 Naive / 439 LSTM origin karşılaştırması geçersizdi: aynı pencere için yeniden kurulan T+1 toplu kazanç **−%1,6961**, eski +%2,7647 iddiası geçersiz. Tam LSTM tahminleri bulunmadığından paired güven aralığı yok. TCN sentetik öğrenme kontrolünün yanlış `passed` yorumu ayrıca düzeltildi. [Düzeltme kayıtları](research/evidence/) eski artifact'leri değiştirmez.

[İlk işlem hedefi teslimatı](docs/TRADING_WASDE_DELIVERY_20261005.md), mevcut tahminlerde 2.006 T+1 origin'i yeni fit olmadan inceledi. Seçilmiş müdahalenin brüt proxy sonucu tek aktif yıldan geliyor; kontrol katkısı belirsiz ve gerçek giriş/maliyet doğrulanmış değil. Bölgesel WASDE bilgisi 95 rapor/26 alanlı aday pakete geri alındı, eğitim kabulü verilmedi. Sürümlü karar saati ve CLI/engine tazelik eşitliği düzeltildi; 690 eski dosyanın SHA-256'sı korundu. [Yeni payload Release'i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/research-delivery-20261005) eski yedeğe ek kanıttır.

[Fiyat alanı/seans denetimi](docs/PRICE_SEMANTICS_AUDIT_20261005.md): 268/2.006 hedef barda `close` high–low aralığı dışında. Bu settlement/mark ayrımını gerektirir; doğrudan bozuk tahmin etiketi demek değildir. Ham proxy kazancı yalnız bu barlardan gelmiyor. Veri değiştirilmedi; tahmin araştırması broker/komisyon bilgisine bağlanmıyor.

[Bölgesel WASDE doğrulaması](docs/WASDE_REGIONAL_VERIFICATION_20261008.md), 95 rapor/26 alanlı adayın kaynak değerlerini ve nedensel revizyonlarını ayrı denetler. 117 bilinmeyen revizyon hücresi korunur. Compile/import artık kaynak, tam değer, vintage ve saat kanıtını birbirine bağlar; yeniden checksum'lanan sonraki değerler eski kanıtı kullanamaz. Tarihsel erişilebilirlik onayı ve yeni piyasa becerisi iddiası yoktur; yeni fit başlatılmadı.

## Veriler, modeller ve tam kanıt yedeği

[Bölgesel WASDE T+1 sonucu](docs/WASDE_REGIONAL_T1_RESULT_20261008.md): ön kayıt değişmeden, dört kol × 1.254 origin üzerinde 424 piyasa fit'i tamamlandı. Sayısal D0/D1 seçilmiş Naive kazancı −%0,5229 / −%0,0107; ham kazanç −%10,2134 / −%9,3435. Pratik %5 hedefi bu sabit tarifte desteklenmiyor; **Naive korunur, program otomatik büyütülmez**. 2.682 eski dosya korundu. Saat/vintage varsayımı gerçek erişim kanıtı değildir. [Ek sonuç paketi](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/wasde-regional-result-20261008) tahminleri, 424 fit checkpoint'ini ve ayrı sentetik kontrolü saklar.

[11 Aralık 2018 WASDE kontrolü](docs/WASDE_CLOCK_CASE_20261008.md) tamamlandı: 11/14 Aralık PDF/XML çiftlerinde 392 hücre eşleşti; repost Cotton değerlerini değiştirmedi. Bağımsız üniversite kopyasının on değeri de eşleşti, fakat karar kesiminden önce erişim kanıtı kurulamadı. 2.190 eski dosya değişmedi. Tarihsel kabul ve yeni fit yok.

[Resmî WASDE geçmişiyle son mutabakat](docs/WASDE_AS_REPORTED_20261008.md): 95 raporun 950 temel değeri ve 2.470 toplam alan kontrolü eşleşti; fark yok, 117 bilinmeyen revizyon korundu. CSV teslim saati rapor saatinden ayrı tutuldu. Eski veriler değiştirilmedi; tarihsel eğitim kabulü ve yeni fit yok. [Ek kanıt paketi](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/wasde-as-reported-20261008) yeniden kontrol için resmî export'ları ve çıktıları saklar.

Büyük ham veriler, tahminler, checkpoint'ler, kaynak ZIP'leri ve ledger'lar [evidence-20261005 Release](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/evidence-20261005) altında tutulur. [Arşiv açıklaması](research/README.md) ve `research/archive-summary.json`, mevcut dosyaların kapsamını, eksikleri, redaksiyonları ve SHA-256 değerlerini gösterir. Git geçmişi korunur; bütün eski eğitimin eksiksiz geri kazanıldığı iddia edilmez.

GitHub CLI ile indirme ve belirli bir deneyi ayrı dizine geri kurma:

```bash
gh release download evidence-20261005 --repo ErayKulkizaga/CottonLensAI --dir output/evidence-download
python ml/evidence_restore.py --archive-dir output/evidence-download --destination output/recovered --prefix output/full-year/experiments/research-availability-clock-pilot-v1/
```

Geri kurma arşiv ve dosya checksum'larını denetler, mevcut dosyaların üzerine yazmaz. Model yüklemez, eğitim başlatmaz ve canlı artifact'i değiştirmez. Gizli bilgiler ayıklanan kopyalar orijinal hash'leriyle eşit gösterilmez.

## Uygulamayı açma

Docker Desktop'ın Linux motoru açıkken:

```bash
docker compose up --build
```

Uygulama: <http://localhost:8080>. API: <http://localhost:8080/api/v1/openapi.json>.

`Development fixture` banner'ı sentetik gösterim verisi anlamına gelir. Gerçek artifact `runtime/artifacts/current` içinde checksum doğrulamasıyla yüklenir; eski artifact'in metrikleri güncel araştırma performansı gibi sunulmaz. `live` ve `backtest` kayıtları ayrıdır. Backend'e TensorFlow, MLflow, Jupyter veya CUDA kurulmaz.

```bash
docker compose down
```

Bu komut veritabanını silmez. Artifact kullanım/operasyon ayrıntıları [belge indeksinde](docs/README.md).

## Araştırma araçları

Sonraki iş **FAS ülke/commitment kabul denetimi**, yeni eğitim değildir.
[Kaynak kabul koruması ve ortak-origin hizalaması](docs/FAS_REVIEWED_ALIGNMENT_20261009.md)
sentetik veride doğrulandı; bu, gerçek tarihsel sürüm/saat onayı değildir.
Bekleme sırasında [karantina ülke paneli](docs/FAS_COUNTRY_PREPARATION_20261008.md)
hazırlandı: 752 hafta/3.008 satır, negatif satış ve bilinmeyenler korunur.
29 Temmuz 2021 negatif stok bileşeni o haftanın stok paylarını tanımsız kılar.
Bu panel eğitim girdisi değildir; kaynak kabulü, adapter/test ve ön kayıt kalır.
Mevcut 15 ham kaynağın 752 haftalık ulusal toplamları doğrulandı.
[İki haftalık ülke kontrolünde](docs/FAS_COUNTRY_REPORT_REVIEW_20261008.md) 24 alanın
22'si eşleşti; Çin/Pakistan birikimli ihracatında −55/−53 balya fark korunur.
[Alt sınıf aralık kontrolü](docs/FAS_MAY28_VERSION_AUDIT_20261008.md) iki farkın
yuvarlama varsayımıyla uyumlu olduğunu gösterdi; gerçek neden kanıtlanmadı.
[Tarihsel form denetimi](docs/FAS_HISTORICAL_REFERENCE_20261008.md) dört kod için
2020 tanığı buldu; özgün Çin hücresi hâlâ dış kanıt bekliyor. Aynı PDF/form
araması tekrarlanmaz. Sonraki tek test ilk yayın sürümüne bağlı kesin değerdir;
güncel API bunun yerine geçmez. Saat/vintage ve kaynak kabulü tamamlanmadan
özellikler eğitime alınmaz.
[Güncel sıra ve durma koşulları](docs/RESEARCH_DATA_REENTRY_PLAN_20261005.md).
WASDE pilotu, DXY/WTI saat deneyi ve eski FAS ulusal T+5 tarifleri tekrar açılmaz.

Güncel girişler [data workbench](ml/notebooks/cottonlens_data_workbench.ipynb) ve [research workbench](ml/notebooks/cottonlens_research_workbench.ipynb). Varsayılan akış durum/hazırlık kontrolüdür, eğitim değildir. Arşivdeki notebook'lar yalnız tarihsel protokolü incelemek içindir.

```bash
python ml/full_year_cpu.py --help
python ml/source_bundle.py --output output/new-source.zip
```

Yeni deney otomatik açılmaz. İzin verilen küçük CPU tarifleri ayrı CPU ortamında tek süreç ve en fazla iki thread kullanır; gerçek sequence/GPU ve istatistiksel ARIMA eğitimleri Colab sınırında kalır. CI gerçek piyasa eğitimi yapmaz. Kaynak kurulumları ve kimlik doğrulaması [AGENTS.md](AGENTS.md) ile mevcut kilit dosyalarına uyar. [USDA secret kurulumu](docs/USDA_COLAB_SETUP.md); gerçek anahtarlar depoya veya Release'e konmaz.

## Kontroller

```bash
python ml/history.py validate
# ML için mevcut ayrı CPU/test ortamında, PYTHONPATH=ml/src:ml (Windows: ml/src;ml)
python -m pytest -q ml/tests
cd backend
python -m pytest -q
python -m ruff check .
cd ../frontend
npm ci
npm run build
```

Frontend Node 24.15+ ister. Bağımlılık kimlikleri `ml/uv.lock`, `ml/constraints/`, backend manifesti ve npm lockfile'ında kayıtlıdır. Sentetik test veya build başarısı tahmin/işlem üstünlüğünün kanıtı değildir.

Eski `research-v2-tf-placement` için 7.331 fit ve kısmi on-call için 102 fit makbuzunun model payload kopyaları yerelde yok: toplam 22.299 dosya referansı. Makbuzlar/tahmin kanıtı korunur; bunlar yeniden kullanılabilir tam checkpoint gibi gösterilmez. Yeni zamanlama deneyinin 6.157 dosyası checksum doğrulamasıyla geri kurulmuştur. Kaybolan eski dosyalar yeni eğitimle yeniden üretilmedi.
