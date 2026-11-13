# Range-EWMA risk tahmini: doğrulayıcı ön kayıt — 13 Kasım 2026

Profil: `range-ewma-confirmatory-t1-v1`. Durum: **PREREGISTERED_NOT_RUN**.
Veri indirilmedi, skor hesaplanmadı, fit yok. Kod: `research/range_ewma.py`;
testler: `research/tests/test_range_ewma.py`. Hash'ler:
`research/evidence/range-ewma-prereg-20261113.json`.

Bu ön kayıt fiyat kapılarını (%5 MAE, %53/%55 yön, 6/8 yıl) ve EWMA
referansını değiştirmez. Başarılı sonuç bile EWMA'nın yerine geçme, ileri
başarı veya işlem becerisi kanıtı değildir.

## Soru

Günlük yüksek–düşük aralığından (Parkinson) gelen bilgi, değiştirilmemiş proje
EWMA'sının (α=0,06, kapanış getirisinin karesi) kaçırdığı **ertesi seans**
varyans bilgisini, keşfin yapılmadığı piyasalarda da taşıyor mu?

## Bu hipotez nereden geldi (keşif, doğrulama değil)

Claude'un 13 Kasım sıfır-fit tanıları (`output/claude-findings-20261113/`, git-ignored):
CT=F 2010–15 / 2016–23 / 2024–26 Eylül ve ZC=F, ZS=F 2010–23 verilerinde
0,5·EWMA(r²) + 0,5·EWMA(k·Parkinson) karışımı EWMA'yı QLIKE'ta geçti. Örneğin
CT 2016–23'te +0,0236, CI20 [0,0115; 0,0378] ve 8/8 yıl. Saf büzülme kontrolleri
ve geçmiş-gün plasebosu geçmedi.

Bu sonuçlar **görülmüş veridir**. Yaklaşık 20 varyant denendi ve çoklu
karşılaştırma düzeltmesi yapılmadı. Özgün form, yani sabit 2010–15 ölçeği, sonra
genişleyen geçmiş-only ölçeğe çevrildi. Bu tercih, 2024–26 dahil sonuçlar
görüldükten sonra yapıldı. Bu nedenle CT=F, ZC=F ve ZS=F bu hipotez için
**temiz değildir** ve doğrulayıcı panele girmez. CT 2024+ dönemi de bu hipotez
için artık bağımsız denetim sayılmaz. Bilinen zayıflıklar:

- Kazanç kuyruklarda yoğun. Üst %5 baseline kaybı kırpıldığında CT'de kazanç negatif.
- Geçersiz barlarda ffill geri dönüşü kazancı yok ediyor.
- CT barlarının %19,6'sı geçersiz.

Önceki ilgili kayıt: `RISK_INFORMATION_PREFLIGHT_20261013.md` içindeki log
range/EWMA moment screen'i geçmedi. O kayıt bir tahmin karşılaştırması değil,
yıl içi merkezlenmiş kovaryans tanısıydı; sonucu korunur. `history.py check`:
`parkinson` için NO_REGISTERED_MATCH; `ewma` sorgusu iki named-EWMA-scale kaydı
döndürür. Eşleşme olmaması yenilik kanıtı değildir.

## Dondurulmuş tahminci (sıfır fit)

Her origin t ve sonraki seans hedefi için:

- `r²_t = ln(C_t/C_{t−1})²`.
- Geçerli bar koşulu: `L>0`, `H>L`, `H ≥ max(O,C)` ve `L ≤ min(O,C)` (1e−9 tolerans).
- `P_t = ln(H_t/L_t)² / (4 ln 2)`.
- `k_t = Σ r² / Σ P`; toplamlar yalnız t'den **önceki** geçerli barlardan alınır,
  en az 250 geçerli bar gerekir. Başka kalibrasyon yoktur.
- Proxy: bar geçerliyse `k_t·P_t`, değilse aynı günün `r²_t`'si. ffill yoktur.
- `EWMA_t = EWMA_{α=0,06}(r²)` (proje fonksiyonuyla birebir aynı, testli).
  `RangeEWMA_t = EWMA_{α=0,06}(proxy)`.
- Aday `h_t = 0,5·EWMA_t + 0,5·RangeEWMA_t`.
- T+5 için `5·h_t` ile `Σ_{i=1..5} r²_{t+i}` karşılaştırılır.

α, karışım ağırlığı, 250 eşiği, geri dönüş ve tahminci (Parkinson; GK/RS değil)
değiştirilmez. Doğrulayıcı veride başka varyant skorlanmaz.

## Doğrulayıcı panel (A kolu, tarihsel, görülmemiş piyasalar)

Yahoo Finance sürekli vadeli günlük OHLC (`auto_adjust=False`, mümkün olan en
erken tarihten itibaren) kullanılır. Bu proxy kaynaktır; kontrat kimliği,
seans saati ve vintage doğrulanmış değildir.

Piyasalar (13): ZW=F, KE=F, ZM=F, ZL=F, ZO=F, ZR=F, SB=F, KC=F, CC=F, OJ=F,
LE=F, HE=F, GF=F. Hariç tutulanlar: CT=F, ZC=F, ZS=F (keşifte görüldü).
Liste skorlardan önce kapanır; sonradan piyasa eklenmez veya çıkarılmaz.

Origin'ler: tahmin tanımlıdır ve hedef tarihi en geç **2026-10-30**'dur. Takvim
kaynağın kendi işlem günleridir; satır düşürülmez veya doldurulmaz. Pozitif
olmayan veya sonsuz kapanış görülürse o piyasa durur ve kayıt hata olarak tutulur.

### Aşamalar (sıra zorunlu)

1. **İndirme (ayrı kullanıcı onayı gerekir):** Ham dosyalar ayrı klasöre yazılır
   ve SHA-256'ları kaydedilir. Fiyat incelenmez, grafik çizilmez.
2. **Kabul:** `python research/range_ewma.py admit ...`. Yalnız veri kalitesi
   okunur; kayıp hesaplanmaz. Bir piyasa şu koşulların hepsini sağlarsa kabul
   edilir:
   - en az 2.000 origin,
   - geçerli bar oranı en az %80,
   - origin aralığında 10 takvim gününü aşan boşluk yok.

   Kabul çıktısı commit'lenir. Reddedilen piyasa gerekçesiyle raporlanır.
   %80 eşiği CT'nin %80,4 oranı bilinirken seçildi; bu açıkça kaydedilir.
3. **Skor:** `python research/range_ewma.py score --admission ... --prereg
   docs/RANGE_EWMA_PREREGISTRATION_20261113.md --expected-prereg-sha256 ...`.
   Kod, ön kayıt hash'i veya kabul sonrası değişen veri hash'i uyuşmazsa durur.

### Karar kuralı

Kayıp: QLIKE `log p + a/p` (proje fonksiyonu). Kazanç = EWMA − aday.

- **Birincil (T+1, havuzlanmış):** Kabul edilen piyasaların piyasa-başı ortalama
  kazançlarının eşit ağırlıklı ortalaması. Güven aralığı, 20 ve 60 günlük
  **ortak tarih blokları** ile yapılır; böylece piyasalar arası bağımlılık
  korunur (10.000 örnek, seed 42). Geçme koşulu: ortalama > 0, CI20 alt sınırı
  > 0 ve CI60 alt sınırı > 0.
- **Genişlik:** Kabul edilen piyasaların en az ⌈2/3⌉'ünde ortalama kazanç > 0.
- **CONFIRMED** = birincil ve genişlik koşullarının ikisi de sağlanır.
  Aksi halde **NOT_CONFIRMED**.
- **Raporlanır, kapı değildir:**
  - piyasa-başı proje üç-koşul kapısı (pozitif ortalama, iki pozitif CI alt
    sınırı, yılların en az %75'i pozitif; en az 100 origin'li yıllar),
  - üst %1 ve %5 baseline-kayıp kırpma,
  - T+5 ikincil sonucu (aynı kurallarla).

  Kırpmalarda negatif sonuç gizlenmez. Kuyruk bağımlılığı sonucun zayıf yanı
  olarak yazılır.

NOT_CONFIRMED sonucu negatif kanıt olarak korunur. Ardından α, ağırlık, piyasa
veya tahminci üzerinde arama yapılmaz; CT keşif bulgusu "tekrarlanmadı" diye
not edilir. CONFIRMED sonucu yalnızca B koluna geçişi gerekçelendirir.

## B kolu: ileri doğrulama (tanımlı, başlamadı)

Aynı dondurulmuş tahminci CT=F ve A kolunda kabul edilen piyasalar için ileri
kayda alınır. Origin'ler kilitten sonraki ilk seanstan itibaren başlar; her
origin karar anından önce zaman damgalı olarak arşivlenir. Eksik origin
doldurulmaz ve ara skor hesaplanmaz. Tek piyasada 252 origin'lik gücün yaklaşık
%11 olduğu hesaplandı; bu nedenle karar havuzlanmış panelde, A kolunun kuralıyla
verilir. Bitiş tarihi ve toplama mekanizması (iki Windows görevi 4 Kasım'da
kapatıldı) **ayrı kullanıcı onayı** olmadan kurulmaz.

## Onay sınırı

Bu commit yalnız ön kaydı dondurur. Aşağıdaki adımların her biri ayrı onay ister:

- Yahoo indirmesi,
- kabul ve skor koşuları,
- private GitHub Release / push,
- B kolu toplaması,
- sicil (`registry.json`/`trials.json`) kaydı.
