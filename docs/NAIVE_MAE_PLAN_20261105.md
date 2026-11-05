# Naive-MAE araştırması — onaylanan plan, 5 Kasım 2026

Güncel durum: ilk deney tamamlandı, **STOP**. [Sonuç ve kanıt](PRICE_ONLY_MEDIAN_RESULT_20261105.md). Aşağıdaki hazırlık ifadeleri ilk kayıt anını anlatır.

Durum: bu kayıt hazırlanırken yeni tarif uygulanmadı veya eğitilmedi. Önce GitHub
birleştirme teslimatı tamamlanır; sonuçlar ayrı kimlikli ek kayıtla yayımlanır.

## Faz 1–2: kanıt ve Naive diagnostikleri

Metindeki her işi deney/kod/kanıt ile eşleştir. Tamamlandı, rapor-only, eksik,
kısmi ve compromised kanıtları ayır. Eksik eski tahminleri yeniden fit ederek üretme.
T+1/T+5 hedeflerini aynı frozen Cotton gözlemlerinden yeniden kur; train-only
özellik/ölçekleme, olgunluk/purge, geçmiş seçim ve kaynak saat/vintage sınırlarını denetle.
Yıl/ay MAE, medyan/p90/p99 hata, signed/absolute ACF(1,5,20), geçmiş volatilite
tercile'ları ve |ret20|/(vol20*sqrt(20)) trend gücü gruplarını raporla.
Gerçekleşmiş büyük hareketler, rapor ve doğrulanmış kontrat geçişleri açıklayıcıdır;
karar anında kullanılabilir özellik varsayılmaz. Unknown ayrı tutulur.

## Faz 3–4: tek sabit T+1 fiyat-only hipotezi

P(t+1)=P(t)+tahmin(ΔP). Girdiler ret1/ret5/ret20 log getirileri, 20 gözlemlik
getiri std(ddof=1), (P-SMA20)/STD20(P). Sıfır payda eksik; tüm dönüşümler train-only.
Aday QuantileRegressor(quantile=.5,alpha=.01,fit_intercept=True,solver='highs');
kontrol Ridge(alpha=1). Mevcut price_delta hedef ölçekleme ve getiri uzayındaki
[0,.25,.5,.75,1] geçmiş küçültme korunur; eşitlikte küçük ağırlık seçilir.
Birincil seçilmiş Quantile, Ridge kontrol ve ham tahminler ayrıca raporlanır.
Bu karşılaştırma yalnız loss etkisini izole ettiği iddiası taşımaz.

2016–2023 aynı 2006 origin, expanding history,119warmup,5purge/strict H5 olgunluğu,
3x63 iç blok,21refit. 2*(72inner+97outer)=338piyasa/16yıllık çıktı.
En fazla6 ayrı sentetik fit,tek süreç/<=2thread,açık yerel CPU izni.
İlk15dk checkpoint dilimi,toplam60dk; hata/timeout fail closed,otomatik fallback yok.
Quantile yalnız bu araştırma profiline izinli; CI gerçek eğitim yapamaz.
Tam recipe/scope sicil kontrolü, frozen kaynak/ortam/veri/feature tasarımı ve
sentetik kontroller ilk piyasa fit'inden önce tamamlanır. Alpha/window/seed grid yok.

## Faz 5–6: karar

Aynı origin/hedef/fiyat doğrulanır; sessiz intersection yok. Fiyat-MAE,
Naive-MAE,kazanç,yön,aktif oran,yıllar/rejimler,paired farklar ve 10k seed42
yıl sınırlarını koruyan recentered moving-block20/60 CI raporlanır.
CONTINUE yalnız >=5%MAE,>=53%tüm-origin yön,>=6/8yıl ve iki CI alt sınırı>0.
Bu tarihsel aday koşuludur; bağımsız forward başarısı değildir.
Veri/metod kusurunda REVISE; geçerli tamamlanan deney koşulları sağlayamazsa STOP.
Top1%trim duyarlılıktır; ana cohort/gate değişmez.
Yalnız CONTINUE halinde üç frozen Quantile ablation:ret5/20 çıkar,vol çıkar,zscore çıkar;
507ekfit/90dk üst sınırı,sonradan kazanan seçimi yok. T+5 eğitim/bağımsız forward
ayrı ön kayıt; incelenmiş2024+seçime girmez. Otomasyon/inference değişmez.

## Test ve teslimat

Mevcut ledger ve prepare/pilot/compare/verify arayüzleri kullanılır. Future-feature
invariance,zero correction=Naive,target roundtrip,origin mismatch,checksum/native
replay,solver failure ve duplicate-free resume test edilir. Audit matrisi,
diagnostik CSV/grafikler,frozen config,native model/scaler/tahmin/CI ve karar
private Release+sicil ile korunur. Ürün API/artifact şeması değişmez.
