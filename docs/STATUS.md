# Güncel kararlar — 5 Ekim 2026

Bu dosya ve [işlem araştırması sözleşmesi](TRADING_RESEARCH_CONTRACT_20261005.md) güncel karar kaynaklarıdır. `archive/` belgeleri tarihsel tanıklık/kanıttır; eski “sonraki deney” talimatları etkin değildir.

- Ürün amacı: vadeli piyasada işlem yönü ve pozisyon. Henüz doğrulanmış maliyet sonrası işlem üstünlüğü yok.
- İlk uygulama teslimatı: sürümlü 00:15 UTC yayın hizalaması ve CLI/engine tazelik eşitliği düzeltildi; kayıtlı 2.006 T+1 origin'de sıfır-fit işlem proxy analizi tamamlandı. Seçilmiş müdahalenin 251 işlemi yalnız 2017'de; kontrol katkısı belirsiz. Ham sonuç ayrı tanıdır, strateji seçimi değildir. 95 rapor/26 alanlı bölgesel WASDE aday paketine geri alındı; erişilebilirlik/vintage doğrulanmadan eğitime kabul edilmedi. 690 eski dosyanın hash'i değişmedi. [Sonuçlar, kanıt ve tek sonraki iş](TRADING_WASDE_DELIVERY_20261005.md).
- Mevcut fiyat programında T+1 ve T+5: **Naive korunur**. %5 fiyat-MAE, %53/%55 yön ve 6/8 tam-yıl kazanımı eşikleri değiştirilmedi.
- Son tamamlanan çalışma `research-availability-clock-pilot-v1`: 676 fit, 32 yıllık çıktı, 8.024 tahmin satırı. Daha güncel DXY/WTI bilgisi pratik hedefi kurtarmadı. [Sonuç ve sınırlar](AVAILABILITY_CLOCK_RESULT_20261005.md).
- Zamanlama varsayımına bağlı bu çalışma üretim erişilebilirliği kanıtı değildir. Yeni kodla eski cache devam ettirilmez.
- Legacy LSTM ortak origin'lerde ölçülmemişti; aynı pencere toplu T+1 kazancı −%1,6961. Tam tahmin yok; yeniden üretim/paired CI uydurulmaz.
- TCN bilinen-sinyal kontrolü başarısızken genel durum geçiyordu. Yeni aile bazlı politika ve eski cache yeniden değerlendirmesi uygulanmıştır; TCN yeniden eğitilmedi.
- Kaynak T+5 testlerinden T+1 çıkarımı yapılmaz. Ham, küçültülmüş ve aktif tahmin metrikleri ayrıdır. Sıfır ağırlık kaynakta sıfır sinyal kanıtı değildir.
- 2016–2023 ve görülmüş 2024+ geliştirme/inceleme tarihidir; bağımsız holdout olarak kullanılmaz.

## Devam etmeden önce

1. `python ml/history.py check` ile aile, ufuk ve kaynak/tarif geçmişini kontrol et; [sicil kullanımını](../research/README.md) oku.
2. Önceki sonuç/tahmin kapsamını ve dondurulmuş kimliği incele. Tekrar gerekiyorsa gerekçeyi kaydet; daha büyük grid/compute kendiliğinden gerekçe değildir.
3. İşlem hedefinde araştırmaya başlamak için sözleşme kimliği, karar/emir anı, uygulanabilir giriş/çıkış ve maliyetler açık olmalı. CT=F günlük barı gerçek işlem fiyatı kanıtı sayılmaz.

Yeni eğitim, yeni veri veya otomatik canlı model yayını bu depo düzenlemesinin parçası değildir. [Tarihsel belge indeksi](archive/README.md) ve [GitHub kanıt Release'i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/evidence-20261005) geçmişi korur.
