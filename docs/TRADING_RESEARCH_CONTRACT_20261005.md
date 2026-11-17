# CottonLensAI: işlem yönü ve pozisyon araştırma sözleşmesi

Kullanıcı kararı: vadeli piyasada işlem yönü ve pozisyon. Tarih: 5 Ekim 2026.
Durum: araştırma hedefi; canlı işlem veya model yayını onayı değildir.

## Amaç

Geçmişte gerçekten kullanılabilir bilgiyle, işlem maliyetleri sonrası pozitif
fayda sağlayan **long / flat / short** kararları üretmek. Fiyat-MAE ve doğru yön
oranı tanısal ölçümlerdir. Kazancın büyüklüğü, kaybın büyüklüğü ve maliyet olmadan
yön doğruluğu stratejinin değerini belirleyemez.

Önceki fiyat tahmini deneylerinin %5 MAE, %53/%55 yön ve 6/8 yıl eşikleri
değişmez. Yeni işlem hedefi, o deneyleri başarılı saymak veya bir modeli eski
yayın kapısından geçirmek için kullanılamaz. Hiçbir eski modelin işlem becerisi
bu sözleşmeyle kabul edilmiş olmaz.

## İlk araştırmanın sınırı

- Çalışma varsayımı: ICE Cotton No. 2 (CT). Kullanıcının işlem yaptığı ürün farklıysa
  fiyatlama, maliyet ve seans sözleşmesi değiştirilip değerlendirmeden önce kilitlenir.
- T+1 birincil: kararın ardından ilk Cotton seansında giriş ve aynı seans sonunda çıkış.
  T+5 için ilk teslimatta pozisyon stratejisi veya ufuk seçimi yapılmaz.
- Karar anı: Cotton kaynak tarihini izleyen gün 00:15 UTC; mevcut kaynak
  erişilebilirliği varsayımları açıkça korunur, doğrulanmış yayın zamanı sayılmaz.
- Giriş: karar anından **sonra**, işlem yapılabilir ilk seansta uygulanabilir fiyat.
  Önceki Cotton kapanışı giriş fiyatı olarak kullanılamaz. UTC takvim günü,
  borsa seans etiketi ve veri sağlayıcının günlük bar tarihi eşit varsayılmaz.
- İlk tanısal pozisyonlar sabit `−1 / 0 / +1` kontrat birimidir. Sermaye, kaldıraç
  ve kullanıcıya uygun pozisyon büyüklüğü henüz belirlenmez. Overlap, stop/limit
  emirleri ve intraday yol varsayımları eklenmez.
- Kontrat açık pozisyonda değiştirilemez. Daha uzun tutuş veya roll içeren ileriki
  araştırma, gerçek kontrat fiyatları ve önceden kilitli geçiş kuralı gerektirir.

## İşlem sonucu ve başarı

CT'nin kontrat büyüklüğü 50.000 lb; fiyat cent/lb, tick 0,01 cent/lb = 5 USD.
Dolayısıyla bir kontrat için:

`gross_pnl_usd = position × 500 × (exit_price_cents − entry_price_cents)`

`net_pnl_usd = gross_pnl_usd − commissions − spread/slippage/execution_cost`

Maliyetler çift yönlüdür; doldurma fiyatı spread/slippage'ı zaten içeriyorsa
ikinci kez düşülmez. Flat pozisyonun işlem maliyeti sıfırdır. Limitte işlem
gerçekleşememesi veya eksik fiyat, otomatik gerçekleşmiş işlem sayılmaz.

Birincil ölçüm, **bütün uygun karar günleri dahil** maliyet sonrası ortalama
kontrat-birim PnL'dir. İşlem başına PnL, işlem sayısı, aktif oran, turnover,
yıllık sonuç ve dolar drawdown ayrıca raporlanır. Yalnız kazandıran işlem günleri
veya en iyi yıllar seçilmez. Sermaye ve marjin tanımı olmadan yüzde hesap getirisi
ve kaldıraçlı performans iddiası üretilmez.

Referanslar: flat/no-trade; aynı giriş/çıkışta sabit long ve sabit short.
Aynı aktif günlerde sabit yön karşılaştırması, getirinin yön seçiminden mi yoksa
hangi günlerde işlem yapıldığından mı geldiğini ayırır. Basit referanslardan
üstünlük henüz ölçülmüş veya kabul edilmiş değildir.

Tarihsel bir adayın hedeflenmiş ileri doğrulamaya geçebilmesi için uygulanabilir
fiyat/veri kimliği, gerçekçi maliyet hesabı, pozitif net sonuç ve 20/60 günlük
bloklarda eşlenmiş üstünlük kanıtı gerekir. Mevcut sekiz yıllık kapsamda en az
6/8 yıl pozitif net sonuç, bu yeni araştırma için **önerilen ve ilk hesap öncesi
kilitlenecek** istikrar koşuludur. Bu koşullar yeterli canlı yayın kanıtı değildir.
Kabul edilebilir drawdown ve sermaye riski kullanıcı kullanımına göre ayrıca
kilitlenmeden pozisyon büyüklüğü veya canlı kullanıma GO verilemez.

2016–2023 ve daha önce incelenen 2024+ gelişim/geçmiş araştırmadır. Bu yıllarda
PnL hesaplamak onları yeni holdout yapmaz. İleri doğrulama, tarif dondurulduktan
sonra zamanında arşivlenen sinyallerle yapılır; izlenmiş geçmiş tarihten başlatılamaz.

## Şimdi yapılacak tek analiz

Yeni fit olmadan, `research-availability-clock-pilot-v1` kayıtlarındaki **T+1**
kontrol ve müdahale tahminleri kullanılacak. Her kol için kural önceden sabit:
`position = sign(selected_predicted_return)`; sıfır flat. Ham tahminlerin işareti
ayrı tanısal çıktı olarak verilecek, sonuçlara bakılarak strateji seçilmeyecek.

Mevcut market dosyasının sonraki kayıtlı Cotton `open → close` değişimine bu
kararlar uygulanacak. Eski hedefteki `close → next close` hareketi ile sinyalden
sonra yakalanabilecek hareket ayrılacak. Ortak tarih ve hedef kimlikleri
zorunlu; eksik açılışlarda sessiz intersection veya fiyat doldurma yok.

Rapor, brüt sonuç, işlem sayısı, yıllık istikrar ve işlem başına başa baş toplam
maliyeti gösterecek. Maliyet bilinmediğinde net başarı uydurulmayacak. Birincil
hesap brüt olarak bile pozitif değilse, bu sabit stratejinin pozitif maliyetle
kurtulmadığı söylenebilir; bütün olası modeller hakkında sonuç çıkarılamaz.
Pozitif brüt sonuç bulunması da gerçek işlem becerisi kanıtı sayılmaz.

Mevcut veri sınırı: raw market dosyasının kolonları yalnız
`date, series, open, high, low, close, volume`; Cotton kaynağı `CT=F`.
Kontrat kimliği, barın kesin seans/açılış zaman damgası, quote/gerçekleşme fiyatı
ve tarihsel erişilebilirlik kanıtı yok. Analiz bu yüzden **proxy duyarlılığıdır**.
Günlük open değerinin 00:15 UTC'den sonra uygulanabilir olduğu doğrulanmış sayılmaz.
Gerçek kontrat/seans/işlem maliyeti doğrulanmadan net strateji GO'su verilemez.

Bu analiz model araması, eşik optimizasyonu, daha uzun ufuk taraması veya yeni
veri edinimini otomatik başlatmaz. Hesap ve kısa karar raporu, bir sonraki sınırlı
teslimat olarak hazırlanır; bu sözleşme yazılırken yeni fit/backtest çalıştırılmadı.

## Birincil dayanaklar

- `ml/src/cottonlens_ml/data.py`: Cotton ticker `CT=F`; günlük OHLCV alanları.
- `ml/src/cottonlens_ml/features.py`: mevcut hedef `log(close[t+h]/close[t])`.
- `docs/AVAILABILITY_CLOCK_RESULT_20261005.md`: son deney ve kanıt sınırları.
- ICE kontrat büyüklüğü, tick ve New York seans saatleri:
  [Cotton No. 2 Futures](https://www.ice.com/products/254).
  Normal seans açılışı 21:00 New York; UTC karşılığı yaz/kış saati ve özel
  borsa takvimine göre ele alınır. Bugünkü spesifikasyon tarihsel takvim kanıtı değildir.
