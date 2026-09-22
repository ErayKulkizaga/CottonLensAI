# CottonLens AI — görüşme provası

## Beş dakikalık akış

1. **Problem (0:00–0:40):** “Cotton No. 2 için bir ve beş işlem günü sonrasının log getirisini tahmin eden araştırma uygulaması geliştirdim. Yahoo sürekli vadeli serisi resmî ICE settlement değil; yatırım tavsiyesi sunmuyorum.”
2. **Veri (0:40–1:30):** Dashboard’da kaynak tarihi ve sürümü göster. Cotton işlem günlerini ana takvim yap; DXY/WTI kapanışlarını bir Cotton seansı geciktir. CFTC arşivindeki gerçek yayın anı doğrulanamadığı için bu sürümde modele katma. `data_quality.json` içindeki geçersiz değerleri ve her serinin ilk tarihini göster.
3. **Eğitim (1:30–2:40):** Tek Colab notebook’unda izole Python 3.12, train-only ölçekleme, 60-adımlı iki çıktılı LSTM, Ridge referansı, XGBoost ve Naive’ı anlat. Model Lab’de dört ayrı 126-günlük walk-forward fold’u, epoch loss/val_loss eğrisini ve feature ablation’ı göster. Bu ekranlar yalnızca yeni Colab artifact’ı kurulduğunda ölçülmüş kanıttır.
4. **Gerçek sonuç (2:40–3:30):** MAE birimini ¢/lb olarak, yön doğruluğunu çoğunluk yönü referansıyla beraber açıkla. %5 MAE, %53/%55 yön, 3/4 fold tutarlılığına bak. Sağlanmıyorsa Naive seçimini savun. 2024 sonrası tarihsel audit daha önce görüldüğü için bağımsız test diye adlandırma.
5. **Ürün (3:30–4:30):** Why ekranında gerçek TreeSHAP tabanı, “diğer” katkılar ve varsa LSTM yaklaşık açıklama hatasını göster. Sensitivity Lab’de değişiklik öncesi/sonrası **aynı modelin** tahminini karşılaştır; nedensel sonuç olmadığını söyle. Replay’in geçmişten yeniden üretilmiş backtest olduğunu, live kaydı olmadığını belirt.
6. **Dağıtım (4:30–5:00):** Colab → checksum’lı ZIP → FastAPI/ONNX veya XGBoost CPU runtime → Angular akışını göster. TensorFlow/MLflow yerel backend imajında bulunmaz.

## Sık teknik sorulara kısa cevaplar

- **Neden shuffle yok?** Gelecekteki fiyat/pozisyon bilgisini geçmiş eğitimine taşıyabilir. Fold’lar ve hedef tarihleri kronolojik; T+5 sınırlarında beş seans purge var.
- **LSTM nasıl öğreniyor?** 60 seanslık feature dizisini LSTM katmanı iki ölçeklenmiş log-getiri çıktısına dönüştürür. Adam/MAE ile `compile` edilir; 100 epoch üst sınırı, 10 epoch early stopping ve en iyi checkpoint kullanılır. Train–validation ayrışması overfitting işaretidir.
- **XGBoost neden var?** Yaklaşık 20 yapılandırılmış piyasa feature’ı için güçlü ve hafif bir tabular referans. Öğrenilmiş modelin Naive’ı geçmesi gerektiğinden karmaşık model otomatik kazanmaz.
- **Neden fiyat yerine log getiri?** Ufuklar arasında tutarlı bir hedef ve `close × exp(predicted_return)` ile doğrudan fiyat dönüşümü sağlar.
- **SHAP nedensellik mi?** Hayır. Katkı, eğitilmiş modelin belirli tahminindeki ilişkisel ayrıştırmasıdır. LSTM GradientExplainer yaklaşık olduğundan artık hata ayrı gösterilir.
- **Sonuç şirketin açıkladığı MAPE ile kıyaslanabilir mi?** Veri, dönem ve protokol aynı olmadığı için hayır. Ben yalnızca kendi Naive/Ridge/XGBoost/LSTM karşılaştırmamı aynı tarihlerde savunurum.
- **Model ne zaman güncellenir?** Colab Run All manuel başlatılır; artifact doğrulanıp yerelde içe alınır. Otomatik gece eğitimi kuruluymuş gibi anlatılmaz.

## Demo öncesi dürüstlük kontrolü

- `GET /api/v1/health/ready` `200`, artifact sürümü beklenen sürüm olmalı.
- Model Lab’de “Legacy evaluation” veya “Development fixture” görünüyorsa yeni walk-forward sonucu varmış gibi sunma.
- Colab final hücresinde ZIP, checksum ve TensorFlow’suz backend parity doğrulaması geçmiş olmalı.
- İnternet bağlantısı kapalıyken tahmin, Why, Sensitivity, Model Lab ve Replay yerel servisten açılmalı.
- Yedek olarak ZIP ve `.zip.sha256` ayrı saklanmalı; sunumda asıl veri/forecast kayıtları simülasyonla değişmemeli.
