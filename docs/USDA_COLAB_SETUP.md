# USDA ücretsiz API kurulumu

Veri aboneliği veya ödeme gerekmez. Üç servis üç ayrı anahtar kullanır.
Anahtarları sohbet, notebook kodu, URL veya Git dosyasına yapıştırma.

## 1. Önce FAS ve NASS

**USDA FAS — Export Sales**

- https://api.data.gov/signup/ adresinden kişisel anahtar al.
- Servis bilgisi: https://apps.fas.usda.gov/opendatawebV2/ .
- Colab Secrets adı: `USDA_FAS_API_KEY`.

**USDA NASS — Crop Progress/Condition**

- https://quickstats.nass.usda.gov/api/ adresinde `Request API Key` bölümünü aç.
- Kullanım koşullarını okuyup e-posta adresinle başvur; gelen anahtarı kullan.
- Colab Secrets adı: `USDA_NASS_API_KEY`.
- Uygulamada gerekli atıf: “This product uses the NASS API but is not endorsed or certified by NASS.”

## 2. AMS — fiziksel pamuk spot raporları

- https://mymarketnews.ams.usda.gov/ adresinde Login seç.
- Resmî giriş/kayıt yönlendirmesini tamamla; girişten sonra adını ve `Show API key` seçeneğini aç.
- Colab Secrets adı: `USDA_AMS_API_KEY`.
- Resmî kılavuz: https://mymarketnews.ams.usda.gov/mymarketnews-api/authentication .
- Bu anahtar FAS anahtarıyla aynı değildir. AMS hesabı hazır değilse diğer iki kaynakla devam edilebilir;
  halka açık rapor dosyaları için ayrı, anahtarsız arşivleme komutu da vardır.

## 3. Colab'a ekleme

Yeni notebook'ta sol kenar çubuğundaki anahtar simgesini (Secrets) aç.
Yukarıdaki adlarla üç kayıt oluştur; değerleri ilgili servislerden kopyala.
Her kayıt için bu notebook'un erişimini etkinleştir. GPU runtime yeniden bağlandığında
anahtarları tekrar kod içine yazma; kaynak hücresi Secrets'tan yeniden okur.

`SOURCE_ACTION = 'status'` ile kaynak hücresini çalıştır. Yalnız mevcut/eksik adları
görünür; değerler yazdırılmaz. Eksik bir anahtar model eğitimini sessizce başka veriye düşürmez.

`SOURCE_ACTION = 'catalog'` AMS pamuk rapor kimliklerini ve FAS pamuk ürün kodlarını arşivler.
Kimlikler katalogdan seçilir; eski bir slug veya ürün kodu tahmin edilmez.

NASS için kaynak hücresinde `SOURCE_ACTION = 'nass'`, `SOURCE_YEAR = 2020` örneğini
kullanabilirsin. Yıl başına sorgu yapılır; 50.000 kayıt sınırı aşılırsa indirme durur.
İlk smoke için tek yıl yeterlidir; araştırma arşivi daha sonra 2010–2023 yıllarına genişletilir.

## İndirilen veri neden hemen eğitime girmiyor?

API bugün revize edilmiş tarihsel değerleri döndürebilir. NASS `load_time` veri tabanına
yüklenme zamanıdır; ilk yayın kanıtı değildir. FAS release metadata'sı da tek başına
her tarihsel satırın orijinal sürümünü kanıtlamaz. İlk indirme ham arşiv/şema kontrolüdür.
Yayın ve vintage kanıtı tamamlanan paket `PUBLICATION_PACKAGES` listesine eklenir ve
yeni deney kimliğiyle dondurulur. Arşivlenmiş dosyanın bulunması performans kanıtı değildir.

Ücretli alternatif gerekmez. Servis erişimi sorun çıkarırsa hata kodu paylaşılabilir;
anahtar veya anahtar içeren URL paylaşılmaz.
