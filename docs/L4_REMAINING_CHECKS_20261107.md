# Kalan L4 kontrolleri — tek Drive notebook'u

Drive klasörü: https://drive.google.com/drive/u/0/folders/1rtMBiFl2lMBR2DVCB0Ez5UlB75j5uqy2
Dosya: CottonLens_L4_Remaining_Checks_Drive_v2.ipynb.
Kaynak yanındadır; kullanıcı ZIP indirme/yükleme yapmaz.

1. Yeni L4 çalışma zamanında notebook'u açıp Tümünü çalıştır.
2. Hazır mesajı geldikten sonra Çalışma zamanı > Bağlantıyı kes ve çalışma
   zamanını sil işlemini yap. Yalnız Python kernel'ini yeniden başlatma yeterli değil.
3. Aynı notebook'u yeniden L4'e bağla, Tümünü çalıştır. Kaynak ve checkpoint
   Drive'dan okunur. Yeni boot kimliği ve eski yerel marker'ın yokluğu zorunludur.
4. Sonuçlar aynı klasörde cottonlens-session-rng-result.zip ve checksum JSON'a
   kaydedilir. Sonuçlar bağımsız denetimden sonra kabul edilir.

Bu akış toplam3sentetik train_on_batch çağrısı yapar: referans için2, yeni
oturumda devam için1. Önceki preflight/worker-restart/OOM çalışmalarının üç
makbuzu taşınır; bu testlerin modelleri tekrar eğitilmez. Birinci aşama başarısız
olursa veya resume yarım kalırsa aynı adla otomatik yeniden eğitim yapılmaz.
Dosya/kimlik uyuşmazlığında mevcut kanıt korunur. Başarılı cache tekrar fit yapmaz.

Birinci güncellemeden sonra native model/Adam optimizer, NumPy PCG64 durumu ve
TF Philox üreteci durumu kaydedilir. İkinci güncelleme önce kesintisiz referansta,
sonra yeni runtime'da yapılır. Rastgele batch sırası ve maskeler tam eşit;
model/optimizer ve tahminler en fazla1e-6mutlak toleransla karşılaştırılır.
Eksik/bozuk checkpoint reddedilir; kaynak veya cihaz değişirse devam edilmez.

Rastgelelik kontrollü NumPy batch seçimi ve açık TF üreteciyle oluşturulan input
maskesidir. Bu Keras Dropout, tf.data shuffle, çoklu GPU veya tüm rastgelelik
kaynakları için genel resume garantisi değildir. Boot kimliği ve yerel marker
runtime yenilenme tanıklarıdır; uzaktan donanım/VM attestation değildir.

Yerelde TensorFlow/CUDA kurulmadı ve GPU eğitimi çalıştırılmadı. Yerel kontroller
phase/cache/bozulma ve notebook korunmasını; ayrı verifier kontrolleri hash'leri
yeniden hesaplanmış yanlış boot/marker/source/fit/prior/mask kayıtlarını denetler.
Pozitif fixture native HDF5 yerine açık mock kullanır; gerçek native kabulü,
kullanıcının dönen model dosyalarında mevcut güvenli HDF5 okuyucuyla yapılacak.

Offline verifier: research/verify_gpu_session.py. TF import etmez, executable
model config yüklemez; JSON ve native HDF5 durumlarını kontrol eder. NumPy batch
sırasını bağımsız yeniden üretir; TF generator'u yerelde çalıştırmaz. Model
referans/resume state ve tahminlerinin eşitliği ayrıca denetlenir.

Kaynak dayanağı: https://www.tensorflow.org/api_docs/python/tf/random/Generator
State/algorithm açıkça saklanıp from_state ile yüklenir. Depo TF2.20.0 kilidi
korunur; güncel doküman sayfasının2.21olması paket yükseltmesi gerekçesi sayılmadı.

Bu teslimat G3 hazırlığıdır, gerçek session/RNG kabulü henüz yoktur. Piyasa
fit'i0; G1/G2 kaynak kabulleri ve bilimsel kampanya bu notebook ile açılmaz.

Kullanılacak notebook v2: kurulumun mevcut /content/CottonLensAI proje yolu
korunur. İlk teslimat arşivi tarihçe olarak kalır; çalıştırma için v2 kullanılır.
