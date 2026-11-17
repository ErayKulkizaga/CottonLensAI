# İlk gerçek risk yayınının bağımsız denetimi — 31 Ekim 2026

**VERIFIED:** 30 Ekim origin'inden 2 Kasım seansına yapılan ilk risk
tahmini, ham veri ve native model katsayılarından yeniden kuruldu.
Gerçekleşmiş ileri hedef veya kayıp hesaplanmadı; **EWMA üstünlüğü henüz
gösterilmiş değildir.** Sınıflandırma bu performans sorusu için INCONCLUSIVE'dur.

| Kanıt | UTC | Türkiye saati |
|---|---|---|
| Ham kaynak makbuzu | 31 Ekim 00:05:23 | 03:05:23 |
| Kaynak/model arşivinin GitHub yüklemesi tamamlandı | 00:05:31 | 03:05:31 |
| Sabit karar anı | 00:15:00 | 03:15:00 |
| Tahmin kaydı tamamlandı | 00:20:03 | 03:20:03 |
| Tahmin arşivinin GitHub yüklemesi tamamlandı | 00:20:10 | 03:20:10 |

GitHub nesne kimlikleri, sunucu zamanları ve SHA256 digest'leri taze API
yanıtı ve yeniden indirilen dosyalarla eşleştirildi. Bu, kaynak/model
paketinin karar anından önce ve tahminin yayın sonundan önce arşivlendiğine
yerel saat dışında destek sağlar; Yahoo'nun ilk tarihsel yayın veya vintage
zamanını kanıtlamaz.

Beş private parça (419+15+13+16+13 üye), 446 çalışma dosyasına birebir geri
kuruldu. Aynı 4.233 fiyat satırından Decimal HAR/EWMA hesabı; 4.108 olgun
eğitim satırı, beş seanslık etiket olgunlaşması, native katsayı/gradient ve
yalnız geçmiş doğrulamadaki ağırlık seçimi denetlendi. Kaydedilen ham ve
küçültülmüş varyans `0.0002530981101449749`, EWMA `0.00023894126163319902`;
yeniden hesaplama sayısal toleransta eşleşti. Bunlar tahmin değerleridir,
performans katkısı değildir.

Çekirdek `a173df8e…`, çalıştırıcı `2747450f…`, denetim kaynağı `9aacd67a…`.
Denetim mevcut Ledger, ham kaynak zinciri ve seçim kontrollerini kullanır;
fit, optimizer, ağdan kaynak alma, tahmin yayını ve ileri kayıp skorlama
girişleri kapatılır. Yeni 8 analitik test ve ortak rapor yolundaki 22 test
geçti. Analitik fixture gerçek eğitilmiş model veya piyasa sonucu değildir.

[Private denetim Release'i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/ct-first-publication-audit-20261031)
466 dosya/sıfır dışlamayla taze indirildi ve geri kuruldu. Paket içindeki
donmuş kodla, ağ ve yeni fit olmadan aynı sonuç dosyası byte eşit üretildi:
`2a710d764c5f8b95798983ba901bbe7228e5db84974002ecf127d3db4c88f45a`.
İlk model durumlarının 92 dosya hash'i de eşleşti. Eski kanıtlar ve model
dosyaları korundu; bu çalışmada yeni fit ve tahmin üretimi sıfırdır.

[Sicil kanıtı](../../../research/evidence/ct-first-publication-audit-20261031.json)
227 aşamayı indeksler; 561 tarif ve 32.500 fit kopyası değişmedi.
CT=F'nin kontrat ve tarihsel vintage sınırları devam eder. Ara skor ve
geçmişe tahmin doldurma yoktur. Sonraki iş, 2026 örtüşmesini koruyarak resmi
2027 Cotton takvimi uzatmasını hazırlamaktır; henüz takvim kabulü yapılmadı.
