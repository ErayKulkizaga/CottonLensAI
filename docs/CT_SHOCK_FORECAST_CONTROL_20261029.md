# CT işaretli/işaretsiz şok tahmini: sabit ön kayıt

Profil: `ct-signed-unsigned-shock-forecast-t1-v1`.

Önceki [koşullu score](CT_SIGNED_RISK_SCORE_20261029.md) tahmin deneyi değildi.
Bu karşılaştırma, yalnız geçmişte öğrenilen şok katsayısının gerçek OOS
tahmin katkısını ölçer. Negatif ilişki önceden görüldü; iki yönü de öğrenmeye
açık katsayının işareti geleceği görerek seçilmeyecek.

| Karar | Sabit değer |
|---|---|
| Hedef ve veri | Eldeki CT proxy, sonraki seans kare log-getirisi; yeni dataset yok |
| Dış dönem | Aynı 2016–2023, 2.006 origin; 2024+ seçimde yok |
| Referans | Aynı EWMA λ=0,94 ve mevcut 169 native dört-parametre kontrolü |
| İki kol | Üç relative-HAR + `r_t²/EWMA_t`; üç relative-HAR + `sign(r_t)*r_t²/EWMA_t` |
| Kapasite | Her kolda intercept dahil beş parametre |
| Eğitim | Ortak QLIKE/BFGS; aynı expanding geçmiş ve beş seans olgunlaşma |
| Seçim | Aynı geçmiş 3×63, 21 seans refit; shrinkage 0/0,25/0,5/0,75/1 |
| Birincil aday | İşaretli kolun geçmişte seçilmiş tahmini, güçlü EWMA'ya karşı |
| Mekanizma | Ham işaretli kol, aynı kapasitedeki ham işaretsiz kola karşı |
| Bütçe | 338 piyasa + en fazla 1 kayıtlı sentetik öğrenme fit'i; tek süreç/iki thread/30 dakika |

40 ilgili sentetik/unit test ve Ruff geçti. Unit fixture optimizasyonları
piyasa eğitimi veya araştırma OOS başarısı değildir. Kayıtlı fit bütçesi
169×2 piyasa işi ile tek native öğrenme kontrolünü kapsar; eski 169 fit
tekrar eğitilmez.

Ortak HAR daha önce üç özelliği sabit kullanıyordu. Yeni `variance_features`
listesi yalnız QLIKE yolu için açık seçimdir; varsayılan eski üç özellik
olarak kalır. Dördüncü özelliğe bağımlı bilinen ilişki, held-out tahmin,
eski varsayılanın eşitliği, gelecek hedeflerin etkisizliği, kaynak/veri
değişiminde cache reddi ve bütçe dolunca piyasa compute'una girilmemesi test edildi.

Başarı kapısı değiştirilmez: pozitif QLIKE katkısı, hem 20 hem 60 blokta
pozitif %95 alt sınır, en az 6/8 pozitif yıl ve en büyük %1 EWMA kaybı
çıkarıldığında pozitif katkı. Bootstrap 10.000 tekrar/seed 42/yıl sınırları
korunarak uygulanır. Kayıp negatif olabildiği için yüzde QLIKE başarısı yoktur.
Kollar sonuçtan sonra değiştirilmez veya kazanan seçilmez; grid/ufuk taraması yoktur.

Bu görülmüş tarihsel CT-proxy araştırmasıdır. Tarihsel saat/ilk-vintage
doğrulanmış değildir; ileri bağımsız başarı, ekonomik fayda veya temiz
aynı-kontrat riski ilan edilemez. Önceki AMS kaynak kusurları burada
kullanılmaz; named-market eğitim kabulü ayrıca kapalı kalır. Saatlik görev
ve ileri yayın otomasyonu yeniden açılmaz.

Yeni native öğrenme kontrolü beş özelliğin tamamını sınar; genel `passed`
cache'iyle yetinilmez. Her market fit'i kimliği kaynak/veri/plan/ortam ve
gerçek train/test tarihlerine bağlıdır. Eksik veya başarısız fit tamamlanmış
sayılmaz; kör retry/fallback yoktur. Ön kayıt taze GitHub geri kurma ve
bağımsız geçmiş/etiket kontrolünden önce eğitim açmaz.
