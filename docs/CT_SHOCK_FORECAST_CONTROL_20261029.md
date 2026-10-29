# CT işaretli/işaretsiz şok tahmini: sabit ön kayıt

**Sonuç: INCONCLUSIVE.** Ön kayıtlı 338 piyasa + 1 native öğrenme fit'i
tamamlandı; aşağıdaki özgün tarif korunur. Sonuç ve doğrulama en alttadır.

**30 Ekim protokol düzeltmesi:** Onaylı insan planında top%1 çıkarma
hassasiyet kontrolüdür; birincil sonucu değiştirmez. Frozen ön kayıt bunu
fazladan pozitif-trim kapısı olarak kopyalamıştı. Aşağıdaki tarihsel tarif
ve model çıktıları korunur. Özgün üç koşul (pozitif ortalama, iki CI alt
sınırı>0, 6/8 pozitif yıl) ayrıca uygulandı: dört EWMA karşılaştırmasının
hiçbiri geçmiyor; birincil hüküm değişmiyor. Gelecek ön kayıtlarda trim
hassasiyet olarak raporlanacaktır. Ek [düzeltme kanıtı](../research/evidence/ct-shock-gate-qualification-20261029.json)
bu ayrımı kaydeder; bağımsız ileri doğrulama koşulu sürer.

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

## Sonuç — 338 piyasa fit'i tamamlandı

**VERIFIED:** İki kol da aynı 2.006 origin üzerinde tamamlandı; başarısız
deneme yok. Eski 169 kontrol yeniden eğitilmedi. Eğitim, tahmin ve kaynak
kimliği aynı frozen tarifte kaldı. Birincil işaretli/seçilmiş kol üstünlük
kapısını geçmedi.

| EWMA'ya karşı aday | QLIKE katkısı | CI20 | CI60 | Pozitif yıl | Top%1 trim |
|---|---:|---|---|---:|---:|
| İşaretsiz ham | +0,02828145 | [−0,03550416; +0,06700475] | [−0,02594715; +0,07119387] | 4/8 | −0,01053549 |
| İşaretsiz geçmişte seçilmiş | +0,00190815 | [−0,00583632; +0,00857791] | [−0,00597584; +0,00863188] | 2/8 | −0,00312958 |
| İşaretli ham | +0,02601805 | [−0,02621195; +0,05914588] | [−0,01844974; +0,06166278] | 6/8 | −0,01112990 |
| **İşaretli geçmişte seçilmiş — birincil** | **+0,00272560** | **[−0,00433905; +0,00918277]** | **[−0,00468041; +0,00917129]** | **4/8** | **−0,00304148** |

Eşit kapasitede ham işaretli−işaretsiz katkı −0,00226339;
CI20 [−0,01471286; +0,01325462], CI60 [−0,01436302; +0,01150471].
Dolayısıyla bu temsil altında işaretin ek tahmin avantajı da gösterilmedi.
Bu, her modelde bütün işaret bilgisinin yararsızlığını kanıtlamaz.

**STRONGLY SUPPORTED — bu tarifin başarısızlık mekanizması:** pozitif ham
ortalama, dönemler boyunca ve uç kayıplar dışarıda kararlı bir üstünlük
oluşturmuyor. İki kolda da 2022'ye geçmiş validation ağırlık 0 verdi; bu
ağırlık geleceği görerek 1'e çevrilmedi. Ancak shrinkage tek başına ana
açıklama değildir: ham işaretli tahmin de iki CI ve trim kapısını geçemez.
Ortak QLIKE loss kullanıldığı için bu sonuç MSE–MAE uyuşmazlığıyla açıklanamaz.

Önceki incelemede sabitlenmiş 2022-06-23 origin'inin tanımlayıcı katkısı,
ham işaretli modelin eski dört-parametre HAR üzerindeki net ek kazancının
%83,58'ini; işaretsiz modelde %120,28'ini taşır. İkinci oran, o gün dışında
net ek katkının negatif olduğunu gösterir. Bu hesap yeni ana kohort değildir;
tek gün silinmedi ve veri kusuru ilan edilmedi. İki tarihli Jul-22 kaynak
tanığının sınırları [önceki raporda](CT_SIGNED_RISK_SCORE_20261029.md) korunur.

### Native doğrulama

Bağımsız kontrol 3.520 close satırından risk özelliklerini kurdu; 338 modelin
beş gerçek katsayısını ve optimizer sertifikasını, 7.036 tahmin scalar'ını,
10.030 Decimal QLIKE loss scalar'ını, 16 geçmiş yıl seçimini ve 12 temporal
CI'ı doğruladı. Tek native öğrenme fit'inin 200 tahmini ayrıca bilinen
ortalama ile eşleşti; genel `passed` cache'i kullanılmadı. En büyük bağımsız
standardized gradient normu 9,7177e−7. Raporlamada fit girişleri kapalıydı;
native dosyaların önce/sonra hash'leri eşittir.

Piyasa fit'lerinin ölçülen compute toplamı 31,54 saniye, checkpoint kopyası
4,80 saniyedir; bunlar tüm inceleme/teslimat süresi değildir. GPU gerekmedi.
Native model/prediction dosyaları, eski veri/kanıt ve başarısız olmayan tüm
fit kayıtları korunur. Bu sonuç ileri bağımsız veya ekonomik başarı değildir.

### Sonraki tek bilimsel ayrım

Bu modellerin ek alanı da Cotton'ın kendi getirisinden türetilir; yeni bir
bilgi kaynağı eklenmiş değildir. Ayrı test edilmesi gereken soru, eldeki
DXY risk state'inin yalnız own-risk-state katsayısının serbestleştirilmesine
göre katkısıdır: üç relative-HAR + serbest log(EWMA) ile üç relative-HAR +
raw-calendar log(beş DXY kare-getirisinin ortalaması), yine beş-parametre kapasite.

Geçmiş `risk-level-source-t1-v1` DXY kontrolü sicilde bulunur; AMS annual
kimlik kusurları nedeniyle temiz named-market hükmü veremez ve CT kontrolünün
yerine geçmez. Yeni çalışma onun yapıldığını gizleyemez. Eldeki raw veri,
varsayımsal availability clock ve source-date/age sınırları yeniden bağlanmalı;
origin düşürülmemeli, eski fiyat-getiri DXY testleri risk testi sayılmamalıdır.
Henüz yeni source/ön kayıt kilidi veya fit açılmadı. Bu sonuç arşivinin taze
geri kurma/replay teslimatı tamamlanmadan sıradaki deney başlamaz.

### Doğrulanmış GitHub teslimatı

Private [sonuç Release'i](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/ct-shock-forecast-result-20261029)
4.184 dosya olarak taze geri kuruldu. Kaynak snapshot'ıyla yeniden karşılaştırma
ve bağımsız kontrol **sıfır fit/ağ isteğiyle** result JSON, eşlenmiş loss CSV
ve bağımsız makbuzu byte eşit üretti. Makbuz ayrıca yüklenip taze geri indirildi.
SHA-256: `dcfb003a8dea74b37ba2e04aeaa3e41ee103d346aa33cf4ed062c821bafeb5c2`.
338 piyasa ve 1 native öğrenme modeli, eski gerekli modeller ve tüm veri/kimlik
bağları korunur. Sicile iki 169-fit piyasa tarifi ve tek öğrenme tarifi eklendi;
eski 554 tarif değiştirilmedi. Tam öneri sorgusu artık her kolun 169 fit'ini
`REPEAT_FIT_RECIPE` olarak bulur. Bu eğitimler yeniden başlatılmaz.
