# Gerçek vade farkı T+1 — kilitli, varsayımlı kaynak katkısı testi

[Kaynak incelemesi](INFORMATION_ADMISSION_20261010.md) tamamlandı;
[eşleştirilmiş loss sonucu](PAIRED_PRICE_LOSS_RESULT_20261010.md) yeniden
eğitilmez. Kullanıcı erişim kanıtı/duyarlılık ayrımında kararı araştırmacıya
bıraktı. Seçim: **açık saat ve ilk sürüm varsayımına bağlı duyarlılık testi**.
Tarihsel point-in-time, leakage yokluğu veya gerçek zamanlı beceri ispatı değildir.
Eski planlar, kaynaklar, deneyler ve Naive korunur; yeni veri/grid/GPU yok.

## Düzeltilen kapsam

997 AMS `FUTURES TODAY` tablosu 2020–2023'te 10 gerçek vade fiyatı içerir.
Eski AMS özelliği spot/Close farkıydı ve T+5 sınanmıştı; gerçek vadeler arası
T+1 farkının sınaması değildir. Şimdiki 24 çekirdek özellikte eğri yok.
UTC yayın saati ve ilk vintage kanıtlanmamıştır; kaynak hâlâ karantinadadır.

İlk 2.006-origin /676-fit taslağı **çalıştırılmadan** korunur: 2016–2019'da
kaynak yok, 2020'nin üç geçmiş iç doğrulama bloğunda da kaynak yok. Bu yıllar
ek kaynağın öğrenildiğine dair kanıt üretemez. Kaynak başında Cotton history
resetlemek ise 500 geçmiş gözlem şartıyla yalnız 2023/246 origin bırakır.

Yeni tasarım bütün 3.520 Cotton history satırını tutar. Bir yıl ancak dış
değerlendirme başlamadan, üç geçmiş iç bloğun her birinde ve iki gecikmede
en az iki kullanılabilir eğitim raporu, iki validation raporu, iki farklı
spread ve pozitif eğitim spread varyansı varsa alınır. Dış performans veya
özellik tamlığı seçimde kullanılmaz; origin düşürülmez. Sonuç **2021–2023,
749 ortak origin** (252/251/246). Bunlar tekrar incelenmiş araştırma yıllarıdır.
**6/8 yıl kapısı ölçülemez; 2/3 veya 3/3 ile değiştirilmez.**

## Tek hipotez ve sabit tasarım

Gerçek ikinci ve ilk vade fiyatlarının `log(F_second/F_first)` oranı,
aynı çekirdek ve kaynak zamanlama/missingness bilgisine ek T+1 fiyat-MAE
katkısı sağlar mı? Farklı teslim ayları 2 veya 3 ay aralıklıdır;
özellik “bir sonraki ay” diye adlandırılmaz, yıllıklaştırılmaz.

| Unsur | Kilit |
|---|---|
| Profil /deney | `contract-curve-t1-pilot-v1` /`research-contract-curve-t1-pilot-v1` |
| Karar | Cotton kaynak günü +1 takvim günü 00:15 UTC |
| Varsayım | `max(referans günü, doğrulanmamış basılı yayın takvim günü)+1 gün 00:00 UTC`; mevcut Final fiyatları o anda vardı varsayımı |
| D0 /D1 | İlk varsayımlı uygun karar /bir sonraki kayıtlı Cotton kararı; D0 birincil, D1 ikincil |
| Eskilik | D0 ilk uygun kararından en fazla 3 ek Cotton gözlemi; D1 süreyi uzatmaz |
| Kollar | `mask_D0`, `numeric_D0`, `mask_D1`, `numeric_D1` |
| Ortak bilgi | 24 çekirdek +kaynak yaşı +unavailable +gerçek vade aralığı |
| Ayrılan tek sütun | Bilgi varsa mask=0/numeric=log oran; yoksa ikisi de NaN |
| Model | Ridge alpha=1, seed=42, window=1; ayrı T+1, existing scaled_log |
| Eğitim | Expanding history, ortak 120 warmup, target_date_5 < refit cutoff |
| İşleme | Eğitimde median/mean/std ve aynı missing indicators; geçmişe doldurma yok |
| Seçim | Geçmiş 3×63 fiyat-MAE/Naive; sadece `[0,.25,.5,.75,1]` ağırlık, eşitlikte küçüğü |
| Refit | 21 Cotton gözleminde; existing Experiment/Ledger |
| Bütçe | 108 iç +144 dış=252 piyasa fit'i; ayrı en fazla 1 sentetik Ridge kontrolü |
| Oturum | Tek CPU süreç, en fazla 2 thread, 30 dakika; 12 yıllık çıktı/2.996 satır |

Geç gelen eski referans daha yeni raporu geri çeviremez. Kaynak tarihi,
kontrat adları, SHA, varsayımlı erişim ve karar saati tahminle saklanır.
Referans günü, yayın saati, vintage ve ingestion birbirinin yerine geçmez.

## Önceden belirlenmiş yorum

Birincil: seçilmiş `numeric_D0` karşısında `mask_D0` fiyat mutlak hata farkı.
Pozitif fark numeric lehine. Yılları geçmeyen paired bootstrap 10.000 tekrar,
seed42, blok20 ve60; iki alt sınır >0 ise bu tarif/varsayıma bağlı kaynak
katkısı adayıdır. D1 başarısız D0'ı kurtarmak için birincil yapılamaz.

Naive karşılaştırması, ham/seçilmiş tahmin, aktif oran, aktif/tüm-origin yön,
yıllık fark ve en büyük `ceil(%1)` Naive hatası çıkarılmış duyarlılık birlikte
raporlanır. Seçilmiş `[0,0]`, kaynakta sinyal yokluğu kanıtı değildir.
İki D0 Naive aralığının üst sınırı %5 altında ise **bu sabit tarife** pratik
hedefi kurtarma iddiası kapanır. Diğer haller belirsiz; otomatik grid yok.
Pozitif kaynak katkısı bile üç yılla 6/8 koşulunu veya üretim kapısını açmaz.

Önce kod/test/input/ortam kimlikleri ve karar sözleşmesi dondurulur; checksum
bağlı sıfır-fit kayıt GitHub'a taşınmadan fit yok. Değişen kimlik eski cache'i
reddeder. Eksik makbuz/payload tamamlanmış bilimsel sonuç sayılmaz.
Sonuç sonrası tek karar bu kaynak katkısını kanıtın gerçek kapsamıyla
sınıflandırmak; yeni model ailesi seçmek değildir.

## Yeniden çalışma

Mevcut ayrı CPU ortamı ve checksum doğrulanmış kayıtla:

```bash
python ml/full_year_cpu.py prepare --profile contract-curve-t1-pilot-v1 --cpu-environment CPU_ENV --drive-root RUN_ROOT --registration-root REGISTRATION --decision-contract REGISTRATION/decision-contract.json
python ml/full_year_cpu.py pilot-plan --profile contract-curve-t1-pilot-v1 --cpu-environment CPU_ENV --drive-root RUN_ROOT
python ml/full_year_cpu.py pilot --profile contract-curve-t1-pilot-v1 --cpu-environment CPU_ENV --drive-root RUN_ROOT --max-minutes 30
python ml/full_year_cpu.py compare --profile contract-curve-t1-pilot-v1 --cpu-environment CPU_ENV --drive-root RUN_ROOT
```

`search`, `lock`, `export`, model yayını ve gerçek T+5 bu profilde reddedilir.
ML kaynağı değişirse mevcut kayıtla devam edilmez; eski kayıt bozulmaz.

## Sıfır-fit kayıt

[Kimlik kanıtı](../research/evidence/contract-curve-preregistration-20261010.json):
source `84d40007e33b75ecff9ca422958daab79d30d67f3adb59b7bb419bc873412d1a`,
registration `e2fac4fa52a20d48ba4a09346ca1b7b9ae32db88133f61dcc8080bcec5ac2698`.
7.040 bağımsız kaynak-zamanı seçimi, 63 farklı refit kesiminde olgunluk,
eski 1.000 ham girdi checksum'ı ve karantina taslağıyla exact frame replay
doğrulandı. Son profil 19 sentetik test; ortak 72 kontrol; geniş ML 1.020
passed/3 skip, Ruff başarılı. Geniş test koşusundan sonra eklenen üç regresyon
son 19-test koşusunda ayrıca doğrulandı. GPU/backend/frontend değişikliği yok.

İlk hazırlama kaydı, geçici `.writer-lock/owner.json` kalıcı checksum listesine
girdiği için tamamlanma doğrulamasında reddedildi; **fit 0**. Silinmedi veya
geçmiş deneylere teşmil edilmedi. Lock ağacı dışlama regresyonuyla düzeltildi;
`registration-r2` yeni source/namespace ile sıfırdan donduruldu.

[Ön kayıt arşivi](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/contract-curve-preregistration-20261010)
564 üye /7.313.091 byte; SHA256
`18520c49d6750a3a6b08dcd9fe69d7f1a04d69cc12b8e35a2e96a2c79956fc1e`.
Bu paket kod, ortam, hazır execution, quotes, karar sözleşmesi ve fit yapmayı
reddeden taşınabilir doğrulayıcıyı içerir. Eski 157 sicil kaydı/508 fit tarifi
değişmedi; yeni ön kayıt performans kanıtı sayılmıyor. Tahmin henüz yok.
