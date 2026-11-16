# 17 · Sunum güncelleme özeti (Codex için, 17 Kasım 2026)

Bu dosya `docs/portfolio/cottonlens-sunum.html` sayfasını güncelleyecek kişi için yazıldı. Kullanıcının "USDA'yı herhangi bir şeyde bile olsa yenemedik mi hala?" sorusundan sonra yapılan tüm işleri, sonuçları ve sayfada değişmesi gerekenleri anlatır. Bütün sayılar `research/evidence/` altındaki kanıt dosyalarından ya da `docs/T5x_…` ön kayıtlarından gelir.

- Repo `ErayKulkizaga/CottonLensAI`, dal `claude/hls-move3`, PR #78 (açık, birleştirilmedi).
- Daha önceki işler için `16_PORTFOLIO_BRIEF_AFTER_REVIEWS.md` dosyasına bak. Bu dosya ondan sonrasını kapsar.

## 1. Kısa cevap: USDA'yı yendik mi?

**Evet, iki yerde. Ama koşulsuz "USDA'yı geçtik" cümlesi kurulamaz.**

| Ne | Sonuç | Durum |
|---|---|---|
| Teksas hasat alanı, gerçek zamanlı, 2008–2025 | USDA hatası %11,4 → bizimki %7,0 (p = 0,0005). Geçmiş yanlılık düzeltmesi de %9,6'da kalıyor. | **Yendik.** Ama Teksas, bulgunun keşfedildiği eyalet. |
| Kör sezonlar 1987–2011, üretim | 128 karşılaştırmanın 88'inde USDA'dan daha yakın (p = 0,007) | **Yendik.** Ama iki dönemi birleştirme kararı sonuçlar görüldükten sonra alındı. |
| USDA'nın bir sonraki revizyonunun yönü, kör pamuk | %67 doğru (62/93, p = 0,012), ön kayıtlı | **Geçti.** 2012 sonrasında %43, yani bugün çalışmıyor. |
| Bugünkü üretim tahmini (2002–2025) | 12 hücrenin 12'sinde ortalama daha iyi, ama Holm/BH düzeltmesinden sonra hiçbiri anlamlı değil | Anlamlı değil |
| Alan düzeltmesiyle Teksas üretimi | 4 ayın 4'ünde daha iyi; p = 0,135 (T51), uzun tarihte p = 0,19, 2008 sonrasında p = 0,12 (T53) | Tutarlı ama anlamlı değil |
| Önceden mühürlenmiş testler (T39, T42) | ABD ve 2025 sonuçları USDA'dan kötü | Başarısız |
| İleriye dönük test 2026 | Kilitler git'te tarihli | 10 Aralık 2026 ve Mayıs 2027'de puanlanacak |

**Sayfada kullanılacak cümle:**
> "USDA'nın pamuk tahminlerinin nerede ve neden yanıldığını bulduk: 2012'den sonra Teksas'ta hata verimden hasat alanına kaydı. O değişkende USDA'yı istatistiksel olarak anlamlı biçimde geçen gerçek zamanlı bir düzeltme kurduk. Üretimdeki kazanç tutarlı ama henüz anlamlı değil. İleriye dönük test git'te tarihli olarak kilitli."

## 2. O sorudan sonra yapılanlar (sırasıyla)

| Adım | Ne yapıldı | Sonuç | Kanıt |
|---|---|---|---|
| Birleşik kör test | 1987–1999 ve 2002–2011'deki 23 kör sezonun hepsi birleştirildi | 88/128 daha iyi, 8 TX/US hücresinin hepsinde ortalama kazanç pozitif, ortalama 0,64 yp, p = 0,007 | `pooled-blind-1987-2011-20261116.json` |
| T50 (ön kayıtlı) | Parametresiz kural: bir sonraki revizyon, son revizyonla aynı yönde olur | Kör pamuk %67 (p = 0,012), Teksas %79, tüm eyaletler %72; mısır %69, soya %67; pamuk 2012–2025 %43. **GEÇTİ** | `docs/T50_REVISION_DIRECTION_PREREG_20261116.md` |
| T51 (ön kayıtlı) | Teksas alanı: referans modeller, tahmin aralıkları, üretim etkisi, yapısal kırılma | Alanda USDA'yı ve yanlılık düzeltmesini geçti; üretimde p = 0,135, **eşik geçilmedi**. %80 aralık gözlemlerin yalnızca %64'ünü kapsıyor. Chow testi 2012'de p = 0,0005; sup-F taraması 2019'da tepe yapıyor (p = 0,055). | `docs/T51_AREA_PRODUCTION_PREREG_20261116.md` |
| USDA takip aracı (monitör) | Herkese açık ESMIS metnini okuyor, alan × verim kontrolü yapıyor, doğrulamadan geçemeyen tahminde USDA rakamına dönüyor; model dosyası donduruldu; kayıtlar SHA-256 ile zincirleniyor; GitHub Actions iş akışı var; 15 test geçiyor | Kasım 2026 raporundan kilitli sayıları birebir yeniden üretti | `research/monitor/`, `docs/adr/0001-usda-forecast-monitor.md` |
| PR #78 | Bütün bu iş `main`'e PR olarak açıldı | Açık. Birleştirme kararı kullanıcının. | |
| T52 (ön kayıtlı) | Revizyon yönü sinyali rapor günü vadeli fiyatı hareket ettiriyor mu? 2010–2021 | 32 olay, +%0,27, isabet %53, p = 0,30. **GEÇMEDİ.** Fiyat verisi yalnızca sinyalin artık çalışmadığı dönemi kapsıyor (bu olaylarda sinyal %47 doğru). | `docs/T52_MARKET_PREREG_20261116.md` |
| T53 (ön kayıtlı) | Alan düzeltmesi → Teksas üretimi, en uzun tarih (2001–2025, 1985–1994 alanları OCR ile) | Tüm dönem 4/4 ay, p = 0,19; 2008–2025 4/4 ay, p = 0,12; 2001–2007 1/4 ay. **GEÇMEDİ.** Seyrelme riski testten önce yazılmıştı. Alan hatası sonucu uzun örneklemde de tutuyor (Eylül %9,3 → %6,6). | `docs/T53_AREA_PRODUCTION_LONG_PREREG_20261116.md` |
| T54 (ön kayıtlı) | Aynı sabitlenme kuru tarım tane sorgumunda da var mı? (Kansas, Teksas, 2012–2025) | İşaret var: korelasyon −0,44, üst sınır −0,10. Düzeltme 4 ayın 4'ünde daha kötü, p = 0,70. **GEÇMEDİ.** USDA'nın sorgum alan hatası zaten küçük (%4–7). | `docs/T54_SORGHUM_PREREG_20261116.md` |
| Çalışma makalesi | İngilizce taslak; başarısız testlerin hepsi içinde | Hiçbir yere gönderilmedi. Yayın kararı kullanıcının. | `docs/paper/WORKING_PAPER.md` |
| Açık veri seti | Kurtarılan USDA aylık tahminleri 1985–2011, 5.088 satır | QuickStats ile çakışan hücrelerde 153/153 ve 441/441 eşleşme | `docs/research_package/data/usda_monthly_forecasts_recovered_1985_2011.csv` |
| Veri seti düzeltmesi (17 Kasım) | 2000–2011 pamuk satırlarının 1.022'sinde kaynak dosya adı boştu | Artık her satırda kaynak rapor adı var. Değerler değişmedi; yalnızca `source_file` sütunu doldu. Not: Eylül 2005 raporunun kayıtlı yayın tarihi 12 Eylül, dosya adı ise `CropProd-09-13-2005.txt`. | `research/package/export_dataset.py` |
| İngilizce portfolyo sayfası | T52–T54 deneme listesine eklendi; makale ve veri seti bölümü açıldı (sürüm 6) | | `docs/portfolio/cotton-usda-research.html` |

## 3. Bulguların bütünü (sorgumla birlikte)
Sabitlenme imzası, yani USDA'nın sezon içi alan hatasının kamuya açık "zayıf + çok zayıf" ürün oranıyla birlikte hareket etmesi, üç yerde görüldü:

| Yer | İşaret | Kullanılabilir düzeltme |
|---|---|---|
| Teksas pamuğu (keşif) | korelasyon −0,82 ile −0,88 arası | **Var:** alan hatası %11,4 → %7,0 |
| Oklahoma + Kansas pamuğu (bağımsız) | korelasyon −0,61 | Kıl payı: 4/4 ay, p = 0,056 |
| Kansas + Teksas sorgumu (bağımsız) | korelasyon −0,44 | Yok: USDA'nın hatası zaten küçük |
| Kışlık buğday | yok | Yok |

**Okuma:** mekanizmanın işareti kuru tarımda genelleşiyor. Kullanılabilir bir düzeltmeye ise yalnızca USDA'nın alan hatasının büyük olduğu yerde dönüşüyor. Sayfada sorgum kartına bu yorum eklenmeli.

## 4. `cottonlens-sunum.html` için yapılacaklar
Sayfa 17 Kasım 01:03 sürümünde T50–T54'ü, makaleyi, veri setini ve monitörü zaten içeriyor. Kalan işler:

1. **"Ne iddia ediyoruz, ne etmiyoruz" kutusu ekle.** Bölüm 1'deki tabloyu ya da kısa bir sürümünü koy, kapanış cümlesi bölüm 1'deki alıntı olsun. Bu sayfanın en çok sorulacak sorusu "USDA'yı geçtiniz mi?".
2. **T54 sorgum kartı:** "İşaret genelleşiyor; düzeltme yalnızca USDA'nın alan hatasının büyük olduğu yerde çalışıyor (sorgumda USDA'nın alan hatası %4–7)" cümlesini ekle.
3. **Veri seti kartı:** "1.022 satırda kaynak dosya adı boş" cümlesini kaldır; artık her satırın kaynağı var.
4. **Model envanteri:** "Üretim aşamasında T01–T46 denemeleri" ifadesini "T01–T54" yap.
5. **Kanıt dosyası sayısı:** sayfada 297 yazıyor; `research/evidence/` altında şu an 331 dosya var. `data/evidence_index.csv` güncellenmediyse sayıyı "297 (dizinde)" diye bırak ya da dizini yeniden üret.
6. **Bağlantılar** `0fb5c4a…` commit'ine sabitlenmiş. Bu dosyanın commit'inden sonra en son commit'e çevir; veri seti bağlantısı özellikle değişti.
7. **Tarihli takvim:**
   - **10 Aralık 2026:** T50 revizyon yönleri (ABD aşağı, Teksas aşağı) ve monitörün Aralık kaydı.
   - **Mayıs 2027:** ABD üretimi 12.763.510 balya, Teksas alanı 3.258.029 akre, Teksas üretimi 4.070.552 balya.

## 5. Kullanılmaması gereken ifadeler
- "USDA'yı geçtik" (koşulsuz), "USDA'dan daha iyi tahmin ediyoruz".
- "Piyasada para kazandırır" (T52 sıfır sonuç verdi).
- "Mekanizma bütün ürünlerde geçerli" (buğdayda yok, sorgumda düzeltme yok).
- "Üretimde anlamlı iyileşme" (p = 0,12–0,19).
- "Teksas sonucu bağımsız doğrulandı" (Teksas keşif eyaleti; Oklahoma/Kansas p = 0,056).

## 6. Codex'e verilecek istem
> `docs/research_package/17_SUNUM_GUNCELLEME_CODEX.md` dosyasını oku ve bölüm 4'teki maddeleri `docs/portfolio/cottonlens-sunum.html` (CSS/JS aynı klasörde) üzerinde uygula. Her sayıyı dosyada adı geçen kanıtla doğrula. Bölüm 5'teki ifadeleri kullanma. Görsel dili ve yapıyı koru; masaüstü, telefon genişliği, açık ve koyu temada kontrol et. Bu klasörün dışındaki hiçbir dosyayı değiştirme, commit yapma.
