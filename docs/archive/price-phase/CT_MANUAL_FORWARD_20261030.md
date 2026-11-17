# Own-level risk: manuel ileri protokol

Profil `ct-own-level-manual-forward-t1-v1`. **Hazırlandı; kohort kilitli
değil, yeni fit veya ileri tahmin yok.** Bu kod bir zamanlanmış görev kurmaz.
Konsol açan eski saatlik görev ve üç tetikleyicisi kapalı kalır.

## Sabit bilimsel tarif

[PR63 tarihsel adayının](CT_LEVEL_DXY_CONTROL_20261030.md) beş parametreli
own-level QLIKE learner'ı kullanılır; yeni model, grid veya kaynak araması
yoktur. Hedef sonraki Cotton seansının kare log-getirisi; referans EWMA
λ=0,94. Fiyat/yön başarısı veya işlem getirisi iddiası yapılmaz.

- Başlangıç, [PR65 kaynak kabulündeki](CT_MANUAL_SOURCE_20261030.md) ayrı,
  hash-bağlı 4.232 satırlık seed sürümüdür. Eski veri değiştirilmez.
- Expanding eğitim 2010-06-24/index119 warmup'ını korur. Beşinci hedef
  tarihi cutoff'tan **kesinlikle önce** olmalıdır; en az 500 olgun satır.
- Her takvim yılının ilk kohort origin'inde son 189 olgun origin, 3×63
  iç doğrulama, 21 gözlemde refit; `[0,.25,.5,.75,1]` aritmetik varyans
  karışımı, eşitlikte küçük ağırlık. Dış refit sayacı eksik origin'leri
  de sayar ve yeni yılda sıfırlanır.
- Önceki **21-fit taslağı eksikti**: iki yıla yayılan 252 slotta yıllık
  seçim en fazla 18 iç + 13 dış = **31 piyasa fiti** gerektirir. Yeni
  sentetik fit yok; tek süreç, en fazla iki thread, oturum en fazla 30 dakika.

252 origin, sınırlı bir ileri kanıt çalışmasıdır; %80 güç veya başarı
garantisi değildir. [Kurulabilirlik hesabının](CT_PROSPECTIVE_READINESS_20261030.md)
yaklaşık %9–10 değeri 126 origin için koşullu tek-aralık senaryosudur.

## Kilit ve karar kuralları

`lock` gerçek UTC saatinden sonraki ilk başlamamış 00:15 karar penceresini
seçer; komutta sahte `--now` veya geçmiş ilk tarih seçimi yoktur. Tarif
2026'da başlayıp en fazla 2027 sonuna uzanan ilk 252 resmi Cotton slotunu
alır. Eski 126-baseline kohortu ayrı ve değişmezdir; kaçan origin yerine
yeni origin eklenmez, çalışma sonuç görülerek uzatılmaz veya yeniden başlatılmaz.

İleri birincil fark `QLIKE(EWMA) − QLIKE(seçilmiş model)`; pozitif model
lehinedir. Doğrulayıcı üstünlük için 252 slotun tamamında zamanında
eşlenmiş yayın ve olgun hedef, pozitif ortalama ve hem 20 hem 60 blok
%95 alt sınırının pozitifliği gerekir. Bootstrap 10.000 tekrar/seed42,
yıl sınırını aşmaz. Eksik yayın doğrulayıcı başarıyı engeller; dönem
sonundaki kısmi eşlenmiş sonuç ancak tanımlayıcı olabilir. Ara sonuçla
tarif seçimi veya erken başarı ilanı yoktur. Top%1 trim hassasiyettir.

Bu ileri kural, tarihsel adayın zaten karşıladığı özgün 6/8 yıl kapısını
değiştirmez; tek ileri yılı sekiz bağımsız yıl gibi sunmaz. Bu teslimattaki
`status` yalnız kayıt sayılarını denetler; dönem sonu loss raporu henüz
çalıştırılabilir bir komut olarak eklenmedi ve hiçbir ara skor üretilmedi.

## Gerçek zaman ve veri zinciri

Karar 00:15 UTC; kaynak makbuzu en geç bu anda, yayın 00:15 dahil–00:30
hariç. `capture` yalnız açık manuel çağrıyla 00:00–00:15 arasında tek Yahoo
isteği yapar; retry, döngü veya görev kaydı yoktur. Yanıt geç gelirse
saklanır fakat o origin'e tahmin üretilemez. Sıfır/negatif fiyat, açık bar,
yanlış seans, tarih boşluğu ve bilinen fiyat revizyonu reddedilir.

`fit`, mevcut Ledger ve learner'ı çağırır. İlk 9 geçmiş-iç fit + bir dış
model durumu önceden hazırlanabilir: origin kapanmadan da beş-gözlem
olgunluk filtresi aynı train'i verir. Dış checkpoint'teki son-train satırı
yalnız `IN_SAMPLE_STATE_CHECK_ONLY` etiketlidir; gelecek tahmin değildir.
Model ve yıllık karışım seçimi karar kesiminden önce tamamlanmış olmalıdır.
Her rolün planı eğitimden önce donar; checkpoint/pointer yazısı kesilirse
aynı rol farklı snapshot'la yeniden eğitilmez.

`admit` ham kaynak/HTTP makbuzu/tam geçmişi hash bağlı saklar. `publish`
origin özelliğini o kaynakla kurar; ham ve seçilmiş varyans, EWMA, kaynak,
model, seçim ve kohort kimliklerini korur. Önceki kaçan slotlar gerçek
oluşturma saatli `MISSING` olaylarıdır; tahmin doldurulmaz. Olaylar hash
zinciridir. Payload yazısı tamamlanıp olay mühürlenmediyse yayın tamamlanmış
sayılmaz; yazma 00:30'u aşarsa makbuz `LATE` olur. Aktif writer görülürse
status eksik yazıyı başarıya çevirmez.

Resmi IFUS Cotton takviminin mevcut kapsamı 1 Ocak 2027'de biter. Gelecek
takvim ancak ayrı ham ICE belgesi, HTTP makbuzu ve açık Cotton-sütunu
incelemesiyle; bilinen takvimi tam koruyarak ve yeni kapsam başlamadan
eklenebilir. Şu an 2027 uzatması yoktur; kapsam dışında durulur. Yahoo
provider-proxy / named-contract / settlement / tarihsel vintage sınırları
aynen geçerlidir. Yeni canlı API modeli yayımlanmaz.

## Doğrulama ve kullanım

77 ilgili test/Ruff geçti: doğru ve geç yayın, kesim eşitliği, gelecek
fiyat/hedef değişimi, olgunluk, eski warmup, eksik slot sayacı, yarım yazı,
payload tahrifi ve takvim revizyonları dahil. Mevcut native öğrenme
kontrolünde kernel byte'ları/ortam/kimlik ve 200 tahmin yeniden doğrulandı;
genel `passed` cache'i kullanılmadı. Bağımsız ordinal denetim ilk dönem
9 iç plan ve dış train listesini birebir doğruladı: minimum 3.914 iç,
4.108 dış olgun train satırı. 30 Ekim ilk origin'i yalnız preflight
örneğidir; gerçek ilk origin sonradan gerçek `lock` saatinde belirlenir.

Hazırlık v1 kaynak/çıktıları korunur; v2, eşlenmiş kayıp işaretini açıkça
yazar ve kaynak/yayın transaction sınırlarını tamamlar. V1 ile kohort veya
eğitim açılmadı. İki hazırlık da piyasa başarısı kanıtı değildir.

```powershell
python -m research.ct_manual_forward verify --repo FROZEN_SOURCE --root STUDY
python -m research.ct_manual_forward lock --repo FROZEN_SOURCE --root STUDY --authorization AUTH
python -m research.ct_manual_forward fit --repo FROZEN_SOURCE --root STUDY --authorization AUTH --origin FUTURE_ORIGIN
python -m research.ct_manual_forward capture --repo FROZEN_SOURCE --root STUDY
python -m research.ct_manual_forward publish --repo FROZEN_SOURCE --root STUDY
python -m research.ct_manual_forward status --repo FROZEN_SOURCE --root STUDY
```

Kilitten önce kaynak, ön kayıt ve gerekli girdilerin private GitHub'dan
taze geri kurması doğrulanmalıdır. Yerel çalışma dosyası veya yeşil test
tek başına bu yetkiyi vermez. Komutları otomatik çalıştırma izni bu
teslimata dahil değildir. Günlük gerçek kaydın nasıl sürdürüleceği net
olmadan 252-origin kohortunu başlatmak eksik yayınları peşinen üretir;
bu yüzden mevcut hazırlık, kilit/eğitim/yayından ayrıdır.

## Teslimat ve etkin sürüm

[Taslak PR66](https://github.com/ErayKulkizaga/CottonLensAI/pull/66) merge
edilmedi. İlk private ön kayıt 1.161 dosyayla taze geri kuruldu, altı çıktı
byte eşit replay geçti. Kilit açılmadan yapılan son iz sürmede tahmin
payload'ına açık `horizon=1` ve resmi sonraki Cotton `target_date` yazılması
tamamlandı; kapsam dışındaki hedef tarihi de artık yayını durdurur. Cuma
30 Ekim → Pazartesi 2 Kasım örneği testte doğrulandı. Eski sürümlerde hiçbir
kohort, fit veya tahmin üretilmedi; veri/tarif/kayıp değişmedi.

**Etkin ön kayıt v4**:
[private hedef-tarih eki](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/ct-manual-forward-targetdate-prereg-20261030).
530 dosya/sıfır dışlama taze restore; indirilen frozen kodla ready, sözleşme,
kaynak manifesti, öğrenme makbuzu, geçmiş sorgusu ve bağımsız plan çıktısı
ağ/fit/optimizer kapalıyken altı byte eşit çıktı verdi.

Source `a173df8e65b42460931018e53ab11778b248698383920231af1f31bb6cb16d41`;
design `2950ca5a0792022c1cea2f584b1599936c94221de3f27aff46af50233e795134`.
ZIP `3ded0d730ff9becc28e3c2724a05784f01ba8a443e2530f44018593310498631`.
Eski ön kayıt ve [ilk kanıt](../../../research/evidence/ct-manual-forward-prereg-20261030.json)
değişmez; [etkin ek kayıt](../../../research/evidence/ct-manual-forward-targetdate-prereg-20261030.json)
ayrı tutulur. Sicil 225 aşama/560 tarif/32.490 fit kopyası; eski trials
byte hash'i değişmedi. Günlük kayıt izni için kullanıcıya açık soru
gönderildi; yanıt beklenirken kilit, fit, yayın veya yeni görev açılmadı.
