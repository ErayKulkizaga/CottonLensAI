# Gerçek eğri T+5 — tek ufuk duyarlılık deneyi

Bu kayıt [tamamlanmış gerçek eğri T+1 sonucunu](CONTRACT_CURVE_RESULT_20261010.md)
ve [sıfır-fit hedef ayrıştırmasını](CURVE_TARGET_INTEGRITY_20261010.md) izler.
Eski plan/sonuç silinmez; T+1, spot-basis T+5 veya grid tekrar edilmez.
**Ön kayıt piyasa sonucu değildir.** Saat/ilk vintage varsayımı korunur;
tarihsel erişimin veya leakage yokluğunun ispatı iddia edilmez.

## Ayrıştırılacak soru

Aynı gerçek vade farkının katkısı bu sabit temsilde beş Cotton gözleminde
ortaya çıkıyor mu? T+1 negatifi bunu elemedi. T+5 kaynak katkısı pozitifse
ufka bağlı **sürekli-seri fiyat** katkısı desteklenir. Sadece kontrat geçişlerinde
görülen katkı, aynı-kontrat fiyat hareketi veya pozisyon becerisi sayılmaz.
Negatif sonuç bu sabit tarifin dışında bütün eğride bilgi yokluğu anlamına gelmez.

## Sabit tasarım

| Unsur | Kilitli karar |
|---|---|
| Profil / namespace | `contract-curve-t5-pilot-v1` / `research-contract-curve-t5-pilot-v1` |
| Veri | Mevcut 997 AMS Final tablosu; yeni veri yok, model-eligible0 korunur |
| Cotton history | Tamamlanmış T+1'in genişletilmiş history'siyle **bütün hücreler birebir**; orijinal history3520 satır/60 sütun |
| Kohort | Aynı2021/2022/2023,252/251/246 =749 origin/kol; kaynak eksikliği origin düşürmez |
| Kollar | Aynı mask/numeric D0/D1;24 çekirdek +3 ortak kaynak göstergesi +1 spread; eksiklik desenleri eşit |
| Saat | Cotton kaynak tarihi +1 takvim günü00:15 UTC |
| Varsayımsal kaynak erişimi | `max(reference date, unverified publication calendar day)+1 day00:00 UTC`; Final vintage varsayılır |
| Eskilik | D0 ilk eligible Cotton kararına bağlı azami3 gözlem; D1 gecikmesi eskiliği yeniden başlatmaz |
| Model | Ridge alpha1, seed42,window1,train-only standardized log-return |
| Tek bilimsel değişiklik | Model hedefi T+1 yerine **T+5**; beş orijinal Cotton gözlemi, takvim günü veya kaynak satırı değil |
| Eğitim | Aynı expanding history/120 warmup; `target_date_5 < cutoff`; train-only imputation/scaler/target dönüşümü |
| Seçim/refit | Geçmiş3×63 validation,21 gözlemde refit; sabit tarif, yeni model/grid yok |
| Shrinkage | Aynı0/.25/.5/.75/1; sadece geçmiş, eşitlikte düşük ağırlık; T+1 OOS ağırlıkları aktarılmaz |
| Baseline | Aynı749 hedefte Naive; birincil kaynak kontrolü mask_D0 |
| Bütçe | 108 iç +144 dış =252 piyasa fit'i; en fazla1 ayrı T+5 sentetik kontrol |
| Kaynak sınırı | Tek süreç/azami2 thread;30 dakikalık checkpoint koruyan oturum;12 çıktı/2.996 tahmin satırı |

2016–2019'da kaynak yok;2020 geçmiş iç bloklarında uygun kaynak yok. Bu
elenme T+1'de **geçmiş kaynak uygunluğu** ile yapılmıştı; şimdi aynı dondurulmuş
origin'ler korunur. T+5 sonuçlarıyla yeni yıl/period seçilmez.2024+ kullanılmaz.
Üç yıl6/8 koşulunu değerlendiremez;2/3 veya3/3 onun yerine konmaz.

## Karar kuralları

Birincil soru bütün749 origin'de seçilmiş numeric_D0–mask_D0 paired mutlak
fiyat hatasıdır; pozitif fark numeric'i destekler. Yılları koruyan mevcut
paired bootstrap10.000 tekrar,seed42,blok20 ve60. İki alt sınır da pozitifse
`CONDITIONAL_SOURCE_CONTRIBUTION_GATE_UNTESTED`; D1 birincil sonucun yerine
geçemez. Diğer durumda Naive kazancının iki üst sınırı5%'in altında ise
`FIXED_RECIPE_BELOW_PRACTICAL_GOAL_SOURCE_INCONCLUSIVE`; aksi hâlde
`INCONCLUSIVE_SOURCE_CONTRIBUTION`. Bu sınıflar otomatik model yayını açmaz.

5% fiyat-MAE,55% T+5 yön ve6/8 yıl koşulları değişmez. Ham/seçilmiş sonuç,
aktif tahmin oranı, aktif yön/yönün sıfır semantiği, yıllık katkı ve en büyük
1% Naive hatası çıkarılmış duyarlılık birlikte raporlanır. Flat ağırlık0/[0,0]
aralığı kaynağın sinyalsiz olduğunun kanıtı değildir. İncelenmiş yıllar ve
önceki insan seçimleri bu aralıkları bağımsız holdout'a dönüştürmez.

## Leakage ve hedef yorum koruması

Birincil skor ve geçmiş seçimleri tamamlanmadan kontrat geçişi alt grubu
hesaplanmaz. Sonuç raporuna ayrı **ex-post** açıklama eklenir: geçiş/aynı
ilk-kontrat/bilinmeyen olarak ham ve seçilmiş paired hata toplamları.
Bütün749 origin ve hata toplamı korunur; bu alan model feature'ına,
train/validation süzmesine, parametre seçimine veya yeni karara giremez.

Önceki tanı T+5'te743 bilinen çift/74 geçiş origin'i/15 ilk-vade değişimi
buldu; altı bilinmeyen silinmez.74 origin bağımsız74 olay değildir. Sonradan
bilinen hedef kontratının%2,891947 gap-only azalması **tahmin veya ulaşılabilir
üst sınır değildir**. Aynı-kontrat hareketi için ayrı sözleşme kanıtı gerekir;
bu deneye başka hedef eklenmez.5 gözlemde overlap, blok20/60 ve yıllık
heterojenlik korunur. Ex-post alt gruplara yeni CI/önemlilik taraması yapılmaz.

## Yürütme ve kimlik

Mevcut `Experiment`, `Ledger`, `path_pilot` ve paired raporlama kullanılır.
T+1 profili T+5'i, T+5 profili T+1'i reddeder. Ayrı source/registration/ready
kimlikleri ve namespace zorunlu; eski checkpoint, shrinkage veya tamamlanma
makbuzu taşınmaz. Yanlış ufuklu sentetik kontrol genel passed sayılamaz.
Eski geçici lock dosyaları completion envanterine giremez.

Kaynak/testler tamamlanınca kod/veri/ortam ve bu kararlar ön kayda bağlanır.
Mevcut CPU ortamının gerçek sürümleri kayda girer; lock'taki bütün sürümlerin
kurulu olduğu iddia edilmez. Kaynak değişirse eski kayda devam edilmez.
Ön kayıt GitHub'a korunmuş checksum paketiyle teslim edilmeden fit başlamaz.
Bu adımda gerçek fit0; büyük arama, GPU, yeni dataset, FAS kaynak kabulü,
canlı görev değişikliği, PR merge veya model yayını yok.

```text
python -m cottonlens_ml.research.contract_curve --profile contract-curve-t5-pilot-v1 --repo REPO --output NEW_REGISTRATION --history ORIGINAL_HISTORY --quotes FROZEN_QUOTES --input-proof SOURCE_PROOF --parent-root COMPLETED_T1_ROOT --mode assumption_sensitivity_not_historical_pit
python ml/full_year_cpu.py prepare --profile contract-curve-t5-pilot-v1 --cpu-environment EXISTING_CPU_ENV --drive-root NEW_RUN_ROOT --registration-root NEW_REGISTRATION --decision-contract NEW_REGISTRATION/decision-contract.json
python ml/full_year_cpu.py pilot-plan --profile contract-curve-t5-pilot-v1 --cpu-environment EXISTING_CPU_ENV --drive-root NEW_RUN_ROOT
```

`prepare`/`pilot-plan` fit yapmaz. `pilot` ancak kimlik/kanıt teslimi doğrulandıktan
sonra tek252+1 tavanla; devam eden fit'te yalnız doğrulanmış checkpoint resume.
`search`, `reproduce`, `lock`, `export`, model yayını ve remote mirror reddedilir.

Sonraki tek iş bu kilitli ufuk karşılaştırmasını sonuçlandırmak; sonuç
belirsiz/negatif diye otomatik ikinci model, grid veya dataset programı açılmaz.
