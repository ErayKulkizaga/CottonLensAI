# Vade eğrisi risk tanısı: kapanış ve kanıt sınırı

**INCONCLUSIVE.** 14 Ekim'de dondurulmuş tek hipotez, ilk iki vadeli
kontratın `log(F1/F2)` farkının aynı ikinci kontratın ertesi seans
`gerçekleşen kare getiri / EWMA − 1` hatasıyla pozitif ilişkili olmasıydı.
29 Ekim'de yeni eğitim yapmadan kaynak, hedef ve istatistik denetimi kapandı.

| Sabit kapsam / istatistik | Sonuç |
|---|---:|
| 2016–2023 ordinal slot / uygun origin | 2.006 / 1.987 |
| Yıl içinde merkezlenen, uygun origin ağırlıklı covariance | +0,002276842168 |
| Blok20 %95 aralık | [−0,000593850901; +0,004640553840] |
| Blok60 %95 aralık | [+0,000071327494; +0,004326770704] |
| Pozitif yıl | 5/8 |
| En büyük %1 EWMA kaybı çıkarılınca covariance | +0,001071457599 |

İki aralığın pozitif alt sınırı ve en az 6/8 yıl şartı birlikte sağlanmadı.
Tek pozitif aralığı seçmek, işareti/ufku değiştirmek veya yeni grid açmak
bu sonucu doğrulamaz. Bu bir **ilişki** ölçümüdür; covariance QLIKE kazancı
değildir. Sonuç, bütün vade yapısı bilgisinin yararsızlığını da kanıtlamaz.

## Bağımsız denetim

`research/verify_ams_curve_intervals.py`, analizin merkezlenmiş çarpım
kodundan ayrı yeterli istatistik hesabıyla iki aralığı yeniden hesaplar.
Kaynak settlement fiyatlarından Decimal ile 1.987 hedef, 1.987 spread ve
1.987 göreli hata (5.961 scalar) kurar. Yıl sınırları aşılmaz; eksik 19 slot
takvimden silinmez. Her iki hesapta 10.000 tekrar, seed42, blok20/60 vardır.
Yıllar, hedef kontratı, kaynak hash'leri ve uç-kayıp duyarlılığı doğrulanır.

[Private Release](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/ams-curve-risk-moment-20261029)
10 dosya/3.711.596 mantıksal baytı saklar. ZIP SHA-256:
`7fbfed917979177cf91e23a76d7a75cad42c4de7dd6cbea75d8ae0fb1f22986e`.
Taze GitHub indirmesi ve restore sonrası doğrulama **geri kurulmuş girdi
dosyaları açıkça verilerek** çalıştırıldı; yereldeki eski mutlak yollardan
okuma yapılmadı. Doğrulama makbuzu bayt eşit yeniden üretildi. Özgün
14 Ekim contract/result/prediction dosyaları değiştirilmedi.

```powershell
python -B research/verify_ams_curve_intervals.py `
  --root RESTORED/bundle/analysis `
  --report RESTORED/bundle/inputs/paired-losses.csv `
  --observations RESTORED/bundle/inputs/observations.csv `
  --output NEW_RECEIPT.json
```

Kaynak günü+1 gün 00:00 UTC erişimi hâlâ varsayımdır; ilk vintage ve tarihsel
publication UTC doğrulanmış değildir. Görülmüş araştırma yılları bağımsız
holdout sayılmaz. Kaynağın geçmişteki koşullu kullanılabilirliği ile model
becerisi ayrı sorulardır. Yeni fit0; eski 204 aşama, 185 kanıt dosyası ve
545 tarif/30.419 fit kopyası korunur. Sicile yalnız sıfır-fit ek kayıt kondu.

## Şimdi bildiğimiz ve açık kalan tek araştırma ayrımı

Resmî günlük arşiv sayesinde sekiz yıllık aynı-kontrat geçmişindeki eski
kapsam açığı kapatıldı; [aynı kaynaktaki HAR kontrolü](AMS_NAMED_EWMA_HAR_20261014.md)
yine güçlü EWMA'yı güvenilir biçimde geçmedi. Loss, sabit scale ve EWMA-state
kontrolleri de üstünlük göstermedi. Dolayısıyla sadece daha fazla aynı tür
veri, optimizer değişikliği veya kontrat temizliği çözüm olarak desteklenmiyor.

Bu bulgular bütün modellerde öğrenilebilir bilgi yokluğunu kanıtlamaz.
Sıradaki ayrım, mevcut bilgi kümesinin küçük fakat kullanılabilir etkisi ile
ekonomik bilgi içeren ve gerçekten karar anında bilinebilen yeni girdidir.
Sonuçlanmamış bu ilişki ekranı artık yeni eğitim gerekçesi olarak kullanılamaz;
aynı HAR/loss/scale tariflerini yeniden çalıştırma. Yeni adayın gerekçesi,
geçmiş sicilde sınanmamış bilgi/temsil farkını ve tek sabit testi açıkça
belirtmelidir. Referans, eski fiyat kapıları ve risk kapısı zayıflatılmaz.

CI düzeltmesi: `unittest discover`, serbest pytest fonksiyonlarını toplamıyordu.
Araştırma adımı artık `pytest research/tests` kullanır; mevcut unittest sınıfları
da toplanır. Bu pakette 117 ilgili araştırma/ML testi ve Ruff geçti. Bu kontroller
hesap doğruluğuna kanıttır; piyasa tahmin başarısına kanıt değildir.
