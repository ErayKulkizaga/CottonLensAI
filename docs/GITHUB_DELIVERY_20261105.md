# GitHub konsolidasyonu — 5 Kasım 2026

## Kapsam

Başlangıç main c38e151, güncel birleşik dal 463cff91;164commit ileride.
108privateRelease/465asset metadata envanteri çıkarıldı; belgelerde bağlantısı
olan Release tag'lerinde eksik yok. Bu bütün arşivlerin yeniden indirildiği
iddiası değildir. Önceki hash-bağlı teslimat kayıtları korunur.

Ana eski çalışma kopyasında kişisel opencode.json dışındaki32yol karşılaştırıldı:
23eşdeğer,9sonraki düzeltmelerle superseded. Sonraki değişiklikler API paket
kimliği ve kaynak sahipliği, ek araştırma profilleri, olgun hedef/rapor/native
makbuz korumaları ve notebook testlerini kapsar. Eski yerel dosyalar korunur.
Provenance dalındaki iki unique commit'in bilimsel kod/test/kanıtları aynıdır;
belge ve sicil snapshot'ları sonraki sürümlerle genişlemiştir. Dallar korunur.

Küçük paket, kapsam metadata'sı, özgün README ve iki resmi takvim PDF önizlemesini
korur. tmp önizlemeleri ham bilimsel fiyat kanıtı değildir. Eski kaynak Release
research-semantics-20261005 yeniden indirilip ZIP hash'i ve211üyenin tamamı doğrulandı.
Eski TF-placement/On-Call için bilinen22299payload referansı hâlâ eksiktir;
makbuzlar korunur, eski eğitimler yeniden oluşturulmadı.

## Belgeler ve kontroller

README güncel amacı, Naive/EWMA durumunu, kurulum/test ve geri yüklemeyi gösterir.
Özgün README byte içeriği docs/archive/README_HISTORY_20261105.md altında korunur.
Günlük/saatlik görevler kapalıdır. Yeni araştırma planı ön kayıt taslağıdır;
bu teslimat yeni fit, aktif model veya otomasyon değişikliği içermez.

Backend47pytest/Ruff geçti. Frontend Node24.19.0 ile kurulum/build doğrulanır.
ML/research sonuçları ve sonCI teslimat makbuzuna eklenir. npm audit mevcut
6bildirim(1low,1moderate,4high)veriyor; bu belge/birleştirme işinde bağımlılık
upgrade'i yapılmadı. Compose config ve sicil doğrulanır.

.env,sırlar,kişisel yapılandırma,incelenmemiş veritabanları,cache/build/venv ve
ilgisiz kullanıcı dosyaları yayın dışıdır. output klasörlerinin tamamı için yeni
byte-level kapsam garantisi verilmez; bilimsel kapsam ilgili mevcut makbuz ve
Release manifestlerinde tanımlıdır. Bu sınır, bilinen eski checkpoint eksiğini gizlemez.

## Yayın

Kapsam/manifest: [makbuz](../research/evidence/github-consolidation-20261105.json).
[Private teslimat paketi](https://github.com/ErayKulkizaga/CottonLensAI/releases/tag/github-consolidation-20261105).
PR75 sonCI sonrası merge commit ile main'e alınır; kapsananPR'lar
commit/content doğrulamasıyla kapatılır. Dallar/Release'ler korunur.
Son mainCI ve kapananPR kimlikleri ayrı sonuç makbuzuyla kaydedilir.
