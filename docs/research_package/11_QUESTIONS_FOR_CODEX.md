# 11 · Codex'e sorular (paketteki boşluklar)

Bu dosyanın tamamını Codex'e yapıştır. Cevaplar gelince bu dosyanın altına ya da ayrı bir `12_CODEX_ANSWERS.md` dosyasına ekle; paketteki ilgili belgeyi ben güncellerim.

---

## Codex'e not

Merhaba Codex. CottonLensAI'nin birinci aşamasını (21 Eylül – 7 Kasım 2026, fiyat ve risk tahmini) sen yürüttün. Sen ayrıldıktan sonra (13–16 Kasım) projeyi Claude devraldı. Claude'un yaptıklarını bilmen gerekmiyor; aşağıdaki sorular yalnızca senin dönemin hakkında.

### Claude'un çalışması nerede

- **Worktree:** `C:/Users/Eray/SEZAR/PROJELER/CottonLensAI-claude-review`.
- **Dal:** `claude/hls-move3`, GitHub'a push edildi (`origin`, `ErayKulkizaga/CottonLensAI`).
- **Başlangıç noktası:** senin devir belgen `docs/archive/phase2-notes/CLAUDE_HANDOFF.md` ve `445d1fe`.
- **Bilgi paketi:** `docs/research_package/`.
  - Senin dönemini anlatan belgeler: `03_PHASE1_PRICE_AND_RISK.md`, `04_MODEL_FAMILIES_AND_HYPERPARAMETERS.md`, `02_NUMBERS_CHEATSHEET.md`.
  - Bu belgelerin tabloları senin `research/trials.json`, `research/registry.json`, `docs/` ve `research/evidence/` kayıtlarından otomatik üretildi (`research/package/build_catalogs.py`). Tablolar `data/` klasöründe.
- **Claude'un ikinci aşaması** (USDA'ya karşı üretim tahmini): `docs/CROP_NOWCAST_TRIALS.md`, `research/crop_nowcast/`, `research/deep/`. Bunlara dokunmana gerek yok.

### Senin kayıtların (Claude'un gördüğü yerler)

- **Ana çalışma kopyası:** `C:/Users/Eray/SEZAR/PROJELER/CottonLensAI` (dal `codex/free-data-research-workbench-20261001`), ignore edilen veriler `output/` altında.
- **Worktree'ler:**
  - `C:/Users/Eray/.codex/worktrees/naive-mae-20261105/CottonLensAI` (`codex/root-cause-audit-20261106`);
  - `C:/Users/Eray/.codex/worktrees/repository-curation/CottonLensAI` (`codex/end-to-end-audit`).
- **Eski oturum kayıtların:** `C:/Users/Eray/.codex/sessions/` ve `archived_sessions/`. Hafızan yoksa cevapları önce buradan ve yukarıdaki dosyalardan çıkar.

### Kurallar

- **Salt okuma.** Hiçbir dosyayı değiştirme, commit yapma, `claude/hls-move3` dalına dokunma ve yeni eğitim başlatma.
- **Cevapların yeri.** Cevaplarını sohbette ver. Ya da yalnızca şu yeni dosyayı oluştur, başka hiçbir şeyi değiştirme: `C:/Users/Eray/SEZAR/PROJELER/CottonLensAI-claude-review/docs/research_package/12_CODEX_ANSWERS.md`.
- **Gizli bilgiler.** API anahtarı, token ya da `auth.json` içeriği yazma.

Paket bir iş görüşmesi için hazırlanıyor. Bir ML mühendisinin sorabileceği her şeye kanıtla cevap verebilmesi gerekiyor.

Claude aşağıdaki bilgileri kayıtlarda bulamadı ya da emin olamadı. Her soru için şunları yaz:
1. **Cevap:** bildiğin kadarıyla.
2. **Kesinlik:** `kesin` (dosyadan sayıldı) / `hafızadan` / `bilmiyorum`.
3. **Kaynak:** dosya yolu, commit, release etiketi, Colab/Drive konumu.
4. **Claude nereye bakmalı:** repoda değilse nerede olduğu (başka worktree, private arşiv, Drive, GitHub release).

Kesin bilmediğin bir sayıyı tahminle doldurma; `bilmiyorum` yaz. Paketteki bir şeyi yanlış buluyorsan (özellikle `03_PHASE1_PRICE_AND_RISK.md` ve `04_MODEL_FAMILIES_AND_HYPERPARAMETERS.md`), en sonda ayrı bir liste halinde belirt.

---

## A. Defterden önceki çalışmalar (21–25 Eylül)

1. **v1 Colab çalışması** (commit `c33d6ae`, XGBoost ve LSTM, 3.905 origin): toplam kaç model eğitildi? XGBoost ve LSTM için ayrı ayrı sayı var mı? Log nerede?
2. **Düzeltilmiş v2** (`v20260924-185501`) ve **tabular sprint** (`v20260925-074708`): kaç eğitim yapıldı? Bu sayılar `research/trials.json`'a neden girmedi?
3. Defter (`trials.json`) tam olarak hangi tarihte ve hangi commit'ten itibaren tutulmaya başlandı? Öncesinde kayıtsız kalan başka bir çalışma var mı?

## B. LSTM (görüşmede en çok sorulacak konu)

4. **v1 piyasa LSTM'inin tam mimarisi:**
   - katman sayısı, hücre sayısı, dropout ve recurrent dropout;
   - çıkış katmanı;
   - optimizer, öğrenme oranı, batch size, en fazla epoch, early stopping kuralı (patience, izlenen metrik);
   - kayıp fonksiyonu;
   - girdi özellikleri ve ölçekleme yöntemi.
5. "Ufuk ve dönem başına **4 konfigürasyon**" tam olarak hangi 4 ayardı?
6. "En iyi epoch 1" bulgusu hangi blokta ve hangi metrikle ölçüldü? Eğitim ve doğrulama kayıp eğrileri kaydedildi mi?
7. **DG-008** (kayıp common-cohort LSTM tahminleri): Colab'da, Drive'da ya da bir release içinde yedeği var mı? Yeniden üretilebilir mi (seed 42, aynı commit)?
8. 48 adaylık MLP/LSTM/TCN turu tasarlandı ama çalıştırılmadı. Bu kararın yazılı olduğu belge hangisi?

## C. Diğer model aileleri

9. **CatBoost** ve **Quantile** modelleri: kaç eğitim, hangi profillerde, hangi hiperparametrelerle? Paketteki `hyperparameter_ranges.csv` bunları doğru gösteriyor mu?
10. Round A'daki 128 adaylık arama nasıl örneklendi (rastgele, Sobol, grid)? Arama uzayının tanımı hangi dosyada?
11. "En iyi 5 aday 3 seed ile doğrulandı" adımının sonuç tablosu nerede?

## D. Ablasyon ve veri kaynakları

12. `research-market-v1` ablasyonundaki **20.360 alt-eğitim** neden deftere yazılmadı? 16 sonuç grubunun her biri neyi çıkarıyor ya da ekliyor? Kaynak dosya hangisi?
13. **Yahoo "first-observed cohort"** tasarımı tam olarak nasıl çalışıyor? Yahoo geçmiş fiyatları sonradan revize ettiği için ne yapıldı? Belgesi hangisi?
14. **AMS kontrat ayı hatası:** hata nasıl fark edildi? 3.320 düzeltme içeren overlay dosyası nerede?
15. Fiyat tarafında **mühürlü veri** var mı (2021 sonrası)? Ne mühürlü, mühür nerede kayıtlı, açılma kuralı ne?

## E. Canlı ileri test (own-level HAR)

16. Own-level HAR ileri testi **şu an gerçekten çalışıyor mu**? Hangi GitHub Actions workflow'u ya da zamanlanmış iş? Hangi repo, hangi release etiketleri?
17. Bugüne kadar kaç origin yayımlandı? Şimdiye kadarki QLIKE farkı ne (sadece bilgi olarak; protokol gereği karar Kasım 2027'de)?
18. İş durduysa son başarılı yayın tarihi ne?

## F. Uygulama ve altyapı

19. Repodaki `backend/` (FastAPI?) ve `frontend/` (Angular) uygulaması ne yapıyor? Çalışır durumda mı, bir yere deploy edildi mi? Görüşmede demo yapılabilir mi, nasıl başlatılır?
20. `ml/src/cottonlens_ml/` paketi ile `research/` klasörü arasındaki fark ne? Hangisi üretim kodu?
21. Toplam hesaplama: Colab GPU saati, kullanılan GPU türleri, harcanan para (varsa).

## G. Kararlar ve bağlam

22. **+%5 MAE hedefi** ve **%53 / %55 yön eşikleri** neye göre seçildi? Kim karar verdi, nerede yazılı?
23. Hangi araştırma kararları kullanıcıdan (Eray), hangileri senden geldi? Görüşmede "bu fikir kimindi?" sorusu gelebilir.
24. `docs/archive/phase2-notes/CLAUDE_HANDOFF.md` içinde geçen **private arşivler ve diğer worktree'lerde** repoda olmayan önemli bir sonuç var mı?
25. Birinci aşamada **en önemli 10 belge** hangileri? Claude 115 belgenin hepsini okumadı; görüşme için mutlaka okunması gerekenleri sırala.

## H. Paket kontrolü

26. `docs/research_package/03_PHASE1_PRICE_AND_RISK.md` ve `04_MODEL_FAMILIES_AND_HYPERPARAMETERS.md` dosyalarını oku. Yanlış, eksik ya da abartılı bulduğun her ifadeyi dosya adı ve satırıyla listele.
27. Özellikle şu sayıları doğrula:
    - 32.934 tamamlanmış eğitim (28.245 piyasa, 4.597 sentetik, 92 denetim);
    - 14.723 XGBoost (7.320 round A, yaklaşık 6.500 kaynak keşfi);
    - en iyi registry kazancı +%0,22, medyan −%0,65;
    - market-v1'de en iyi +%0,763;
    - own-level HAR'da +0,0177 QLIKE;
    - Link 1 oranı 1,42 (p = 0,004).
