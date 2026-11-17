# CottonLensAI

- Kurallar ve güncel durum `AGENTS.md` içinde; o dosya Claude için de geçerli. Belge dizini `docs/README.md`.
- Tek çalışma klasörü burası. Büyük veri ve ara çıktılar git'e girmeyen `output/` altında; `CottonLensAI Colab/` yerel Colab çıktılarıdır.
- Python araçları kullanıcı kurulumu: `python -m ruff`, `python -m pytest`. Testler CI ile aynı `PYTHONPATH=ml/src:ml` (PowerShell: `ml/src;ml`) ve `pyarrow` ister.
- Parquet okuyan betikler için pyarrow gerekir; sistem Python'unda yoksa ayrı bir ortam kullan.
- Commit'ler pathspec ile yapılır; `output/`, `.env` ve anahtar dosyaları hiçbir zaman eklenmez.
- `research/evidence/` değiştirilmez. Yeni test önce `docs/prereg/` altına yazılıp gönderilir.
- Repo public (17 Kasım 2026). Kanıt release'leri private `ErayKulkizaga/CottonLensAI-evidence` deposunda; buraya release, anahtar, `.env`, Drive dökümü, veritabanı ya da ham üçüncü taraf fiyat serisi konmaz. Commit e-postası noreply.
- Eski Codex işi `archive/codex-wip-20261003` dalında saklı (17 Kasım 2026); test edilmedi, birleştirilmedi.
