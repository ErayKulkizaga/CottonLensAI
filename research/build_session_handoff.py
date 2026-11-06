"""One Drive notebook for the remaining session/RNG synthetic controls."""
import ast
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    output = ROOT/'output/l4-session-handoff-20261107'
    output.mkdir(parents=True, exist_ok=True)
    paths = [ROOT/'ml'/n for n in ('gpu_session_probe.py', 'gpu_restart_probe.py', 'a100_preflight.py',
                                  'colab_setup.py', 'pyproject.toml', 'uv.lock')]
    paths += sorted((ROOT/'ml/src').rglob('*.py'))
    receipts = ['l4-technical-verification-20261106.json', 'l4-restart-verification-20261106.json',
                'l4-oom-verification-20261107.json']
    payload = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in paths}
    payload.update({'prior-evidence/'+n: (ROOT/'research/evidence'/n).read_bytes() for n in receipts})
    manifest = {n: hashlib.sha256(raw).hexdigest() for n, raw in payload.items()}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        for n, raw in sorted(payload.items()):
            info = zipfile.ZipInfo(n, (2026, 11, 7, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, raw)
    source = buffer.getvalue()
    digest = hashlib.sha256(source).hexdigest()
    cells = []

    def cell(kind, text):
        if kind == 'code':
            ast.parse(text)
        item = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(keepends=True)}
        if kind == 'code':
            item.update(execution_count=None, outputs=[])
        cells.append(item)

    cell('markdown', '''# CottonLens — kalan L4 kontrolleri, tek notebook
Yalnız sentetik veri; toplam üç küçük minibatch güncellemesi. Önceki başarılı
preflight/restart/OOM testleri yeniden çalışmaz; doğrulama makbuzları taşınır.

**1. çalıştırma:** Yeni L4 runtime ile Tümünü çalıştır. Hazır mesajı gelince
Çalışma zamanı > Bağlantıyı kes ve çalışma zamanını sil işlemini sen yap.
Yalnız kernel'i yeniden başlatmak yeterli değildir.

**2. çalıştırma:** Aynı notebook'u yeniden L4'e bağlayıp Tümünü çalıştır.
Drive checkpoint'inden devam eder, NumPy/TF rastgeleliği, optimizer/model ve
tahminleri kesintisiz referansla karşılaştırır. Sonuçları Drive'a kaydeder.
Başarısızlıkta dosyaları silip yeniden deneme; mevcut kanıt korunmalıdır.
''')
    cell('code', '''from google.colab import drive
drive.mount('/content/drive')
''')
    cell('code', f'''from pathlib import Path
import hashlib, io, json, sys, zipfile
base = Path('/content/drive/MyDrive/CottonLensAI/GPU_Handoff_L4_20261106')
source = (base/'cottonlens-l4-session-source.zip').read_bytes()
assert hashlib.sha256(source).hexdigest() == {digest!r}, 'Source hash mismatch'
expected = {manifest!r}
root = Path('/content/CottonLensSessionProbe')
with zipfile.ZipFile(io.BytesIO(source)) as z:
    assert len(z.namelist()) == len(set(z.namelist())) and set(z.namelist()) == set(expected)
    for name, wanted in expected.items():
        raw = z.read(name)
        assert hashlib.sha256(raw).hexdigest() == wanted
        target = (root/name).resolve()
        assert target.is_relative_to(root.resolve())
        if target.exists(): assert target.read_bytes() == raw, 'Changed local source'
    for name in expected:
        target = root/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists(): target.write_bytes(z.read(name))
sys.path.insert(0, str(root/'ml'))
import colab_setup
python = colab_setup.setup()
''')
    cell('code', '''import subprocess
destination = base/'session-rng-results-20261107'
env = colab_setup.process_env()
env['PYTHONPATH'] = str(root/'ml/src')
subprocess.run([str(python), '-B', str(root/'ml/gpu_session_probe.py'),
               '--output', str(destination), '--gpu', 'L4'], cwd=root, env=env, check=True)
''')
    cell('code', f'''if (destination/'complete.json').exists():
    names = sorted(p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file())
    assert not any('.writer-lock' in n or '.pending' in n for n in names)
    result = base/'cottonlens-session-rng-result.zip'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in names:
            info = zipfile.ZipInfo(name, (2026, 11, 7, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, (destination/name).read_bytes())
        for p in sorted((root/'prior-evidence').glob('*.json')):
            info = zipfile.ZipInfo('prior-evidence/'+p.name, (2026, 11, 7, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
    raw = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        hashes = {{n: hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist()}}
    checksum = result.with_suffix('.sha256.json')
    receipt = json.dumps({{'archive_sha256': hashlib.sha256(raw).hexdigest(),
                          'source_archive_sha256': {digest!r}, 'members': hashes}}, indent=2).encode()
    for target, data in [(result, raw), (checksum, receipt)]:
        if target.exists(): assert target.read_bytes() == data, 'Preserve earlier result'
        else: target.write_bytes(data)
        assert target.read_bytes() == data
    print('Tamamlandı. Sonuç ZIP ve checksum Drive klasöründe. Bağımsız inceleme bekleniyor.')
else:
    print('İlk aşama hazır. Runtime bağlantısını kesip sil; sonra aynı notebooku L4 ile yeniden çalıştır.')
''')
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'accelerator': 'GPU',
        'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}}, 'cells': cells}
    artifacts = {'cottonlens-l4-session-source.zip': source,
                 'CottonLens_L4_Remaining_Checks_Drive.ipynb': (json.dumps(notebook, ensure_ascii=False, indent=2)+'\n').encode()}
    for name, raw in artifacts.items():
        path = output/name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError('Preserve previous handoff; use a new reviewed version')
        if not path.exists():
            path.write_bytes(raw)
    return {'source_sha256': digest, 'members': len(manifest),
            'files': {n: hashlib.sha256(v).hexdigest() for n, v in artifacts.items()},
            'gpu_execution_verified': False, 'prior_checks_retrained': False}


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
