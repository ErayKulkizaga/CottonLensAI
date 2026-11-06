"""Build a code-only, hash-bound Colab handoff. Never includes data or secrets."""
import argparse
import ast
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def build(namespace='a100-handoff-20261106', gpu='A100', probe='preflight'):
    if Path(namespace).name != namespace or namespace in ('.', '..') or gpu not in ('A100', 'L4', 'G4'):
        raise ValueError('Invalid handoff namespace or device')
    if probe not in ('preflight', 'restart', 'oom'):
        raise ValueError('Explicit supported probe required')
    output = ROOT/'output'/namespace
    output.mkdir(parents=True, exist_ok=True)
    paths = [ROOT/'ml'/n for n in ('colab_setup.py', 'a100_preflight.py', 'pyproject.toml', 'uv.lock')]
    if probe in ('restart', 'oom'):
        paths.append(ROOT/'ml/gpu_restart_probe.py')
    if probe == 'oom':
        paths.append(ROOT/'ml/gpu_oom_probe.py')
    paths += sorted((ROOT/'ml/src').rglob('*.py'))
    members = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in paths}
    manifest = {name: sha(data) for name, data in members.items()}
    archive = output/'cottonlens-a100-source.zip'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, (2026, 11, 6, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    payload = buffer.getvalue()
    if archive.exists() and archive.read_bytes() != payload:
        raise ValueError('Delivered source package changed; preserve it and choose a new handoff namespace')
    if not archive.exists():
        archive.write_bytes(payload)
    archive_hash = sha(payload)
    cells = []

    def cell(kind, text):
        if kind == 'code':
            ast.parse(text)
        body = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(keepends=True)}
        if kind == 'code':
            body.update(execution_count=None, outputs=[])
        cells.append(body)

    cell('markdown', f'# CottonLensAI — {gpu} {probe} teknik doğrulama\n'
         f'Çalışma zamanı türünü **{gpu} GPU** seç. Hücreleri sırayla çalıştır. '
         'Yalnız üretilmiş veri; piyasa eğitimi yok. Kaynak ZIP aynı teslimattan olmalı. '
         'Drive izni kurulum içindir; eski deneyleri çalıştırma. '
         'Son hücre sonuç ZIP ve checksum dosyasını indirir; ikisini bu sohbete ekle.\n')
    cell('code', '''from google.colab import drive, files
drive.mount('/content/drive')
uploaded = files.upload()  # Yalnız cottonlens-a100-source.zip dosyasını seç.
''')
    cell('code', f'''import hashlib, zipfile, io, sys
from pathlib import Path
expected_archive = {archive_hash!r}
expected_members = {manifest!r}
name = 'cottonlens-a100-source.zip'
assert set(uploaded) == {{name}}, 'Yalnız aynı teslimattaki kaynak ZIP seçilmeli'
assert hashlib.sha256(uploaded[name]).hexdigest() == expected_archive, 'Kaynak ZIP hash uyuşmazlığı'
root = Path('/content/CottonLensAI')
with zipfile.ZipFile(io.BytesIO(uploaded[name])) as z:
    assert len(z.namelist()) == len(set(z.namelist())), 'Tekrarlı ZIP üyesi'
    assert set(z.namelist()) == set(expected_members), 'Beklenmeyen/eksik kaynak'
    payload = {{n: z.read(n) for n in z.namelist()}}
    for n, data in payload.items():
        assert hashlib.sha256(data).hexdigest() == expected_members[n], n
        target = (root/n).resolve()
        assert target.is_relative_to(root.resolve()), 'Güvensiz dosya yolu'
        if target.exists():
            assert target.read_bytes() == data, 'Eski çalışma alanı farklı; yeni runtime kullan'
    for n, data in payload.items():
        target = root/n
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists(): target.write_bytes(data)
sys.path.insert(0, str(root/'ml'))
import colab_setup
python = colab_setup.setup()
''')
    entrypoint = {'restart': 'gpu_restart_probe.py', 'oom': 'gpu_oom_probe.py',
                  'preflight': 'a100_preflight.py'}[probe]
    cell('code', f'''import subprocess
env = colab_setup.process_env()
env['PYTHONPATH'] = str(root/'ml/src')
probe_output = Path('/content/cottonlens-{probe}-{gpu.lower()}-20261106')
subprocess.run([str(python), '-B', str(root/'ml/{entrypoint}'),
                '--output', str(probe_output), '--gpu', {gpu!r}], cwd=root, env=env, check=True)
''')
    export_names = ("assert (probe_output/'complete.json').is_file(), 'Restart probe tamamlanmadi'\n"
                    "names = sorted(p.relative_to(probe_output).as_posix() for p in probe_output.rglob('*') if p.is_file())"
                    if probe in ('restart', 'oom') else "names = ['complete.json', 'step1.keras', 'step2.keras']")
    cell('code', '''import json, shutil
from datetime import datetime, timezone
result = Path('/content/cottonlens-a100-result.zip')
''' + export_names + '''
assert all((probe_output/n).is_file() for n in names), 'Teknik probe tamamlanmadı'
with zipfile.ZipFile(result, 'w', zipfile.ZIP_DEFLATED) as z:
    for n in names: z.write(probe_output/n, n)
checksum = result.with_suffix('.sha256.json')
checksum.write_text(json.dumps({'archive_sha256': hashlib.sha256(result.read_bytes()).hexdigest(),
    'source_archive_sha256': expected_archive,
    'members': {n: hashlib.sha256((probe_output/n).read_bytes()).hexdigest() for n in names}}, indent=2))
backup = Path('/content/drive/MyDrive/CottonLensAI/a100-handoff-results')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
backup.mkdir(parents=True, exist_ok=False)
for p in [result, checksum]:
    shutil.copyfile(p, backup/p.name)
    assert (backup/p.name).read_bytes() == p.read_bytes()
print('Sonuçlar Drive kopyasında mevcut; bu uzak GitHub doğrulaması değildir.')
files.download(str(result))
files.download(str(checksum))
''')
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'accelerator': 'GPU', 'colab': {'name': 'CottonLens_A100_Preflight.ipynb'},
                             'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}}, 'cells': cells}
    target = output/f'CottonLens_GPU_{probe}_{gpu}.ipynb' if probe in ('restart', 'oom') else output/f'CottonLens_GPU_Preflight_{gpu}.ipynb'
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output/'source-manifest.json').write_text(json.dumps({'archive_sha256': archive_hash, 'files': manifest}, indent=2)+'\n')
    print(json.dumps({'notebook': str(target), 'source_zip': str(archive), 'source_sha256': archive_hash,
                      'source_files': len(members), 'archive_bytes': archive.stat().st_size}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace', default='a100-handoff-20261106')
    parser.add_argument('--gpu', choices=('A100', 'L4', 'G4'), default='A100')
    parser.add_argument('--probe', choices=('preflight', 'restart', 'oom'), default='preflight')
    args = parser.parse_args()
    build(args.namespace, args.gpu, args.probe)
