"""Isolated offline presentation: never opens the default database or runtime."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / 'backend'


def demo_paths(name):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,59}', name):
        raise ValueError('Use a simple lowercase presentation name')
    root = (REPO / 'output/presentation' / name).resolve()
    parent = (REPO / 'output/presentation').resolve()
    if not root.is_relative_to(parent):
        raise ValueError('Presentation must remain in output/presentation')
    return root, root / 'presentation.db', root / 'artifact'


def configure(name):
    root, database, artifact = demo_paths(name)
    os.environ.update(DATABASE_URL='sqlite:///' + database.as_posix(), ARTIFACT_DIR=str(artifact), DEMO_MODE='false')
    sys.path.insert(0, str(BACKEND))
    return root, database, artifact


def install(name, bundle):
    root, database, artifact = configure(name)
    from app.artifacts import install_bundle, sha256_file, verify_directory

    config_path = root / 'presentation.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if config['bundle_sha256'] != sha256_file(bundle):
            raise ValueError('Use a new presentation name for a different release')
        verify_directory(artifact)
        if not database.is_file():
            raise ValueError('Presentation database missing; use a new name')
        print(json.dumps(config, indent=2))
        return
    if root.exists():
        raise ValueError('Incomplete/existing presentation directory: preserve it and use a new name')
    root.mkdir(parents=True)
    manifest = install_bundle(bundle, artifact)
    subprocess.run([sys.executable, '-m', 'alembic', 'upgrade', 'head'], cwd=BACKEND, check=True)
    from app.artifact_importer import import_artifact_directory
    from app.database import SessionLocal, engine
    with SessionLocal() as db:
        version = import_artifact_directory(db, artifact)
        db.commit()
    engine.dispose()
    if version != manifest['artifact_version']:
        raise ValueError('Imported release identity mismatch')
    config = {'artifact_version': version, 'bundle_sha256': sha256_file(bundle),
              'database': str(database), 'artifact': str(artifact), 'demo_mode': False}
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps(config, indent=2))


def serve(name, port):
    root, database, artifact = configure(name)
    if not (root / 'presentation.json').is_file() or not database.is_file():
        raise ValueError('Install the isolated presentation first')
    static = REPO / 'frontend/dist/cottonlens/browser'
    if not (static / 'index.html').is_file():
        raise ValueError('Build frontend first: npm run build')
    import uvicorn
    from fastapi.responses import FileResponse, JSONResponse

    from app.main import app

    @app.get('/{path:path}', include_in_schema=False)
    def frontend(path: str):
        if path.startswith('api/'):
            return JSONResponse({'detail': 'Not found'}, status_code=404)
        requested = (static / path).resolve()
        if not requested.is_relative_to(static.resolve()):
            return JSONResponse({'detail': 'Not found'}, status_code=404)
        return FileResponse(requested if requested.is_file() else static / 'index.html')

    uvicorn.run(app, host='127.0.0.1', port=port)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('install', 'serve'))
    parser.add_argument('--name', default='r2-demo')
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    if args.action == 'install':
        if not args.bundle:
            parser.error('install requires --bundle')
        install(args.name, args.bundle.resolve())
    else:
        serve(args.name, args.port)


if __name__ == '__main__':
    main()
