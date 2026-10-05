"""Create a checksum-bound Colab source ZIP, including working-tree edits. No Git writes."""

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cottonlens_ml.code_identity import digest, manifest_id, safe_member


def write_bundle(repo: Path, output: Path, paths: list[str], *, head: str, dirty: bool) -> dict:
    files = {name: digest(repo / name) for name in sorted(paths)}
    if not files or any(not safe_member(name) for name in files):
        raise ValueError("Bundle requires safe source paths")
    payload = {"format": 1, "git_head": head, "working_tree_dirty": dirty, "files": files}
    manifest = {**payload, "source_id": manifest_id(payload)}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never replace a previous source delivery, even if the requested name matches.
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(repo / name, name)
        archive.writestr(".source-manifest.json", json.dumps(manifest, indent=2))
    output.with_suffix(output.suffix + ".sha256").write_text(digest(output) + "  " + output.name + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--exclude', action='append', default=[], help='Explicit source-relative files to preserve locally only')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]

    def git(*parts):
        return subprocess.check_output(["git", "-C", str(repo), *parts], text=True).strip()

    paths = []
    for name in git("ls-files", "--cached", "--others", "--exclude-standard").splitlines():
        if name in args.exclude:
            continue
        path = Path(name)
        if ((path.parts[0] in {"ml", "backend", "docs", "frontend", "research", ".github"} or name in {"README.md", "AGENTS.md"})
                and (repo / path).is_file()
                and path.suffix in {".py", ".ps1", ".toml", ".lock", ".txt", ".json", ".md", ".ini", ".ipynb", ".ts", ".html", ".css", ".yml", ".yaml", ".mako"}):
            paths.append(path.as_posix())
    manifest = write_bundle(repo, args.output, paths, head=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")))
    print(json.dumps({"zip": str(args.output.resolve()), "zip_sha256": digest(args.output),
                      "source_id": manifest["source_id"], "files": len(manifest["files"]),
                      "git_head": manifest["git_head"], "working_tree_dirty": manifest["working_tree_dirty"]}, indent=2))


if __name__ == "__main__":
    main()
