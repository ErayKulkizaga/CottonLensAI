import argparse
from pathlib import Path

from app.artifacts import install_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and install a CottonLens Colab artifact bundle")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--destination", type=Path, default=Path("../runtime/artifacts/current"))
    args = parser.parse_args()
    manifest = install_bundle(args.bundle.resolve(), args.destination.resolve())
    print(f"Installed {manifest['artifact_version']} at {args.destination.resolve()}")


if __name__ == "__main__":
    main()

