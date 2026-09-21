"""
Build colab/gnn_aml_code.zip, the code bundle the Colab notebook unpacks.

Usage:
    python scripts/make_colab_bundle.py

Only the directories in INCLUDE_DIRS are bundled (an allowlist). An earlier
hand-rolled version used a skip-list that matched any path component named
"data", which silently dropped src/data/ and made Colab fail with
"No module named 'src.data'". tests/test_colab_bundle.py guards that property.
"""

import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
INCLUDE_DIRS = ("src", "scripts", "configs", "tests")
INCLUDE_FILES = ("requirements.txt",)
DEFAULT_OUT = REPO_ROOT / "colab" / "gnn_aml_code.zip"


def build_bundle(out_path: Path = DEFAULT_OUT, root: Path = REPO_ROOT) -> list[str]:
    """Write the code bundle and return the archive names it contains.

    Args:
        out_path: Where to write the zip.
        root: Repository root to bundle from.

    Returns:
        Sorted list of POSIX-style paths stored in the archive.
    """
    files: list[Path] = [root / f for f in INCLUDE_FILES if (root / f).is_file()]
    for d in INCLUDE_DIRS:
        files += [
            p
            for p in sorted((root / d).rglob("*"))
            if p.is_file() and p.suffix != ".pyc" and "__pycache__" not in p.parts
        ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = []
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            arcname = p.relative_to(root).as_posix()
            z.write(p, arcname)
            names.append(arcname)
    return sorted(names)


if __name__ == "__main__":
    names = build_bundle()
    print(f"{len(names)} files -> {DEFAULT_OUT.relative_to(REPO_ROOT)} "
          f"({DEFAULT_OUT.stat().st_size / 1024:.0f} KB)")
