"""
Guard for the Colab code bundle.

The first hand-built bundle silently omitted src/data/ (a skip-list entry for
the top-level data/ directory matched the nested one), and the failure only
surfaced on Colab as "No module named 'src.data'".
"""

import zipfile

from scripts.make_colab_bundle import INCLUDE_DIRS, REPO_ROOT, build_bundle


def test_bundle_contains_every_source_file(tmp_path):
    out = tmp_path / "bundle.zip"
    build_bundle(out)
    in_zip = set(zipfile.ZipFile(out).namelist())

    on_disk = {
        p.relative_to(REPO_ROOT).as_posix()
        for d in INCLUDE_DIRS
        for p in (REPO_ROOT / d).rglob("*.py")
        if "__pycache__" not in p.parts
    }
    assert on_disk, "found no source files to check"
    assert on_disk <= in_zip, f"missing from bundle: {sorted(on_disk - in_zip)}"


def test_bundle_includes_src_data_package(tmp_path):
    """The specific regression."""
    out = tmp_path / "bundle.zip"
    build_bundle(out)
    names = set(zipfile.ZipFile(out).namelist())
    assert "src/data/__init__.py" in names
    assert "src/data/graph_builder.py" in names


def test_bundle_excludes_data_and_outputs(tmp_path):
    out = tmp_path / "bundle.zip"
    build_bundle(out)
    for name in zipfile.ZipFile(out).namelist():
        assert not name.startswith(("data/", "outputs/", ".venv/")), name
