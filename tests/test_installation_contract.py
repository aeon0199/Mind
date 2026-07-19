from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_ci_installs_the_src_layout_package():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()

    assert "pip install -e ." in workflow
    assert "python -m pytest -q" in workflow


def test_public_version_is_consistent():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    citation = (ROOT / "CITATION.cff").read_text()
    version = pyproject["project"]["version"]

    assert version == "0.2.0"
    assert f'version: "{version}"' in citation


def test_setup_docs_install_the_package():
    readme = (ROOT / "README.md").read_text()
    contributing = (ROOT / "CONTRIBUTING.md").read_text()

    assert "python -m pip install -e ." in readme
    assert "python -m pip install -e ." in contributing
