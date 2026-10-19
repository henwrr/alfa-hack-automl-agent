from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGER = REPOSITORY / "scripts" / "package_solution.py"


def _make_fake_project(root: Path) -> set[str]:
    files = {
        "solution.py": "print('fixture')\n",
        "automl_agent/__init__.py": "\"\"\"Fixture package.\"\"\"\n",
        "automl_agent/contracts.py": "TASK_CONTRACT = True\n",
        "automl_agent/nested/worker.py": "def main(): pass\n",
        "README.md": "Fixture documentation\n",
        ".env.example": "OPENAI_MODEL=example\n",
        "requirements-mvp.txt": "numpy==1.26.4\n",
        "tests/test_hidden.py": "assert False\n",
        "docs/research/private-notes.md": "research log\n",
        "data/train.csv": "secret,dataset\n",
        ".env": "OPENAI_API_KEY=not-for-archive\n",
        ".venv/pyvenv.cfg": "home = /private\n",
    }
    for relative, contents in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    return {
        "solution.py",
        "automl_agent/__init__.py",
        "automl_agent/contracts.py",
        "automl_agent/nested/worker.py",
        "README.md",
        ".env.example",
        "requirements-mvp.txt",
    }


def test_packager_includes_only_runtime_allowlist_and_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source"
    expected = _make_fake_project(source)
    first = tmp_path / "one.zip"
    second = tmp_path / "two.zip"

    for output in (first, second):
        completed = subprocess.run(
            [sys.executable, str(PACKAGER), "--root", str(source), "--output", str(output)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert output.is_file()

    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        names = set(archive.namelist())
        assert names == expected
        assert "solution.py" in names
        assert not any(
            name == "train.csv"
            or name == ".env"
            or name.startswith((".venv/", "tests/", "research/", "docs/research/", "data/"))
            or "private" in name
            for name in names
        )
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
        assert archive.read("solution.py") == b"print('fixture')\n"


def test_packager_fails_clearly_when_solution_entrypoint_is_missing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_fake_project(source)
    (source / "solution.py").unlink()

    completed = subprocess.run(
        [sys.executable, str(PACKAGER), "--root", str(source), "--output", str(tmp_path / "out.zip")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "solution.py" in (completed.stdout + completed.stderr)
    assert not (tmp_path / "out.zip").exists()
