"""Сборка архива для платформы: dist/solution.zip — все файлы в корне, без папок.

python scripts/package.py
По инструкции организаторов в ZIP только solution.py, requirements.txt и .py-модули в корне:
вложенные папки не копируются (модуль из подпапки недоступен -> 0 баллов). Данные, тесты, журналы и
секреты в архив не попадают.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["solution.py", "requirements.txt", *sorted(p.name for p in ROOT.glob("agent_*.py"))]


def main() -> Path:
    out = ROOT / "dist" / "solution.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in FILES:
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))  # стабильный архив
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, (ROOT / name).read_bytes())
    names = zipfile.ZipFile(out).namelist()
    assert all("/" not in n and "\\" not in n for n in names), names
    print(f"{out} ({out.stat().st_size // 1024} KB): {', '.join(names)}")
    return out


if __name__ == "__main__":
    main()
