"""Сборка архива для платформы: dist/solution.zip с solution.py в корне.

python scripts/package.py
В архив попадают только solution.py и пакет agent/ — без данных, тестов, журналов и секретов.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["solution.py", *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "agent").glob("*.py"))]


def main() -> Path:
    out = ROOT / "dist" / "solution.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in FILES:
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))  # стабильный архив
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, (ROOT / name).read_bytes())
    print(f"{out} ({out.stat().st_size // 1024} KB): {', '.join(FILES)}")
    return out


if __name__ == "__main__":
    main()
