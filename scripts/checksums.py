#!/usr/bin/env python
"""Write or verify the release's canonical SHA-256 manifest.

The manifest covers released result JSONs, provenance, retained indices,
analysis scripts, manuscript source, and public artifact documentation.
Historical archives and generated build files are deliberately excluded.

    python scripts/checksums.py
    python scripts/checksums.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "results" / "CHECKSUMS.sha256"


def canonical_files() -> list[Path]:
    results = ROOT / "data" / "results"
    files: list[Path] = []

    files += sorted(results.glob("*_chip.json"))
    for extra in ("paired_significance.json",):
        if (results / extra).exists():
            files.append(results / extra)
    files += sorted(results.glob("ftw_proxy_sensitivity_*.json"))
    files += sorted(results.glob("xseason_*.json"))
    files += sorted(results.glob("cross_region_*.json"))
    files += sorted(results.glob("ftw_*.json"))
    files += sorted(results.glob("ftw_*_summary.txt"))

    environment = results / "environment.json"
    if environment.exists():
        files.append(environment)

    files += sorted((ROOT / "data" / "chips").glob("manifest_*.jsonl"))
    files += sorted((ROOT / "data" / "index").glob("*.jsonl"))
    files += sorted((ROOT / "docs").glob("*.md"))
    files += sorted((ROOT / "scripts").glob("*.py"))

    for path in (
        ROOT / "README.md",
        ROOT / "ARTIFACT.md",
        ROOT / "croissant.json",
        ROOT / "pyproject.toml",
        ROOT / "requirements.lock",
        ROOT / "requirements.runtime.txt",
        ROOT / ".gitattributes",
        ROOT / ".gitignore",
        ROOT / "LICENSE",
        ROOT / "scripts" / "reproduce_all.sh",
        ROOT / "paper" / "main_tmlr.tex",
        ROOT / "paper" / "references.bib",
        ROOT / "paper" / "tmlr.sty",
    ):
        if path.exists():
            files.append(path)

    files = list(dict.fromkeys(files))
    return [
        path
        for path in files
        if "archive" not in path.parts and "archive_preaudit" not in path.parts
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Write to a custom output path. This records a reproduced manifest "
            "without overwriting the shipped CHECKSUMS.sha256."
        ),
    )
    args = parser.parse_args()

    current = {relative(path): sha256(path) for path in canonical_files()}

    if not args.verify:
        target = Path(args.out) if args.out else OUT
        target.write_text(
            "".join(f"{digest}  {name}\n" for name, digest in sorted(current.items())),
            encoding="utf-8",
            newline="\n",
        )
        label = relative(target) if target.resolve() == OUT.resolve() else target.name
        print(f"wrote {label} ({len(current)} files)")
        return 0

    if not OUT.exists():
        print(f"[ERROR] {relative(OUT)} missing; run without --verify first")
        return 1

    recorded: dict[str, str] = {}
    for line in OUT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            recorded[name.strip()] = digest

    drift: list[str] = []
    for name, digest in current.items():
        if name not in recorded:
            drift.append(f"NEW (uncommitted): {name}")
        elif recorded[name] != digest:
            drift.append(f"CHANGED: {name}")
    for name in recorded:
        if name not in current:
            drift.append(f"MISSING: {name}")

    if drift:
        print(f"[FAIL] {len(drift)} checksum drift(s):")
        for item in drift:
            print(f"  {item}")
        return 1

    print(f"[OK] {len(current)} canonical artifacts match CHECKSUMS.sha256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
