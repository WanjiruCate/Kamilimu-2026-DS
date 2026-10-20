"""Point the pipeline at a Kaggle kernel before `pipeline` is imported.

Kaggle mounts datasets read-only under `/kaggle/input` and only
`/kaggle/working` can be written. Importing `pipeline.config` freezes the
cache and artifact directories, so this module has to run first.

    import sys
    from pathlib import Path
    sys.path.insert(0, "<folder that contains this file>")
    from bootstrap import configure
    configure()
    from pipeline.train import run
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _case_dir() -> Path:
    here = Path(__file__).resolve().parent
    if (here / "pipeline" / "config.py").exists():
        return here
    return here


def _find_csv() -> Path | None:
    override = os.environ.get("POWER_SOURCE_CSV")
    if override and Path(override).exists():
        return Path(override)
    named: list[Path] = []
    # parents[1] of case_study_3 is the repository root.
    repo = _case_dir().parents[1] if len(_case_dir().parents) > 1 else _case_dir()
    roots = [
        Path("/kaggle/input"),
        Path("/content/drive/MyDrive"),
        Path("/content"),
        repo,
    ]
    for root in roots:
        if root.exists():
            named.extend(root.rglob("labeled_power_source_data.csv"))
    if named:
        return max(named, key=lambda path: path.stat().st_size)
    kaggle = Path("/kaggle/input")
    if not kaggle.exists():
        return None
    csvs = [path for path in kaggle.rglob("*.csv") if path.is_file()]
    if not csvs:
        return None
    return max(csvs, key=lambda path: path.stat().st_size)


def configure() -> dict:
    """Set environment variables and `sys.path`. Safe to call more than once."""
    case_dir = _case_dir()
    if str(case_dir) not in sys.path:
        sys.path.insert(0, str(case_dir))

    on_kaggle = Path("/kaggle/working").exists()
    if on_kaggle:
        os.environ.setdefault("POWER_SOURCE_WORK", "/kaggle/working/work")
        os.environ.setdefault("ARTIFACT_DIR", "/kaggle/working/artifacts")
    else:
        os.environ.setdefault("POWER_SOURCE_WORK", str(case_dir / "work"))
        os.environ.setdefault("ARTIFACT_DIR", str(case_dir / "artifacts"))

    csv_path = _find_csv()
    if csv_path is not None:
        os.environ["POWER_SOURCE_CSV"] = str(csv_path)

    Path(os.environ["POWER_SOURCE_WORK"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["ARTIFACT_DIR"]).mkdir(parents=True, exist_ok=True)
    return {
        "on_kaggle": on_kaggle,
        "case_dir": str(case_dir),
        "csv": os.environ.get("POWER_SOURCE_CSV"),
        "work": os.environ["POWER_SOURCE_WORK"],
        "artifacts": os.environ["ARTIFACT_DIR"],
    }


if __name__ == "__main__":
    for key, value in configure().items():
        print(f"{key}: {value}")
