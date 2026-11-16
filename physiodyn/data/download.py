"""Download the MIT-BIH Arrhythmia Database."""

from __future__ import annotations

from pathlib import Path

try:
    import wfdb
except ImportError as exc:  # pragma: no cover - only for environments without wfdb
    wfdb = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def download_mitdb(dl_dir: str | Path = "data/mitdb") -> Path:
    """Download the MIT-BIH Arrhythmia Database using wfdb."""
    if wfdb is None:
        raise ImportError("wfdb is required to download MIT-BIH data") from _IMPORT_ERROR
    target = Path(dl_dir)
    target.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("mitdb", dl_dir=str(target))
    return target
