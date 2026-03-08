"""Log maintenance — TTL cleanup, migration, and subdirectory management.

Provides:
  - ``cleanup_old_logs()`` — delete log files older than ``LOG_TTL_DAYS``
  - ``ensure_log_subdirectories()`` — create the categorised log layout
  - ``migrate_flat_logs()`` — move legacy flat files into the new structure
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger("march.log_maintenance")

LOG_TTL_DAYS: int = 30

# Canonical subdirectories under ~/.march/logs/
LOG_SUBDIRS = ("agent", "guardian", "turns", "metrics", "dashboard")


def ensure_log_subdirectories(log_dir: Path | None = None) -> Path:
    """Create the categorised log directory tree.

    Returns the resolved *log_dir* for convenience.
    """
    log_dir = (log_dir or Path.home() / ".march" / "logs").expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    for name in LOG_SUBDIRS:
        (log_dir / name).mkdir(exist_ok=True)
    return log_dir


def cleanup_old_logs(log_dir: Path | None = None, ttl_days: int = LOG_TTL_DAYS) -> int:
    """Delete log files older than *ttl_days*.

    Scans every immediate subdirectory of *log_dir* and removes regular files
    whose **mtime** is older than the cutoff.  Returns the number of files
    deleted.
    """
    log_dir = (log_dir or Path.home() / ".march" / "logs").expanduser()
    if not log_dir.is_dir():
        return 0

    cutoff_ts = (datetime.now() - timedelta(days=ttl_days)).timestamp()
    deleted = 0

    for subdir in log_dir.iterdir():
        if not subdir.is_dir():
            continue
        for f in subdir.iterdir():
            if f.is_file():
                try:
                    if f.stat().st_mtime < cutoff_ts:
                        f.unlink()
                        logger.info("Deleted old log: %s", f)
                        deleted += 1
                except OSError as exc:
                    logger.warning("Failed to delete %s: %s", f, exc)

    return deleted


# ── Legacy flat-file migration ────────────────────────────────────────────────

_MIGRATION_MAP = {
    # old filename → (new subdir, new extension)
    "march.log": ("agent", ".log"),
    "guardian.log": ("guardian", ".log"),
    "turns.jsonl": ("turns", ".jsonl"),
    "metrics.jsonl": ("metrics", ".jsonl"),
    "dashboard.log": ("dashboard", ".log"),
}


def migrate_flat_logs(log_dir: Path | None = None) -> int:
    """Move legacy flat log files into the new subdirectory layout.

    Old files are renamed into their respective subdirectory with a
    ``migrated-YYYY-MM-DD`` prefix so they don't collide with fresh
    date-based files.  Returns the number of files migrated.
    """
    log_dir = (log_dir or Path.home() / ".march" / "logs").expanduser()
    if not log_dir.is_dir():
        return 0

    today = datetime.now().strftime("%Y-%m-%d")
    migrated = 0

    for old_name, (subdir, ext) in _MIGRATION_MAP.items():
        old_path = log_dir / old_name
        if not old_path.is_file():
            continue

        dest_dir = log_dir / subdir
        dest_dir.mkdir(exist_ok=True)
        dest_path = dest_dir / f"migrated-{today}{ext}"

        # If destination already exists, append a counter
        counter = 1
        while dest_path.exists():
            dest_path = dest_dir / f"migrated-{today}-{counter}{ext}"
            counter += 1

        try:
            shutil.move(str(old_path), str(dest_path))
            logger.info("Migrated %s → %s", old_path, dest_path)
            migrated += 1
        except OSError as exc:
            logger.warning("Failed to migrate %s: %s", old_path, exc)

    # Also migrate any rotated flat files (e.g. turns.jsonl.1, march.log.2026-03-05)
    for old_name in ("turns.jsonl", "march.log"):
        base_name = old_name.split(".")[0]
        subdir = _MIGRATION_MAP[old_name][0]
        dest_dir = log_dir / subdir
        for f in log_dir.glob(f"{old_name}.*"):
            if f.is_file():
                dest = dest_dir / f"migrated-{f.name}"
                if not dest.exists():
                    try:
                        shutil.move(str(f), str(dest))
                        logger.info("Migrated rotated %s → %s", f, dest)
                        migrated += 1
                    except OSError:
                        pass

    return migrated
