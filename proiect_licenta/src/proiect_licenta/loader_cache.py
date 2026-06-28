"""Opt-in disk cache for the heavy ``load_and_clean_data`` loaders.

These loaders re-stream ``discharge.csv`` (3.3 GB) for the PMH parse plus the
vitals/medication aggregation every call — minutes each. When the
``LOADER_CACHE_DIR`` environment variable is set (e.g. to a Google Drive folder),
the decorated loader saves its cleaned output there on first run and loads it on
every later run, so a multi-loader benchmark (and re-runs across Colab sessions)
pays the parse cost once.

Safety
------
* **Opt-in.** With ``LOADER_CACHE_DIR`` unset the decorator is a no-op — training
  and every existing call site behave exactly as before.
* **Auto-invalidation.** The cache key is a fingerprint of the source files'
  (size, mtime) plus a ``version`` integer, so a changed MIMIC file (or a bumped
  ``version`` when the loader logic changes) rebuilds automatically — it can never
  silently serve stale data.

Bump a loader's ``version=`` argument whenever you change what that loader
produces, so old caches are discarded.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path

import joblib

CACHE_ENV = "LOADER_CACHE_DIR"


def cache_dir() -> "Path | None":
    d = os.getenv(CACHE_ENV)
    return Path(d) if d else None


def _fingerprint(source_files, version) -> str:
    h = hashlib.sha256()
    h.update(f"version={version}".encode())
    for p in sorted(str(x) for x in source_files):
        h.update(p.encode())
        try:
            st = os.stat(p)
            h.update(f"|{st.st_size}|{int(st.st_mtime)}".encode())
        except OSError:
            h.update(b"|MISSING")
    return h.hexdigest()


def disk_cached(key: str, source_files, version: int = 1):
    """Decorator: cache a loader's return value under ``LOADER_CACHE_DIR``.

    ``key`` names the cache file; ``source_files`` is the list of input paths
    whose (size, mtime) fingerprint keys the cache (also include ``version``).
    No-op when ``LOADER_CACHE_DIR`` is unset.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            d = cache_dir()
            if d is None:
                return fn(*args, **kwargs)
            d.mkdir(parents=True, exist_ok=True)
            fp = _fingerprint(source_files, version)
            data_path = d / f"{key}.joblib"
            meta_path = d / f"{key}.meta.json"
            if data_path.exists() and meta_path.exists():
                try:
                    if json.loads(meta_path.read_text()).get("fingerprint") == fp:
                        print(f"[loader-cache] HIT  {key}  <- {data_path}")
                        return joblib.load(data_path)
                    print(f"[loader-cache] STALE {key} (source changed) — rebuilding")
                except Exception as e:  # corrupt meta -> rebuild
                    print(f"[loader-cache] meta unreadable ({e}) — rebuilding")
            result = fn(*args, **kwargs)
            try:
                joblib.dump(result, data_path, compress=3)
                meta_path.write_text(json.dumps(
                    {"key": key, "version": version, "fingerprint": fp}))
                print(f"[loader-cache] SAVED {key}  -> {data_path}")
            except Exception as e:  # never let a cache write break the run
                print(f"[loader-cache] save failed ({e}); continuing uncached")
            return result
        return wrapper
    return deco
