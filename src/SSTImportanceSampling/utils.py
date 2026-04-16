# utils.py

from pathlib import Path
from typing import List
import contextlib
import os
import pystac

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as devnull:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)

def collect_dss_file_paths(path: str | Path) -> List[str]:
    """
    Collect DSS file paths from either:
      - A directory containing .dss files
      - A STAC catalog JSON (searching its directory on disk)

    For STAC catalogs, this ignores asset hrefs and instead searches
    the catalog's folder recursively for .dss files.

    Returns paths relative to current working directory.
    """

    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    dss_files: List[Path] = []

    # ----------------------------------
    # Case 1: Directory
    # ----------------------------------
    if path.is_dir():
        search_root = path

    # ----------------------------------
    # Case 2: STAC catalog JSON
    # ----------------------------------
    elif path.suffix.lower() == ".json":
        # Load catalog just to validate it is STAC (optional)
        try:
            pystac.Catalog.from_file(str(path))
        except Exception as exc:
            raise ValueError(f"Invalid STAC catalog: {path}") from exc

        # Instead of using hrefs, search the catalog folder
        search_root = path.parent

    else:
        raise ValueError("Must be directory or STAC catalog JSON.")

    # ----------------------------------
    # Search recursively for DSS files
    # ----------------------------------
    dss_files = [f.resolve() for f in search_root.rglob("*.dss")]

    if not dss_files:
        raise ValueError(f"No DSS files found under: {search_root}")

    # ----------------------------------
    # Return relative to current working directory
    # ----------------------------------
    cwd = Path.cwd().resolve()

    relative_paths = []
    for f in dss_files:
        try:
            rel = f.relative_to(cwd)
            relative_paths.append(str(Path(".") / rel))
        except ValueError:
            # If not under cwd, return absolute
            relative_paths.append(str(f))

    return sorted(relative_paths)