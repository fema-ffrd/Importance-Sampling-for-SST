"""Utility functions for SST Importance Sampling."""

import contextlib
import logging
import os
from pathlib import Path
from typing import List

import pandas as pd
import pystac

# =============================================================================
# LOGGING
# =============================================================================


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure and return a logger instance.

    Parameters
    ----------
    name : str
        Logger name (typically __name__).
    level : int, optional
        Logging level, by default logging.INFO.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Examples
    --------
    >>> logger = setup_logger(__name__)
    >>> logger.info("Processing started")
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)

    return logger


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================


def validate_config(config: dict) -> None:
    """
    Validate required configuration keys.

    Parameters
    ----------
    config : dict
        Configuration dictionary to validate.

    Raises
    ------
    KeyError
        If required keys are missing.
    TypeError
        If required sections are not dictionaries.

    Examples
    --------
    >>> validate_config({"project": {}, "output": {}, "paths": {}})
    """
    required_sections = {"project", "output", "paths", "preprocess"}
    missing = required_sections - set(config.keys())

    if missing:
        raise KeyError(f"Missing required configuration sections: {missing}")

    for section in required_sections:
        if not isinstance(config[section], dict):
            raise TypeError(f"Configuration section '{section}' must be a dict")


# =============================================================================
# PATH UTILITIES
# =============================================================================


def resolve_path(path: str | Path, base_dir: Path | None = None) -> Path:
    """
    Resolve relative or absolute path.

    Relative paths are resolved relative to base_dir if provided,
    otherwise relative to current working directory.

    Parameters
    ----------
    path : str | Path
        Path to resolve.
    base_dir : Path | None, optional
        Base directory for relative paths. Defaults to None (uses cwd).

    Returns
    -------
    Path
        Resolved absolute path.

    Examples
    --------
    >>> resolve_path("data/input.csv")
    PosixPath('/home/user/project/data/input.csv')

    >>> resolve_path("/absolute/path.csv")
    PosixPath('/absolute/path.csv')
    """
    path = Path(path)

    if path.is_absolute():
        return path

    if base_dir is None:
        base_dir = Path.cwd()

    return (base_dir / path).resolve()


# =============================================================================
# FILE FORMAT UTILITIES
# =============================================================================


def get_table_extension(export_format: str) -> str:
    """
    Get file extension for tabular data format.

    Parameters
    ----------
    export_format : str
        Export format ('csv' or 'parquet').

    Returns
    -------
    str
        File extension ('csv' or 'parquet').

    Raises
    ------
    ValueError
        If format is not supported.

    Examples
    --------
    >>> get_table_extension("parquet")
    'parquet'

    >>> get_table_extension("csv")
    'csv'
    """
    fmt = export_format.lower()
    if fmt == "parquet":
        return "parquet"
    elif fmt == "csv":
        return "csv"
    else:
        raise ValueError(f"Unsupported format: {fmt}. Use 'csv' or 'parquet'")


# =============================================================================
# DATAFRAME I/O
# =============================================================================


def save_dataframe(
    df: pd.DataFrame,
    output_path: str | Path,
    export_format: str,
) -> None:
    """
    Save DataFrame in specified format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    output_path : str | Path
        Output file path.
    export_format : str
        Export format ('csv' or 'parquet').

    Raises
    ------
    ValueError
        If format is not supported.
    IOError
        If write operation fails.

    Examples
    --------
    >>> df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    >>> save_dataframe(df, "data.parquet", "parquet")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = export_format.lower()

    try:
        if fmt == "parquet":
            df.to_parquet(output_path, index=False, compression="snappy")
        elif fmt == "csv":
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {fmt}. Use 'csv' or 'parquet'")
    except Exception as e:
        raise IOError(f"Failed to save DataFrame to {output_path}: {e}") from e


def read_dataframe(file_path: str | Path) -> pd.DataFrame:
    """
    Read DataFrame from CSV or Parquet file.

    Automatically detects format from file extension.

    Parameters
    ----------
    file_path : str | Path
        Path to data file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If format is not supported.
    IOError
        If read operation fails.

    Examples
    --------
    >>> df = read_dataframe("data.parquet")
    >>> df = read_dataframe("data.csv")
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        if file_path.suffix.lower() == ".parquet":
            return pd.read_parquet(file_path)
        elif file_path.suffix.lower() == ".csv":
            return pd.read_csv(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. "
                f"Expected .csv or .parquet"
            )
    except Exception as e:
        raise IOError(f"Failed to read file {file_path}: {e}") from e


# =============================================================================
# PROCESS SUPPRESSION
# =============================================================================


@contextlib.contextmanager
def suppress_stdout_stderr():
    """
    Context manager to suppress stdout and stderr output.

    Useful for silencing verbose third-party libraries.

    Examples
    --------
    >>> with suppress_stdout_stderr():
    ...     verbose_function()  # Output is suppressed
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)

        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)


# =============================================================================
# DSS FILE COLLECTION
# =============================================================================


def collect_dss_file_paths(path: str | Path) -> List[str]:
    """
    Collect DSS file paths from directory or STAC catalog.

    Supports two input types:
    - Directory: Recursively searches for .dss files
    - STAC Catalog JSON: Searches the catalog's directory for .dss files

    Parameters
    ----------
    path : str | Path
        Path to directory or STAC catalog JSON file.

    Returns
    -------
    List[str]
        Sorted list of DSS file paths (relative to current working directory).

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    ValueError
        If path type is not supported or no DSS files found.

    Examples
    --------
    >>> files = collect_dss_file_paths("./dss_data")
    >>> len(files)
    42

    >>> files = collect_dss_file_paths("./catalog.json")
    >>> files[0]
    './dss_data/storm_001.dss'
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    # Determine search root
    if path.is_dir():
        search_root = path
    elif path.suffix.lower() == ".json":
        # Validate STAC catalog
        try:
            pystac.Catalog.from_file(str(path))
        except Exception as exc:
            raise ValueError(f"Invalid STAC catalog: {path}") from exc

        search_root = path.parent
    else:
        raise ValueError(
            f"Input must be a directory or STAC catalog JSON, got: {path}"
        )

    # Find all DSS files
    dss_files = sorted([f.resolve() for f in search_root.rglob("*.dss")])

    if not dss_files:
        raise ValueError(f"No DSS files found in: {search_root}")

    # Convert to relative paths
    cwd = Path.cwd().resolve()
    relative_paths = []

    for dss_file in dss_files:
        try:
            rel = dss_file.relative_to(cwd)
            relative_paths.append(str(Path(".") / rel))
        except ValueError:
            # File not under cwd, use absolute path
            relative_paths.append(str(dss_file))

    return sorted(relative_paths)


# =============================================================================
# TYPE CHECKING & VALIDATION
# =============================================================================


def validate_file_exists(path: str | Path, name: str = "File") -> Path:
    """
    Validate that a file exists.

    Parameters
    ----------
    path : str | Path
        Path to validate.
    name : str, optional
        Name for error message, by default "File".

    Returns
    -------
    Path
        Resolved path if valid.

    Raises
    ------
    FileNotFoundError
        If file does not exist.

    Examples
    --------
    >>> path = validate_file_exists("data.csv", "Input file")
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"{name} is not a file: {path}")

    return path


def validate_directory_exists(path: str | Path, name: str = "Directory") -> Path:
    """
    Validate that a directory exists.

    Parameters
    ----------
    path : str | Path
        Path to validate.
    name : str, optional
        Name for error message, by default "Directory".

    Returns
    -------
    Path
        Resolved path if valid.

    Raises
    ------
    FileNotFoundError
        If directory does not exist.

    Examples
    --------
    >>> path = validate_directory_exists("./data", "Input directory")
    """
    path = Path(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")

    if not path.is_dir():
        raise FileNotFoundError(f"{name} is not a directory: {path}")

    return path


# =============================================================================
# NUMERIC VALIDATION
# =============================================================================


def validate_positive(value: float, name: str = "Value") -> float:
    """
    Validate that a value is positive.

    Parameters
    ----------
    value : float
        Value to validate.
    name : str, optional
        Name for error message, by default "Value".

    Returns
    -------
    float
        The value if valid.

    Raises
    ------
    ValueError
        If value is not positive.

    Examples
    --------
    >>> spacing = validate_positive(1000, "Grid spacing")
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got: {value}")
    return value


def validate_range(
    value: float, min_val: float, max_val: float, name: str = "Value"
) -> float:
    """
    Validate that a value is within a range.

    Parameters
    ----------
    value : float
        Value to validate.
    min_val : float
        Minimum allowed value (inclusive).
    max_val : float
        Maximum allowed value (inclusive).
    name : str, optional
        Name for error message, by default "Value".

    Returns
    -------
    float
        The value if valid.

    Raises
    ------
    ValueError
        If value is outside range.

    Examples
    --------
    >>> workers = validate_range(4, 1, 32, "Max workers")
    """
    if not (min_val <= value <= max_val):
        raise ValueError(
            f"{name} must be between {min_val} and {max_val}, got: {value}"
        )
    return value