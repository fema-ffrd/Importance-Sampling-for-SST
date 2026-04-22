"""Configuration validation utilities."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


class ConfigValidator:
    """Validate configuration files against JSON schema."""

    def __init__(self, schema_path: Path) -> None:
        """
        Initialize validator with schema file.

        Parameters
        ----------
        schema_path : Path
            Path to JSON schema file.

        Raises
        ------
        FileNotFoundError
            If schema file does not exist.
        json.JSONDecodeError
            If schema file is not valid JSON.
        """
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        with open(schema_path) as f:
            self.schema = json.load(f)

    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate config against schema.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary to validate.

        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, error_messages) tuple.
        """
        try:
            import jsonschema
        except ImportError:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                "jsonschema not installed. Install with: pip install jsonschema"
            )
            return True, []

        errors = []
        validator = jsonschema.Draft7Validator(self.schema)

        # Collect all validation errors
        for error in sorted(validator.iter_errors(config), key=str):
            # Build path to error
            path = " → ".join(str(p) for p in error.absolute_path)
            if path:
                error_msg = f"{path}: {error.message}"
            else:
                error_msg = error.message

            errors.append(error_msg)

        is_valid = len(errors) == 0
        return is_valid, errors