import pytest
from pathlib import Path
import tempfile
import shutil

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp = tempfile.mkdtemp()
    yield Path(temp)
    shutil.rmtree(temp)

@pytest.fixture
def sample_config():
    """Load sample config for testing."""
    return {
        "project": {"name": "test", "run_type": "test"},
        "paths": {
            "watershed_geojson": "test.geojson",
            "domain_geojson": "test.geojson",
            "dss_catalog": "test",
        },
        "output": {"base_folder": "./test_output", "export_format": "csv"},
        "preprocess": {
            "mode": "force",
            "grid_spacing": 1000,
            "dss_variable_keyword": "PRECIPITATION",
            "dss_grid_keyword": "SHG",
            "random_seed": 42,
            "validity": {
                "enabled": True,
                "export_format": "csv",
                "max_workers": 2,
                "fishnet_chunk_size": 1000,
            },
        },
    }