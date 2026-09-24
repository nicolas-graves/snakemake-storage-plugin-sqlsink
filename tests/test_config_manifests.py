from pathlib import Path

import yaml

from sqlsink.manifest import load_manifest

CONFIG = Path(__file__).parent.parent / "config" / "config.yaml"


def test_configured_datasets_form_valid_manifests():
    specs = yaml.safe_load(CONFIG.read_text())["datasets"]
    manifests = [load_manifest(spec) for spec in specs]
    assert len({m.name for m in manifests}) == len(manifests) == 4
    for manifest in manifests:
        assert manifest.geometry_column not in manifest.compact_columns()
        assert manifest.fact_join_columns == ("Code ZE",)
