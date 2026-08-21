from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from latency_meta_mdp.resources import ResourceRecord, load_resource_manifest


def valid_record() -> ResourceRecord:
    return ResourceRecord(
        resource_id="procedural_calibration_bottle_v1",
        source_repository="project-owned",
        source_revision="project-owned-v1",
        source_path="generated/procedural_bottle",
        local_path="assets/project/procedural_bottle.xml",
        upstream_origin="Latency-Meta-MDP-LIBERO",
        sha256="0" * 64,
        license_id="project-owned",
        license_url="docs/design/2026-08-21-robosuite-libero-infra-design.md",
        license_text_path="assets/THIRD_PARTY_NOTICES.md",
        attribution="Project-owned calibration geometry",
        modifications="None",
    )


class ResourceRecordTest(unittest.TestCase):
    def test_complete_record_is_valid(self) -> None:
        valid_record().validate()

    def test_malformed_checksum_is_rejected(self) -> None:
        record = valid_record()
        with self.assertRaisesRegex(ValueError, "sha256"):
            ResourceRecord(**{**record.to_mapping(), "sha256": "abc"})

    def test_absolute_local_path_is_rejected(self) -> None:
        record = valid_record()
        with self.assertRaisesRegex(ValueError, "relative path"):
            ResourceRecord(**{**record.to_mapping(), "local_path": "/tmp/bottle.xml"})

    def test_missing_license_is_rejected(self) -> None:
        record = valid_record()
        with self.assertRaisesRegex(ValueError, "license_id"):
            ResourceRecord(**{**record.to_mapping(), "license_id": ""})

    def test_real_manifest_verifies_file_license_and_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "assets/project/bottle.xml"
            license_text = root / "assets/THIRD_PARTY_NOTICES.md"
            asset.parent.mkdir(parents=True)
            asset.write_text("<mujoco/>")
            license_text.write_text("Project-owned\n")
            checksum = hashlib.sha256(asset.read_bytes()).hexdigest()
            record = {
                **valid_record().to_mapping(),
                "local_path": "assets/project/bottle.xml",
                "sha256": checksum,
            }
            manifest = root / "assets/resource_manifest.json"
            manifest.write_text(json.dumps({"schema_version": 1, "resources": [record]}))

            loaded = load_resource_manifest(manifest, repository_root=root)
            self.assertEqual(loaded, (ResourceRecord(**record),))

            asset.write_text("changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_resource_manifest(manifest, repository_root=root)

    def test_third_party_revision_must_be_full_commit(self) -> None:
        record = valid_record().to_mapping()
        record.update(source_repository="https://example.com/repo", source_revision="main")
        with self.assertRaisesRegex(ValueError, "40-character"):
            ResourceRecord(**record)


if __name__ == "__main__":
    unittest.main()
