import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_global_splits import assign_network_splits, select_cohort
from build_global_tensors import pack_tensors, read_feature_manifest
from global_data_contract import GlobalDataContract


class GlobalPipelineTests(unittest.TestCase):
    def test_catalog_input_explains_required_next_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "global_station_catalog.csv"
            path.write_text(
                "source_file,station,depth_class,good_quality_rows\n"
                "network/station.stm,station,near_surface,10\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "ISMN station catalog"):
                read_feature_manifest(path)

    def test_network_splits_are_disjoint_and_deterministic(self) -> None:
        rows = [
            {
                "source_file": f"{network}/{station}.stm",
                "network": network,
                "station": station,
                "depth_class": "near_surface",
                "exclusion_reason": "",
                "good_quality_rows": "10",
            }
            for network in ("A", "B", "C", "D", "E")
            for station in ("one", "two")
        ]
        cohort = select_cohort(rows)
        first = assign_network_splits(cohort, seed=7)
        second = assign_network_splits(cohort, seed=7)
        self.assertEqual(first, second)
        networks = {split: {row["network"] for row in first if row["split"] == split}
                    for split in ("train", "validation", "test")}
        self.assertTrue(networks["train"].isdisjoint(networks["validation"]))
        self.assertTrue(networks["train"].isdisjoint(networks["test"]))
        self.assertTrue(networks["validation"].isdisjoint(networks["test"]))

    def test_tensor_pack_preserves_shapes_and_metadata(self) -> None:
        contract = GlobalDataContract()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature = root / "feature.npz"
            np.savez(
                feature,
                spatial=np.zeros((contract.spatial_channel_count, 13, 13), dtype=np.float32),
                temporal=np.zeros((2, 4), dtype=np.float32),
                static=np.zeros(contract.static_channel_count, dtype=np.float32),
            )
            manifest = root / "features.csv"
            fields = ["station_id", "network", "sample_date", "split", "feature_path", "target", "anchor"]
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for split in ("train", "validation", "test"):
                    writer.writerow({
                        "station_id": split, "network": split, "sample_date": "2020-01-01",
                        "split": split, "feature_path": "feature.npz", "target": "0.2", "anchor": "0.1",
                    })
            outputs = pack_tensors(
                read_feature_manifest(manifest), manifest_dir=root, output_dir=root / "out"
            )
            self.assertEqual(set(outputs), {"train", "validation", "test"})
            with np.load(outputs["train"]) as data:
                self.assertEqual(data["spatial"].shape, (1, contract.spatial_channel_count, 13, 13))
                self.assertEqual(data["target"].shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
