import tempfile
import unittest
from pathlib import Path

from build_global_ismn_catalog import build_catalog, parse_header, scan_rows, write_outputs


class GlobalCatalogTests(unittest.TestCase):
    def test_depth_classes_are_explicit(self) -> None:
        near = parse_header(
            "NET Network Station 1.0 2.0 3.0 0.0000 0.0500 'Sensor'",
            "NET/Station/file.stm",
        )
        deeper = parse_header(
            "NET Network Station 1.0 2.0 3.0 0.0500 0.1000 'Sensor'",
            "NET/Station/file.stm",
        )
        excluded = parse_header(
            "NET Network Station 1.0 2.0 3.0 0.1000 0.2000 'Sensor'",
            "NET/Station/file.stm",
        )

        self.assertEqual(near.depth_class, "near_surface")
        self.assertEqual(deeper.depth_class, "fallback_deeper")
        self.assertEqual(excluded.depth_class, "excluded")
        self.assertEqual(excluded.exclusion_reason, "depth_outside_0_to_10_cm")

    def test_scan_rows_counts_good_and_malformed_records(self) -> None:
        result = scan_rows(
            [
                "2020/01/01 00:00 0.2 G M\n",
                "2020/01/01 01:00 0.3 D01 M\n",
                "not-a-date malformed\n",
                "2020/01/02 00:00 0.4 G M\n",
            ]
        )
        self.assertEqual(result, (3, 2, 1, "2020-01-01", "2020-01-02"))

    def test_directory_catalog_and_summary_include_provenance_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            station = source / "NET" / "Station"
            station.mkdir(parents=True)
            (station / "NET_NET_Station_sm_0.000000_0.050000_Sensor.stm").write_text(
                "NET NET Station 1.0 2.0 3.0 0.0000 0.0500 'Sensor'\n"
                "2020/01/01 00:00 0.2 G M\n",
                encoding="utf-8",
            )
            output_csv = source / "catalog.csv"
            output_json = source / "summary.json"
            records = build_catalog(source, scan_observations=True)
            write_outputs(records, output_csv, output_json)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].good_quality_rows, 1)
            self.assertIn("static_variables.csv", records[0].static_variables_file or "")
            self.assertIn('"contract"', output_json.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
