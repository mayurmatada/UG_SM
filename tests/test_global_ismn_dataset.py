import json
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from build_global_ismn_dataset import build_dataset


class GlobalIsmnDatasetTests(unittest.TestCase):
    def test_converts_hourly_good_rows_to_daily_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            station = root / "NET" / "Station"
            station.mkdir(parents=True)
            (station / "NET_NET_Station_static_variables.csv").write_text(
                "quantity_name;unit;depth_from[m];depth_to[m];value;\n"
                "clay fraction;%;0;0.3;18;\n"
                "climate classification;;;;Csa;\n",
                encoding="utf-8",
            )
            (station / "sample.stm").write_text(
                "NET Network Station 1.0 2.0 3.0 0.0000 0.0500 'Sensor'\n"
                "2020/01/01 00:00 0.2 G M\n"
                "2020/01/01 01:00 0.4 G M\n"
                "2020/01/01 02:00 0.9 D01 M\n"
                "2020/01/02 00:00 0.3 G M\n",
                encoding="utf-8",
            )
            output = root / "daily.parquet"
            summary = root / "summary.json"
            stats = build_dataset(root, output, summary)

            table = pq.read_table(output).to_pydict()
            self.assertEqual(stats["daily_rows"], 2)
            for value in table["soil_moisture_m3_m3"]:
                self.assertAlmostEqual(value, 0.3, places=6)
            self.assertEqual(table["observation_count"], [2, 1])
            self.assertEqual(table["latitude"], [1.0, 1.0])
            self.assertEqual(table["climate_zone"], ["Csa", "Csa"])
            self.assertEqual(json.loads(summary.read_text())["non_good_rows"], 1)


if __name__ == "__main__":
    unittest.main()
