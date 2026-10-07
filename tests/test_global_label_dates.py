import tempfile
import unittest
from pathlib import Path

from build_global_label_dates import build_label_dates


class GlobalLabelDateTests(unittest.TestCase):
    def test_builds_only_good_compatible_dates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.stm"
            path.write_text(
                "NET NET Station 1.0 2.0 3.0 0.0000 0.0500 'Sensor'\n"
                "2020/01/01 00:00 0.2 G M\n"
                "2020/01/02 00:00 0.2 D01 M\n"
                "2020/01/03 00:00 0.2 G M\n",
                encoding="utf-8",
            )
            rows = build_label_dates(Path(directory))
            self.assertEqual([row["sample_date"] for row in rows], ["2020-01-01", "2020-01-03"])


if __name__ == "__main__":
    unittest.main()
