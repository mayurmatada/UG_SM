import tempfile
import unittest
from datetime import date
from pathlib import Path

from gee_global_export import adaptive_tiles, build_tasks, month_windows, read_stations, time_windows


class GeeGlobalExportTests(unittest.TestCase):
    def test_tiles_are_deterministic_and_bounded(self) -> None:
        from gee_global_export import Station

        stations = [
            Station(f"s{i}", "N", f"s{i}", float(i), float(i * 2), f"cell{i}")
            for i in range(11)
        ]
        first = adaptive_tiles(stations, 3)
        second = adaptive_tiles(list(reversed(stations)), 3)
        self.assertEqual(first, second)
        self.assertTrue(all(len(items) <= 3 for items in first.values()))
        self.assertEqual(sum(map(len, first.values())), 11)

    def test_year_windows_are_half_open(self) -> None:
        self.assertEqual(
            time_windows(date(2020, 1, 15), date(2023, 1, 1)),
            [
                (date(2020, 1, 15), date(2021, 1, 1)),
                (date(2021, 1, 1), date(2022, 1, 1)),
                (date(2022, 1, 1), date(2023, 1, 1)),
            ],
        )

    def test_task_count_is_tiles_times_months(self) -> None:
        from gee_global_export import Station

        tiles = adaptive_tiles(
            [Station(str(i), "N", str(i), float(i), float(i), f"cell{i}") for i in range(4)],
            2,
        )
        tasks = build_tasks(tiles, date(2020, 1, 1), date(2022, 1, 1), "folder")
        self.assertEqual(len(tasks), 6)
        self.assertEqual(sum(task.export_kind == "static" for task in tasks), 2)
        self.assertTrue(all(task.compressed for task in tasks))

    def test_year_filter_selects_only_requested_window(self) -> None:
        from gee_global_export import Station

        tiles = adaptive_tiles(
            [Station("s", "N", "s", 1.0, 1.0, "cell")],
            2,
        )
        tasks = build_tasks(
            tiles,
            date(2020, 1, 1),
            date(2023, 1, 1),
            "folder",
            only_year=2021,
        )
        self.assertEqual(
            [
                (task.export_kind, task.start_date, task.end_date)
                for task in tasks
            ],
            [
                ("static", "2020-01-01", "2023-01-01"),
                ("daily", "2021-01-01", "2022-01-01"),
            ],
        )

    def test_static_features_can_be_submitted_once(self) -> None:
        from gee_global_export import Station

        tiles = adaptive_tiles(
            [Station("s", "N", "s", 1.0, 1.0, "cell")],
            2,
        )
        tasks = build_tasks(
            tiles,
            date(2020, 1, 1),
            date(2021, 1, 1),
            "folder",
            label_dates={"cell": {"2020-03-01"}},
            include_static=False,
        )
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].export_kind, "daily")


if __name__ == "__main__":
    unittest.main()
