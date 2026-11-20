from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from models.energy import DatabaseManager
from tools import get_weather_forecast, query_solar_generation


class ForecastSolarPeakTests(TestCase):
    def test_each_daily_peak_matches_the_highest_hourly_irradiance(self):
        with patch("tools.datetime", wraps=datetime) as clock:
            clock.now.return_value = datetime(2026, 11, 19, 9)
            forecast = get_weather_forecast.invoke({"location": "San Francisco, CA", "days": 2})

        peaks = forecast.get("daily_solar_peaks", [])
        self.assertEqual([peak["date"] for peak in peaks], ["2026-11-19", "2026-11-20"])
        for peak in peaks:
            hourly = [row for row in forecast["hourly"] if row["date"] == peak["date"]]
            expected = max(hourly, key=lambda row: row["solar_irradiance"])
            self.assertEqual(peak["hour"], expected["hour"])
            self.assertEqual(peak["solar_irradiance"], expected["solar_irradiance"])

        self.assertEqual(peaks[1], {"date": "2026-11-20", "hour": 11, "solar_irradiance": 534})


class DailySolarGenerationTests(TestCase):
    def setUp(self):
        directory = Path(self.enterContext(TemporaryDirectory()))
        self.database = DatabaseManager(str(directory / "energy.db"))
        self.addCleanup(self.database.engine.dispose)
        self.database.create_tables()
        self.enterContext(patch("tools.db_manager", self.database))

        samples = [
            ("2026-01-15T00:00:00", 2.0),
            ("2026-01-15T13:00:00", 3.5),
            ("2026-01-17T00:00:00", 1.25),
            ("2026-01-17T23:59:59.999999", 4.25),
            ("2026-01-18T00:00:00", 99.0),
            ("2026-01-19T12:00:00", 0.0),
        ]
        for timestamp, generation in samples:
            self.database.add_generation_record(datetime.fromisoformat(timestamp), generation)

    def test_reports_daily_totals_and_all_tied_highest_dates(self):
        result = query_solar_generation.invoke({"start_date": "2026-01-15", "end_date": "2026-01-17"})

        self.assertEqual(result["total_records"], 4)
        self.assertEqual(result["total_generation_kwh"], 11.0)
        self.assertEqual(result["average_daily_generation"], 3.67)
        self.assertEqual(result["daily_generation"], [
            {"date": "2026-01-15", "total_generation_kwh": 5.5, "total_records": 2},
            {"date": "2026-01-17", "total_generation_kwh": 5.5, "total_records": 2},
        ])
        self.assertEqual(result["highest_generation_dates"], ["2026-01-15", "2026-01-17"])

    def test_distinguishes_a_missing_day_from_measured_zero_generation(self):
        missing = query_solar_generation.invoke({"start_date": "2026-01-16", "end_date": "2026-01-16"})
        measured = query_solar_generation.invoke({"start_date": "2026-01-19", "end_date": "2026-01-19"})

        self.assertEqual(missing["total_records"], 0)
        self.assertEqual(missing["daily_generation"], [])
        self.assertEqual(missing["highest_generation_dates"], [])
        self.assertEqual(measured["daily_generation"], [
            {"date": "2026-01-19", "total_generation_kwh": 0.0, "total_records": 1},
        ])
