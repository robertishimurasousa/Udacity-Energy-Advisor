from unittest import TestCase

from tools import get_electricity_prices


class TariffWindowTests(TestCase):
    def test_tariff_windows_end_after_the_last_hour_they_include(self):
        prices = get_electricity_prices.invoke({"date": "2026-11-20"})

        self.assertEqual(prices.get("rate_windows"), [
            {"period": "off_peak", "start_time": "00:00", "end_time_exclusive": "06:00", "rate": 0.12},
            {"period": "peak", "start_time": "06:00", "end_time_exclusive": "22:00", "rate": 0.28},
            {"period": "off_peak", "start_time": "22:00", "end_time_exclusive": "24:00", "rate": 0.12},
        ])

    def test_expanding_windows_reproduces_all_hourly_rates_in_order(self):
        prices = get_electricity_prices.invoke({"date": "2026-11-20"})
        expanded = []
        for window in prices.get("rate_windows", []):
            start_hour = int(window["start_time"].split(":")[0])
            end_hour = int(window["end_time_exclusive"].split(":")[0])
            expanded.extend(
                (hour, window["period"], window["rate"])
                for hour in range(start_hour, end_hour)
            )

        hourly = [(row["hour"], row["period"], row["rate"]) for row in prices["hourly_rates"]]
        self.assertEqual(expanded, hourly)
        self.assertEqual([hour for hour, _, _ in expanded], list(range(24)))
