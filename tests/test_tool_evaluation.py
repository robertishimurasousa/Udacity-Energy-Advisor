import json
from datetime import datetime, timedelta
from math import isclose
from pathlib import Path
from unittest import TestCase

from langchain_core.messages import AIMessage, ToolMessage


class ToolEvaluationTests(TestCase):
    @classmethod
    def setUpClass(cls):
        notebook_path = Path(__file__).resolve().parents[1] / "03_run_and_evaluate.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        scope = {
            "AIMessage": AIMessage, "ToolMessage": ToolMessage, "json": json,
            "Path": Path, "datetime": datetime, "timedelta": timedelta, "isclose": isclose,
        }
        for marker in (
            "def extract_tool_calls(", "def evaluate_tool_usage(", "def response_checks(",
            "def evaluate_response(", "def generate_evaluation_report(",
        ):
            source = next(
                "".join(cell["source"]) for cell in notebook["cells"]
                if cell["cell_type"] == "code" and marker in "".join(cell["source"])
            )
            exec(compile(source, str(notebook_path), "exec"), scope)
        cls.evaluate = staticmethod(scope["evaluate_tool_usage"])
        cls.extract = staticmethod(scope["extract_tool_calls"])
        cls.report = staticmethod(scope["generate_evaluation_report"])

    def score(self, trace, required, **expectations):
        requests, responses = [], []
        for index, (name, args, output) in enumerate(trace, start=1):
            call_id = f"call_{index}"
            requests.append({"name": name, "args": args, "id": call_id})
            if output is not None:
                responses.append(ToolMessage(content=json.dumps(output), tool_call_id=call_id))
        messages = [AIMessage(content="", tool_calls=requests), *reversed(responses)]
        return self.evaluate(messages, required, **expectations)

    def assert_scores(self, result, appropriateness, completeness):
        self.assertEqual(result["appropriateness"], appropriateness)
        self.assertEqual(result["completeness"], completeness)

    def test_expected_and_optional_calls_are_matched_by_id(self):
        result = self.score(
            [("prices", {"date": "2026-11-21"}, {"rate": 0.12}), ("weather", {}, {"source": "simulated"})],
            ["prices"], optional_tools=["weather"],
            expected_values={"prices": {"rate": 0.12}},
        )
        self.assert_scores(result, 1.0, 1.0)
        self.assertEqual(result["issues"], [])

    def test_missing_tool_reduces_completeness(self):
        result = self.score([("prices", {}, {"rate": 0.12})], ["prices", "weather"])
        self.assert_scores(result, 1.0, 0.5)
        self.assertIn("weather", result["missing_tools"])

    def test_unexpected_tool_reduces_appropriateness(self):
        result = self.score([("prices", {}, {"rate": 0.12}), ("weather", {}, {})], ["prices"])
        self.assert_scores(result, 0.5, 1.0)
        self.assertTrue(any("Unexpected tool" in issue for issue in result["issues"]))

    def test_wrong_date_does_not_count_for_either_metric(self):
        result = self.score(
            [("prices", {"date": "2026-11-20"}, {"rate": 0.12})], ["prices"],
            expected_args={"prices": {"date": "2026-11-21"}},
        )
        self.assert_scores(result, 0.0, 0.0)
        self.assertIn("prices", result["unusable_tools"])
        self.assertTrue(any("2026-11-20" in issue and "2026-11-21" in issue for issue in result["issues"]))

    def test_forecast_must_reach_target_day_and_can_use_default_horizon(self):
        tool = "get_weather_forecast"
        expected_args = {tool: {"location": "San Francisco, CA", "days": range(2, 8)}}
        expected_values = {tool: {"forecast_days": range(2, 8)}}
        for args, days, score in (({}, 3, 1.0), ({"days": 1}, 1, 0.0)):
            with self.subTest(args=args):
                result = self.score(
                    [(tool, {"location": "San Francisco, CA", **args}, {"forecast_days": days})],
                    [tool], expected_args=expected_args, expected_values=expected_values,
                )
                self.assert_scores(result, score, score)

    def test_household_usage_cannot_be_satisfied_by_a_device_filter(self):
        tool = "query_energy_usage"
        for args, score in (({}, 1.0), ({"device_type": "EV"}, 0.0)):
            with self.subTest(args=args):
                result = self.score(
                    [(tool, args, {"total_records": 0})],
                    [tool], expected_args={tool: {"device_type": None}},
                )
                self.assert_scores(result, score, score)

    def test_failed_or_missing_result_does_not_satisfy_required_tool(self):
        for output in ({"error": "Service unavailable"}, None, ["invalid payload"], {}):
            with self.subTest(output=output):
                result = self.score([("prices", {}, output)], ["prices"])
                self.assert_scores(result, 1.0, 0.0)
                self.assertIn("prices", result["unusable_tools"])
                self.assertTrue(result["issues"])

    def test_failed_call_followed_by_success_preserves_retry_and_feedback(self):
        result = self.score(
            [("prices", {}, {"error": "Service unavailable"}), ("prices", {}, {"rate": 0.12})],
            ["prices"],
        )
        self.assert_scores(result, 1.0, 1.0)
        self.assertTrue(any("Tool returned an error" in issue for issue in result["issues"]))

    def test_repeating_a_usable_call_reduces_appropriateness(self):
        call = ("prices", {"date": "2026-11-21"}, {"rate": 0.12})
        result = self.score([call, call], ["prices"])
        self.assert_scores(result, 0.5, 1.0)
        self.assertTrue(any("Redundant call" in issue for issue in result["issues"]))

    def test_mismatched_reference_values_cannot_be_combined_across_calls(self):
        result = self.score(
            [("usage", {}, {"total": 5, "count": 1}), ("usage", {}, {"total": 7, "count": 2})],
            ["usage"], expected_values={"usage": {"total": 5, "count": 2}},
        )
        self.assert_scores(result, 1.0, 0.0)
        self.assertTrue(any("count" in issue for issue in result["issues"]))
        self.assertTrue(any("total" in issue for issue in result["issues"]))

    def test_wrong_guide_is_not_usable_evidence(self):
        result = self.score(
            [("search_energy_tips", {}, {"tips": [{"source": "data/documents/tip_hvac_efficiency.txt"}]})],
            ["search_energy_tips"], expected_sources=["tip_battery_storage.txt"],
        )
        self.assert_scores(result, 1.0, 0.0)
        self.assertTrue(any("tip_battery_storage.txt" in issue for issue in result["issues"]))

    def test_tariff_date_alone_does_not_satisfy_required_rate_evidence(self):
        reference = {"prices": {"date": "2026-11-21", "hours.0.rate": 0.12}}
        for returned, completeness in (({"date": "2026-11-21"}, 0.0),
                                       ({"date": "2026-11-21", "hours": [{"rate": 0.12}]}, 1.0)):
            with self.subTest(returned=returned):
                result = self.score([("prices", {}, returned)], ["prices"], expected_values=reference)
                self.assert_scores(result, 1.0, completeness)

    def test_malformed_forecast_keeps_report_and_identifies_missing_evidence(self):
        forecast = "get_weather_forecast"
        tariff = "get_electricity_prices"
        date = "2026-11-21"
        messages = [
            AIMessage(content="", tool_calls=[
                {"name": forecast, "args": {"days": 2}, "id": "forecast_1"},
                {"name": tariff, "args": {"date": date}, "id": "tariff_1"},
            ]),
            ToolMessage(content='["invalid forecast"]', tool_call_id="forecast_1"),
            ToolMessage(content='{"rate": 0.12}', tool_call_id="tariff_1"),
        ]
        result = {
            "test_id": "ev_charging_1", "label": "EV charging",
            "response": {"messages": messages}, "tool_calls": self.extract(messages),
            "expected_tools": [forecast, tariff], "expected_args": {tariff: {"date": date}},
            "question": "When should I charge my EV tomorrow?",
            "final_response": f"Charge on {date} at 12:00 if there is surplus solar.",
            "expected_response": "Use the forecast and rates for the target date.",
        }
        report = self.report([result])
        self.assert_scores(result["tool_evaluation"], 1.0, 0.5)
        self.assertIn("forecast_1", report)
        self.assertIn("Tool result must be a nonempty JSON object", report)
        self.assertEqual(result["response_evaluation"]["scores"]["accuracy"], 0.5)

    def test_solar_daily_calls_must_cover_every_requested_day(self):
        tool = "query_solar_generation"
        days = ("2026-11-19", "2026-11-20")
        trace = [(tool, {"start_date": day, "end_date": day}, {"total_records": 13}) for day in days]
        args = {tool: {"start_date": days[0], "end_date": days[1]}}
        values = {tool: {"total_records": 26}}
        complete = self.score(trace, [tool], expected_args=args, expected_values=values)
        incomplete = self.score(trace[:1], [tool], expected_args=args, expected_values=values)
        self.assert_scores(complete, 1.0, 1.0)
        self.assert_scores(incomplete, 1.0, 0.0)
        self.assertTrue(any(days[1] in issue for issue in incomplete["issues"]))

    def test_daily_call_after_a_complete_solar_range_is_redundant(self):
        tool = "query_solar_generation"
        full_range = {"start_date": "2026-11-19", "end_date": "2026-11-20"}
        single_day = {"start_date": "2026-11-19", "end_date": "2026-11-19"}
        result = self.score(
            [(tool, full_range, {"total_records": 26}), (tool, single_day, {"total_records": 13})],
            [tool], expected_args={tool: full_range}, expected_values={tool: {"total_records": 26}},
        )
        self.assert_scores(result, 0.5, 1.0)
        self.assertTrue(any("Redundant call" in issue for issue in result["issues"]))

    def test_no_call_is_appropriate_when_no_tool_is_required(self):
        self.assert_scores(self.score([], []), 1.0, 1.0)
