import json
from datetime import datetime
from pathlib import Path
from unittest import TestCase


class ResponseEvaluationTests(TestCase):
    @classmethod
    def setUpClass(cls):
        notebook_path = Path(__file__).resolve().parents[1] / "03_run_and_evaluate.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = next(
            "".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code" and "def evaluate_response(" in "".join(cell["source"])
        )
        scope = {"datetime": datetime}
        exec(compile(source, str(notebook_path), "exec"), scope)
        checks_source = next(
            "".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code" and "def response_checks(" in "".join(cell["source"])
        )
        exec(compile(checks_source, str(notebook_path), "exec"), scope)
        cls.evaluate = staticmethod(scope["evaluate_response"])
        cls.response_checks = staticmethod(scope["response_checks"])

    def test_scores_new_answers_and_explains_a_missing_citation(self):
        question = "How long can the battery last? Cite its guide."
        expectation = {
            "description": "Give a four-hour estimate with a source.",
            "checks": {
                "accuracy": [("Four-hour estimate", r"\b4 hours\b", True)],
                "relevance": [("Discusses the battery", r"\bbattery\b", True)],
                "completeness": [("Cites the battery guide", r"tip_battery_storage\.txt", True)],
                "usefulness": [("States the load", r"\b1 kW\b", True)],
            },
        }
        answer = "The battery can last 4 hours at 1 kW."
        incomplete = self.evaluate(question, answer, expectation)
        complete = self.evaluate(question, answer + " See tip_battery_storage.txt.", expectation)

        self.assertEqual(incomplete["scores"], {"accuracy": 2, "relevance": 2, "completeness": 0, "usefulness": 2})
        self.assertIn("Cites the battery guide", incomplete["feedback"])
        self.assertTrue(all(score == 2 for score in complete["scores"].values()))
        self.assertFalse(any(self.evaluate(question, "", expectation)["scores"].values()))

    def test_distinguishes_no_recorded_usage_from_zero_consumption(self):
        result = {
            "test_id": "missing_history",
            "expected_args": {"query_energy_usage": {"start_date": "2026-10-19"}},
        }
        expectation = {
            "description": "Explain that the records are unavailable.",
            "checks": self.response_checks(result),
        }
        question = "How much energy was used on October 19?"
        missing = self.evaluate(question, "There is no recorded electricity usage on October 19.", expectation)
        measured = self.evaluate(question, "No electricity was used. The total was 0 kWh on October 19.", expectation)
        self.assertEqual(missing["scores"]["accuracy"], 2)
        self.assertEqual(measured["scores"]["accuracy"], 0)
