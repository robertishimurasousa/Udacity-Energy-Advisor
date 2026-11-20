from unittest import TestCase

from tools import calculate_energy_savings


class EnergySavingsTests(TestCase):
    def test_uses_explicit_investment_horizon_and_emissions_factor(self):
        result = calculate_energy_savings.invoke({
            "device_type": "HVAC",
            "current_usage_kwh": 10,
            "optimized_usage_kwh": 8,
            "price_per_kwh": 0.2,
            "investment_cost_usd": 100,
            "analysis_years": 2,
            "emissions_factor_kg_co2e_per_kwh": 0.4,
        })
        self.assertEqual(result["annual_savings_usd"], 146)
        self.assertEqual(result["investment"]["roi_percentage"], 192)
        self.assertEqual(result["investment"]["simple_payback_years"], 0.68)
        self.assertEqual(result["investment"]["net_savings_usd"], 192)
        self.assertEqual(result["emissions"]["annual_avoided_kg_co2e"], 292)

    def test_keeps_simple_savings_and_rejects_invalid_assumptions(self):
        inputs = {"device_type": "HVAC", "current_usage_kwh": 10, "optimized_usage_kwh": 8}
        result = calculate_energy_savings.invoke(inputs)
        self.assertEqual(result["savings_usd"], 0.24)
        self.assertNotIn("investment", result)
        self.assertNotIn("emissions", result)
        invalid = calculate_energy_savings.invoke({**inputs, "investment_cost_usd": 0})
        self.assertIn("error", invalid)
