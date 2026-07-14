import unittest

from src.financial_sim import FinancialRiskSimulator


class FinancialRiskSimulatorTests(unittest.TestCase):
    def setUp(self):
        self.simulator = FinancialRiskSimulator(
            capex=50_000,
            opex_annual=15_000,
            discount_rate=0.10,
        )

    def test_base_case_npv(self):
        result = self.simulator.simulate_3year_npv(
            return_handling_cost=20,
            target_reduction=0.10,
            total_returns_annual=30_000,
        )
        self.assertAlmostEqual(result, 61_908.34, places=2)

    def test_base_case_breakeven(self):
        result = self.simulator.breakeven_reduction(
            return_handling_cost=20,
            total_returns_annual=30_000,
        )
        self.assertAlmostEqual(result, 0.058509567, places=8)

    def test_reduction_must_be_a_fraction(self):
        with self.assertRaises(ValueError):
            self.simulator.simulate_3year_npv(
                return_handling_cost=20,
                target_reduction=1.10,
                total_returns_annual=30_000,
            )


if __name__ == "__main__":
    unittest.main()
