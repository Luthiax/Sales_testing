"""Transparent financial sensitivity calculations for the portfolio case study."""


class FinancialRiskSimulator:
    """Estimate NPV for assumed reductions in annual product returns.

    The class does not infer impact from model predictions. Callers must supply
    an explicit target-reduction assumption.
    """

    def __init__(
        self,
        capex: float = 50_000.0,
        opex_annual: float = 15_000.0,
        discount_rate: float = 0.10,
        horizon_years: int = 3,
    ):
        if capex < 0 or opex_annual < 0:
            raise ValueError("Costs must be non-negative.")
        if discount_rate <= -1:
            raise ValueError("discount_rate must be greater than -1.")
        if horizon_years < 1:
            raise ValueError("horizon_years must be at least 1.")

        self.capex = capex
        self.opex_annual = opex_annual
        self.discount_rate = discount_rate
        self.horizon_years = horizon_years

    @property
    def annuity_present_value_factor(self) -> float:
        """Present-value factor for an equal annual cash flow."""
        return sum(
            1 / ((1 + self.discount_rate) ** year)
            for year in range(1, self.horizon_years + 1)
        )

    def simulate_3year_npv(
        self,
        return_handling_cost: float,
        target_reduction: float = 0.10,
        total_returns_annual: int = 5_000,
    ) -> float:
        """Calculate NPV under explicit volume, cost, and reduction assumptions."""
        if return_handling_cost < 0 or total_returns_annual < 0:
            raise ValueError("Return cost and annual volume must be non-negative.")
        if not 0 <= target_reduction <= 1:
            raise ValueError("target_reduction must be between 0 and 1.")

        annual_savings = total_returns_annual * return_handling_cost * target_reduction
        net_annual_cash_flow = annual_savings - self.opex_annual
        return -self.capex + net_annual_cash_flow * self.annuity_present_value_factor

    def breakeven_reduction(
        self,
        return_handling_cost: float,
        total_returns_annual: int = 5_000,
    ) -> float:
        """Return the annual reduction fraction required for NPV = 0."""
        if return_handling_cost <= 0 or total_returns_annual <= 0:
            raise ValueError("Return cost and annual volume must be positive.")

        annual_savings_needed = (
            self.capex / self.annuity_present_value_factor
        ) + self.opex_annual
        return annual_savings_needed / (total_returns_annual * return_handling_cost)
