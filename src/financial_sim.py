import pandas as pd
import numpy as np
import logging

class FinancialRiskSimulator:
    """Worker #4: Translates machine learning metrics directly into Net Present Value (NPV)."""
    
    def __init__(self, capex: float = 50000.0, opex_annual: float = 15000.0, discount_rate: float = 0.10):
        self.capex = capex
        self.opex_annual = opex_annual
        self.discount_rate = discount_rate

    def simulate_3year_npv(self, return_handling_cost: float, target_reduction: float = 0.10, total_returns_annual: int = 5000) -> float:
        """Calculates the 3-Year NPV return for a specific business operational cost structure."""
        # Calculate gross operational savings per year
        annual_savings = total_returns_annual * return_handling_cost * target_reduction
        
        # Net annual cash flow after maintaining the software model
        net_annual_cash_flow = annual_savings - self.opex_annual
        
        # Calculate NPV over 3 years
        npv = -self.capex
        for year in [1, 2, 3]:
            npv += net_annual_cash_flow / ((1 + self.discount_rate) ** year)
            
        return npv

    def run_breakeven_analysis(self, total_returns_annual: int = 5000):
        """Computes executive thresholds for operational costs."""
        logging.info("--- Executive Financial Breakeven Analysis ---")
        pv_factor = 2.48685  # Present value multiplier for 3 years at 10%
        
        for cost in [10, 15, 20]:
            annual_savings_needed = (self.capex / pv_factor) + self.opex_annual
            req_reduction_pct = (annual_savings_needed / (total_returns_annual * cost)) * 100
            
            logging.info(f"At a ${cost} handling cost per return:")
            logging.info(f" -> Model must reduce returns by at least {req_reduction_pct:.2f}% to break even over 3 years.")