"""
Financial calculations for solar payback analysis.
Supports cash purchase, solar loan, lease, and PPA scenarios.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class CashPurchaseResult:
    """Results for cash purchase scenario."""
    gross_cost: float
    federal_itc: float
    state_rebate: float
    net_cost: float
    annual_savings: float
    simple_payback_years: float
    cumulative_savings: List[float]  # By year
    roi_25_year: float


@dataclass
class LoanResult:
    """Results for solar loan scenario."""
    loan_amount: float
    monthly_payment: float
    annual_payment: float
    total_interest: float
    annual_solar_value: float
    net_annual_savings: float  # Solar value minus loan payment
    cumulative_savings: List[float]  # By year
    breakeven_year: int  # Year when cumulative savings turn positive


@dataclass
class LeaseResult:
    """Results for solar lease scenario."""
    monthly_payment_year1: float
    annual_escalator: float
    lease_term_years: int
    year1_savings: float
    lifetime_savings: float
    payments_by_year: List[float]
    savings_by_year: List[float]
    cumulative_savings: List[float]


@dataclass
class PPAResult:
    """Results for PPA (Power Purchase Agreement) scenario."""
    ppa_rate: float
    utility_rate: float
    annual_escalator: float
    contract_term: int
    year1_savings: float
    lifetime_savings: float
    ppa_costs_by_year: List[float]
    utility_costs_by_year: List[float]
    savings_by_year: List[float]
    cumulative_savings: List[float]


def calculate_cash_purchase(
    system_size_kw: float,
    cost_per_watt: float,
    federal_itc_rate: float,
    state_rebate: float,
    annual_production_kwh: float,
    electricity_rate: float,
    analysis_years: int = 25
) -> CashPurchaseResult:
    """
    Calculate financials for cash purchase of solar system.

    Args:
        system_size_kw: System size in kW
        cost_per_watt: Installed cost per watt ($/W)
        federal_itc_rate: Federal ITC rate (e.g., 0.30 for 30%)
        state_rebate: State rebate amount in dollars
        annual_production_kwh: Predicted annual production
        electricity_rate: Electricity rate ($/kWh)
        analysis_years: Number of years to analyze

    Returns:
        CashPurchaseResult with financial projections
    """
    # Calculate costs
    gross_cost = system_size_kw * 1000 * cost_per_watt
    federal_itc = gross_cost * federal_itc_rate
    net_cost = gross_cost - federal_itc - state_rebate

    # Annual savings (assume 0.5% annual degradation)
    annual_savings_year1 = annual_production_kwh * electricity_rate

    # Simple payback (ignoring degradation and rate increases)
    if annual_savings_year1 > 0:
        simple_payback = net_cost / annual_savings_year1
    else:
        simple_payback = float('inf')

    # Cumulative savings with degradation (0.5%/year) and rate increase (2%/year)
    cumulative_savings = []
    total_savings = -net_cost  # Start negative (initial investment)

    for year in range(1, analysis_years + 1):
        degradation_factor = (1 - 0.005) ** (year - 1)
        rate_factor = (1 + 0.02) ** (year - 1)
        year_production = annual_production_kwh * degradation_factor
        year_rate = electricity_rate * rate_factor
        year_savings = year_production * year_rate
        total_savings += year_savings
        cumulative_savings.append(total_savings)

    # 25-year ROI
    total_value = sum(
        annual_production_kwh * (1 - 0.005) ** (y - 1) * electricity_rate * (1 + 0.02) ** (y - 1)
        for y in range(1, 26)
    )
    roi_25_year = (total_value - net_cost) / net_cost * 100 if net_cost > 0 else 0

    return CashPurchaseResult(
        gross_cost=gross_cost,
        federal_itc=federal_itc,
        state_rebate=state_rebate,
        net_cost=net_cost,
        annual_savings=annual_savings_year1,
        simple_payback_years=simple_payback,
        cumulative_savings=cumulative_savings,
        roi_25_year=roi_25_year
    )


def calculate_loan(
    loan_amount: float,
    interest_rate: float,
    loan_term_years: int,
    annual_production_kwh: float,
    electricity_rate: float,
    analysis_years: int = 25
) -> LoanResult:
    """
    Calculate financials for solar loan.

    Args:
        loan_amount: Principal loan amount
        interest_rate: Annual interest rate (e.g., 0.06 for 6%)
        loan_term_years: Loan term in years
        annual_production_kwh: Predicted annual production
        electricity_rate: Electricity rate ($/kWh)
        analysis_years: Number of years to analyze

    Returns:
        LoanResult with payment and savings projections
    """
    # Monthly payment calculation (standard amortization)
    monthly_rate = interest_rate / 12
    num_payments = loan_term_years * 12

    if monthly_rate > 0:
        monthly_payment = loan_amount * (
            monthly_rate * (1 + monthly_rate) ** num_payments
        ) / ((1 + monthly_rate) ** num_payments - 1)
    else:
        monthly_payment = loan_amount / num_payments

    annual_payment = monthly_payment * 12
    total_payments = monthly_payment * num_payments
    total_interest = total_payments - loan_amount

    # Annual solar value
    annual_solar_value = annual_production_kwh * electricity_rate

    # Net annual savings during loan term
    net_annual_savings = annual_solar_value - annual_payment

    # Cumulative savings analysis
    cumulative_savings = []
    total_savings = 0
    breakeven_year = 0

    for year in range(1, analysis_years + 1):
        degradation_factor = (1 - 0.005) ** (year - 1)
        rate_factor = (1 + 0.02) ** (year - 1)
        year_production = annual_production_kwh * degradation_factor
        year_rate = electricity_rate * rate_factor
        year_solar_value = year_production * year_rate

        if year <= loan_term_years:
            year_savings = year_solar_value - annual_payment
        else:
            year_savings = year_solar_value  # No more loan payments

        total_savings += year_savings
        cumulative_savings.append(total_savings)

        if breakeven_year == 0 and total_savings > 0:
            breakeven_year = year

    return LoanResult(
        loan_amount=loan_amount,
        monthly_payment=monthly_payment,
        annual_payment=annual_payment,
        total_interest=total_interest,
        annual_solar_value=annual_solar_value,
        net_annual_savings=net_annual_savings,
        cumulative_savings=cumulative_savings,
        breakeven_year=breakeven_year if breakeven_year > 0 else analysis_years
    )


def calculate_lease(
    monthly_payment: float,
    annual_escalator: float,
    lease_term_years: int,
    annual_production_kwh: float,
    electricity_rate: float,
    analysis_years: int = 25
) -> LeaseResult:
    """
    Calculate financials for solar lease.

    Args:
        monthly_payment: Initial monthly lease payment
        annual_escalator: Annual payment escalator (e.g., 0.02 for 2%)
        lease_term_years: Lease term in years
        annual_production_kwh: Predicted annual production
        electricity_rate: Electricity rate ($/kWh)
        analysis_years: Number of years to analyze

    Returns:
        LeaseResult with payment and savings projections
    """
    payments_by_year = []
    savings_by_year = []
    cumulative_savings = []
    total_savings = 0

    for year in range(1, analysis_years + 1):
        # Lease payment with escalator
        if year <= lease_term_years:
            escalator_factor = (1 + annual_escalator) ** (year - 1)
            year_payment = monthly_payment * 12 * escalator_factor
        else:
            year_payment = 0  # Lease ended

        # Solar value (production with degradation, rate with inflation)
        degradation_factor = (1 - 0.005) ** (year - 1)
        rate_factor = (1 + 0.02) ** (year - 1)
        year_production = annual_production_kwh * degradation_factor
        year_rate = electricity_rate * rate_factor
        year_solar_value = year_production * year_rate

        # Net savings
        year_savings = year_solar_value - year_payment
        total_savings += year_savings

        payments_by_year.append(year_payment)
        savings_by_year.append(year_savings)
        cumulative_savings.append(total_savings)

    return LeaseResult(
        monthly_payment_year1=monthly_payment,
        annual_escalator=annual_escalator,
        lease_term_years=lease_term_years,
        year1_savings=savings_by_year[0] if savings_by_year else 0,
        lifetime_savings=total_savings,
        payments_by_year=payments_by_year,
        savings_by_year=savings_by_year,
        cumulative_savings=cumulative_savings
    )


def calculate_ppa(
    ppa_rate: float,
    annual_escalator: float,
    contract_term: int,
    annual_production_kwh: float,
    electricity_rate: float,
    analysis_years: int = 25
) -> PPAResult:
    """
    Calculate financials for Power Purchase Agreement (PPA).

    Args:
        ppa_rate: PPA rate ($/kWh)
        annual_escalator: Annual rate escalator (e.g., 0.02 for 2%)
        contract_term: Contract term in years
        annual_production_kwh: Predicted annual production
        electricity_rate: Utility electricity rate ($/kWh)
        analysis_years: Number of years to analyze

    Returns:
        PPAResult with cost and savings projections
    """
    ppa_costs_by_year = []
    utility_costs_by_year = []
    savings_by_year = []
    cumulative_savings = []
    total_savings = 0

    for year in range(1, analysis_years + 1):
        # Production with degradation
        degradation_factor = (1 - 0.005) ** (year - 1)
        year_production = annual_production_kwh * degradation_factor

        # PPA rate with escalator (only during contract term)
        if year <= contract_term:
            ppa_escalator_factor = (1 + annual_escalator) ** (year - 1)
            year_ppa_rate = ppa_rate * ppa_escalator_factor
            year_ppa_cost = year_production * year_ppa_rate
        else:
            # After contract ends, assume ownership or market rate
            year_ppa_cost = 0

        # Utility rate with inflation
        utility_escalator_factor = (1 + 0.02) ** (year - 1)
        year_utility_rate = electricity_rate * utility_escalator_factor
        year_utility_cost = year_production * year_utility_rate

        # Savings = what you would have paid utility - what you pay PPA
        year_savings = year_utility_cost - year_ppa_cost
        total_savings += year_savings

        ppa_costs_by_year.append(year_ppa_cost)
        utility_costs_by_year.append(year_utility_cost)
        savings_by_year.append(year_savings)
        cumulative_savings.append(total_savings)

    return PPAResult(
        ppa_rate=ppa_rate,
        utility_rate=electricity_rate,
        annual_escalator=annual_escalator,
        contract_term=contract_term,
        year1_savings=savings_by_year[0] if savings_by_year else 0,
        lifetime_savings=total_savings,
        ppa_costs_by_year=ppa_costs_by_year,
        utility_costs_by_year=utility_costs_by_year,
        savings_by_year=savings_by_year,
        cumulative_savings=cumulative_savings
    )


def estimate_usage_from_bill(monthly_bill: float, electricity_rate: float) -> float:
    """
    Estimate annual electricity usage from monthly bill.

    Args:
        monthly_bill: Average monthly electricity bill ($)
        electricity_rate: Electricity rate ($/kWh)

    Returns:
        Estimated annual usage in kWh
    """
    if electricity_rate <= 0:
        return 0
    monthly_usage = monthly_bill / electricity_rate
    return monthly_usage * 12


def calculate_offset_percentage(
    annual_production_kwh: float,
    annual_usage_kwh: float
) -> float:
    """
    Calculate what percentage of electricity usage is offset by solar.

    Args:
        annual_production_kwh: Annual solar production
        annual_usage_kwh: Annual electricity usage

    Returns:
        Offset percentage (0-100+, can exceed 100 if overproducing)
    """
    if annual_usage_kwh <= 0:
        return 0
    return (annual_production_kwh / annual_usage_kwh) * 100
