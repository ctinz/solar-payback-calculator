"""Utility modules for Solar Payback Calculator."""

from .api_calls import (
    geocode_address,
    get_building_insights,
    get_solar_imagery_url,
    get_nrel_solar_resource,
    extract_best_roof_segment,
    GeocodingResult,
    SolarInsightsResult
)

from .financial_calcs import (
    calculate_cash_purchase,
    calculate_loan,
    calculate_lease,
    calculate_ppa,
    estimate_usage_from_bill,
    calculate_offset_percentage,
    CashPurchaseResult,
    LoanResult,
    LeaseResult,
    PPAResult
)

from .state_data import (
    ELECTRICITY_RATES,
    STATE_NAMES,
    FEDERAL_ITC_RATE,
    STATE_INCENTIVES,
    DEFAULT_COST_PER_WATT,
    get_electricity_rate,
    get_state_incentive,
    estimate_state_rebate,
    detect_state_from_address
)
