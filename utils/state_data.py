"""
State-specific electricity rates and solar incentives for New England.
"""

# Average retail electricity rates ($/kWh) - as of 2024
# Source: EIA State Electricity Profiles
ELECTRICITY_RATES = {
    'MA': 0.26,  # Massachusetts - highest in region
    'CT': 0.24,  # Connecticut
    'RI': 0.24,  # Rhode Island
    'NH': 0.22,  # New Hampshire
    'VT': 0.20,  # Vermont
    'ME': 0.20,  # Maine
}

# State names for display
STATE_NAMES = {
    'MA': 'Massachusetts',
    'CT': 'Connecticut',
    'RI': 'Rhode Island',
    'NH': 'New Hampshire',
    'VT': 'Vermont',
    'ME': 'Maine',
}

# Federal Investment Tax Credit (ITC)
FEDERAL_ITC_RATE = 0.30  # 30% through 2032

# State incentives (simplified - actual programs have complex eligibility rules)
# Values represent approximate $/watt or description
STATE_INCENTIVES = {
    'MA': {
        'name': 'SMART Program',
        'description': 'Solar Massachusetts Renewable Target - performance-based incentive',
        'type': 'performance',
        'rate_per_kwh': 0.05,  # Approximate SMART compensation rate
        'term_years': 10,
        'notes': 'Rate varies by capacity block and utility territory'
    },
    'CT': {
        'name': 'RSIP (Ended) / SCEF',
        'description': 'Residential Solar Investment Program ended; Shared Clean Energy available',
        'type': 'rebate',
        'rebate_per_watt': 0.00,  # RSIP has ended
        'notes': 'RSIP closed to new applications. Net metering available.'
    },
    'RI': {
        'name': 'REF Program',
        'description': 'Renewable Energy Fund - rebates for solar installations',
        'type': 'rebate',
        'rebate_per_watt': 0.65,  # Approximate
        'max_rebate': 7000,
        'notes': 'Subject to funding availability'
    },
    'NH': {
        'name': 'Residential Solar Rebate',
        'description': 'Utility rebates through NH utilities',
        'type': 'rebate',
        'rebate_per_watt': 0.20,
        'max_rebate': 1000,
        'notes': 'Varies by utility; check with your provider'
    },
    'VT': {
        'name': 'Net Metering',
        'description': 'Strong net metering program with 1:1 credits',
        'type': 'net_metering',
        'rebate_per_watt': 0.00,
        'notes': 'No upfront rebate but excellent net metering rates'
    },
    'ME': {
        'name': 'Net Energy Billing',
        'description': 'Net energy billing at retail rate',
        'type': 'net_metering',
        'rebate_per_watt': 0.00,
        'notes': 'No state rebate; federal ITC applies'
    },
}

# Default system cost assumptions
DEFAULT_COST_PER_WATT = 3.00  # $/watt before incentives (installed cost)
DEFAULT_PANEL_WATTAGE = 400  # Watts per panel (modern panels)


def get_electricity_rate(state_code: str) -> float:
    """Get electricity rate for a state, with fallback to regional average."""
    return ELECTRICITY_RATES.get(state_code.upper(), 0.22)


def get_state_incentive(state_code: str) -> dict:
    """Get state incentive information."""
    return STATE_INCENTIVES.get(state_code.upper(), {
        'name': 'None',
        'description': 'No state incentive program found',
        'type': 'none',
        'rebate_per_watt': 0.00,
        'notes': 'Federal ITC still applies'
    })


def estimate_state_rebate(state_code: str, system_size_kw: float) -> float:
    """
    Estimate state rebate amount based on system size.

    Args:
        state_code: Two-letter state code
        system_size_kw: System size in kW

    Returns:
        Estimated rebate amount in dollars
    """
    incentive = get_state_incentive(state_code)
    system_size_watts = system_size_kw * 1000

    if incentive.get('type') == 'rebate':
        rebate = system_size_watts * incentive.get('rebate_per_watt', 0)
        max_rebate = incentive.get('max_rebate', float('inf'))
        return min(rebate, max_rebate)

    return 0.0


def estimate_annual_smart_payment(state_code: str, annual_production_kwh: float) -> float:
    """
    Estimate annual SMART program payment (MA only).

    Args:
        state_code: Two-letter state code
        annual_production_kwh: Annual production in kWh

    Returns:
        Estimated annual SMART payment in dollars
    """
    if state_code.upper() != 'MA':
        return 0.0

    incentive = get_state_incentive('MA')
    if incentive.get('type') == 'performance':
        return annual_production_kwh * incentive.get('rate_per_kwh', 0)

    return 0.0


def detect_state_from_address(address: str) -> str:
    """
    Attempt to detect state from address string.

    Args:
        address: Full address string

    Returns:
        Two-letter state code or empty string if not found
    """
    address_upper = address.upper()

    # Check for state codes
    for code in ELECTRICITY_RATES.keys():
        # Look for state code with common delimiters
        if f', {code}' in address_upper or f' {code} ' in address_upper:
            return code

    # Check for full state names
    state_name_to_code = {v.upper(): k for k, v in STATE_NAMES.items()}
    for name, code in state_name_to_code.items():
        if name in address_upper:
            return code

    return ''
