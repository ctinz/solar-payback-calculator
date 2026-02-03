"""
New England Solar Payback Calculator
Streamlit application for estimating solar production and financial returns.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Import utility modules
from utils.api_calls import (
    geocode_address,
    get_building_insights,
    extract_best_roof_segment
)
from utils.financial_calcs import (
    calculate_cash_purchase,
    calculate_loan,
    calculate_lease,
    calculate_ppa,
    estimate_usage_from_bill,
    calculate_offset_percentage
)
from utils.state_data import (
    ELECTRICITY_RATES,
    STATE_NAMES,
    FEDERAL_ITC_RATE,
    DEFAULT_COST_PER_WATT,
    get_electricity_rate,
    get_state_incentive,
    estimate_state_rebate
)

# Page configuration
st.set_page_config(
    page_title="New England Solar Calculator",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_PATH = Path(__file__).parent / "models" / "ridge_model.joblib"


def get_api_key() -> str:
    """Get Google API key from Streamlit secrets or environment variable."""
    # Try Streamlit secrets first (for deployment)
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

    # Fall back to environment variable
    import os
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return api_key

    # No key found
    st.error(
        "Google API key not found. Please add it to `.streamlit/secrets.toml` "
        "or set the `GOOGLE_API_KEY` environment variable."
    )
    st.stop()


@st.cache_resource
def load_model():
    """Load the trained Ridge model."""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error("Model file not found. Please run train_ridge_model.py first.")
        return None


def predict_production(model_package, features: dict) -> float:
    """
    Predict annual solar production using the trained model.

    Args:
        model_package: Loaded model package with model, scaler, and feature_cols
        features: Dictionary of feature values

    Returns:
        Predicted annual production in kWh
    """
    model = model_package['model']
    scaler = model_package['scaler']
    feature_cols = model_package['feature_cols']

    # Build feature vector in correct order
    X = np.array([[features.get(col, 0) for col in feature_cols]])

    # Scale and predict
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]

    return max(0, prediction)  # Ensure non-negative


def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        'step': 1,
        'address': '',
        'geocoding_result': None,
        'solar_insights': None,
        'system_size_kw': 0.0,
        'tilt': 20.0,
        'azimuth': 180.0,
        'shade': 0,
        'electricity_rate': 0.24,
        'state_code': 'CT',
        'annual_usage_kwh': 10000.0,
        'financing_type': 'Cash Purchase',
        'predicted_production': 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_step_indicator():
    """Render the step progress indicator."""
    steps = [
        "1. Property",
        "2. Solar Specs",
        "3. Usage",
        "4. Financing",
        "5. Results"
    ]

    cols = st.columns(5)
    for i, (col, step_name) in enumerate(zip(cols, steps), 1):
        if i < st.session_state.step:
            col.markdown(f"✅ **{step_name}**")
        elif i == st.session_state.step:
            col.markdown(f"🔵 **{step_name}**")
        else:
            col.markdown(f"⚪ {step_name}")

    st.divider()


def step1_property_input():
    """Step 1: Property address input and geocoding."""
    st.header("📍 Step 1: Enter Your Property Address")

    st.markdown("""
    Enter your street address to begin. We'll use Google's APIs to:
    - Locate your property
    - Analyze your roof for solar potential
    - Estimate optimal system configuration
    """)

    col1, col2 = st.columns([2, 1])

    with col1:
        address = st.text_input(
            "Street Address",
            value=st.session_state.address,
            placeholder="123 Main Street, Hartford, CT",
            help="Enter your full street address including city and state"
        )

        if st.button("🔍 Analyze Property", type="primary", use_container_width=True):
            if not address:
                st.error("Please enter an address.")
                return

            with st.spinner("Geocoding address..."):
                geo_result = geocode_address(address, get_api_key())

                if not geo_result.success:
                    st.error(f"Could not find address: {geo_result.error}")
                    return

                st.session_state.address = address
                st.session_state.geocoding_result = geo_result
                st.session_state.state_code = geo_result.state_code

                # Set default electricity rate based on state
                if geo_result.state_code in ELECTRICITY_RATES:
                    st.session_state.electricity_rate = ELECTRICITY_RATES[geo_result.state_code]

            with st.spinner("Fetching solar insights..."):
                solar_result = get_building_insights(
                    geo_result.latitude,
                    geo_result.longitude,
                    get_api_key()
                )

                if not solar_result.success:
                    st.warning(f"Limited solar data: {solar_result.error}")
                else:
                    st.session_state.solar_insights = solar_result

                    # Set defaults from solar insights
                    st.session_state.system_size_kw = solar_result.max_capacity_kw

                    # Extract best roof segment for tilt/azimuth
                    roof_info = extract_best_roof_segment(solar_result.roof_segments)
                    st.session_state.tilt = roof_info['tilt']
                    st.session_state.azimuth = roof_info['azimuth']

            st.success("✅ Property analyzed successfully!")
            st.session_state.step = 2
            st.rerun()

    with col2:
        # Show map if we have coordinates
        if st.session_state.geocoding_result:
            geo = st.session_state.geocoding_result
            st.markdown(f"**Found:** {geo.formatted_address}")

            # Create a simple map using plotly
            fig = go.Figure(go.Scattermapbox(
                lat=[geo.latitude],
                lon=[geo.longitude],
                mode='markers',
                marker=dict(size=14, color='red'),
                text=[geo.formatted_address]
            ))

            fig.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=geo.latitude, lon=geo.longitude),
                    zoom=17
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)


def step2_solar_attributes():
    """Step 2: Display and edit solar attributes."""
    st.header("☀️ Step 2: Solar System Specifications")

    if not st.session_state.geocoding_result:
        st.warning("Please complete Step 1 first.")
        if st.button("← Back to Step 1"):
            st.session_state.step = 1
            st.rerun()
        return

    geo = st.session_state.geocoding_result
    solar = st.session_state.solar_insights

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Google Solar API Data")

        if solar and solar.success:
            st.metric("Max Panel Count", f"{solar.max_panel_count} panels")
            st.metric("Max Roof Capacity", f"{solar.max_capacity_kw:.1f} kW")
            st.metric("Annual Sunshine", f"{solar.max_sunshine_hours:,.0f} hours")
            st.metric("Usable Roof Area", f"{solar.roof_area_m2:.0f} m²")

            if solar.imagery_date:
                st.caption(f"Imagery date: {solar.imagery_date}")
        else:
            st.info("Detailed solar data not available. Using defaults.")

    with col2:
        st.subheader("System Configuration")
        st.markdown("*Adjust these values based on your actual or planned system:*")

        # System size (key input for prediction)
        default_size = st.session_state.system_size_kw
        if default_size == 0:
            default_size = 8.0  # Reasonable default

        system_size = st.number_input(
            "System Size (kW)",
            min_value=1.0,
            max_value=50.0,
            value=float(default_size),
            step=0.5,
            help="DC capacity of your solar system in kilowatts"
        )
        st.session_state.system_size_kw = system_size

        # Tilt angle
        tilt = st.number_input(
            "Roof Tilt (degrees)",
            min_value=0.0,
            max_value=90.0,
            value=float(st.session_state.tilt),
            step=1.0,
            help="Angle of your roof from horizontal (flat = 0°)"
        )
        st.session_state.tilt = tilt

        # Azimuth
        azimuth = st.number_input(
            "Azimuth (degrees)",
            min_value=0.0,
            max_value=360.0,
            value=float(st.session_state.azimuth),
            step=5.0,
            help="Direction panels face (180° = due South)"
        )
        st.session_state.azimuth = azimuth

        # Shade level
        shade = st.selectbox(
            "Shading Level",
            options=[0, 1, 2, 3],
            index=st.session_state.shade,
            format_func=lambda x: ["None", "Slight", "Moderate", "Heavy"][x],
            help="Amount of shading on your roof"
        )
        st.session_state.shade = shade

    st.divider()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with col3:
        if st.button("Continue →", type="primary", use_container_width=True):
            if st.session_state.system_size_kw <= 0:
                st.error("Please enter a valid system size.")
            else:
                st.session_state.step = 3
                st.rerun()


def step3_electricity_usage():
    """Step 3: Electricity usage input."""
    st.header("⚡ Step 3: Your Electricity Usage")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Enter Your Usage")

        input_method = st.radio(
            "How would you like to enter your usage?",
            ["Monthly Bill ($)", "Annual Usage (kWh)"],
            horizontal=True
        )

        if input_method == "Monthly Bill ($)":
            monthly_bill = st.number_input(
                "Average Monthly Electric Bill",
                min_value=0.0,
                max_value=2000.0,
                value=150.0,
                step=10.0,
                format="%.0f"
            )

            # Calculate usage from bill
            annual_usage = estimate_usage_from_bill(
                monthly_bill,
                st.session_state.electricity_rate
            )
            st.session_state.annual_usage_kwh = annual_usage

            st.info(f"Estimated annual usage: **{annual_usage:,.0f} kWh**")

        else:
            annual_usage = st.number_input(
                "Annual Electricity Usage (kWh)",
                min_value=0.0,
                max_value=100000.0,
                value=st.session_state.annual_usage_kwh,
                step=500.0
            )
            st.session_state.annual_usage_kwh = annual_usage

    with col2:
        st.subheader("Electricity Rate")

        # State selection
        state_code = st.selectbox(
            "State",
            options=list(STATE_NAMES.keys()),
            index=list(STATE_NAMES.keys()).index(st.session_state.state_code)
                if st.session_state.state_code in STATE_NAMES else 0,
            format_func=lambda x: f"{STATE_NAMES[x]} ({x})"
        )
        st.session_state.state_code = state_code

        # Default rate for state
        default_rate = get_electricity_rate(state_code)

        electricity_rate = st.number_input(
            "Electricity Rate ($/kWh)",
            min_value=0.05,
            max_value=0.50,
            value=default_rate,
            step=0.01,
            format="%.3f"
        )
        st.session_state.electricity_rate = electricity_rate

        # Show state incentive info
        incentive = get_state_incentive(state_code)
        st.markdown(f"**State Program:** {incentive['name']}")
        st.caption(incentive['description'])

    st.divider()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with col3:
        if st.button("Continue →", type="primary", use_container_width=True):
            st.session_state.step = 4
            st.rerun()


def step4_financing():
    """Step 4: Financing scenario selection."""
    st.header("💰 Step 4: Choose Your Financing Option")

    financing_type = st.radio(
        "Select Financing Type",
        ["Cash Purchase", "Solar Loan", "Lease", "PPA"],
        horizontal=True,
        index=["Cash Purchase", "Solar Loan", "Lease", "PPA"].index(
            st.session_state.financing_type
        )
    )
    st.session_state.financing_type = financing_type

    st.divider()

    # Store financing parameters in session state
    if 'financing_params' not in st.session_state:
        st.session_state.financing_params = {}

    if financing_type == "Cash Purchase":
        st.subheader("💵 Cash Purchase Details")

        col1, col2 = st.columns(2)

        with col1:
            # System cost
            default_cost = st.session_state.system_size_kw * 1000 * DEFAULT_COST_PER_WATT
            system_cost = st.number_input(
                "Total System Cost",
                min_value=1000.0,
                max_value=200000.0,
                value=default_cost,
                step=500.0,
                format="%.0f",
                help=f"Default estimate: {st.session_state.system_size_kw:.1f} kW × $3.00/W"
            )

            federal_itc = st.slider(
                "Federal ITC (%)",
                min_value=0,
                max_value=40,
                value=30,
                help="Federal Investment Tax Credit (30% through 2032)"
            )

        with col2:
            state_rebate = st.number_input(
                "State Rebate / Incentive ($)",
                min_value=0.0,
                max_value=50000.0,
                value=estimate_state_rebate(
                    st.session_state.state_code,
                    st.session_state.system_size_kw
                ),
                step=100.0,
                format="%.0f"
            )

        st.session_state.financing_params = {
            'system_cost': system_cost,
            'federal_itc': federal_itc / 100,
            'state_rebate': state_rebate
        }

    elif financing_type == "Solar Loan":
        st.subheader("🏦 Solar Loan Details")

        col1, col2 = st.columns(2)

        with col1:
            # Calculate default loan amount (after ITC)
            gross_cost = st.session_state.system_size_kw * 1000 * DEFAULT_COST_PER_WATT
            default_loan = gross_cost * 0.7  # After 30% ITC

            loan_amount = st.number_input(
                "Loan Amount",
                min_value=1000.0,
                max_value=200000.0,
                value=default_loan,
                step=500.0,
                format="%.0f"
            )

            interest_rate = st.slider(
                "Interest Rate (%)",
                min_value=0.0,
                max_value=15.0,
                value=6.0,
                step=0.25
            )

        with col2:
            loan_term = st.selectbox(
                "Loan Term (years)",
                options=[5, 7, 10, 12, 15, 20, 25],
                index=5  # Default to 20 years
            )

        st.session_state.financing_params = {
            'loan_amount': loan_amount,
            'interest_rate': interest_rate / 100,
            'loan_term': loan_term
        }

    elif financing_type == "Lease":
        st.subheader("📋 Solar Lease Details")

        col1, col2 = st.columns(2)

        with col1:
            monthly_payment = st.number_input(
                "Monthly Lease Payment",
                min_value=0.0,
                max_value=500.0,
                value=100.0,
                step=10.0,
                format="%.0f"
            )

            lease_term = st.selectbox(
                "Lease Term (years)",
                options=[10, 15, 20, 25],
                index=2  # Default to 20 years
            )

        with col2:
            escalator = st.slider(
                "Annual Escalator (%)",
                min_value=0.0,
                max_value=5.0,
                value=2.0,
                step=0.5,
                help="Annual increase in lease payment"
            )

        st.session_state.financing_params = {
            'monthly_payment': monthly_payment,
            'lease_term': lease_term,
            'escalator': escalator / 100
        }

    elif financing_type == "PPA":
        st.subheader("⚡ Power Purchase Agreement (PPA)")

        col1, col2 = st.columns(2)

        with col1:
            ppa_rate = st.number_input(
                "PPA Rate ($/kWh)",
                min_value=0.05,
                max_value=0.30,
                value=0.12,
                step=0.01,
                format="%.3f"
            )

            contract_term = st.selectbox(
                "Contract Term (years)",
                options=[10, 15, 20, 25],
                index=2
            )

        with col2:
            escalator = st.slider(
                "Annual Rate Escalator (%)",
                min_value=0.0,
                max_value=5.0,
                value=2.0,
                step=0.5
            )

        st.session_state.financing_params = {
            'ppa_rate': ppa_rate,
            'contract_term': contract_term,
            'escalator': escalator / 100
        }

    st.divider()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 3
            st.rerun()
    with col3:
        if st.button("Calculate Results →", type="primary", use_container_width=True):
            st.session_state.step = 5
            st.rerun()


def step5_results(model_package):
    """Step 5: Results dashboard."""
    st.header("📊 Step 5: Your Solar Analysis Results")

    # Get all session state values
    geo = st.session_state.geocoding_result

    if not geo:
        st.warning("Please complete all previous steps.")
        if st.button("← Start Over"):
            st.session_state.step = 1
            st.rerun()
        return

    # Prepare features for prediction
    features = {
        'latitude': geo.latitude,
        'longitude': geo.longitude,
        'dc_capacity_kW': st.session_state.system_size_kw,
        'tilt': st.session_state.tilt,
        'azimuth': st.session_state.azimuth,
        'shade': st.session_state.shade,
        'azimuth_deviation': abs(st.session_state.azimuth - 180),
        'tilt_deviation': abs(st.session_state.tilt - geo.latitude)
    }

    # Make prediction
    predicted_production = predict_production(model_package, features)
    st.session_state.predicted_production = predicted_production

    # Calculate offset
    offset_pct = calculate_offset_percentage(
        predicted_production,
        st.session_state.annual_usage_kwh
    )

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Annual Production",
            f"{predicted_production:,.0f} kWh",
            help="Predicted annual solar production"
        )

    with col2:
        st.metric(
            "Monthly Average",
            f"{predicted_production/12:,.0f} kWh"
        )

    with col3:
        st.metric(
            "Usage Offset",
            f"{min(offset_pct, 100):.0f}%",
            help="Percentage of your usage covered by solar"
        )

    with col4:
        annual_value = predicted_production * st.session_state.electricity_rate
        st.metric(
            "Annual Value",
            f"${annual_value:,.0f}",
            help="Value of electricity produced at current rates"
        )

    st.divider()

    # Financial analysis based on selected financing type
    financing_type = st.session_state.financing_type
    params = st.session_state.financing_params

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"💰 {financing_type} Analysis")

        if financing_type == "Cash Purchase":
            result = calculate_cash_purchase(
                system_size_kw=st.session_state.system_size_kw,
                cost_per_watt=params['system_cost'] / (st.session_state.system_size_kw * 1000),
                federal_itc_rate=params['federal_itc'],
                state_rebate=params['state_rebate'],
                annual_production_kwh=predicted_production,
                electricity_rate=st.session_state.electricity_rate
            )

            st.metric("Gross System Cost", f"${result.gross_cost:,.0f}")
            st.metric("Federal ITC (30%)", f"-${result.federal_itc:,.0f}")
            st.metric("State Rebate", f"-${result.state_rebate:,.0f}")
            st.metric("**Net Cost**", f"${result.net_cost:,.0f}")
            st.divider()
            st.metric("Year 1 Savings", f"${result.annual_savings:,.0f}")
            st.metric("Simple Payback", f"{result.simple_payback_years:.1f} years")
            st.metric("25-Year ROI", f"{result.roi_25_year:.0f}%")

            cumulative_savings = result.cumulative_savings

        elif financing_type == "Solar Loan":
            result = calculate_loan(
                loan_amount=params['loan_amount'],
                interest_rate=params['interest_rate'],
                loan_term_years=params['loan_term'],
                annual_production_kwh=predicted_production,
                electricity_rate=st.session_state.electricity_rate
            )

            st.metric("Loan Amount", f"${result.loan_amount:,.0f}")
            st.metric("Monthly Payment", f"${result.monthly_payment:,.0f}")
            st.metric("Annual Payment", f"${result.annual_payment:,.0f}")
            st.metric("Total Interest", f"${result.total_interest:,.0f}")
            st.divider()
            st.metric("Annual Solar Value", f"${result.annual_solar_value:,.0f}")
            st.metric("Net Annual Cash Flow", f"${result.net_annual_savings:,.0f}",
                     delta="During loan term")
            st.metric("Break-even Year", f"Year {result.breakeven_year}")

            cumulative_savings = result.cumulative_savings

        elif financing_type == "Lease":
            result = calculate_lease(
                monthly_payment=params['monthly_payment'],
                annual_escalator=params['escalator'],
                lease_term_years=params['lease_term'],
                annual_production_kwh=predicted_production,
                electricity_rate=st.session_state.electricity_rate
            )

            st.metric("Monthly Payment (Year 1)", f"${result.monthly_payment_year1:,.0f}")
            st.metric("Annual Escalator", f"{result.annual_escalator*100:.1f}%")
            st.metric("Lease Term", f"{result.lease_term_years} years")
            st.divider()
            st.metric("Year 1 Savings", f"${result.year1_savings:,.0f}")
            st.metric("Lifetime Savings", f"${result.lifetime_savings:,.0f}")

            cumulative_savings = result.cumulative_savings

        else:  # PPA
            result = calculate_ppa(
                ppa_rate=params['ppa_rate'],
                annual_escalator=params['escalator'],
                contract_term=params['contract_term'],
                annual_production_kwh=predicted_production,
                electricity_rate=st.session_state.electricity_rate
            )

            st.metric("PPA Rate", f"${result.ppa_rate:.3f}/kWh")
            st.metric("Utility Rate", f"${result.utility_rate:.3f}/kWh")
            st.metric("Rate Difference", f"${result.utility_rate - result.ppa_rate:.3f}/kWh")
            st.divider()
            st.metric("Year 1 Savings", f"${result.year1_savings:,.0f}")
            st.metric(f"{result.contract_term}-Year Savings", f"${result.lifetime_savings:,.0f}")

            cumulative_savings = result.cumulative_savings

    with col2:
        st.subheader("📈 Cumulative Savings Over Time")

        years = list(range(1, len(cumulative_savings) + 1))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=years,
            y=cumulative_savings,
            mode='lines+markers',
            name='Cumulative Savings',
            line=dict(color='green', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)'
        ))

        # Add break-even line
        fig.add_hline(y=0, line_dash="dash", line_color="red",
                     annotation_text="Break-even")

        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Cumulative Savings ($)",
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Model confidence note
    metrics = model_package.get('metrics', {})
    st.caption(
        f"Model Performance: R² = {metrics.get('test_r2', 0):.3f}, "
        f"MAE = {metrics.get('test_mae', 0):,.0f} kWh"
    )

    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back to Financing", use_container_width=True):
            st.session_state.step = 4
            st.rerun()
    with col3:
        if st.button("🔄 Start New Analysis", use_container_width=True):
            # Reset session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def main():
    """Main application entry point."""

    # Initialize session state
    initialize_session_state()

    # Load model
    model_package = load_model()
    if model_package is None:
        st.stop()

    # Sidebar
    with st.sidebar:
        st.title("☀️ Solar Calculator")
        st.markdown("**New England Solar Payback Calculator**")
        st.divider()

        st.markdown("### About")
        st.markdown("""
        This tool helps you estimate:
        - Solar production for your property
        - Financial returns under different scenarios
        - Payback period and lifetime savings

        **Powered by:**
        - Google Solar API
        - Ridge Regression ML Model
        - NREL PVWatts data
        """)

        st.divider()

        st.markdown("### Model Info")
        metrics = model_package.get('metrics', {})
        st.markdown(f"- **Test R²:** {metrics.get('test_r2', 0):.3f}")
        st.markdown(f"- **Test MAE:** {metrics.get('test_mae', 0):,.0f} kWh")

        st.divider()

        # Quick navigation
        st.markdown("### Quick Jump")
        step = st.radio(
            "Go to step:",
            [1, 2, 3, 4, 5],
            index=st.session_state.step - 1,
            format_func=lambda x: [
                "1. Property", "2. Solar Specs", "3. Usage",
                "4. Financing", "5. Results"
            ][x-1],
            label_visibility="collapsed"
        )
        if step != st.session_state.step:
            st.session_state.step = step
            st.rerun()

    # Main content
    st.title("☀️ New England Solar Payback Calculator")

    # Step indicator
    render_step_indicator()

    # Render current step
    if st.session_state.step == 1:
        step1_property_input()
    elif st.session_state.step == 2:
        step2_solar_attributes()
    elif st.session_state.step == 3:
        step3_electricity_usage()
    elif st.session_state.step == 4:
        step4_financing()
    elif st.session_state.step == 5:
        step5_results(model_package)


if __name__ == "__main__":
    main()
