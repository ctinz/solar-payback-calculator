"""
Google API integration for solar production prediction.
Handles Geocoding API and Solar API calls.
"""

import requests
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class GeocodingResult:
    """Result from geocoding an address."""
    latitude: float
    longitude: float
    formatted_address: str
    state_code: str
    success: bool
    error: Optional[str] = None


@dataclass
class SolarInsightsResult:
    """Result from Google Solar API building insights."""
    max_panel_count: int
    max_capacity_kw: float
    max_sunshine_hours: float
    roof_area_m2: float
    panel_configs: list
    roof_segments: list
    imagery_date: Optional[str]
    imagery_url: Optional[str]
    success: bool
    error: Optional[str] = None
    raw_data: Optional[Dict] = None


def geocode_address(address: str, api_key: str) -> GeocodingResult:
    """
    Convert street address to coordinates using Google Geocoding API.

    Args:
        address: Street address to geocode
        api_key: Google Geocoding API key

    Returns:
        GeocodingResult with coordinates and metadata
    """
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": api_key
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data['status'] == 'OK':
            result = data['results'][0]
            location = result['geometry']['location']

            # Extract state code from address components
            state_code = ''
            for component in result.get('address_components', []):
                if 'administrative_area_level_1' in component.get('types', []):
                    state_code = component.get('short_name', '')
                    break

            return GeocodingResult(
                latitude=location['lat'],
                longitude=location['lng'],
                formatted_address=result.get('formatted_address', address),
                state_code=state_code,
                success=True
            )
        else:
            return GeocodingResult(
                latitude=0, longitude=0, formatted_address='',
                state_code='', success=False,
                error=f"Geocoding failed: {data['status']}"
            )

    except requests.exceptions.Timeout:
        return GeocodingResult(
            latitude=0, longitude=0, formatted_address='',
            state_code='', success=False,
            error="Request timed out. Please try again."
        )
    except requests.exceptions.RequestException as e:
        return GeocodingResult(
            latitude=0, longitude=0, formatted_address='',
            state_code='', success=False,
            error=f"Network error: {str(e)}"
        )
    except Exception as e:
        return GeocodingResult(
            latitude=0, longitude=0, formatted_address='',
            state_code='', success=False,
            error=f"Unexpected error: {str(e)}"
        )


def get_building_insights(
    latitude: float,
    longitude: float,
    api_key: str,
    required_quality: str = "LOW"
) -> SolarInsightsResult:
    """
    Get building solar insights from Google Solar API.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
        api_key: Google Solar API key
        required_quality: Minimum imagery quality (LOW, MEDIUM, HIGH)

    Returns:
        SolarInsightsResult with solar potential data
    """
    base_url = "https://solar.googleapis.com/v1/buildingInsights:findClosest"
    params = {
        "location.latitude": latitude,
        "location.longitude": longitude,
        "requiredQuality": required_quality,
        "key": api_key
    }

    try:
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if 'solarPotential' not in data:
            return SolarInsightsResult(
                max_panel_count=0, max_capacity_kw=0, max_sunshine_hours=0,
                roof_area_m2=0, panel_configs=[], roof_segments=[],
                imagery_date=None, imagery_url=None, success=False,
                error="No solar potential data available for this location"
            )

        solar = data['solarPotential']
        panel_configs = solar.get('solarPanelConfigs', [])

        # Get optimal configuration (maximum panels)
        max_panels = 0
        max_capacity = 0
        if panel_configs:
            best_config = max(panel_configs, key=lambda x: x.get('panelsCount', 0))
            max_panels = best_config.get('panelsCount', 0)
            panel_wattage = solar.get('panelCapacityWatts', 400)
            max_capacity = (max_panels * panel_wattage) / 1000

        # Extract roof segment info
        roof_segments = solar.get('roofSegmentStats', [])

        # Get imagery info if available
        imagery_date = data.get('imageryDate', {})
        if imagery_date:
            imagery_date_str = f"{imagery_date.get('year', '')}-{imagery_date.get('month', ''):02d}-{imagery_date.get('day', ''):02d}"
        else:
            imagery_date_str = None

        return SolarInsightsResult(
            max_panel_count=max_panels,
            max_capacity_kw=max_capacity,
            max_sunshine_hours=solar.get('maxSunshineHoursPerYear', 0),
            roof_area_m2=solar.get('maxArrayAreaMeters2', 0),
            panel_configs=panel_configs,
            roof_segments=roof_segments,
            imagery_date=imagery_date_str,
            imagery_url=None,  # Would need separate API call for imagery
            success=True,
            raw_data=data
        )

    except requests.exceptions.Timeout:
        return SolarInsightsResult(
            max_panel_count=0, max_capacity_kw=0, max_sunshine_hours=0,
            roof_area_m2=0, panel_configs=[], roof_segments=[],
            imagery_date=None, imagery_url=None, success=False,
            error="Request timed out. Please try again."
        )
    except requests.exceptions.HTTPError as e:
        error_msg = f"API error: {e.response.status_code}"
        if e.response.status_code == 404:
            error_msg = "No solar data available for this location"
        elif e.response.status_code == 403:
            error_msg = "API key invalid or quota exceeded"
        return SolarInsightsResult(
            max_panel_count=0, max_capacity_kw=0, max_sunshine_hours=0,
            roof_area_m2=0, panel_configs=[], roof_segments=[],
            imagery_date=None, imagery_url=None, success=False,
            error=error_msg
        )
    except Exception as e:
        return SolarInsightsResult(
            max_panel_count=0, max_capacity_kw=0, max_sunshine_hours=0,
            roof_area_m2=0, panel_configs=[], roof_segments=[],
            imagery_date=None, imagery_url=None, success=False,
            error=f"Unexpected error: {str(e)}"
        )


def get_solar_imagery_url(
    latitude: float,
    longitude: float,
    api_key: str,
    image_type: str = "MASK"
) -> Optional[str]:
    """
    Get URL for solar imagery (roof mask, flux map, etc.).

    Args:
        latitude: Latitude
        longitude: Longitude
        api_key: Google Solar API key
        image_type: Type of image (MASK, DSM, RGB, ANNUAL_FLUX, MONTHLY_FLUX)

    Returns:
        URL string or None if unavailable
    """
    base_url = "https://solar.googleapis.com/v1/dataLayers:get"
    params = {
        "location.latitude": latitude,
        "location.longitude": longitude,
        "radiusMeters": 50,
        "view": image_type,
        "requiredQuality": "LOW",
        "key": api_key
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Return the appropriate URL based on image type
        if image_type == "RGB":
            return data.get('rgbUrl')
        elif image_type == "MASK":
            return data.get('maskUrl')
        elif image_type == "ANNUAL_FLUX":
            return data.get('annualFluxUrl')
        elif image_type == "DSM":
            return data.get('dsmUrl')

        return None

    except Exception:
        return None


def get_nrel_solar_resource(
    latitude: float,
    longitude: float,
    api_key: str = "DEMO_KEY"
) -> Optional[float]:
    """
    Get solar resource data from NREL PVWatts API.

    Args:
        latitude: Latitude
        longitude: Longitude
        api_key: NREL API key (DEMO_KEY works with rate limits)

    Returns:
        Annual AC output in kWh per kW installed, or None if failed
    """
    url = "https://developer.nrel.gov/api/pvwatts/v6.json"
    params = {
        'api_key': api_key,
        'lat': latitude,
        'lon': longitude,
        'system_capacity': 1,  # 1 kW for normalized output
        'azimuth': 180,        # South-facing
        'tilt': latitude,      # Optimal tilt ~ latitude
        'array_type': 1,       # Fixed (open rack)
        'module_type': 0,      # Standard
        'losses': 14           # System losses (%)
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data['outputs']['ac_annual']
    except Exception:
        return None


def extract_best_roof_segment(roof_segments: list) -> Dict[str, float]:
    """
    Extract the best roof segment for solar installation.

    Args:
        roof_segments: List of roof segment stats from Solar API

    Returns:
        Dict with tilt (pitch) and azimuth of best segment
    """
    if not roof_segments:
        return {'tilt': 20.0, 'azimuth': 180.0}  # Defaults

    # Find segment with most sunshine hours
    best_segment = max(
        roof_segments,
        key=lambda x: x.get('stats', {}).get('sunshineQuantiles', [0])[-1]
    )

    return {
        'tilt': best_segment.get('pitchDegrees', 20.0),
        'azimuth': best_segment.get('azimuthDegrees', 180.0)
    }
