"""
Geocoding services for PhotoPipe.

Uses geopy with Nominatim (OpenStreetMap) for free geocoding.
"""

import time
from typing import Optional

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

from photopipe.models import Location, LocationAccuracy


# Rate limiting: Nominatim requires max 1 request per second
_last_request_time = 0.0

# Manual caches (not lru_cache): only successes are stored, so a transient
# network failure can't permanently mark a location un-geocodable.
_location_cache: dict[tuple[str, str], Location] = {}
_components_cache: dict[tuple[float, float], dict] = {}


def _rate_limit():
    """Enforce rate limiting for Nominatim API."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)
    _last_request_time = time.time()


def get_geocoder() -> Nominatim:
    """Get configured Nominatim geocoder instance."""
    return Nominatim(
        user_agent="photopipe/0.1.0",
        timeout=10,
    )


def geocode_location(location_text: str, country_hint: str = "USA") -> Optional[Location]:
    """
    Convert user-provided location text to coordinates.

    Args:
        location_text: Location description (e.g., "Toledo, OH" or "123 Main St, Toledo, OH")
        country_hint: Country context for disambiguation

    Returns:
        Location object with coordinates, or None if geocoding failed
    """
    if not location_text or not location_text.strip():
        return None

    cache_key = (location_text.strip().lower(), country_hint)
    cached = _location_cache.get(cache_key)
    if cached is not None:
        return cached

    geolocator = get_geocoder()

    # Try original query first
    queries = [
        location_text,
        f"{location_text}, {country_hint}",
    ]

    for query in queries:
        # One retry on timeout for the SAME query before moving on
        for _attempt in range(2):
            _rate_limit()
            try:
                result = geolocator.geocode(query, exactly_one=True, addressdetails=True)
            except GeocoderTimedOut:
                time.sleep(2)
                continue
            except GeocoderServiceError:
                break
            except Exception:
                break

            if result:
                # Determine accuracy based on result type
                accuracy = LocationAccuracy.APPROXIMATE
                if result.raw.get("addresstype") in ("house", "building", "place"):
                    accuracy = LocationAccuracy.EXACT
                elif result.raw.get("addresstype") in ("city", "town", "village"):
                    accuracy = LocationAccuracy.REGION

                location = Location(
                    description=location_text,
                    address=result.address,
                    latitude=result.latitude,
                    longitude=result.longitude,
                    accuracy=accuracy,
                )
                _location_cache[cache_key] = location
                return location
            break  # query resolved to nothing — try the next query form

    return None


def reverse_geocode(latitude: float, longitude: float) -> Optional[str]:
    """
    Get address from coordinates.

    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate

    Returns:
        Address string, or None if reverse geocoding failed
    """
    _rate_limit()
    geolocator = get_geocoder()

    try:
        result = geolocator.reverse((latitude, longitude), exactly_one=True)
        if result:
            return result.address
    except Exception:
        pass

    return None


def parse_location_components(location: Location) -> dict:
    """
    Parse location into IPTC-compatible components.

    Args:
        location: Location object

    Returns:
        Dictionary with city, state, country fields
    """
    components = {
        "city": None,
        "state": None,
        "country": None,
    }

    if not location.address:
        return components

    # Finalizing a batch calls this once per photo with the SAME batch
    # location — cache per coordinate so a 300-photo batch issues one
    # Nominatim request, not 300 identical ones.
    cache_key = (location.latitude, location.longitude)
    cached = _components_cache.get(cache_key)
    if cached is not None:
        return dict(cached)

    # Try to extract from reverse geocoded address
    _rate_limit()
    geolocator = get_geocoder()

    try:
        result = geolocator.reverse(
            (location.latitude, location.longitude),
            exactly_one=True,
            addressdetails=True,
        )

        if result and result.raw.get("address"):
            addr = result.raw["address"]
            components["city"] = addr.get("city") or addr.get("town") or addr.get("village")
            components["state"] = addr.get("state")
            components["country"] = addr.get("country")
            _components_cache[cache_key] = dict(components)

    except Exception:
        # Fall back to parsing the address string (not cached — the network
        # lookup may succeed next time)
        parts = location.address.split(",")
        if len(parts) >= 3:
            components["city"] = parts[-3].strip()
            components["state"] = parts[-2].strip()
            components["country"] = parts[-1].strip()
        elif len(parts) >= 2:
            components["city"] = parts[-2].strip()
            components["country"] = parts[-1].strip()

    return components


def validate_coordinates(latitude: float, longitude: float) -> bool:
    """
    Check if coordinates are valid.

    Args:
        latitude: Latitude value
        longitude: Longitude value

    Returns:
        True if coordinates are valid
    """
    return -90 <= latitude <= 90 and -180 <= longitude <= 180


def clear_geocoding_cache():
    """Clear the geocoding caches."""
    _location_cache.clear()
    _components_cache.clear()
