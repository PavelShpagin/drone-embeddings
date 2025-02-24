import math
from ..utils.constants import EARTH_RADIUS
from ..api.elevation import get_elevation

def get_xyz_coordinates(lat, lng, altitude_above_ground=100):
    """Convert lat/lng to XYZ coordinates with elevation data"""
    ground_elevation = get_elevation(lat, lng)
    print(f"Ground elevation: {ground_elevation}")
    if ground_elevation is None:
        return None
    
    total_altitude = ground_elevation + altitude_above_ground
    
    lat_rad = math.radians(lat)
    lng_rad = math.radians(lng)
    
    x = (EARTH_RADIUS + total_altitude) * math.cos(lat_rad) * math.cos(lng_rad)
    y = (EARTH_RADIUS + total_altitude) * math.cos(lat_rad) * math.sin(lng_rad)
    z = (EARTH_RADIUS + total_altitude) * math.sin(lat_rad)
    
    return {
        'x': x,
        'y': y,
        'z': z,
        'ground_elevation': ground_elevation,
        'total_altitude': total_altitude
    }
