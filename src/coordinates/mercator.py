import math
from ..utils.constants import EARTH_RADIUS

def lat_to_y(lat_deg):
    """Convert latitude to Y coordinate in Mercator projection"""
    lat_rad = math.radians(lat_deg)
    y = math.log(math.tan(math.pi/4 + lat_rad/2))
    return y * EARTH_RADIUS

def lng_to_x(lng_deg):
    """Convert longitude to X coordinate in Mercator projection"""
    lng_rad = math.radians(lng_deg)
    return lng_rad * EARTH_RADIUS

def y_to_lat(y):
    """Convert Y coordinate back to latitude"""
    y = y / EARTH_RADIUS
    lat_rad = 2 * math.atan(math.exp(y)) - math.pi/2
    return math.degrees(lat_rad)

def x_to_lng(x):
    """Convert X coordinate back to longitude"""
    lng_rad = x / EARTH_RADIUS
    return math.degrees(lng_rad)
