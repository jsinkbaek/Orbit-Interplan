import numpy as np
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body_barycentric, get_body, get_moon


class CelestialObject:
    solar_system_ephemeris.set('de432s')

    def __init__(self, name, mass, kind, parent):
        self.name = name
        self.mass = mass
        self.kind = kind
        self.parent = parent

    def get_barycentric(self, t):
        pos = get_body_barycentric(self.name, t).to_value      # cartesian representation in km
        return pos
