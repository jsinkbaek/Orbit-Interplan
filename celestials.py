import numpy as np
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body_barycentric, get_body_barycentric_posvel, get_body, get_moon


class CelestialBody:
    solar_system_ephemeris.set('de432s')

    def __init__(self, name, mass, kind, parent, a):
        self.name = name
        self.mass = mass
        self.kind = kind
        self.parent = parent
        self.a = a

    def get_barycentric(self, t):
        pos = get_body_barycentric(self.name, t).to_value      # cartesian representation in km
        return pos

    def get_barycentric_vel(self, t):
        vel = get_body_barycentric_posvel(self.name, t).to_value
        return vel

    def sphere_of_influence(self):
        if self.parent is not None:
            r_soi = self.a * (self.mass / self.parent.mass) ** (2/5)
            return r_soi
        else:
            raise AttributeError('sphere_of_influence. body of name ', self.name, ' has no parent')


class CelestialGroup:

    def __init__(self, *args):
        self.objects = np.empty((0, ), dtype=type(CelestialBody))
        self.names = np.empty((0, ))
        self.mass = np.empty((0, ))
        self.parents = np.empty((0, ))
        self.kinds = np.empty((0, ))
        for arg in args:
            if type(arg) is not type(CelestialBody):
                raise TypeError('CelestialGroup __init__. Type of argument is not CelestialBody.')
            self.objects = np.append(self.objects, arg)
            self.names = np.append(self.names, arg.name)
            self.mass = np.append(self.mass, arg.mass)
            self.parents = np.append(self.parents, arg.parent)
            self.kinds = np.append(self.kinds, arg.kind)
        self.n = len(self.names)

    def ephemeris(self, t):
        pos = np.empty((3, self.n))
        for i in range(0, self.n):
            pos[i] = self.objects[i].get_barycentric(t)
        return pos

    def get_body(self, name):
        return self.objects[np.where(self.names == name)]


