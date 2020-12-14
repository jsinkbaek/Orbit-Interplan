import numpy as np
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body_barycentric, get_body_barycentric_posvel, get_body, get_moon


class CelestialBody:
    """
    Class of objects called CelestialBody. Used to define celestial bodies necessary to calculate trajectory of
    spacecraft.
    """
    solar_system_ephemeris.set('de430')

    def __init__(self, name, mass, kind, parent, a, unit_converter):
        self.name = name
        self.mass = mass
        self.kind = kind
        self.parent = parent
        self.a = a           # semi-major axis in m
        self.unitc = unit_converter

    def get_barycentric(self, t, t0=None, tformat=None):
        if tformat is None:
            tformat = self.unitc.tname
        if tformat == 'jd' and t0 is None:
            t0 = 2451545        # jan 1 2000 in jd
        t_ = Time(t+t0, format=tformat)
        pos = get_body_barycentric(self.name, t_).xyz.to(self.unitc.dname).to_value()  # cartesian representation
        return pos

    def get_barycentric_vel(self, t, t0=None, tformat=None):
        if tformat is None:
            tformat = self.unitc.tname
        if tformat == 'jd' and t0 is None:
            t0 = 2451545        # jan 1 2000 in jd
        t_ = Time(t + t0, format=tformat)
        vel = get_body_barycentric_posvel(self.name, t_)[1].xyz.to(self.unitc.vname).to_value()
        return vel

    def sphere_of_influence(self):
        if self.parent is not None:
            mass = self.mass * self.unitc.m
            pmass = self.parent.mass * self.parent.unitc.m
            r_soi = self.a * (mass / pmass) ** (2/5)
            return r_soi
        else:
            raise AttributeError('sphere_of_influence. body of name ', self.name, ' has no parent')


class CelestialGroup:
    """
    Class used to group multiple CelestialObjects together for convenient storage and access of values.
    """

    def __init__(self, *args):
        self.objects = np.empty((0, ), dtype=type(CelestialBody))
        self.names = np.empty((0, ))
        self.mass = np.empty((0, ))
        self.parents = np.empty((0, ))
        self.kinds = np.empty((0, ))
        for arg in args:
            if type(arg) is not CelestialBody:
                raise TypeError('CelestialGroup __init__. Type of argument is not CelestialBody.')
            self.objects = np.append(self.objects, arg)
            self.names = np.append(self.names, arg.name)
            self.mass = np.append(self.mass, arg.mass)
            self.parents = np.append(self.parents, arg.parent)
            self.kinds = np.append(self.kinds, arg.kind)
        self.n = len(self.names)
        self.unitc = self.objects[0].unitc

    def ephemeris(self, t, t0=None, tformat=None):
        if tformat is None:
            tformat = self.unitc.tname
        if tformat == 'jd' and t0 is None:
            t0 = 2451545        # jan 1 2000 in jd
        pos = np.empty((3, self.n))
        for i in range(0, self.n):
            pos[:, i] = self.objects[i].get_barycentric(t, t0, tformat)
        return pos

    def get_body(self, name):
        return self.objects[np.where(self.names == name)][0]


