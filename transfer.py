from spacecraft import SpaceCraft
from celestials import CelestialBody, CelestialGroup
import numpy as np
import numpy.linalg as la
from scipy import constants as cnst


class Hohmann:
    def __init__(self, scraft):
        self.spacecraft = scraft

    def simple(self, r1=None, r2=None, body=None):
        if r1 is None:
            r1 = la.norm(self.spacecraft.pos, axis=0)
        if body is None:
            body = self.spacecraft.get_current_body(self.spacecraft.system_bodies)
        if r2 is None:
            la.norm(body.get_barycentric(self.spacecraft.t), axis=0)
        dv1 = np.sqrt(cnst.G * body.mass / r1) * (np.sqrt(2*r2/(r1+r2)) - 1)
        dv2 = np.sqrt(cnst.G * body.mass / r2) * (1 - np.sqrt(2*r1/(r1+r2)))
        dv = dv1 + dv2
        tH = np.pi * np.sqrt((r1+r2)**3 / (8*cnst.G*body.mass))
        return dv, tH

    def angular_alignment(self, target, parent, r1=None):
        """
        For calculating angular alignment between target planet for interplanetary Hohmann transfer rendezvous.
        Assumes no inclination difference between the spacecraft and target.
        """
        if r1 is None:
            r1 = la.norm(self.spacecraft.pos, axis=0)
        r2 = la.norm(target.get_barycentric(self.spacecraft.t), axis=0)
        target_angvel = np.sqrt(cnst.G * parent.mass / r2**3)
        tH = np.pi * np.sqrt((r1+r2)**3 / (8*cnst.G*parent.mass))
        # Angular alignment alpha in radians
        alpha = np.pi - target_angvel * tH
        return alpha, tH


class Rendezvous:
    def __init__(self, scraft, target, parent):
        self.spacecraft = scraft
        self.hohmann = Hohmann(scraft)
        self.target = target
        self.parent = parent        # parent of target (f.ex. Sun if target is Jupiter)

    def relative_angle_xy(self, t):
        """
        Finds xy relative angle between spacecraft and target. Most useful for orbits close to the xy-plane.
        """
        pos_target = self.target.get_barycentric(t)
        y_target, x_target = pos_target[1], pos_target[0]
        angle_target = np.arctan(y_target/x_target)
        y_scraft, x_scraft = self.spacecraft.pos[1], self.spacecraft.pos[0]
        angle_scraft = np.arctan(y_scraft/x_scraft)
        return angle_target - angle_scraft

    def initialburn_simple:
        """
        Approximates time for initial burn for interplanetary Hohmann transfer using relative angle in the xy-plane.
        (Only good for low inclination of target, and un-inclined periastron/apoastron of spacecraft)
        """
