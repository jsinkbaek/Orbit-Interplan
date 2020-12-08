import numpy as np
import numpy.linalg as la
from scipy import constants as cnst


class Elliptical:
    def __init__(self, pos, vel, t, body):
        """
        Initialize using a known position and velocity at a given time.
        """
        # Find eccentricity vector
        p = pos*body.unitc.d
        v = vel*body.unitc.v
        h = np.cross(p, v)
        mu = cnst.G*body.mass*body.unitc.m
        e_vec = p/la.norm(p) - (np.cross(v, h))/mu
        # Find semi-major axis a
        a = mu*la.norm(p)/(2*mu - la.norm(p)*(la.norm(v**2)))
        # Find f2_vec
        f2_vec = 2*a*e_vec

        # Reorder to x,y coordinates instead of x,y,z without loss of generality:
