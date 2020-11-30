import numpy as np
from celestials import CelestialBody
import numpy.linalg as la
from scipy import constants as cnst


class SpaceCraft:

    def __init__(self, initial_pos, initial_t, initial_v, system_bodies):
        self.pos = initial_pos
        self.t = initial_t
        self.current_body = self.get_current_body(system_bodies)
        self.pos_cb = self.get_cb_pos()
        self.velocity = initial_v
        self.velocity_cb = self.get_cb_vel()

    def __update__(self, pos, t, v, system_bodies):
        self.pos = pos
        self.t = t
        self.current_body = self.get_current_body(system_bodies)
        self.velocity = v
        self.velocity_cb = self.get_cb_vel()

    def get_current_body(self, system_bodies):
        within_sphere = np.empty((0, ), dtype=type(CelestialBody))
        for body in system_bodies.objects:
            if body.name is not 'sun':
                r_soi = body.sphere_of_influence()
                if np.abs(body.get_barycentric - self.pos) < r_soi:
                    within_sphere = np.append(within_sphere, body)
        if len(within_sphere) == 0:
            return system_bodies.get_body('sun')
        elif len(within_sphere) == 1:
            return within_sphere[0]
        else:
            planets = np.empty((0, ), dtype=type(CelestialBody))
            moons = np.empty((0, ), dtype=type(CelestialBody))
            for body in within_sphere:
                    if body.kind == 'planet':
                        planets = np.append(planets, body)
                    elif body.kind == 'moon':
                        moons = np.append(moons, body)
                    else:
                        raise AttributeError('get_current_body. body.kind does not match moon or planet')
            if len(moons) == 1:
                return moons[0]
            elif len(moons) > 1:
                raise ValueError('get_current_body. Spacecraft within SOI of multiple moons.')
            elif len(planets) == 1:
                return planets[0]
            elif len(planets) > 1:
                raise ValueError('get_current_body. Spacecraft within SOI of multiple planets')

    def get_cb_pos(self):
        current_body_pos = self.current_body.get_barycentric(self.t)
        relative_pos = self.pos - current_body_pos
        return relative_pos

    def circular_speed(self, body, relative_pos):
        distance = la.norm(relative_pos, axis=0)
        speed = np.sqrt(cnst.G * body.mass / distance)
        return speed

    def get_cb_vel(self):
        current_body_vel = self.current_body.get_barycentric_vel(self.t)
        relative_vel = self.velocity - current_body_vel
        return relative_vel



