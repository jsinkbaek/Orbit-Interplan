import numpy as np
from celestials import CelestialBody
import numpy.linalg as la
from scipy import constants as cnst
import ode


class SpaceCraft:
    """
    Class of object called SpaceCraft. Used for the spacecraft that we wish to do trajectory planning on.
    """
    def __init__(self, initial_pos, initial_t, initial_v, system_bodies):
        self.pos = initial_pos
        self.t = initial_t
        self.current_body = self.get_current_body(system_bodies)
        self.pos_cb = self.get_cb_pos()
        self.velocity = initial_v
        self.velocity_cb = self.get_cb_vel()
        self.system_bodies = system_bodies

    def update(self, pos, t, v, system_bodies):
        self.pos = pos
        self.t = t
        self.current_body = self.get_current_body(system_bodies)
        self.velocity = v
        self.velocity_cb = self.get_cb_vel()

    def calculate_trajectory(self, expected_endtime, expected_endpos, system_bodies):
        """
        Calls ode.driver to integrate path until nearest expected_endpos.
        """
        # Wrapper
        def ode_equations(_t, _r):
            return ode.equations(_t, _r, system_bodies)

        initial_y = np.append(self.pos, self.velocity)
        ys, ts = ode.driver(ode_equations, self.t, initial_y, expected_endtime, expected_endpos, h=0.1, acc=1e-3,
                            eps=1e-3, stepper=ode.rk45_step, limit=2000, max_factor=2)
        return ts, ys

    def circular_speed(self, body):
        """
        Estimates speed around indicated body if SpaceCraft is in circular orbit around it.
        """
        relative_pos = self.pos - body.get_barycentric(self.t)
        distance = la.norm(relative_pos, axis=0)
        speed = np.sqrt(cnst.G * body.mass / distance)
        return speed

    def get_current_body(self, system_bodies):
        """
        Finds out which body SpaceCraft is primarily within sphere of influence of (in case of both moon and planet,
        moon is selected).
        """
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

    def get_cb_vel(self):
        current_body_vel = self.current_body.get_barycentric_vel(self.t)
        relative_vel = self.velocity - current_body_vel
        return relative_vel



