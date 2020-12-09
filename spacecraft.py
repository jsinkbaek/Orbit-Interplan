import numpy as np
from celestials import CelestialBody
import numpy.linalg as la
from scipy import constants as cnst
import ode


class SpaceCraft:
    """
    Class of object called SpaceCraft. Used for the spacecraft that we wish to do trajectory planning on.
    """
    def __init__(self, initial_pos, initial_t, initial_v, system_bodies, unit_converter):
        self.pos = initial_pos
        self.t = initial_t
        self.current_body = self.get_current_body(system_bodies)
        self.pos_cb = self.get_cb_pos()
        self.velocity = initial_v
        self.velocity_cb = self.get_cb_vel()
        self.system_bodies = system_bodies
        self.unitc = unit_converter

    def update(self, pos, t, v, system_bodies=None):
        if system_bodies is None:
            system_bodies = self.system_bodies
        self.pos = pos
        self.t = t
        self.current_body = self.get_current_body(system_bodies)
        self.velocity = v
        self.velocity_cb = self.get_cb_vel()

    def calculate_trajectory(self, expected_endtime, expected_endpos=None, system_bodies=None, max_stepsize=None):
        """
        Calls ode.driver to integrate path until nearest expected_endpos.
        """
        if system_bodies is None:
            system_bodies = self.system_bodies

        # Wrapper
        def ode_equations(_t, _r):
            return ode.equations(_t, _r, system_bodies)

        if expected_endpos is None:
            estimate_endpos = False
        else:
            estimate_endpos = True
        initial_y = np.append(self.pos, self.velocity)
        ys, ts = ode.driver(ode_equations, self.t, initial_y, expected_endtime, expected_endpos, h=0.1, acc=1e-3,
                            eps=1e-3, stepper=ode.rk45_step, limit=5000, max_factor=1.5,
                            estimate_endpos=estimate_endpos,
                            max_stepsize=max_stepsize)
        return ts, ys

    def circular_speed(self, body=None):
        """
        Estimates speed around indicated body if SpaceCraft is in circular orbit around it.
        """
        if body is None:
            body = self.get_current_body()
        relative_pos = self.pos - body.get_barycentric(self.t)
        distance = la.norm(relative_pos, axis=0) * self.unitc.d
        mass = body.mass * body.unitc.m
        speed = np.sqrt(cnst.G * mass / distance)
        return speed * 1/self.unitc.v

    def get_current_body(self, system_bodies=None):
        """
        Finds out which body SpaceCraft is primarily within sphere of influence of (in case of both moon and planet,
        moon is selected).
        """
        if system_bodies is None:
            system_bodies = self.system_bodies
        within_sphere = np.empty((0, ), dtype=type(CelestialBody))
        for body in system_bodies.objects:
            if body.name is not 'sun':
                r_soi = body.sphere_of_influence()
                if la.norm(body.get_barycentric(self.t) - self.pos) < r_soi:
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

    def get_2body_pos(self, t_wanted, body, pos=None, t=None, vel=None):
        if t is None:
            t = self.t
        if pos is None:
            pos = self.pos
        if vel is None:
            vel = self.velocity
        relative_pos = pos - body.get_barycentric(t)
        relative_vel = vel - body.get_barycentric_vel(t)

    def get_cb_pos(self):
        current_body_pos = self.current_body.get_barycentric(self.t)
        relative_pos = self.pos - current_body_pos
        return relative_pos

    def get_cb_vel(self):
        current_body_vel = self.current_body.get_barycentric_vel(self.t)
        relative_vel = self.velocity - current_body_vel
        return relative_vel




