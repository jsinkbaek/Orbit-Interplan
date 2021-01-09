import numpy as np
from celestials import CelestialBody
import numpy.linalg as la
from scipy import constants as cnst
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
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
        self.system_bodies = system_bodies
        self.unitc = unit_converter

    def update(self, pos, t, v, system_bodies=None):
        if system_bodies is None:
            system_bodies = self.system_bodies
        self.pos = pos
        self.t = t
        self.current_body = self.get_current_body(system_bodies)
        self.velocity = v

    def calculate_trajectory(self, expected_endtime, expected_endpos=None, system_bodies=None, max_stepsize=0.01,
                             scipy=False, limit=5000):
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
        if scipy is True:
            print()
            print('self.t', self.t)
            print('expc_t', expected_endtime)
            ode_res = solve_ivp(ode_equations, (self.t, expected_endtime), initial_y, dense_output=False)
            ys, ts = ode_res.y, ode_res.t
            print('ys.shape', ys.shape)
            print('ts.shape', ts.shape)

        else:
            ys, ts = ode.driver(ode_equations, self.t, initial_y, expected_endtime, expected_endpos, h=0.1, acc=1e-6,
                                eps=1e-6, stepper=ode.rk45_step, limit=limit, max_factor=2,
                                estimate_endpos=estimate_endpos,
                                max_stepsize=max_stepsize)
        # plt.figure()
        # plt.plot(ys[0, :], ys[1, :])
        # plt.show()

        return ts, ys

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

    def get_cb_pos(self):
        current_body_pos = self.current_body.get_barycentric(self.t)
        relative_pos = self.pos - current_body_pos
        return relative_pos




