from spacecraft import SpaceCraft
from celestials import CelestialBody, CelestialGroup
import numpy as np
import numpy.linalg as la
from scipy import constants as cnst
from scipy.optimize import minimize, minimize_scalar


class Hohmann:
    def __init__(self, scraft):
        self.spacecraft = scraft

    def simple(self, r1=None, r2=None, body=None):
        if r1 is None:
            r1 = la.norm(self.spacecraft.pos, axis=0)
        if body is None:
            body = self.spacecraft.get_current_body(self.spacecraft.system_bodies)
        if r2 is None:
            r2 = la.norm(body.get_barycentric(self.spacecraft.t), axis=0)
        r1 = r1 * self.spacecraft.unitc.d
        r2 = r2 * self.spacecraft.unitc.d
        mass = body.mass * self.spacecraft.unitc.m
        dv1 = np.sqrt(cnst.G * mass / r1) * (np.sqrt(2.*r2/(r1+r2)) - 1) * 1/self.spacecraft.unitc.v
        dv2 = np.sqrt(cnst.G * mass / r2) * (1 - np.sqrt(2*r1/(r1+r2))) * 1/self.spacecraft.unitc.v
        dv = dv1 + dv2
        tH = np.pi * np.sqrt((r1+r2)**3 / (8*cnst.G*mass)) * 1/self.spacecraft.unitc.t
        return dv, dv1, dv2, tH

    def angular_alignment(self, target, parent, r1=None):
        """
        For calculating angular alignment between target planet for interplanetary Hohmann transfer rendezvous.
        Assumes no inclination difference between the spacecraft and target.
        """
        if r1 is None:
            r1 = la.norm(self.spacecraft.pos, axis=0)
        r2 = la.norm(target.get_barycentric(self.spacecraft.t), axis=0)
        r1 = r1 * self.spacecraft.unitc.d
        r2 = r2 * self.spacecraft.unitc.d
        pmass = parent.mass * parent.unitc.m
        target_angvel = np.sqrt(cnst.G * pmass / r2**3)
        tH = np.pi * np.sqrt((r1+r2)**3 / (8*cnst.G*pmass))
        # Angular alignment alpha in radians
        alpha = np.pi - target_angvel * tH
        return alpha, tH * 1/target.unitc.t


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

    def initialburn_simple(self, timebound=None):
        """
        Approximates time for initial burn for interplanetary Hohmann transfer using relative angle in the xy-plane.
        (Only good for low inclinations of target and spacecraft).
        Assumes spacecraft starts in circular orbit around sun.
        """
        t_ini = self.spacecraft.t
        # angle_ini = self.relative_angle_xy(t_ini)
        # alpha_angle = self.hohmann.angular_alignment(self.target, self.parent)
        if timebound is None:
            sma_target = la.norm(self.target.get_barycentric(t_ini), axis=0) * self.target.unitc.d
            # Upper timelimit is then 2 orbits of target
            timebound = 2 * 2*np.pi*np.sqrt(sma_target**3 / (cnst.G*self.parent.mass * self.parent.unitc.m))
            timebound = timebound * 1/self.target.unitc.t + t_ini

        def minimize_func(t_):
            ang_align, _ = self.hohmann.angular_alignment(self.target, self.parent)
            return np.abs(self.relative_angle_xy(t_) - ang_align)

        optimize_res = minimize_scalar(minimize_func, t_ini, bounds=(t_ini, timebound), method='Bounded')
        t_first = np.min(optimize_res.x)
        print('t_first', t_first)
        return t_first, optimize_res

    def initialburn_interplan(self, timebound=None):
        """
        Assumes spacecraft starts in circular orbit around a planet. Tries to find optimal time for burn to utilize
        current planet gravity.
        """
        t_simple, _ = self.initialburn_simple(timebound)

        _, tH = self.hohmann.angular_alignment(self.target, self.parent)
        target_pos = self.target.get_barycentric(t_simple+tH)             # location of target at approximate rendezvous

        def minimize_func(t_):
            rel_pos_cb = self.spacecraft.get_cb_pos()
            rel_angle_cb = np.arctan(rel_pos_cb[1]/rel_pos_cb[0])
            angle_target = np.arctan(target_pos[1]/target_pos[0])
            return np.abs(angle_target - np.pi - rel_angle_cb)

        rpos_cb = self.spacecraft.get_cb_pos()
        rel_distance_cb = la.norm(rpos_cb, axis=0) * self.spacecraft.unitc.d
        mu_cb = (cnst.G * self.spacecraft.get_current_body().mass * self.spacecraft.unitc.m)
        tbound = 0.5 * 2 * np.pi * np.sqrt(rel_distance_cb**3 / mu_cb) * 1/self.spacecraft.unitc.t
        optimize_res = minimize_scalar(minimize_func, t_simple, bounds=(t_simple-tbound, t_simple+tbound),
                                       method='Bounded')
        print('rpos', rpos_cb)
        print('t_bound', tbound)
        t_res = np.min(optimize_res.x)
        return t_res, optimize_res

    def integrate_optimize(self, target_distance=1000, timebound=None):
        t_initialburn, t_possible = self.initialburn_interplan(timebound=timebound)
        ts_, ys_ = self.spacecraft.calculate_trajectory(expected_endtime=t_initialburn)
        t_begin = ts_[-1]
        pos_begin = ys_[0:2, -1]
        vel_begin = ys_[3:5, -1]
        tempcraft = SpaceCraft(pos_begin, t_begin, vel_begin, self.spacecraft.system_bodies, self.spacecraft.unitc)
        thohmann_ = Hohmann(tempcraft)
        _, dv1h, dv2h, th = thohmann_.simple()
        v_unit = vel_begin/la.norm(vel_begin, axis=0)
        tempcraft.update(pos_begin, t_begin, vel_begin+v_unit*dv1h)
        target_endpos = self.target.get_barycentric(th)
        target_endpos[0] += target_distance

        # function to minimize, input 1D array (time, velocity, position), so (7, ) length
        def minimize_func(tvp):
            ts_temp, ys_temp = tempcraft.calculate_trajectory(th, target_endpos)
            # Minimize difference from target_distance
            endpos_temp = ys_temp[0:2, -1]
            relative_distance = la.norm(endpos_temp - self.target.get_barycentric(ts_temp[-1]), axis=0)
            return np.abs(relative_distance - target_distance)

        tvp_initial = np.empty((7, ))
        tvp_initial[0] = tempcraft.t
        tvp_initial[1:3] = tempcraft.pos
        tvp_initial[4:6] = tempcraft.velocity
        optimize_res = minimize(minimize_func, tvp_initial)
        return optimize_res


