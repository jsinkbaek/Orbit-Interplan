from spacecraft import SpaceCraft
from celestials import CelestialBody, CelestialGroup
import numpy as np
import numpy.linalg as la
from scipy import constants as cnst
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt


class Hohmann:
    def __init__(self, scraft, target):
        self.spacecraft = scraft
        self.target = target
        self.body = self.spacecraft.get_current_body()
        self.dvtot, self.dv1, self.dv2, self.tH, self.r1, self.r2 = self.simple(body=self.target.parent)

    def simple(self, r1=None, r2=None, body=None):
        if r1 is None:
            r1 = la.norm(self.spacecraft.pos, axis=0)
        if body is None:
            body = self.spacecraft.get_current_body(self.spacecraft.system_bodies)
        if r2 is None:
            r2 = la.norm(self.target.get_barycentric(self.spacecraft.t), axis=0)
        r1 = r1 * self.spacecraft.unitc.d
        r2 = r2 * self.spacecraft.unitc.d
        mass = body.mass * self.spacecraft.unitc.m
        dv1 = np.sqrt(cnst.G * mass / r1) * (np.sqrt(2.*r2/(r1+r2)) - 1) * 1/self.spacecraft.unitc.v
        dv2 = np.sqrt(cnst.G * mass / r2) * (1 - np.sqrt(2*r1/(r1+r2))) * 1/self.spacecraft.unitc.v
        dv = dv1 + dv2
        tH = np.pi * np.sqrt((r1+r2)**3 / (8*cnst.G*mass)) * 1/self.spacecraft.unitc.t
        r1 = r1 / self.spacecraft.unitc.d
        r2 = r2 / self.spacecraft.unitc.d
        return dv, dv1, dv2, tH, r1, r2

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

    def integrate(self, pos0, vel0, dv, t0, scraft=None, limit=5000):
        if scraft is None:
            scraft = self.spacecraft
        scraft.update(pos0, t0, vel0+dv)
        ts, ys = scraft.calculate_trajectory(t0+1.5*self.tH, limit=limit)
        return ts, ys


class Rendezvous:
    def __init__(self, scraft, target, parent):
        self.spacecraft = scraft
        self.hohmann = Hohmann(scraft, target)
        self.target = target
        self.parent = parent        # parent of target (f.ex. Sun if target is Jupiter)

    def relative_angle_xy(self, t):
        """
        Finds xy relative angle between spacecraft and target. Most useful for orbits close to the xy-plane.
        Not useful for finding initialburn, as spacecraft position is static (needs to be integrated)
        """
        pos_target = self.target.get_barycentric(t)[0:2]
        y_target, x_target = pos_target[1], pos_target[0]
        angle_target = np.mod(np.arctan2(y_target, x_target), 2*np.pi)
        y_scraft, x_scraft = self.spacecraft.pos[1], self.spacecraft.pos[0]
        angle_scraft = np.mod(np.arctan2(y_scraft, x_scraft), 2*np.pi)
        return angle_target - angle_scraft

    def relative_angle_planets(self, t, body1, body2):
        """
        Finds xy relative angle between two planets. Can be used to estimate relative angle between a target and space-
        craft in case spacecraft is in orbit around the other body
        """
        pos_body1 = body1.get_barycentric(t)[0:2]
        pos_body2 = body2.get_barycentric(t)[0:2]
        # angle_body1 = np.arctan(pos_body1[1]/pos_body1[0])
        # angle_body2 = np.arctan(pos_body2[1]/pos_body2[0])
        angle_body1 = np.mod(np.arctan2(pos_body1[1], pos_body1[0]), 2*np.pi)
        angle_body2 = np.mod(np.arctan2(pos_body2[1], pos_body2[0]), 2*np.pi)
        return angle_body1 - angle_body2
        # if np.cross(pos_body2, pos_body1) < 0:
        #    return angle_body1 - angle_body2
        # else:
        #    return angle_body1 - angle_body2

    def initialburn_simple(self, timebound=None, plot=False):
        """
        Approximates time for initial burn for interplanetary Hohmann transfer using relative angle in the xy-plane.
        (Only good for low inclinations of target and spacecraft).
        Assumes spacecraft position can be approximated with a body (e.g. Earth) in circular orbit around sun.
        """
        t_ini = self.spacecraft.t
        # angle_ini = self.relative_angle_xy(t_ini)
        # alpha_angle = self.hohmann.angular_alignment(self.target, self.parent)
        if timebound is None:
            # sma_target = la.norm(self.target.get_barycentric(t_ini), axis=0) * self.target.unitc.d
            sma_self = la.norm(self.spacecraft.pos) * self.target.unitc.d
            # Upper timelimit is then 2 orbits of self
            timebound = 2 * 2*np.pi*np.sqrt(sma_self**3 / (cnst.G*self.parent.mass * self.parent.unitc.m))
            timebound = timebound * 1/self.target.unitc.t + t_ini

        def minimize_func(t_):
            ang_align, _ = self.hohmann.angular_alignment(self.target, self.parent)
            body1 = self.target
            body2 = self.spacecraft.get_current_body()
            return np.abs(self.relative_angle_planets(t_, body1, body2) - ang_align)

        t_linspace = np.linspace(t_ini, timebound, 10000)
        y_linspace = minimize_func(t_linspace)
        y_inverted = 1-y_linspace/(2*np.pi)
        peaks_idx = find_peaks(y_inverted)[0]
        if plot:
            plt.figure()
            plt.plot(t_linspace, 1 - y_linspace / (2 * np.pi), 'r.')
            plt.plot(t_linspace[peaks_idx], y_inverted[peaks_idx], 'b*')
            plt.show()
        t_first = np.min(t_linspace[peaks_idx])
        return t_first

    def initialburn_interplan(self, timebound=None, plot=False):
        """
        Assumes spacecraft starts in circular orbit around a planet with z=0.
        Tries to find optimal time for burn to utilize current planet gravity.
        """
        t_simple = self.initialburn_simple(timebound, plot)

        _, tH = self.hohmann.angular_alignment(self.target, self.parent)
        target_pos = self.target.get_barycentric(t_simple+tH)             # location of target at approximate rendezvous
        rpos_cb = self.spacecraft.get_cb_pos()
        rel_distance_cb = la.norm(rpos_cb, axis=0) * self.spacecraft.unitc.d
        mu_cb = (cnst.G * self.spacecraft.get_current_body().mass * self.spacecraft.unitc.m)
        tbound = np.pi * np.sqrt(rel_distance_cb**3 / mu_cb) * 1/self.spacecraft.unitc.t
        period = 2 * tbound
        a_sma = rel_distance_cb

        def x_cosine_phasematch(x_, v_x, t_):
            if v_x < 0:
                shift = 0
            else:
                shift = np.pi
            t0s = np.linspace(0, period/2, 5000)
            phase_idx = np.argmin(np.abs(x_ - a_sma*np.cos(2*np.pi*(t_-t0s)/period + shift)))
            return t0s[phase_idx], shift

        t0_phase, pshift = x_cosine_phasematch(self.spacecraft.pos[0], self.spacecraft.velocity[0], self.spacecraft.t)

        def find_tfirst(t_ini):
            ts = np.linspace(t_ini - tbound, t_ini + tbound, 5000)
            xt = a_sma * np.cos(2 * np.pi * (ts - t0_phase) / period + pshift)
            yt = a_sma * np.sin(2 * np.pi * (ts - t0_phase) / period + pshift)

            angle_rel_cb = np.mod(np.arctan2(xt, yt), 2 * np.pi)
            cb_pos = self.spacecraft.get_current_body().get_barycentric(t_ini)
            angle_cb = np.mod(np.arctan2(cb_pos[1], cb_pos[0]), 2 * np.pi)
            target_pos_ = self.target.get_barycentric(t_ini+tH)
            angle_target = np.mod(np.arctan2(target_pos_[1], target_pos_[0]), 2 * np.pi)

            angle_match = np.abs(angle_target - np.pi - (angle_rel_cb - angle_cb))
            angle_match = 1 - angle_match / (2 * np.pi)
            peaks_idx = find_peaks(angle_match)[0]
            min_idx = np.min(peaks_idx)
            t_first = ts[min_idx]

            if plot:
                plt.figure()
                cb_pos_ = self.spacecraft.get_current_body().get_barycentric(t_first)
                plt.plot(cb_pos_[0], cb_pos_[1], 'b.')
                target_pos_ = self.target.get_barycentric(t_first + tH)
                plt.plot(target_pos_[0], target_pos_[1], 'r.')
                plt.plot(cb_pos_[0] + xt[min_idx], cb_pos_[1] + yt[min_idx], 'g.')
                sun_pos = self.target.parent.get_barycentric(t_first)
                plt.plot(sun_pos[0], sun_pos[1], 'y.')
                plt.show(block=False)
            return t_first
        t1 = find_tfirst(t_simple)
        # t1 = find_tfirst(t1)
        return t1

    def integrate_optimize(self, target_distance=0.1, timebound=None):
        # Calculate timestamp for estimated initial delta-v
        t_initialburn = self.initialburn_interplan(timebound=timebound)
        # Integrate until initial delta-v
        ts_, ys_ = self.spacecraft.calculate_trajectory(expected_endtime=t_initialburn)
        # Initialize temporary spacecraft copy to work with instead of primary one
        t_begin = ts_[-1]
        pos_begin = ys_[0:3, -1]
        vel_begin = ys_[3:6, -1]
        tempcraft = SpaceCraft(pos_begin, t_begin, vel_begin, self.spacecraft.system_bodies, self.spacecraft.unitc)

        # Integrate forward in orbit to have additional data points to use
        ts_2, ys_2 = tempcraft.calculate_trajectory(expected_endtime=(t_begin + 2*(t_begin-ts_[0])))
        ts_ = np.append(ts_, ts_2)
        ys_ = np.append(ys_, ys_2, axis=1)
        ts_, u_idx = np.unique(ts_, return_index=True)
        ys_ = ys_[:, u_idx]

        # Make interpolation function to use for constraints on minimization function
        f_interp = interp1d(ts_, ys_, kind='cubic')

        # Estimate initial delta-v using simple Hohmann transfer equations and escape velocity of current_body
        thohmann_ = Hohmann(tempcraft, self.target)
        _, dv1h, dv2h, th, _, _ = thohmann_.simple(body=self.parent)
        v_unit = vel_begin/la.norm(vel_begin, axis=0)
        cb_mu = tempcraft.current_body.mass * tempcraft.unitc.m * cnst.G
        pos_rel_cb = tempcraft.get_cb_pos()
        vel_rel_cb = vel_begin - tempcraft.current_body.get_barycentric_vel(t_begin)
        v_escape = np.sqrt(2*cb_mu/la.norm(pos_rel_cb * tempcraft.unitc.d)) * 1/tempcraft.unitc.v
        dv_initialburn = v_unit * (dv1h + (v_escape - la.norm(vel_rel_cb)))

        # function to minimize, input 1D array (time (1), delta-v (3)), so (4, ) length
        def minimize_func(t_dv):
            try:
                i += 1
            except NameError:
                i = 0
            t_, dv_ = t_dv[0], t_dv[1:]
            y_interp = f_interp(t_)
            pos_, vel_ = y_interp[0:3], y_interp[3:]
            tempcraft.update(pos_, t_, vel_+dv_)
            ts_temp, ys_temp = tempcraft.calculate_trajectory(th*2.5)
            f_interp_temp = interp1d(ts_temp, ys_temp[0:3, :])
            ts_interp = np.linspace(ts_temp[0], ts_temp[-1], 2000)
            ys_interp = f_interp_temp(ts_interp)
            # # Find places furthest from initial position # #
            pos_rel_ini = la.norm(f_interp_temp(ts_interp)-pos_.reshape(3, 1), axis=0)
            apoapsis_idx, _ = find_peaks(pos_rel_ini)
            if apoapsis_idx.size > 1:
                print()
                print('Apoapsis error, multiple found. Taking first one.')
                print('apoapsis_idx', apoapsis_idx)
                apoapsis_idx = apoapsis_idx[0]
                print('apoapsis_idx[0]', apoapsis_idx)
            if not apoapsis_idx:
                print()
                print('Apoapsis error, none found. Taking furthest point as apoapsis.')
                apoapsis_idx = np.argmax(pos_rel_ini)
                print('apoapsis_idx', apoapsis_idx)
            apoapsis_t = ts_interp[apoapsis_idx]
            apoapsis_pos = f_interp_temp(apoapsis_t)
            # # Minimize difference from target_distance # #
            target_pos = self.target.get_barycentric(ts_interp)
            parent_pos = self.target.parent.get_barycentric(ts_interp)
            cb_pos = tempcraft.get_current_body().get_barycentric(ts_interp)
            distance = la.norm(apoapsis_pos - target_pos[:, apoapsis_idx])

            # # Print and Plot # #
            print()
            print('t0', t_)
            print('dv', la.norm(dv_))
            print('diff ', np.abs(distance - target_distance))
            plt.clf()
            plt.xlim([-6.5, 6.5])
            plt.ylim([-6.5, 6.5])
            plt.plot(target_pos[0, :], target_pos[1, :], 'r')
            plt.plot(parent_pos[0, :], parent_pos[0, :], 'y')
            plt.plot(cb_pos[0, :], cb_pos[1, :], 'b')
            plt.plot(parent_pos[0, apoapsis_idx], parent_pos[1, apoapsis_idx], 'y.', markersize=20)
            plt.plot(target_pos[0, apoapsis_idx], target_pos[1, apoapsis_idx], 'r.', markersize=14)
            plt.plot(target_pos[0, 0], target_pos[1, 0], 'g.', markersize=12)
            plt.plot(target_pos[0, -1], target_pos[1, -1], 'm.', markersize=12)
            plt.plot(cb_pos[0, apoapsis_idx], cb_pos[1, apoapsis_idx], 'b.', markersize=8)
            plt.plot(cb_pos[0, 0], cb_pos[1, 0], 'g.', markersize=8)
            plt.plot(cb_pos[0, -1], cb_pos[1, -1], 'm.', markersize=8)
            plt.plot(ys_interp[0, :], ys_interp[1, :], 'k')
            plt.plot(ys_interp[0, apoapsis_idx], ys_interp[1, apoapsis_idx], 'k.', markersize=8)
            plt.plot(ys_interp[0, 0], ys_interp[1, 0], 'k.', markersize=8)
            plt.draw()
            plt.pause(0.0001)
            plt.savefig(f'fig/pic_{i}.png')
            return np.abs(distance - target_distance)

        # Constraints on minimizer
        def con1(t_dv):
            # Require that dv does not set spacecraft on an orbit to escape solar system
            vel_ = f_interp(t_dv[0])[3:]
            speed = la.norm(vel_ + t_dv[1:])
            # Sun escape velocity
            mu_sun = self.target.parent.mass * self.target.unitc.m * cnst.G
            v_esc = np.sqrt(2*mu_sun/(la.norm(tempcraft.pos * tempcraft.unitc.d))) * 1/tempcraft.unitc.v
            print('v_esc ', v_esc)
            return v_esc - speed

        def con2(t_dv):
            # Require that dv does not decrease the speed of the spacecraft
            vel_ = f_interp(t_dv[0])[3:]
            vel_new = vel_+t_dv[1:]
            return la.norm(vel_new) - la.norm(vel_)

        def con3(t_dv):
            # Require that orbital energy in relation to sun is <0
            y_interp = f_interp(t_dv[0])
            speed = la.norm(y_interp[3:6] + t_dv[1:4])
            dist_ = la.norm(y_interp[0:3])
            kin = speed**2 / 2
            mu_sun = self.target.parent.mass * self.target.unitc.m * cnst.G
            pot = mu_sun/dist_
            return pot - kin

        def con4(t_dv):
            if t_dv[0] > ts_[-1] or t_dv[0] < ts_[0]:
                return 1
            else:
                return 0

        constraints = [# {'type': 'ineq', 'fun': con1},
                       {'type': 'ineq', 'fun': con2},
                       {'type': 'ineq', 'fun': con3}
                       ]

        # Initial values and boundaries
        t_dv_initial = np.empty((4, ))
        t_dv_initial[0] = t_begin
        t_dv_initial[1:4] = dv_initialburn

        cb_dist = la.norm(tempcraft.get_current_body().get_barycentric(t_begin)) * tempcraft.unitc.d
        parent_mu = self.target.parent.mass * self.target.unitc.m * cnst.G
        v_escape_sun = np.sqrt(2 * parent_mu/(la.norm(pos_begin * tempcraft.unitc.d))) * 1/tempcraft.unitc.v
        dv_bound = v_escape_sun + (v_escape - la.norm(vel_rel_cb))
        bounds = ((ts_[0], ts_[-1]), (-dv_bound, dv_bound), (-dv_bound, dv_bound),
                  (-dv_bound, dv_bound))
        if la.norm(dv_initialburn) < dv_bound:
            print()
            print('within bound')
            print('dv', la.norm(dv_initialburn))
            print('dv_bound', dv_bound)
            print('dv1h', dv1h)
            print('v_escape_sun', v_escape_sun)
            print('v_escape', v_escape)
            print('norm(v_unit)', la.norm(v_unit))
            print('norm(vel_begin)', la.norm(vel_begin))
        else:
            print()
            print('initial dv not within bound')
            print('dv', la.norm(dv_initialburn))
            print('dv_bound', dv_bound)
            print('dv1h', dv1h)
            print('v_escape_sun', v_escape_sun)
            print('v_escape', v_escape)
            print('norm(v_unit)', la.norm(v_unit))
            print('norm(vel_begin)', la.norm(vel_begin))

        # Call minimizer
        plt.figure()
        optimize_res = minimize(minimize_func, t_dv_initial, method='L-BFGS-B', bounds=bounds
                                # , constraints=constraints
                                )
        # method L-BFGS-B converges, but seems to ignore constraints and bounds
        print('target_distance', target_distance)
        print('values ', optimize_res.values())
        print('status ', optimize_res.message)

        # # Plot Solution # #
        plt.show()
        return optimize_res


