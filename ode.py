import numpy as np
import numpy.linalg as la
from scipy import constants as cnst
import pandas as pd


def calculate_trajectory():
    print('test')


def equations(t, r, objects):
    """
    Sets up 6 ordinary differential equations to solve the problem by calculating the acceleration on the satellite due
    to gravitational effects from objects.
    :param t: Current time
    :param r: Current r vector (x,y,z,vx,vy,vz)
    :param objects: an object storing information about the planetoid objects. object.mass should give a 1d array of
                    all the planetoid masses needed, in order. object.ephemeris(t) should give a (3, nobjects) array
                    of current position of all planetoids needed, in solar cartesian coordinates.
    :return: vector dr with dx,dy,dz,ddx,ddy,ddz
    """
    dr = np.empty(r.shape)
    dr[0:2] = r[3:]
    obj_pos = objects.ephemeris(t)
    positions = np.empty((3, obj_pos[0, :].size+1))
    positions[:, 0] = r[0:2]
    positions[:, 1:] = obj_pos
    dr[3:] = acceleration(positions, objects.mass)
    return dr


def acceleration(positions, mass):
    """
    Calculates satellite acceleration given newtonian forces from multiple masses.
    :param positions: position in solar cartesian system. [:,0] is satellite, [:,1] is sun, then other objects.
                      numpy 3d array size (3, len(mass[0,:])+1)
    :param mass: masses of objects. mass[0] is sun, then other objects. numpy 2d array size (1, nobjects)
    :return: returns resulting acceleration a at point r[:, 0]. Should be ndarray with size (3, 1)
    """
    rij = positions[:, 1:] - positions[:, 0]
    distance = la.norm(rij, axis=0)  # distance[0] is distance to sun
    print(np.size(distance))
    a = cnst.G * np.sum(mass * rij / (distance**3), axis=1)
    return a


def driver(f, a, ya, b, yb, h, acc, eps, stepper, limit, max_factor):
    """
    Adaptive ode driver that advances solution by calling rkXX stepper and adjusts to appropriate step sizes.
    Stops at closest approach to a wanted position yb.
    :param f: function that returns a vector, right-hand side of dy/dt=f(t,y) (vector length 2*coordinates (3))
    :param a: start-point a
    :param ya: initial values y(a)
    :param b: expected end-point t
    :param yb: wanted end-value
    :param h: initial stepsize
    :param acc: absolute accuracy goal
    :param eps: relative accuracy goal
    :param stepper: stepping function rkXX_step
    :param limit: limit on number of steps allowed
    :param max_factor: limit on step factor increase
    :return: r, t: returns vector r with x,y,z,vx,vy,vz and time points t
    """
    nsteps = 0
    r = np.nan(shape=(3, 1000))
    t = np.nan(shape=(1000,))

    x = a   # current last step
    x0 = a  # the step before that
    yx = ya
    yx0 = ya

    while la.norm(yx - yb, axis=0) < la.norm(yx0 - yb, axis=0):
        nsteps += 1
        if x+h > b:
            h = b-x
        # Attempt step
        yh, err = stepper(f, x, yx, h)
        # Calculate step tolerances (it is possible that a small error on the tolerance arises from b being an estimate)
        # yh0 = la.norm(yh[0:2])      # to calculate length tolerances for x,y,z and dx,dy,dz separately
        # yh1 = la.norm(yh[3:-1])
        # ytau = np.array([yh0, yh1])
        # errtau = np.array([la.norm(err[0:2]), la.norm(err[3:-1])])
        tau = (eps*np.abs(yh)+acc) * np.sqrt(h/(b-a))
        # Determine if error is within tolerances and calculate tolerance ratio
        accept = True
        tol_ratio = np.ones(err.size)
        for i in range(0, tau.size):
            tol_ratio[i] = tau[i]/err[i]
            if tol_ratio[i] < 1:
                accept = False

        # Update if step was accepted
        if accept:
            x += h
            yx0 = yx
            yx = yh
            t[:, nsteps] = x
            r[:, nsteps] = yx
        else:
            print("Bad step at x = ", x)
            print("Step rejected")
        # Adjust stepsize based on empirical formula
        tol_min = tol_ratio[0]
        for i in range(1, tol_ratio.size):
            tol_min = np.min(tol_min, tol_ratio[i])
        adj_factor = np.power(tol_min, 0.25)*0.95
        if adj_factor > max_factor:
            adj_factor = max_factor
        h = h*adj_factor
        if nsteps >= limit:
            print("Step limit exceeded, returning current result")
            break

    print("ODE solved in ", nsteps, " steps")
    return r, t


def rk45_step(f, t, yt, h):
    """
    Implementation of Runge Kutta Fehlberg 45 stepper that advances a step t+h
    :param f: function that returns a vector, right-hand side of dy/dt=f(t,y)
    :param t: current point t
    :param yt: current vector value y(t)
    :param h: step to take
    :return: yh, err (value at y(t+h), and associated error)
    """
    # Butcher's table
    k = np.zeros((yt.size, 6))
    k[:, 0] = h * f(t, yt)
    k[:, 1] = h * f(t+h*1.0/4, yt+k[:, 0]*1.0/4)
    k[:, 2] = h * f(t+h*3.0/8, yt+k[:, 0]*3.0/32+k[:, 1]*9.0/32)
    k[:, 3] = h * f(t+h*12.0/13, yt+k[:, 0]*1932.0/2197-k[:, 1]*7200.0/2197*+k[:, 2]*7296.0/2197)
    k[:, 4] = h * f(t+h*1.0, yt+k[:, 0]*439.0/216-k[:, 1]*8.0+k[:, 2]*3680.0/513-k[:, 3]*845.0/4104)
    k[:, 5] = h * f(t+h*0.5, yt-k[:, 0]*8.0/27+k[:, 1]*2.0-k[:, 2]*3544.0/2565+k[:, 3]*1859.0/4104-k[:, 4]*11.0/40)
    b5 = np.array([16.0/135, 0, 6656.0/12825, 28561.0/56430, -9.0/50, 2.0/55])
    b4 = np.array([25.0/216, 0, 1408.0/2565, 2197.0/4104, -1.0/5, 0])

    yh = yt + np.sum(k*b5, axis=1)
    err = yh - yt - np.sum(k*b4, axis=1)
    return yh, err






