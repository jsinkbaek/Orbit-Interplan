from spacecraft import SpaceCraft
from celestials import CelestialBody, CelestialGroup
import transfer
from scipy import constants as cnst
import numpy as np
from unitconverter import UnitConverter
import matplotlib.pyplot as plt
import ode
import transfer
# m_factor = 5.9722 * 1e24
a_factor = 1.496 * 1e8      # AU to km
unitc = UnitConverter(m_unit='m_earth', d_unit='km', t_unit='days', v_unit='km/d')
# Stars
sun = CelestialBody('sun', 332946.0487, 'star', None, 0.000001*a_factor, unitc)
# Planets
mercury = CelestialBody('mercury', 0.05527, 'planet', sun, 0.3870993*a_factor, unitc)
venus = CelestialBody('venus', 0.81500, 'planet', sun, 0.723336*a_factor, unitc)
earth = CelestialBody('earth', 1.0, 'planet', sun, 1.000003*a_factor, unitc)
mars = CelestialBody('mars', 0.10745, 'planet', sun, 1.52371*a_factor, unitc)
jupiter = CelestialBody('jupiter', 317.83, 'planet', sun, 5.2029*a_factor, unitc)
saturn = CelestialBody('saturn', 95.159, 'planet', sun, 9.537*a_factor, unitc)
uranus = CelestialBody('uranus', 14.500, 'planet', sun, 19.189*a_factor, unitc)
neptune = CelestialBody('neptune', 17.204, 'planet', sun, 30.0699*a_factor, unitc)
# Dwarf planets
# ceres = CelestialBody('ceres', 0.00016, 'planet', sun, 2.7658*a_factor, unitc)
# Moons
moon = CelestialBody('moon', 0.0123000371, 'moon', earth, 0.002572*a_factor, unitc)

solar_system = CelestialGroup(sun, earth, moon, mars, jupiter)
# solar_system = CelestialGroup(sun, mercury, venus, earth, mars, jupiter, saturn, uranus, neptune, moon)

distance = 15000   # km
t_jd = np.double(800)
v_circ = np.sqrt(cnst.G * earth.mass*earth.unitc.m / (distance*earth.unitc.d)) * 1/earth.unitc.v
satellite_pos = earth.get_barycentric(t_jd) + np.array([distance, 0, 0])
satellite_vel = earth.get_barycentric_vel(t_jd) + np.array([0, -v_circ, 0])
billy = SpaceCraft(satellite_pos, t_jd, satellite_vel, solar_system, unitc)
dt = 100

v_esc = np.sqrt(2*cnst.G*earth.mass/distance)
dv_earthescape = v_esc - v_circ
billy.update(billy.pos, billy.t, billy.velocity + np.array([0, -dv_earthescape, 0]))
hohmann1 = transfer.Hohmann(billy)
dvh1, dvh1_1, dvh1_2, tH1 = hohmann1.simple(r2=jupiter.a, body=sun)
print('hoh1 ang_align jup ', hohmann1.angular_alignment(jupiter, jupiter.parent))


jrendz = transfer.Rendezvous(billy, jupiter, sun)
print('rendv rel anglexy ', jrendz.relative_angle_xy(billy.t))
print('initalburn_simple ', jrendz.initialburn_simple())
print('initialburn_interplan', jrendz.initialburn_interplan())
"""
ts, ys = billy.calculate_trajectory(t_jd+dt)
pos_res = ys[0:3, :]
pos_rela = pos_res  # - earth.get_barycentric(ts)
moon_rela = moon.get_barycentric(ts)  - earth.get_barycentric(ts)
earth_pos = earth.get_barycentric(ts)
sun_pos = sun.get_barycentric(ts)
# plt.plot(moon_rela[0, :], moon_rela[1, :], 'k.')
# plt.plot(earth_pos[0, :], earth_pos[1, :], 'b.')
# plt.plot(sun_pos[0, :], sun_pos[1, :], 'y.')
plt.plot(pos_rela[0, :], pos_rela[1, :], 'r*')
plt.show()
"""

"""
def eqsin(x, y):
    return np.array([y[1], -y[0]])


a = 0
ya = np.array([0, 1])
b = 3*np.pi
yb = 1

r_, t_ = ode.driver(eqsin, a, ya, b, yb=yb, stepper=ode.rk45_step, estimate_endpos=False)

t_sin = np.linspace(0, 3*np.pi)
plt.plot(t_sin, np.sin(t_sin))
plt.plot(t_, r_[0, :], '*')
print(r_.shape)
plt.show()
"""
