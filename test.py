from spacecraft import SpaceCraft
from celestials import CelestialBody, CelestialGroup
from scipy import constants as cnst
import numpy as np
import matplotlib.pyplot as plt
import ode
import transfer

# Stars
sun = CelestialBody('sun', 332946.0487, 'star', None, 0.000001)
# Planets
mercury = CelestialBody('mercury', 0.05527, 'planet', sun, 0.3870993)
venus = CelestialBody('venus', 0.81500, 'planet', sun, 0.723336)
earth = CelestialBody('earth', 1, 'planet', sun, 1.000003)
mars = CelestialBody('mars', 0.10745, 'planet', sun, 1.52371)
jupiter = CelestialBody('jupiter', 317.83, 'planet', sun, 5.2029)
saturn = CelestialBody('saturn', 95.159, 'planet', sun, 9.537)
uranus = CelestialBody('uranus', 14.500	, 'planet', sun, 19.189)
neptune = CelestialBody('neptune', 17.204, 'planet', sun, 30.0699)
# Dwarf planets
# ceres = CelestialBody('ceres', 0.00016, 'planet', sun, 2.7658)
# Moons
moon = CelestialBody('moon', 0.0123000371, 'moon', earth, 0.002572)

solar_system = CelestialGroup(sun, mercury, venus, earth, mars, jupiter, saturn, uranus, neptune, moon)

distance = 381000
t_jd = np.double(800)
v_circ = np.sqrt(cnst.G * earth.mass / distance)
satellite_pos = earth.get_barycentric(t_jd) + np.array([distance, 0, 0])
satellite_vel = earth.get_barycentric_vel(t_jd) + np.array([0, v_circ, 0])
billy = SpaceCraft(satellite_pos, t_jd, satellite_vel, solar_system)
dt = 3000


plt.plot()
ts, ys = billy.calculate_trajectory(t_jd+dt, max_stepsize=0.01)
pos_res = ys[0:3, :]
pos_rela = pos_res # - earth.get_barycentric(ts)
moon_rela = moon.get_barycentric(ts) # - earth.get_barycentric(ts)
# plt.plot(moon_rela[0, :], moon_rela[1, :], 'k.')
plt.plot(pos_rela[0, :], pos_rela[1, :], 'b*')
ts, ys = billy.calculate_trajectory(t_jd+dt)
pos_res = ys[0:3, :]
pos_rela = pos_res # - earth.get_barycentric(ts)
# plt.plot(pos_rela[0, :], pos_rela[1, :], 'r*')
print(pos_rela[:, 23])
print(pos_rela[:, 24])
print(pos_rela[:, 25])
print(pos_rela[:, 26])
plt.show()
