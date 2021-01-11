from spacecraft import SpaceCraft
from celestials import CelestialBody, CelestialGroup
from scipy import constants as cnst
import numpy as np
from unitconverter import UnitConverter
import matplotlib.pyplot as plt
import transfer
import numpy.linalg as la

# # Initialize unitconverter, bodies, and solar system # #
unitc = UnitConverter(m_unit='m_earth', d_unit='au', t_unit='days', v_unit='au/d')
sun = CelestialBody('sun', 332946.0487, 'star', None, 0.000000001, unitc)
earth = CelestialBody('earth', 1.0, 'planet', sun, 1.000003, unitc)
# earth = CelestialBody('earth', .00001, 'planet', sun, 1.000003, unitc)
jupiter = CelestialBody('jupiter', 317.83, 'planet', sun, 5.2029, unitc)
# jupiter = CelestialBody('jupiter', .00001, 'planet', sun, 5.2029, unitc)
solar_system = CelestialGroup(sun, earth, jupiter)

# # Find a time when earth crosses y=0 # #
ts = np.linspace(-200, 0, 10000)
ymin_idx = np.argmin(np.abs(earth.get_barycentric(ts)[1]))

# # Input parameters and initialize spacecraft # #
distance = 0.001
t_0 = ts[ymin_idx]
print('t_0', t_0)
earth_mu = cnst.G * earth.mass*earth.unitc.m
v_circ = np.sqrt(earth_mu / (distance*earth.unitc.d)) * 1/earth.unitc.v
pos = earth.get_barycentric(t_0) + np.array([distance, 0, 0])
vel = earth.get_barycentric_vel(t_0) + np.array([0, v_circ, 0])
scraft = SpaceCraft(pos, t_0, vel, solar_system, unitc)

# # Plot system with velocity vectors # #
vel_vec_scr = 4 * vel  # / la.norm(vel)
vel_vec_ear = 4*earth.get_barycentric_vel(t_0)  # / la.norm(earth.get_barycentric_vel(t_0))
vel_vec_jup = 4*jupiter.get_barycentric_vel(t_0)  # / la.norm(jupiter.get_barycentric_vel(t_0))
vel_vec_sun = 4*sun.get_barycentric_vel(t_0)  # / la.norm(sun.get_barycentric_vel(t_0))
pos_ear = earth.get_barycentric(t_0)
pos_jup = jupiter.get_barycentric(t_0)
pos_sun = sun.get_barycentric(t_0)
pos_scr_rel = pos - pos_ear
vec_scr_rel = (vel_vec_scr - vel_vec_ear)/4

fig = plt.figure()
ax = plt.plot(pos[0], pos[1], 'k.', markersize=5)
plt.xlim([-0.5, 5])
plt.ylim([-0.5, 2.5])
plt.ylabel('Barycenctric Y [AU]', fontsize=18)
plt.xlabel('Barycentric X [AU]', fontsize=18)
plt.title('Initial System', fontsize=25)
plt.arrow(pos[0], pos[1], vel_vec_scr[0], vel_vec_scr[1], width=0.0005, color='k', label='_nolegend_')
plt.arrow(pos_ear[0], pos_ear[1], vel_vec_ear[0], vel_vec_ear[1], width=0.0005, color='g', label='_nolegend_')
plt.arrow(pos_jup[0], pos_jup[1], vel_vec_jup[0], vel_vec_jup[1], width=0.0005, color='r', label='_nolegend_')
plt.arrow(pos_sun[0], pos_sun[1], vel_vec_sun[0], vel_vec_sun[1], width=0.0005, color='y', label='_nolegend_')
plt.plot(pos_ear[0], pos_ear[1], 'g.', markersize=18)
plt.plot(pos_jup[0], pos_jup[1], 'r.', markersize=24)
plt.plot(pos_sun[0], pos_sun[1], 'y.', markersize=30)
plt.legend(['Spacecraft', 'Earth', 'Jupiter', 'Sun'], fontsize=18)

ax_new = fig.add_axes([0.6, 0.2, 0.2, 0.2])
plt.xlim(pos_ear[0]-distance*0.4, pos_ear[0]+distance*1.1)
plt.ylim(pos_ear[1]-distance*1.1, pos_ear[1]+distance*1.1)
plt.arrow(pos_ear[0], pos_ear[1], vel_vec_ear[0]/100, vel_vec_ear[1]/100, width=0.00001, color='g')
plt.arrow(pos[0], pos[1], vel_vec_scr[0]/100, vel_vec_scr[1]/100, width=0.00001, color='k')
plt.plot(pos_ear[0], pos_ear[1], 'g.', markersize=18)
plt.plot(pos[0], pos[1], 'k.', markersize=10)
plt.show(block=False)

plt.figure()
plt.xlim([-distance-0.1*distance, distance+0.1*distance])
plt.ylim([-distance-0.1*distance, distance+0.1*distance])
plt.arrow(pos_scr_rel[0], pos_scr_rel[1], vec_scr_rel[0], vec_scr_rel[1], width=0.000005, color='b')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.title('Initial System relative to Earth')
plt.plot(0, 0, 'g.', markersize=24)
plt.plot(pos_scr_rel[0], pos_scr_rel[1], 'k.', markersize=8)
plt.show(block=False)

# # Make Hohmann transfer calculations # #
hohmann = transfer.Hohmann(scraft, jupiter)
_, dv1, _, _, _, _ = hohmann.simple(body=sun)
print('dv1', dv1)
print('hohmann.dv1', hohmann.dv1)
vel_scr_rel = vel - earth.get_barycentric_vel(t_0)
v_unit = vel_scr_rel / la.norm(vel_scr_rel)
v_esc = np.sqrt(2*earth_mu / (la.norm(pos_scr_rel)*unitc.d)) * 1/unitc.v
dv_initial = v_unit * (dv1 + (v_esc - la.norm(vel_scr_rel)))
print('dv_initial', (dv1 + (v_esc - la.norm(vel_scr_rel))))
print('dv_E_escape', v_esc - la.norm(vel_scr_rel))
print('v_esc', v_esc)
print('vel_scr_rel', la.norm(vel_scr_rel))
print('v_circ', v_circ)
# # Integrate resulting ODE # #
ts0, ys0 = hohmann.integrate(pos, vel, dv_initial, t_0)
ts1, ys1 = hohmann.integrate(pos, vel, dv_initial*0.8, t_0)
# # Plot result of integration # #

plt.figure()
plt.ylabel('y [AU]', fontsize=18)
plt.xlabel('x [AU]', fontsize=18)
plt.plot(ys0[0, :], ys0[1, :], 'k')
plt.plot(ys1[0, :], ys1[1, :], 'k--')
plt.plot(earth.get_barycentric(ts0)[0, :], earth.get_barycentric(ts0)[1, :], 'g', label='_nolegend_')
plt.plot(earth.get_barycentric(ts0)[0, -1], earth.get_barycentric(ts0)[1, -1], 'g.', markersize=8)
plt.plot(jupiter.get_barycentric(ts0)[0, :], jupiter.get_barycentric(ts0)[1, :], 'r', label='_nolegend_')
plt.plot(jupiter.get_barycentric(ts0)[0, -1], jupiter.get_barycentric(ts0)[1, -1], 'r.', markersize=14)
plt.plot(sun.get_barycentric(ts0)[0, :], sun.get_barycentric(ts0)[1, :], 'y', label='_nolegend_')
plt.plot(sun.get_barycentric(ts0)[0, -1], sun.get_barycentric(ts0)[1, -1], 'y.', markersize=20)
plt.plot(ys0[0, -1], ys0[1, -1], 'k.', markersize=12, label='_nolegend_')
plt.plot(ys1[0, -1], ys1[1, -1], 'k.', markersize=12, label='_nolegend_')
plt.legend(['Trajectory 1*dv', 'Trajectory 0.8*dv', 'Earth', 'Jupiter', 'Sun'], fontsize=18)
plt.show(block=True)


# # Reset spacecraft # #
scraft = SpaceCraft(pos, t_0, vel, solar_system, unitc)

# # Create Rendezvous object # #
rendezvous = transfer.Rendezvous(scraft, jupiter, sun)
t1 = rendezvous.initialburn_interplan(plot=True)
print('t_0', t_0)
print('t1', t1)

# # Try for rendezvous, integrate and update # #
ts, ys = scraft.calculate_trajectory(t1)
scraft.update(ys[0:3, -1], ts[-1], ys[3:6, -1])

# # Integrate Hohmann Transfer # #
ts, ys = hohmann.integrate(scraft.pos, scraft.velocity, dv_initial, scraft.t)

# # Plot and show # #

plt.figure()
plt.ylabel('y [AU]', fontsize=18)
plt.xlabel('x [AU]', fontsize=18)
plt.plot(ys[0, :], ys[1, :], 'k', label='_nolegend_')
plt.plot(earth.get_barycentric(ts)[0, :], earth.get_barycentric(ts)[1, :], 'g', label='_nolegend_')
plt.plot(earth.get_barycentric(ts)[0, -1], earth.get_barycentric(ts)[1, -1], 'g.', markersize=18)
plt.plot(jupiter.get_barycentric(ts)[0, :], jupiter.get_barycentric(ts)[1, :], 'r', label='_nolegend_')
plt.plot(jupiter.get_barycentric(ts)[0, -1], jupiter.get_barycentric(ts)[1, -1], 'r.', markersize=24)
plt.plot(sun.get_barycentric(ts)[0, :], sun.get_barycentric(ts)[1, :], 'y', label='_nolegend_')
plt.plot(sun.get_barycentric(ts)[0, -1], sun.get_barycentric(ts)[1, -1], 'y.', markersize=30)
plt.plot(ys[0, -1], ys[1, -1], 'k.', markersize=12)
plt.legend(['Earth', 'Jupiter', 'Sun', 'Spacecraft'], fontsize=18)
plt.show()


# # Reset and try to optimize result # #
scraft = SpaceCraft(pos, t_0, vel, solar_system, unitc)
rendezvous = transfer.Rendezvous(scraft, jupiter, sun)
print(rendezvous.integrate_optimize(target_distance=0.05, fudge_factor_initial_dv=0.9))
