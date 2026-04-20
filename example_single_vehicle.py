"""
Example: Single vehicle crossing a simply supported bridge
===========================================================
Demonstrates both coupled (iterative) and decoupled (moving load) VBI analysis
using a quarter-car vehicle on a 25 m simply supported beam.

This reproduces Benchmark 1 from:
    Talebi-Kalaleh & Mei, "An Open-Source Framework for Coupled Vehicle-Bridge
    Interaction Analysis Using OpenSees"
"""

import numpy as np
import matplotlib.pyplot as plt
from opensees_vbi import (
    initial_calcs,
    generate_vehicle_props_pool,
    road_roughness_generator,
    generate_random_traffic_nodal_forces,
    main_vehicle_bridge,
)

# ── Bridge parameters ────────────────────────────────────────────────────────
bridge_span = 25.0            # m
bridge_supports = [0, bridge_span]
bridge_E = 27.5e9             # Pa
bridge_Iz = 0.12              # m^4
bridge_A = 2.0                # m^2
bridge_mbar = 4800.0          # kg/m
bridge_damping = 0.0          # damping ratio
bridge_disc = 250             # number of elements
speed = 10.0                  # m/s
sampling_rate = 500           # Hz

# ── Derived quantities ───────────────────────────────────────────────────────
road_window = [-10, bridge_span + 10]       # approach + bridge + exit (m)
run_time = bridge_span / speed + 1.5        # s

# ── Road roughness (smooth road for this benchmark) ──────────────────────────
np.random.seed(42)
roughness = road_roughness_generator(Gd_n0=0, smoothing_window=0.05,
                                     roughness_window=road_window)

# ── Background traffic (none) ───────────────────────────────────────────────
bp, bric, ts, dt = initial_calcs(bridge_span, bridge_supports, bridge_Iz,
                                  bridge_A, bridge_E, bridge_damping,
                                  bridge_mbar, bridge_disc, run_time,
                                  sampling_rate)
traffic = generate_random_traffic_nodal_forces(0, bridge_disc + 1, ts)

# ── Vehicle properties ──────────────────────────────────────────────────────
# Uses the built-in Yang 2004 quarter-car configuration
vp, vr = generate_vehicle_props_pool(
    ts, [road_window[0], 0], [0.7, 1.3], [0.7, 1.3], [0.7, 1.3],
    base_vehicle_name='quarter_car_simple_Yang2004',
    num_gen_vehicle=1, base_speed=speed)

# ── Coupled (iterative) analysis ────────────────────────────────────────────
print("Running coupled analysis...")
bp_c, bric_c, _, _ = initial_calcs(bridge_span, bridge_supports, bridge_Iz,
                                    bridge_A, bridge_E, bridge_damping,
                                    bridge_mbar, bridge_disc, run_time,
                                    sampling_rate)
ba_c, va_c, _, iters, _ = main_vehicle_bridge(
    bp_c, vp, traffic, roughness, bric_c, vr, ts, dt,
    vehicle_model='moving_mass', plot_mode_shapes=False, print_iterations=False)
print(f"  Done — max iterations per step: {iters[iters > 0].max()}")

# ── Decoupled (moving load) analysis ────────────────────────────────────────
print("Running decoupled analysis...")
vp_d, vr_d = generate_vehicle_props_pool(
    ts, [road_window[0], 0], [0.7, 1.3], [0.7, 1.3], [0.7, 1.3],
    base_vehicle_name='quarter_car_simple_Yang2004',
    num_gen_vehicle=1, base_speed=speed)
bp_d, bric_d, _, _ = initial_calcs(bridge_span, bridge_supports, bridge_Iz,
                                    bridge_A, bridge_E, bridge_damping,
                                    bridge_mbar, bridge_disc, run_time,
                                    sampling_rate)
ba_d, va_d, _, _, _ = main_vehicle_bridge(
    bp_d, vp_d, traffic, roughness, bric_d, vr_d, ts, dt,
    vehicle_model='moving_load', plot_mode_shapes=False, print_iterations=False)
print("  Done")

# ── Plot results ─────────────────────────────────────────────────────────────
mid = bridge_disc // 2
t_c = ba_c['time'].flatten()
t_d = ba_d['time'].flatten()

# On-bridge time window
rear = vp['vehicle_1']['axle_loc_rear']
front = rear + vp['vehicle_1']['axle_local_coords'][-1]
t_enter = ts[np.searchsorted(front, 0)]
t_exit = ts[min(np.searchsorted(rear, bridge_span), len(ts) - 1)]

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# Bridge midspan displacement
axes[0].plot(t_c, ba_c['disp'][mid, :] * 1e3, 'b-', lw=1.5, label='Coupled')
axes[0].plot(t_d, ba_d['disp'][mid, :] * 1e3, 'r--', lw=1.3, label='Decoupled')
axes[0].set_ylabel('Midspan displacement (mm)')
axes[0].set_title('Bridge midspan displacement')
axes[0].legend()

# Vehicle body displacement
t_vc = va_c['vehicle_1']['time'].flatten()
t_vd = va_d['vehicle_1']['time'].flatten()
axes[1].plot(t_vc, va_c['vehicle_1']['disp'][1, :] * 1e3, 'b-', lw=1.5, label='Coupled')
axes[1].plot(t_vd, va_d['vehicle_1']['disp'][1, :] * 1e3, 'r--', lw=1.3, label='Decoupled')
axes[1].set_ylabel('Vehicle body displacement (mm)')
axes[1].set_xlabel('Time (s)')
axes[1].set_title('Vehicle body displacement')
axes[1].legend()

for ax in axes:
    ax.set_xlim(t_enter, t_exit)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('example_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nResults saved to example_results.png")
