# OpenSees-Moving-Load

A Python framework for vehicle-bridge interaction (VBI) analysis using [OpenSees](https://opensees.berkeley.edu/) as the finite element solver.

The framework models the bridge and vehicle as separate OpenSees finite element subsystems coupled through an iterative partitioned scheme. No custom element formulations or modifications to OpenSees are required.

## Features

- **Coupled iterative analysis** — full vehicle-bridge interaction with convergence at each time step
- **Decoupled moving load analysis** — computationally efficient approach for lightweight vehicles
- **Five vehicle model types** — from simple quarter-car to full half-car with pitch, axle masses, and independent tyre-suspension assemblies
- **ISO 8608 road roughness** — synthetic roughness profile generation for Classes A through E
- **Multi-span bridges** — simply supported and continuous bridges with arbitrary support locations
- **Background traffic** — random nodal forces representing ambient traffic loading

## Vehicle Models

| Model | Description | DOFs |
|-------|-------------|------|
| `one_axle_simple` | Quarter-car: single spring-mass | 2 |
| `one_axle_comp` | Quarter-car: tyre + suspension + body | 3 |
| `two_axle_comp1` | Two independent quarter-cars | 6 |
| `two_axle_comp2` | Half-car with body pitch | 8 |
| `two_axle_comp3` | Half-car with pitch and axle masses | 10 |

## Installation

### Requirements

- Python 3.8+
- [OpenSeesPy](https://openseespydoc.readthedocs.io/) (`pip install openseespy`)
- NumPy
- SciPy
- Pandas

### Setup

```bash
pip install openseespy numpy scipy pandas
```

Clone or download this repository, then import the module:

```python
from opensees_vbi import initial_calcs, generate_vehicle_props_pool, main_vehicle_bridge
```

## Quick Start

```python
import numpy as np
from opensees_vbi import *

# Bridge: 25 m simply supported beam
bp, bric, ts, dt = initial_calcs(
    bridge_span_total=25.0, bridge_supports_locs=[0, 25],
    bridge_Iz=0.12, bridge_area=2.0, bridge_modulusE=27.5e9,
    bridge_damping_ratio=0.0, bridge_unit_mass=4800,
    bridge_disc_number=250, run_end_time=4.0, sampling_rate=500)

# Smooth road, no traffic
rr = road_roughness_generator(0, 0.05, [-10, 35])
tf = generate_random_traffic_nodal_forces(0, 251, ts)

# Quarter-car vehicle at 10 m/s
vp, vr = generate_vehicle_props_pool(
    ts, [-10, 0], [0.7, 1.3], [0.7, 1.3], [0.7, 1.3],
    base_vehicle_name='quarter_car_simple_Yang2004',
    num_gen_vehicle=1, base_speed=10.0)

# Coupled analysis
ba, va, forces, iters, vr_out = main_vehicle_bridge(
    bp, vp, tf, rr, bric, vr, ts, dt,
    vehicle_model='moving_mass')

# Bridge midspan displacement
mid = 125
print(f"Peak midspan displacement: {abs(ba['disp'][mid, :]).max() * 1e3:.2f} mm")
```

See `example_single_vehicle.py` for a complete working example with plotting.

## Vehicle Configurations

The file `vehicle_configs.json` contains ready-to-use vehicle configurations from the literature. To use a configuration, specify the vehicle name when calling `generate_vehicle_props_pool`:

```python
vp, vr = generate_vehicle_props_pool(
    ts, [-10, 0], [0.7, 1.3], [0.7, 1.3], [0.7, 1.3],
    base_vehicle_name='half_car_with_pitch_Yang2019',
    num_gen_vehicle=1, base_speed=10.0)
```

Available configurations:

| Name | Type | Source |
|------|------|--------|
| `quarter_car_simple_Yang2004` | Simple spring-mass | Yang et al. (2004) |
| `quarter_car_composite_Eshkevari2020_commercial` | Composite (536 kg) | Eshkevari et al. (2020) |
| `quarter_car_composite_Eshkevari2020_heavy` | Composite (18 t) | Eshkevari et al. (2020) |
| `quarter_car_composite_Cantero_V1` | Composite (NuBe V1) | Cantero et al. (2024) |
| `half_car_with_pitch_Yang2019` | Pitch, no axle mass | Yang et al. (2019) |
| `half_car_with_pitch_damped` | Pitch, with damping | Yang et al. (2019) adapted |
| `half_car_full_composite_Cantero_V2` | Full composite (NuBe V2) | Cantero et al. (2024) |
| `half_car_full_composite_OBrien` | Full composite | O'Brien et al. |

## Analysis Modes

### Coupled (iterative)

The coupled mode solves the vehicle and bridge as interacting systems. At each time step, the algorithm iterates between the bridge and vehicle solvers until the bridge displacement converges (measured by NRMSE). Use this mode when the vehicle mass is significant relative to the bridge mass.

```python
ba, va, forces, iters, vr = main_vehicle_bridge(
    bp, vp, tf, rr, bric, vr, ts, dt,
    vehicle_model='moving_mass')
```

### Decoupled (moving load)

The decoupled mode applies the vehicle's static weight as a moving load and solves the bridge in a single pass. The bridge response is then used as input to the vehicle model. This is faster but neglects the vehicle's dynamic feedback on the bridge.

```python
ba, va, forces, iters, vr = main_vehicle_bridge(
    bp, vp, tf, rr, bric, vr, ts, dt,
    vehicle_model='moving_load')
```

## Output Format

Both analysis modes return the same data structures:

- **`ba`** (bridge all): `dict` with keys `'time'`, `'disp'`, `'vel'`, `'acc'` — each is a 2D array of shape `(n_nodes, n_timesteps)`
- **`va`** (vehicle all): `dict` keyed by vehicle name (e.g., `'vehicle_1'`), each containing `'time'`, `'disp'`, `'vel'`, `'acc'` arrays of shape `(n_dofs, n_timesteps)`
- **`forces`**: interaction force history
- **`iters`**: number of iterations per time step (coupled mode only)
- **`vr`**: vehicle roughness input history

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{TalebiKalaleh2025opensees_vbi,
  title   = {An Open-Source Framework for Coupled Vehicle-Bridge Interaction
             Analysis Using {OpenSees}},
  author  = {Talebi-Kalaleh, Mohammad and Mei, Qipei},
  journal = {},
  year    = {2025},
  note    = {Manuscript in preparation}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Contact

**Mohammad Talebi-Kalaleh**
Department of Civil and Environmental Engineering
University of Alberta, Edmonton, Canada
Email: [talebika@ualberta.ca](mailto:talebika@ualberta.ca)

## References

- Yang, Y.B., Yau, J.D. & Wu, Y.S. (2004). *Vehicle-Bridge Interaction Dynamics*. World Scientific.
- Yang, Y.B., Wang, Z.L., Shi, K., Xu, H., Mo, X.Q. & Wu, Y.T. (2019). Two-axle test vehicle for bridges. *International Journal of Structural Stability and Dynamics*.
- Cantero, D. et al. (2024). NuBe: Numerical Benchmark for vehicle-bridge interaction. *Engineering Structures*.
- Eshkevari, S.S., Matarazzo, T.J. & Pakzad, S.N. (2020). Bridge modal identification using acceleration measurements within moving vehicles. *Mechanical Systems and Signal Processing*.
