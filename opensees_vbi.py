# OpenSees VBI Framework — Mohammad Talebi-Kalaleh, University of Alberta
"""
OpenSees Vehicle-Bridge Interaction (VBI) Framework
====================================================
A Python framework for coupled and decoupled vehicle-bridge interaction
analysis using OpenSees as the finite element solver.

Author: Mohammad Talebi-Kalaleh (talebika@ualberta.ca)
        University of Alberta

License: MIT
"""

import openseespy.opensees as ops
import numpy as np
import os
import copy
import json

import pandas as pd
from scipy.stats import truncnorm

######################################### main functions ###################################################
def initial_calcs(bridge_span_total, bridge_supports_locs, bridge_Iz, bridge_area, bridge_modulusE, bridge_damping_ratio, bridge_unit_mass,
                  bridge_disc_number, run_end_time, sampling_rate, damage_scenario=dict()):
    dt_analysis = 1 / sampling_rate
    number_nodes_bridge = bridge_disc_number + 1
    bridge_mesh_size = bridge_span_total / bridge_disc_number
    base_time_stamps = np.arange(0, run_end_time + dt_analysis, dt_analysis)
    bridge_nodes_dist = np.linspace(0, bridge_span_total, number_nodes_bridge)

    support_node_names = []
    for support_loc in bridge_supports_locs:
        support_node_names.append(int(np.argsort(abs(bridge_nodes_dist - support_loc))[0]) + 1)

    bridge_props = dict(span=bridge_span_total, Iz=bridge_Iz, sec_area=bridge_area, E_modulus=bridge_modulusE, damping_ratio=bridge_damping_ratio,
                        unit_mass=bridge_unit_mass, mesh_number=bridge_disc_number, bridge_mesh_size=bridge_mesh_size,
                        nodes_coord=bridge_nodes_dist, support_node_names=support_node_names, damage=damage_scenario)
    bridge_responses_initial_cond = dict(time=np.zeros((1, 1)), disp=np.zeros((number_nodes_bridge, 1)),
                                         vel=np.zeros((number_nodes_bridge, 1)), acc=np.zeros((number_nodes_bridge, 1)))

    return bridge_props, bridge_responses_initial_cond, base_time_stamps, dt_analysis

def generate_vehicle_props_pool(base_time_stamps, start_location_range_list, speed_variation_range_list,
                                mass_variation_range_list, spring_damp_variation_range_list,
                                base_vehicle_name='base_one_axle_Mei_2019', num_gen_vehicle=1, all_same_speed='no',
                                min_spacing=2, base_speed="default"):

    _config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vehicle_configs.json")
    with open(_config_path) as file:
        vehicles_config_dict = json.load(file)

    all_vehicle_props_locs = dict()
    vehicle_responses_psteps_all = dict()

    axle_dists = vehicles_config_dict[base_vehicle_name]['axle_local_coords'][-1]

    all_random_initial_locs = start_location_range_list[0]
    if num_gen_vehicle != 1:
        vehicles_spacing = max(2*axle_dists, min_spacing)
        all_random_initial_locs = np.sort(np.random.choice(list(range(start_location_range_list[0], start_location_range_list[1]-axle_dists+1, vehicles_spacing)),
                                                           size=num_gen_vehicle, replace=False))
    mean = (speed_variation_range_list[1] + speed_variation_range_list[0]) / 2
    std = speed_variation_range_list[1] - speed_variation_range_list[0]
    if all_same_speed == 'no':
        all_random_const_speed = np.sort(truncnorm.rvs((speed_variation_range_list[0]-mean) / std, (speed_variation_range_list[1]-mean)/std,
                                         loc=mean, scale=std, size=num_gen_vehicle))
    else:
        all_random_const_speed = np.ones(num_gen_vehicle)


    if base_speed == "default":
        base_speed = vehicles_config_dict[base_vehicle_name]['speed']

    for ID in range(num_gen_vehicle):  #here (higher speed and )
        if num_gen_vehicle == 1:
            start_loc = start_location_range_list[0]  # the rear axle is origin
            const_speed = base_speed

        else:
            start_loc = all_random_initial_locs[ID] # the back axle is origin
            const_speed = all_random_const_speed[ID] * base_speed

        rear_axle_location_history = start_loc + const_speed * base_time_stamps  # it does not have to be with constant speed

        mean = (mass_variation_range_list[1] + mass_variation_range_list[0]) / 2
        std = mass_variation_range_list[1] - mass_variation_range_list[0]
        mass_modif = truncnorm.rvs((mass_variation_range_list[0] - mean) / std, (mass_variation_range_list[1] - mean) / std,
                                   loc=mean, scale=std)
        mean = (spring_damp_variation_range_list[1] + spring_damp_variation_range_list[0]) / 2
        std = spring_damp_variation_range_list[1] - spring_damp_variation_range_list[0]
        spring_damping_modif = truncnorm.rvs((spring_damp_variation_range_list[0] - mean) / std,
                                             (spring_damp_variation_range_list[1] - mean) / std, loc=mean, scale=std)

        if 'simple' in vehicles_config_dict[base_vehicle_name]['opensees_model_name']:
            if num_gen_vehicle == 1:
                mass = vehicles_config_dict[base_vehicle_name]['mass']
                spring = vehicles_config_dict[base_vehicle_name]['stiff']
                damping = vehicles_config_dict[base_vehicle_name]['damping']

            else:
                mass = mass_modif * vehicles_config_dict[base_vehicle_name]['mass']
                spring = spring_damping_modif * vehicles_config_dict[base_vehicle_name]['stiff']
                damping = spring_damping_modif * vehicles_config_dict[base_vehicle_name]['damping']

            vehicle_props = dict(mass=mass,
                                 spring=spring,
                                 damping=damping,
                                 num_axles=len(vehicles_config_dict[base_vehicle_name]['axle_local_coords']),
                                 axle_local_coords=vehicles_config_dict[base_vehicle_name]['axle_local_coords'],
                                 model_opensees=vehicles_config_dict[base_vehicle_name]['opensees_model_name'],
                                 tot_num_dofs=vehicles_config_dict[base_vehicle_name]['tot_num_dofs'],
                                 axle_loc_rear=rear_axle_location_history,
                                 time_loc=base_time_stamps)

        elif vehicles_config_dict[base_vehicle_name]['opensees_model_name'] == 'one_axle_comp':
            if num_gen_vehicle == 1:
                mass_body = vehicles_config_dict[base_vehicle_name]['body_mass']
                mass_axle = vehicles_config_dict[base_vehicle_name]['axle_mass']

                spring_susp = vehicles_config_dict[base_vehicle_name]['susp_stiff']
                damping_susp = vehicles_config_dict[base_vehicle_name]['susp_damping']
                spring_tyre = vehicles_config_dict[base_vehicle_name]['tyre_stiff']
                damping_tyre = vehicles_config_dict[base_vehicle_name]['tyre_damp']

            else:
                mass_body = mass_modif * vehicles_config_dict[base_vehicle_name]['body_mass']
                mass_axle = mass_modif * vehicles_config_dict[base_vehicle_name]['axle_mass']

                spring_susp = spring_damping_modif * vehicles_config_dict[base_vehicle_name]['susp_stiff']
                damping_susp = spring_damping_modif * vehicles_config_dict[base_vehicle_name]['susp_damping']
                spring_tyre = spring_damping_modif * vehicles_config_dict[base_vehicle_name]['tyre_stiff']
                damping_tyre = spring_damping_modif * vehicles_config_dict[base_vehicle_name]['tyre_damp']

            vehicle_props = dict(mass_body=mass_body,
                                 mass_axle=mass_axle,
                                 spring_susp=spring_susp,
                                 damping_susp=damping_susp,
                                 spring_tyre=spring_tyre,
                                 damping_tyre=damping_tyre,
                                 num_axles=len(vehicles_config_dict[base_vehicle_name]['axle_local_coords']),
                                 axle_local_coords=vehicles_config_dict[base_vehicle_name]['axle_local_coords'],
                                 model_opensees=vehicles_config_dict[base_vehicle_name]['opensees_model_name'],
                                 tot_num_dofs=vehicles_config_dict[base_vehicle_name]['tot_num_dofs'],
                                 axle_loc_rear=rear_axle_location_history,
                                 time_loc=base_time_stamps)
        elif vehicles_config_dict[base_vehicle_name]['opensees_model_name'] == 'two_axle_comp1':
            if num_gen_vehicle == 1:
                mass_body = vehicles_config_dict[base_vehicle_name]['body_mass']
                mass_axle = vehicles_config_dict[base_vehicle_name]['axle_mass']

                spring_susp = vehicles_config_dict[base_vehicle_name]['susp_stiff']
                damping_susp = vehicles_config_dict[base_vehicle_name]['susp_damping']
                spring_tyre = vehicles_config_dict[base_vehicle_name]['tyre_stiff']
                damping_tyre = vehicles_config_dict[base_vehicle_name]['tyre_damp']

            else:
                mass_body = mass_modif * vehicles_config_dict[base_vehicle_name]['body_mass']
                mass_axle = mass_modif * vehicles_config_dict[base_vehicle_name]['axle_mass']

                spring_susp = spring_damping_modif * vehicles_config_dict[base_vehicle_name]['susp_stiff']
                damping_susp = spring_damping_modif * vehicles_config_dict[base_vehicle_name]['susp_damping']
                spring_tyre = spring_damping_modif * vehicles_config_dict[base_vehicle_name]['tyre_stiff']
                damping_tyre = spring_damping_modif * vehicles_config_dict[base_vehicle_name]['tyre_damp']

            vehicle_props = dict(mass_body=mass_body,
                                 mass_axle=mass_axle,
                                 spring_susp=spring_susp,
                                 damping_susp=damping_susp,
                                 spring_tyre=spring_tyre,
                                 damping_tyre=damping_tyre,
                                 num_axles=len(vehicles_config_dict[base_vehicle_name]['axle_local_coords']),
                                 axle_local_coords=vehicles_config_dict[base_vehicle_name]['axle_local_coords'],
                                 model_opensees=vehicles_config_dict[base_vehicle_name]['opensees_model_name'],
                                 tot_num_dofs=vehicles_config_dict[base_vehicle_name]['tot_num_dofs'],
                                 axle_loc_rear=rear_axle_location_history,
                                 time_loc=base_time_stamps)
        elif vehicles_config_dict[base_vehicle_name]['opensees_model_name'] == 'two_axle_comp2':
            if num_gen_vehicle == 1:
                mass_body = vehicles_config_dict[base_vehicle_name]['body_mass']

                pitch_moment_inertia = vehicles_config_dict[base_vehicle_name]['pitch_moment_inertia']

                spring_susp_front = vehicles_config_dict[base_vehicle_name]['susp_stiff_front']
                spring_susp_rear = vehicles_config_dict[base_vehicle_name]['susp_stiff_rear']

                damping_susp_front = vehicles_config_dict[base_vehicle_name]['susp_damping_front']
                damping_susp_rear = vehicles_config_dict[base_vehicle_name]['susp_damping_rear']


            else:
                mass_body = mass_modif * vehicles_config_dict[base_vehicle_name]['body_mass']
                pitch_moment_inertia = mass_modif * vehicles_config_dict[base_vehicle_name]['pitch_moment_inertia']

                spring_susp_front = spring_damping_modif * vehicles_config_dict[base_vehicle_name]['susp_stiff_front']
                spring_susp_rear = spring_damping_modif * vehicles_config_dict[base_vehicle_name]['susp_stiff_rear']

                damping_susp_front = spring_damping_modif * vehicles_config_dict[base_vehicle_name]['susp_damping_front']
                damping_susp_rear = spring_damping_modif * vehicles_config_dict[base_vehicle_name]['susp_damping_rear']


            vehicle_props = dict(mass_body=mass_body,
                                 pitch_moment_inertia=pitch_moment_inertia,
                                 spring_susp_front=spring_susp_front,
                                 spring_susp_rear=spring_susp_rear,
                                 damping_susp_front=damping_susp_front,
                                 damping_susp_rear=damping_susp_rear,
                                 num_axles=len(vehicles_config_dict[base_vehicle_name]['axle_local_coords']),
                                 axle_local_coords=vehicles_config_dict[base_vehicle_name]['axle_local_coords'],
                                 centermass_local_coord=vehicles_config_dict[base_vehicle_name]['center_mass_local_coord'],
                                 model_opensees=vehicles_config_dict[base_vehicle_name]['opensees_model_name'],
                                 tot_num_dofs=vehicles_config_dict[base_vehicle_name]['tot_num_dofs'],
                                 axle_loc_rear=rear_axle_location_history,
                                 time_loc=base_time_stamps)

        elif vehicles_config_dict[base_vehicle_name]['opensees_model_name'] == 'two_axle_comp3':
            if num_gen_vehicle == 1:
                mass_body = vehicles_config_dict[base_vehicle_name]['body_mass']
                mass_axle_front = vehicles_config_dict[base_vehicle_name]['axle_mass_front']
                mass_axle_rear = vehicles_config_dict[base_vehicle_name]['axle_mass_rear']
                pitch_moment_inertia = vehicles_config_dict[base_vehicle_name]['pitch_moment_inertia']

                spring_susp_front = vehicles_config_dict[base_vehicle_name]['susp_stiff_front']
                spring_susp_rear = vehicles_config_dict[base_vehicle_name]['susp_stiff_rear']

                damping_susp_front = vehicles_config_dict[base_vehicle_name]['susp_damping_front']
                damping_susp_rear = vehicles_config_dict[base_vehicle_name]['susp_damping_rear']

                spring_tyre_front = vehicles_config_dict[base_vehicle_name]['tyre_stiff_front']
                spring_tyre_rear = vehicles_config_dict[base_vehicle_name]['tyre_stiff_rear']

                damping_tyre_front = vehicles_config_dict[base_vehicle_name]['tyre_damping_front']
                damping_tyre_rear = vehicles_config_dict[base_vehicle_name]['tyre_damping_rear']

            else:
                mass_body = mass_modif * vehicles_config_dict[base_vehicle_name]['body_mass']
                mass_axle_front = mass_modif * vehicles_config_dict[base_vehicle_name]['axle_mass_front']
                mass_axle_rear = mass_modif * vehicles_config_dict[base_vehicle_name]['axle_mass_rear']
                pitch_moment_inertia = mass_modif * vehicles_config_dict[base_vehicle_name]['pitch_moment_inertia']

                spring_susp_front = spring_damping_modif * vehicles_config_dict[base_vehicle_name]['susp_stiff_front']
                spring_susp_rear = spring_damping_modif * vehicles_config_dict[base_vehicle_name]['susp_stiff_rear']

                damping_susp_front = spring_damping_modif * vehicles_config_dict[base_vehicle_name]['susp_damping_front']
                damping_susp_rear = spring_damping_modif * vehicles_config_dict[base_vehicle_name]['susp_damping_rear']

                spring_tyre_front = spring_damping_modif * vehicles_config_dict[base_vehicle_name]['tyre_stiff_front']
                spring_tyre_rear = spring_damping_modif * vehicles_config_dict[base_vehicle_name]['tyre_stiff_rear']

                damping_tyre_front = spring_damping_modif * vehicles_config_dict[base_vehicle_name]['tyre_damping_front']
                damping_tyre_rear = spring_damping_modif * vehicles_config_dict[base_vehicle_name]['tyre_damping_rear']

            vehicle_props = dict(mass_body=mass_body,
                                 mass_axle_front=mass_axle_front,
                                 mass_axle_rear=mass_axle_rear,
                                 pitch_moment_inertia=pitch_moment_inertia,
                                 spring_susp_front=spring_susp_front,
                                 spring_susp_rear=spring_susp_rear,
                                 damping_susp_front=damping_susp_front,
                                 damping_susp_rear=damping_susp_rear,
                                 spring_tyre_front=spring_tyre_front,
                                 spring_tyre_rear=spring_tyre_rear,
                                 damping_tyre_front=damping_tyre_front,
                                 damping_tyre_rear=damping_tyre_rear,
                                 num_axles=len(vehicles_config_dict[base_vehicle_name]['axle_local_coords']),
                                 axle_local_coords=vehicles_config_dict[base_vehicle_name]['axle_local_coords'],
                                 centermass_local_coord=vehicles_config_dict[base_vehicle_name]['center_mass_local_coord'],
                                 model_opensees=vehicles_config_dict[base_vehicle_name]['opensees_model_name'],
                                 tot_num_dofs=vehicles_config_dict[base_vehicle_name]['tot_num_dofs'],
                                 axle_loc_rear=rear_axle_location_history,
                                 time_loc=base_time_stamps)

        #initialization
        all_vehicle_props_locs['vehicle_' + str(ID + 1)] = vehicle_props
        tot_measured_dofs = vehicles_config_dict[base_vehicle_name]['tot_num_dofs']
        vehicle_responses_psteps_all['vehicle_' + str(ID + 1)] = dict(time=np.zeros((1, 1)),
                                                                      disp=np.zeros((tot_measured_dofs, 1)),
                                                                      vel=np.zeros((tot_measured_dofs, 1)),
                                                                      acc=np.zeros((tot_measured_dofs, 1)),
                                                                      reaction_force=np.zeros((vehicle_props['num_axles'], 1)))
    return all_vehicle_props_locs, vehicle_responses_psteps_all

def road_roughness_generator(Gdn0, road_roughness_mesh_size, analysis_road_window_range, n_l=0.01, n_u = 10, delta_n=0.01,
                             refine_method="moving_average", window_size=5):
    n0 = 0.1
    w = 2
    x = np.arange(analysis_road_window_range[0], analysis_road_window_range[1]+0.001, road_roughness_mesh_size)
    N = x.shape[0]

    n = np.arange(n_l, n_u+0.00001, delta_n)  # np.arange(1, 100.00001, delta_n)

    ''' Backup
    delta_n = 1 / L  # 0.04
    n_max = delta_n * N
    n = np.arange(0.01, n_max+0.00001, delta_n)  # np.arange(1, 100.00001, delta_n)
    '''

    Gdn = Gdn0 * (n / n0) ** (-w)
    A = (2 * Gdn * delta_n) ** 0.5
    phi = np.array([np.random.uniform(0, 2 * np.pi) for _ in range(n.shape[0])])

    y = np.zeros(N)
    for i in range(N):
        y[i] = sum(A * np.cos(2 * np.pi * n * x[i] + phi))

    if refine_method != None:
        random_signal_series = pd.Series(y)
        # Calculate the moving average using pandas rolling function
        moving_average = random_signal_series.rolling(window=window_size, min_periods=1).mean()
        # Convert the moving average pandas Series to a NumPy array
        moving_average_y = moving_average.values
        y = copy.deepcopy(moving_average_y)
    road_rough_profile = np.vstack((x, y)).T

    road_rough_profile = dict(loc=road_rough_profile[:, 0], value=road_rough_profile[:, 1])

    return road_rough_profile

def generate_random_traffic_nodal_forces(total_other_vehicles_mass, number_nodes_bridge, base_time_stamps):
    #  Generate a force matrix with Gaussian-distributed values
    g = 9.81
    random_nodal_force = np.random.rand(number_nodes_bridge, base_time_stamps.shape[0])
    scaled_random_force = total_other_vehicles_mass / np.sum(random_nodal_force, axis=0)
    random_traffic_force_time = dict(time=base_time_stamps, force=random_nodal_force * scaled_random_force * g)

    return random_traffic_force_time


def bridge_dynamic_step_forward_opensees(nodal_forces_current, bridge_responses_psteps, bridge_props,
                                         dt_analysis, num_steps_analysis=1, do_modal_analysis='no', plot_mode_shapes=False):
    # Define bridge model opensees
    # All the units are in N, m, Pa
    # tested it is working very well either step by step or all steps in the same time
    # multi-span bridges are also supported

    ops.wipe()  # clear memory of past model definitions
    num_modes_recorded = 5
    ops.model('BasicBuilder', '-ndm', 2, '-ndf', 3)  # Define the model builder, ndm = #dimension, ndf = #dofs

    # define nodal coordinates
    number_nodes = bridge_props['mesh_number'] + 1
    for j in range(1, number_nodes + 1):
        ops.node(j, (j - 1) * bridge_props['bridge_mesh_size'], 0)
    all_node_names = ops.getNodeTags()
    # Boundary Conditions -- assign hinge and roller supports
    for i, node_name in enumerate(bridge_props['support_node_names']):
        if i == 0:
            ops.fix(node_name, 1, 1, 0)
        else:
            ops.fix(node_name, 0, 1, 0)

    # assign mass
    bridge_segment_mass = bridge_props['unit_mass'] * bridge_props['bridge_mesh_size']
    ops.mass(1, 0., bridge_segment_mass / 2., 0.)
    for j in range(2, number_nodes):
        ops.mass(j, 0., bridge_segment_mass, 0.)
    ops.mass(number_nodes, 0., bridge_segment_mass / 2., 0.)

    # define geometric transformation
    TransfTag = 1
    ops.geomTransf('Linear', TransfTag)

    # Define Elements
    if len(bridge_props['damage']) == 0:
        for j in range(1, number_nodes):
            ops.element('elasticBeamColumn', j, j, j + 1, bridge_props['sec_area'], bridge_props['E_modulus'], bridge_props['Iz'], TransfTag)
    else:
        for j in range(1, number_nodes):
            modif = 1
            if j in bridge_props['damage']['elements_names']:
                index = bridge_props['damage']['elements_names'].index(j)
                modif = bridge_props['damage']['damage_severity'][index]
            ops.element('elasticBeamColumn', j, j, j + 1, bridge_props['sec_area'], bridge_props['E_modulus']*modif, bridge_props['Iz'], TransfTag)

    # Modal analysis
    if do_modal_analysis != 'no':
        omega2 = ops.eigen(num_modes_recorded)
        omega_n = [(om ** 0.5) for om in omega2]

        natural_freqs = [omega / (2 * np.pi) for omega in omega_n]

        mode_shapes_lists = []
        for i in range(num_modes_recorded):
            mode_shape = np.array([ops.nodeEigenvector(nodeTag, i+1, 2) for nodeTag in all_node_names])
            mode_shape = mode_shape / max(abs(mode_shape))
            mode_shapes_lists.append(list(mode_shape))
        bridge_props['exact_mode_shapes'] = mode_shapes_lists

    else:
        natural_freqs = bridge_props['natural_freqs']
        omega_n = [2 * np.pi * freq for freq in natural_freqs]


    # generating timeSeries (nodal dynamic forces)
    for j in range(1, number_nodes + 1):
        if nodal_forces_current['force'].ndim == 1:  #constant is working for step by step
            ops.timeSeries('Constant', j, '-factor', nodal_forces_current['force'][j - 1])

        else:
            ops.timeSeries('Path', j, '-time', *nodal_forces_current['time'].tolist(),
                                      '-values', *nodal_forces_current['force'][j - 1, :].tolist())
        ops.pattern('Plain', j, j)
        ops.load(j, *[0., -1.0, 0.])



    # dynamic analysis settings
    alphaM = bridge_props['damping_ratio'] * (2. * omega_n[0] * omega_n[1]) / (omega_n[0] + omega_n[1])
    betaKcomm = 2. * bridge_props['damping_ratio'] / (omega_n[0] + omega_n[1])


    ops.rayleigh(alphaM, 0., 0., betaKcomm)
    ops.constraints('Transformation')
    ops.numberer('Plain')
    ops.system('BandGeneral')
    ops.algorithm('Linear')
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')

    # dynamic analysis and outputs
    #  last step response
    ops.setTime(bridge_responses_psteps['time'][0, -1])
    for j in range(1, number_nodes + 1):
        ops.setNodeDisp(j, 2, bridge_responses_psteps['disp'][j - 1, -1], '-commit')
        ops.setNodeVel(j, 2, bridge_responses_psteps['vel'][j - 1, -1], '-commit')
        ops.setNodeAccel(j, 2, bridge_responses_psteps['acc'][j - 1, -1], '-commit')

    # all_node_tages = ops.getNodeTags()
    bridge_responses_psteps_new = copy.deepcopy(bridge_responses_psteps)

    time_vector = np.zeros((1, num_steps_analysis))
    all_nodes_disp = np.zeros((number_nodes, num_steps_analysis))
    all_nodes_vel = np.zeros((number_nodes, num_steps_analysis))
    all_nodes_acc = np.zeros((number_nodes, num_steps_analysis))
    for step in range(num_steps_analysis):
        ops.analyze(1, dt_analysis)

        time_vector[0, step] = ops.getTime()
        for node_tag in range(number_nodes):
            all_nodes_disp[node_tag, step] = ops.nodeDisp(node_tag + 1, 2)
            all_nodes_vel[node_tag, step] = ops.nodeVel(node_tag + 1, 2)
            all_nodes_acc[node_tag, step] = ops.nodeAccel(node_tag + 1, 2)

    bridge_responses_psteps_new['time'] = np.concatenate((bridge_responses_psteps_new['time'], time_vector), axis=1)
    bridge_responses_psteps_new['disp'] = np.concatenate((bridge_responses_psteps_new['disp'], all_nodes_disp), axis=1)
    bridge_responses_psteps_new['vel'] = np.concatenate((bridge_responses_psteps_new['vel'], all_nodes_vel), axis=1)
    bridge_responses_psteps_new['acc'] = np.concatenate((bridge_responses_psteps_new['acc'], all_nodes_acc), axis=1)

    ops.wipe()
    return bridge_responses_psteps_new, natural_freqs


def vehicle_dynamic_step_forward(vehicle_responses_psteps, vehicle_props, vehicle_input, dt_analysis, num_steps_analysis=1,
                                 analysis_type='dynamic'):
    # tested it is working very well either step by step or all steps in the same time

    ops.wipe()
    g = 9.81
    if 'simple' in vehicle_props['model_opensees']:
            ops.model('basic', '-ndm', 1)
            # defines
            num_axles = len(vehicle_props['axle_local_coords'])
            num_modes = vehicle_props['tot_num_dofs'] - num_axles
            axles_node_names_LeftToRight = list(range(1, num_axles+1))
            active_dof_axles = 1
            not_axle_nodes_names = list(range(num_axles+1, vehicle_props['tot_num_dofs']+1))
            active_dofs_for_not_axle_nodes = [1] * len(not_axle_nodes_names)

            # define nodes, boundry condition
            for ntag in axles_node_names_LeftToRight:
                ops.node(ntag, 0)
                ops.fix(ntag, 1)

            for ntag in not_axle_nodes_names:
                ops.node(ntag, 0)
                ops.mass(ntag, vehicle_props['mass'])

            # define elements and materials
            ops.uniaxialMaterial('Elastic', 1, vehicle_props['spring'], vehicle_props['damping'])
            for i, _ in enumerate(axles_node_names_LeftToRight):
                ops.element('zeroLength', i, axles_node_names_LeftToRight[i], not_axle_nodes_names[i], '-mat', 1, '-dir', 1)

            if analysis_type != 'dynamic':
                # only for static analysis
                ops.timeSeries('Linear', 110)
                ops.pattern('Plain', 100, 110)
                for ntag in not_axle_nodes_names:
                    ops.load(ntag, -vehicle_props['mass']*g)

    elif vehicle_props['model_opensees'] == 'one_axle_comp':
            ops.model('basic', '-ndm', 1)

            # defines
            num_modes = 2
            axles_node_names_LeftToRight = [1]
            active_dof_axles = 1
            not_axle_nodes_names = [2, 3]
            active_dofs_for_not_axle_nodes = [1, 1]

            # define nodes
            ops.node(1, 0)
            ops.node(2, 0)
            ops.node(3, 0)

            # boundry condition
            ops.fix(1, 1)
            # define mass
            ops.mass(2, vehicle_props['mass_axle'])
            ops.mass(3, vehicle_props['mass_body'])
            # define elements and materials
            ops.uniaxialMaterial('Elastic', 1, vehicle_props['spring_tyre'], vehicle_props['damping_tyre'])
            ops.uniaxialMaterial('Elastic', 2, vehicle_props['spring_susp'], vehicle_props['damping_susp'])
            ops.element('zeroLength', 1, 1, 2, '-mat', 1, '-dir', 1)
            ops.element('zeroLength', 2, 2, 3, '-mat', 2, '-dir', 1)

            if analysis_type != 'dynamic':
                # only for static analysis
                ops.timeSeries('Linear', 110)
                ops.pattern('Plain', 100, 110)
                ops.load(2, -vehicle_props['mass_axle']*g)
                ops.load(3, -vehicle_props['mass_body']*g)

    elif vehicle_props['model_opensees'] == 'two_axle_comp1':
            ops.model('basic', '-ndm', 1)

            # defines
            num_modes = 4
            axles_node_names_LeftToRight = [1, 2]
            active_dof_axles = 1
            not_axle_nodes_names = [3, 4, 5, 6]
            active_dofs_for_not_axle_nodes = [1, 1, 1, 1]

            # define nodes
            ops.node(1, 0)
            ops.node(3, 0)
            ops.node(5, 0)

            ops.node(2, 0)
            ops.node(4, 0)
            ops.node(6, 0)

            # boundry condition
            ops.fix(1, 1)
            ops.fix(2, 1)
            # define mass
            ops.mass(3, vehicle_props['mass_axle'])
            ops.mass(5, vehicle_props['mass_body'])

            ops.mass(4, vehicle_props['mass_axle'])
            ops.mass(6, vehicle_props['mass_body'])
            # define elements and materials
            ops.uniaxialMaterial('Elastic', 1, vehicle_props['spring_tyre'], vehicle_props['damping_tyre'])
            ops.uniaxialMaterial('Elastic', 2, vehicle_props['spring_tyre'], vehicle_props['damping_tyre'])

            ops.uniaxialMaterial('Elastic', 3, vehicle_props['spring_susp'], vehicle_props['damping_susp'])
            ops.uniaxialMaterial('Elastic', 4, vehicle_props['spring_susp'], vehicle_props['damping_susp'])


            ops.element('zeroLength', 1, 1, 3, '-mat', 1, '-dir', 1)
            ops.element('zeroLength', 2, 2, 4, '-mat', 2, '-dir', 1)

            ops.element('zeroLength', 3, 3, 5, '-mat', 3, '-dir', 1)
            ops.element('zeroLength', 4, 4, 6, '-mat', 4, '-dir', 1)

            if analysis_type != 'dynamic':
                # only for static analysis
                ops.timeSeries('Linear', 110)
                ops.pattern('Plain', 100, 110)
                ops.load(3, -vehicle_props['mass_axle']*g)
                ops.load(5, -vehicle_props['mass_body']*g)
                ops.load(4, -vehicle_props['mass_axle']*g)
                ops.load(6, -vehicle_props['mass_body']*g)

    elif vehicle_props['model_opensees'] == 'two_axle_comp2':
        ops.model('BasicBuilder', '-ndm', 2, '-ndf', 3)

        # defines
        num_modes = 2
        axles_node_names_LeftToRight = [1, 2]
        active_dof_axles = 2
        not_axle_nodes_names = [5, 6, 7, 8, 9, 9]
        active_dofs_for_not_axle_nodes = [2, 2, 2, 2, 2, 3]

        # define nodes
        body_center_height = 0.
        ops.node(1, vehicle_props['axle_local_coords'][0], 0)
        ops.node(2, vehicle_props['axle_local_coords'][1], 0)

        ops.node(5, vehicle_props['axle_local_coords'][0], body_center_height)
        ops.node(6, vehicle_props['axle_local_coords'][1], body_center_height)

        ops.node(7, vehicle_props['axle_local_coords'][0], body_center_height)
        ops.node(9, vehicle_props['centermass_local_coord'], body_center_height)
        ops.node(8, vehicle_props['axle_local_coords'][1], body_center_height)

        # boundary condition
        ops.fix(1, *[1, 1, 1])
        ops.fix(5, *[1, 0, 1])

        ops.fix(2, *[1, 1, 1])
        ops.fix(6, *[1, 0, 1])

        ops.fix(9, *[1, 0, 0])


        # define mass
        ops.mass(9, *[0, vehicle_props['mass_body'], vehicle_props['pitch_moment_inertia']])   # Body
        # define elements and materials

        ops.uniaxialMaterial('Elastic', 3, vehicle_props['spring_susp_rear'], vehicle_props['damping_susp_rear'])
        ops.uniaxialMaterial('Elastic', 4, vehicle_props['spring_susp_front'], vehicle_props['damping_susp_front'])
        ops.uniaxialMaterial('Elastic', 5, vehicle_props['spring_susp_front']*10e5)




        ops.element('twoNodeLink', 3, *[1, 5], '-mat', 3, '-dir', 2)
        ops.element('twoNodeLink', 4, *[2, 6], '-mat', 4, '-dir', 2)

        ops.element('twoNodeLink', 5, *[7, 9], '-mat', *[5, 5, 5], '-dir', *[1, 2, 3])
        ops.element('twoNodeLink', 6, *[9, 8], '-mat', *[5, 5, 5], '-dir', *[1, 2, 3])

        ops.element('zeroLength', 7, *[5, 7], '-mat', *[5, 5], '-dir', *[1, 2])
        ops.element('zeroLength', 8, *[6, 8], '-mat', *[5, 5], '-dir', *[1, 2])

        if analysis_type != 'dynamic':
            # only for static analysis
            ops.timeSeries('Linear', 110)
            ops.pattern('Plain', 100, 110)
            ops.load(9, 0., -vehicle_props['mass_body'] * g, 0.)



    elif vehicle_props['model_opensees'] == 'two_axle_comp3':
        ops.model('BasicBuilder', '-ndm', 2, '-ndf', 3)

        # defines
        num_modes = 3
        axles_node_names_LeftToRight = [1, 2]
        active_dof_axles = 2
        not_axle_nodes_names = [3, 4, 5, 6, 7, 8, 9, 9]
        active_dofs_for_not_axle_nodes = [2, 2, 2, 2, 2, 2, 2, 3]

        # define nodes
        body_center_height = 0.
        ops.node(1, vehicle_props['axle_local_coords'][0], 0)
        ops.node(2, vehicle_props['axle_local_coords'][1], 0)

        ops.node(3, vehicle_props['axle_local_coords'][0], 0)
        ops.node(4, vehicle_props['axle_local_coords'][1], 0)

        ops.node(5, vehicle_props['axle_local_coords'][0], body_center_height)
        ops.node(6, vehicle_props['axle_local_coords'][1], body_center_height)

        ops.node(7, vehicle_props['axle_local_coords'][0], body_center_height)
        ops.node(9, vehicle_props['centermass_local_coord'], body_center_height)
        ops.node(8, vehicle_props['axle_local_coords'][1], body_center_height)

        # boundary condition
        ops.fix(1, *[1, 1, 1])
        ops.fix(3, *[1, 0, 1])
        ops.fix(5, *[1, 0, 1])

        ops.fix(2, *[1, 1, 1])
        ops.fix(4, *[1, 0, 1])
        ops.fix(6, *[1, 0, 1])

        ops.fix(9, *[1, 0, 0])


        # define mass
        ops.mass(3, *[0, vehicle_props['mass_axle_rear'], 0])   # rear axle
        ops.mass(4, *[0, vehicle_props['mass_axle_front'], 0])   # front axle
        ops.mass(9, *[0, vehicle_props['mass_body'], vehicle_props['pitch_moment_inertia']])   # Body
        # define elements and materials
        ops.uniaxialMaterial('Elastic', 1, vehicle_props['spring_tyre_rear'], vehicle_props['damping_tyre_rear'])
        ops.uniaxialMaterial('Elastic', 2, vehicle_props['spring_tyre_front'], vehicle_props['damping_tyre_front'])
        ops.uniaxialMaterial('Elastic', 3, vehicle_props['spring_susp_rear'], vehicle_props['damping_susp_rear'])
        ops.uniaxialMaterial('Elastic', 4, vehicle_props['spring_susp_front'], vehicle_props['damping_susp_front'])
        ops.uniaxialMaterial('Elastic', 5, vehicle_props['spring_susp_front']*10e5)


        ops.element('twoNodeLink', 1, *[1, 3], '-mat', 1, '-dir', 2)
        ops.element('twoNodeLink', 2, *[2, 4], '-mat', 2, '-dir', 2)

        ops.element('twoNodeLink', 3, *[3, 5], '-mat', 3, '-dir', 2)
        ops.element('twoNodeLink', 4, *[4, 6], '-mat', 4, '-dir', 2)

        ops.element('twoNodeLink', 5, *[7, 9], '-mat', *[5, 5, 5], '-dir', *[1, 2, 3])
        ops.element('twoNodeLink', 6, *[9, 8], '-mat', *[5, 5, 5], '-dir', *[1, 2, 3])

        ops.element('zeroLength', 7, *[5, 7], '-mat', *[5, 5], '-dir', *[1, 2])
        ops.element('zeroLength', 8, *[6, 8], '-mat', *[5, 5], '-dir', *[1, 2])

        if analysis_type != 'dynamic':
            # only for static analysis
            ops.timeSeries('Linear', 110)
            ops.pattern('Plain', 100, 110)
            ops.load(3, 0., -vehicle_props['mass_axle_rear'] * g, 0.)
            ops.load(4, 0., -vehicle_props['mass_axle_front'] * g, 0.)
            ops.load(9, 0., -vehicle_props['mass_body'] * g, 0.)


    # initial calcs
    all_nodes_tags = axles_node_names_LeftToRight + not_axle_nodes_names
    all_active_dof_nodes = [active_dof_axles] * vehicle_props['num_axles'] + active_dofs_for_not_axle_nodes

    if analysis_type == 'dynamic':
        # Modal analysis
        omega2 = ops.eigen('-fullGenLapack', num_modes)
        omega_n = [(om ** 0.5) for om in omega2]
        natural_freqs = [omega / (2 * np.pi) for omega in omega_n]

        # support excitation (all axles will be excited)
        ops.pattern('MultipleSupport', 1)
        for i, axle_node_name in enumerate(axles_node_names_LeftToRight):
            if vehicle_input['disp'].shape[1] == 1:  # this is working very well in step by step

                ops.timeSeries('Constant', 2*i+1, '-factor', vehicle_input['disp'][i, 0])
                ops.timeSeries('Constant', 2*i+2, '-factor', vehicle_input['vel'][i, 0])


            else:
                ops.timeSeries('Path', 2*i+1, '-time', *vehicle_input['time'].tolist(), '-values', *vehicle_input['disp'][i, :].tolist())
                ops.timeSeries('Path', 2*i+2, '-time', *vehicle_input['time'].tolist(), '-values', *vehicle_input['vel'][i, :].tolist())
            ops.groundMotion(i+1, 'Plain', '-disp', 2*i+1, '-vel', 2*i+2)
            ops.imposedMotion(axle_node_name, active_dof_axles, i+1)

        #  last step response
        ops.setTime(vehicle_responses_psteps['time'][0, -1])
        # all nodes must be measured
        for i, node_tag in enumerate(not_axle_nodes_names):
            active_dof = active_dofs_for_not_axle_nodes[i]
            ops.setNodeDisp(node_tag, active_dof, vehicle_responses_psteps['disp'][i + vehicle_props['num_axles'], -1], '-commit')
            ops.setNodeVel(node_tag, active_dof, vehicle_responses_psteps['vel'][i + vehicle_props['num_axles'], -1], '-commit')
            ops.setNodeAccel(node_tag, active_dof, vehicle_responses_psteps['acc'][i + vehicle_props['num_axles'], -1], '-commit')

        # dynamic analysis settings

        ops.constraints('Transformation')
        ops.numberer('RCM')
        ops.system('BandGeneral')
        ops.algorithm('Linear')
        ops.integrator('Newmark', 0.5, 0.25)
        ops.analysis('Transient')


        # dynamic analysis (forward) and outputs
        time_vector = np.zeros((1, num_steps_analysis))
        disp = np.zeros((len(all_nodes_tags), num_steps_analysis))
        vel = np.zeros((len(all_nodes_tags), num_steps_analysis))
        acc = np.zeros((len(all_nodes_tags), num_steps_analysis))
        reaction_force = np.zeros((vehicle_props['num_axles'], num_steps_analysis))
        for step in range(num_steps_analysis):
            ops.analyze(1, dt_analysis)

            time_vector[step] = ops.getTime()
            for i, node_tag in enumerate(all_nodes_tags):
                active_dof = all_active_dof_nodes[i]
                disp[i, step] = ops.nodeDisp(node_tag, active_dof)
                vel[i, step] = ops.nodeVel(node_tag, active_dof)
                acc[i, step] = ops.nodeAccel(node_tag, active_dof)

            for i, node_tag in enumerate(axles_node_names_LeftToRight):
                ops.reactions('-dynamic', '-rayleigh')
                reaction_force[i, step] = ops.nodeReaction(node_tag, active_dof_axles)

        vehicle_responses_psteps_new = copy.deepcopy(vehicle_responses_psteps)
        vehicle_responses_psteps_new['time'] = np.concatenate((vehicle_responses_psteps_new['time'], time_vector), axis=1)
        vehicle_responses_psteps_new['disp'] = np.concatenate((vehicle_responses_psteps_new['disp'], disp), axis=1)
        vehicle_responses_psteps_new['vel'] = np.concatenate((vehicle_responses_psteps_new['vel'], vel), axis=1)
        vehicle_responses_psteps_new['acc'] = np.concatenate((vehicle_responses_psteps_new['acc'], acc), axis=1)
        vehicle_responses_psteps_new['reaction_force'] = np.concatenate((vehicle_responses_psteps_new['reaction_force'], reaction_force),
                                                                        axis=1)
        vehicle_responses_psteps_new['recorded_node_names'] = all_nodes_tags
        vehicle_responses_psteps_new['natural_freqs'] = natural_freqs
    else:
        ops.constraints('Transformation')  # how it handles boundary conditions
        ops.numberer('RCM')  # renumber dof's to minimize band-width (optimization), if you want to
        ops.system('BandGeneral')  # how to store and solve the system of equations in the analysis
        ops.algorithm('Linear')  # use Linear algorithm for linear analysis
        ops.integrator('LoadControl', 1)  # determine the next time step for an analysis, # apply gravity in 1 step
        ops.analysis('Static')  # define type of analysis static or transient
        ops.analyze(1)
        axles_static_forces = np.zeros(vehicle_props['num_axles'])
        for i, node_tag in enumerate(axles_node_names_LeftToRight):
            ops.reactions()
            axles_static_forces[i] = ops.nodeReaction(node_tag, active_dof_axles)

        vehicle_responses_psteps_new = copy.deepcopy(vehicle_responses_psteps)
        vehicle_responses_psteps_new['axles_static_forces'] = axles_static_forces

    ops.wipe()
    return vehicle_responses_psteps_new

def determine_vehicle_location_roughness_alt(road_rough_profile, all_vehicle_props_locs):
    # roughness input tested, it is assigning the lagged responses correctly
    vehicle_axles_roughness_input = dict()
    for vehicle_id in all_vehicle_props_locs.keys():
        num_axles = all_vehicle_props_locs[vehicle_id]['num_axles']
        axle_local_coords_list = all_vehicle_props_locs[vehicle_id]['axle_local_coords']
        vehicle_loc_time_stamps = all_vehicle_props_locs[vehicle_id]['time_loc']
        vehicle_axles_roughness_input[vehicle_id] = dict()
        vehicle_axles_roughness_input[vehicle_id]['rough_input'] = np.empty((num_axles, vehicle_loc_time_stamps.shape[0]))
        vehicle_axles_roughness_input[vehicle_id]['rough_input_derv'] = np.empty((num_axles, vehicle_loc_time_stamps.shape[0]))

        for i in range(num_axles):
            current_axle_loc_history = all_vehicle_props_locs[vehicle_id]['axle_loc_rear'] + axle_local_coords_list[i]
            vehicle_axles_roughness_input[vehicle_id]['rough_input'][i, :] = np.interp(current_axle_loc_history, road_rough_profile['loc'], road_rough_profile['value'])
            vehicle_axles_roughness_input[vehicle_id]['rough_input_derv'][i, :] = np.gradient(vehicle_axles_roughness_input[vehicle_id]['rough_input'][i, :], vehicle_loc_time_stamps)
    return vehicle_axles_roughness_input

def determine_vehicle_location_roughness(road_rough_profile, all_vehicle_props_locs):
    vehicle_axles_roughness_input = dict()
    for vehicle_id in all_vehicle_props_locs.keys():
        num_axles = all_vehicle_props_locs[vehicle_id]['num_axles']
        axle_local_coords_list = all_vehicle_props_locs[vehicle_id]['axle_local_coords']
        vehicle_loc_time_stamps = all_vehicle_props_locs[vehicle_id]['time_loc']
        vehicle_axles_roughness_input[vehicle_id] = dict()
        vehicle_axles_roughness_input[vehicle_id]['rough_input'] = np.empty((num_axles, vehicle_loc_time_stamps.shape[0]))
        vehicle_axles_roughness_input[vehicle_id]['rough_input_derv'] = np.empty((num_axles, vehicle_loc_time_stamps.shape[0]))

        ref_axle_loc_history = all_vehicle_props_locs[vehicle_id]['axle_loc_rear'] + \
                               all_vehicle_props_locs[vehicle_id]['axle_local_coords'][0]
        vehicle_axles_roughness_input[vehicle_id]['rough_input'][0, :] = np.interp(ref_axle_loc_history, road_rough_profile['loc'],
                                                                                   road_rough_profile['value'])
        vehicle_axles_roughness_input[vehicle_id]['rough_input_derv'][0, :] = np.gradient(
            vehicle_axles_roughness_input[vehicle_id]['rough_input'][0, :], vehicle_loc_time_stamps)
        for i in range(num_axles):
            current_axle_loc_history = all_vehicle_props_locs[vehicle_id]['axle_loc_rear'] + axle_local_coords_list[i]
            vehicle_axles_roughness_input[vehicle_id]['rough_input'][i, :] = np.interp(current_axle_loc_history,
                                                                                       ref_axle_loc_history,
                                                                                       vehicle_axles_roughness_input[vehicle_id][
                                                                                           'rough_input'][0, :],
                                                                                       left=None, right=0)
            vehicle_axles_roughness_input[vehicle_id]['rough_input_derv'][i, :] = np.interp(current_axle_loc_history,
                                                                                            ref_axle_loc_history,
                                                                                            vehicle_axles_roughness_input[vehicle_id][
                                                                                                'rough_input_derv'][0, :],
                                                                                            left=None, right=0)
    return vehicle_axles_roughness_input

def determine_axles_static_force(all_vehicle_props_locs, vehicle_responses_psteps_all):
    for vehicle_id in all_vehicle_props_locs.keys():
        vehicle_responses_psteps_all[vehicle_id] = vehicle_dynamic_step_forward(vehicle_responses_psteps_all[vehicle_id],
                                                                                  all_vehicle_props_locs[vehicle_id],
                                                                                  [], [], [], analysis_type='static')

    return vehicle_responses_psteps_all



def vehicle_current_input_calc(vehicle_prop_locs, vehicle_axles_roughness_input, bridge_props, bridge_current_responses,
                               current_time, current_time_index):
    num_axles = vehicle_prop_locs['num_axles']
    axle_local_coords_list = vehicle_prop_locs['axle_local_coords']
    vehicle_input_current = dict(time=np.array([[current_time]]),
                                 disp=np.empty((num_axles, 1)),
                                 vel=np.empty((num_axles, 1)))
    for i in range(num_axles):
        axle_current_location = vehicle_prop_locs['axle_loc_rear'][current_time_index] + axle_local_coords_list[i]
        # Use hermitain function
        bridge_disp_axle_loc_c = np.interp(axle_current_location, bridge_props['nodes_coord'], bridge_current_responses['disp'][:, -1])
        bridge_vel_axle_loc_c = np.interp(axle_current_location, bridge_props['nodes_coord'], bridge_current_responses['vel'][:, -1])
        rough_axle_loc_c = vehicle_axles_roughness_input['rough_input'][i, current_time_index]
        der_rough_axle_loc_c = vehicle_axles_roughness_input['rough_input_derv'][i, current_time_index]
        if (axle_current_location > max(bridge_props['nodes_coord'])) or (axle_current_location < min(bridge_props['nodes_coord'])):
            # the input of the vehicle before or after bridge is only road roughness
            bridge_disp_axle_loc_c = 0
            bridge_vel_axle_loc_c = 0

        vehicle_input_current['disp'][i, :] = bridge_disp_axle_loc_c + rough_axle_loc_c
        vehicle_input_current['vel'][i, :] = bridge_vel_axle_loc_c + der_rough_axle_loc_c
    return vehicle_input_current

def map_axle_force_to_bridge_nodes(axle_current_force, axle_current_location, nodal_inter_force_current, bridge_props):

    closest_nodes = np.argsort(abs(bridge_props['nodes_coord'] - axle_current_location))[:2]
    x_nodes = bridge_props['nodes_coord'][closest_nodes]
    sorting_index = np.argsort(x_nodes)

    closest_nodes = closest_nodes[sorting_index]
    x_nodes = x_nodes[sorting_index]

    l = np.abs(x_nodes[1] - x_nodes[0])
    x_t = axle_current_location - x_nodes[0]
    if (axle_current_location < max(bridge_props['nodes_coord'])) and (axle_current_location > min(bridge_props['nodes_coord'])):
        nodal_inter_force_current[closest_nodes[0]] += (1 - x_t / l) * axle_current_force
        nodal_inter_force_current[closest_nodes[1]] += (x_t / l) * axle_current_force

        #  vehicle prior and after entering the bridge is considered
    return nodal_inter_force_current

def predict_current_vehicle_bridge_response_iterative(current_time, current_index, bridge_force_traffic_current, bridge_current_responses,
                                                      bridge_last_step_responses, vehicle_axles_roughness_input,
                                                      vehicle_responses_psteps_all, bridge_props, all_vehicle_props_locs,
                                                      dt_analysis, convergence_tol=1.0e-6, resp_conve='disp'):
    criteria_convergence = 1
    num_iter_step = 0
    vehicles_reponse_steps_updated = copy.deepcopy(vehicle_responses_psteps_all)
    interaction_nodal_force_all_current = dict()
    interaction_nodal_force_all_current['time'] = np.array([current_time])
    while criteria_convergence > convergence_tol:
        num_iter_step += 1
        nodal_inter_force_current = np.zeros(bridge_force_traffic_current.shape)
        for vehicle_id in all_vehicle_props_locs.keys():
            num_axles = all_vehicle_props_locs[vehicle_id]['num_axles']
            axle_local_coords_list = all_vehicle_props_locs[vehicle_id]['axle_local_coords']

            vehicle_input_current = vehicle_current_input_calc(all_vehicle_props_locs[vehicle_id], vehicle_axles_roughness_input[vehicle_id],
                                                               bridge_props, bridge_current_responses, current_time, current_index)

            vehicles_reponse_steps_updated[vehicle_id] = vehicle_dynamic_step_forward(vehicle_responses_psteps_all[vehicle_id],
                                                                                      all_vehicle_props_locs[vehicle_id],
                                                                                      vehicle_input_current, dt_analysis)


            # Vehicle_bridge Interaction Force mapping on bridge nodes
            axles_current_force = copy.deepcopy(vehicle_responses_psteps_all[vehicle_id]['axles_static_forces'])
            for i in range(num_axles):
                axles_current_force[i] = vehicle_responses_psteps_all[vehicle_id]['axles_static_forces'][i] + \
                                         vehicles_reponse_steps_updated[vehicle_id]['reaction_force'][i, -1]
                axles_current_force[i] = axles_current_force[i] * (axles_current_force[i] > 0)  # considering uplift

                axle_current_location = all_vehicle_props_locs[vehicle_id]['axle_loc_rear'][current_index] + axle_local_coords_list[i]

                nodal_inter_force_current = map_axle_force_to_bridge_nodes(axles_current_force[i], axle_current_location,
                                                                           nodal_inter_force_current, bridge_props)

        interaction_nodal_force_all_current['force'] = bridge_force_traffic_current + nodal_inter_force_current

        bridge_responses_current_updated, _ = bridge_dynamic_step_forward_opensees(interaction_nodal_force_all_current,
                                                                                   bridge_last_step_responses, bridge_props, dt_analysis)

        # conv criteria and update parameters
        criteria_convergence = np.sqrt(
            np.mean(np.square(bridge_responses_current_updated[resp_conve][:, -1].reshape(-1, 1) - bridge_current_responses[resp_conve]),
                    axis=0))
        max_resp = max(abs(bridge_responses_current_updated[resp_conve][:, -1]))
        criteria_convergence = criteria_convergence[0] / max_resp if max_resp > 0 else 0.0
        for key, _ in bridge_responses_current_updated.items():
            bridge_current_responses[key] = bridge_responses_current_updated[key][:, -1].reshape(-1, 1)  # deepcopy may be required

    return vehicles_reponse_steps_updated, bridge_current_responses, interaction_nodal_force_all_current['force'], num_iter_step

def predict_current_vehicle_bridge_response_noniterative(current_time, current_index, bridge_force_traffic_current,
                                                         bridge_last_step_responses, vehicle_axles_roughness_input,
                                                         vehicle_responses_psteps_all, bridge_props, all_vehicle_props_locs,
                                                         dt_analysis):
    vehicles_reponse_steps_updated = copy.deepcopy(vehicle_responses_psteps_all)
    nodal_inter_force_current = np.zeros(bridge_force_traffic_current.shape)
    interaction_nodal_force_all_current = dict()
    interaction_nodal_force_all_current['time'] = np.array([current_time])

    for vehicle_id in all_vehicle_props_locs.keys():

        num_axles = all_vehicle_props_locs[vehicle_id]['num_axles']
        axle_local_coords_list = all_vehicle_props_locs[vehicle_id]['axle_local_coords']

        # Vehicle_bridge Interaction Force mapping on bridge nodes
        axles_current_force = copy.deepcopy(vehicle_responses_psteps_all[vehicle_id]['axles_static_forces'])
        for i in range(num_axles):
            axles_current_force[i] = vehicle_responses_psteps_all[vehicle_id]['axles_static_forces'][i]
            axle_current_location = all_vehicle_props_locs[vehicle_id]['axle_loc_rear'][current_index] + axle_local_coords_list[i]

            nodal_inter_force_current = map_axle_force_to_bridge_nodes(axles_current_force[i], axle_current_location,
                                                                       nodal_inter_force_current, bridge_props)

    interaction_nodal_force_all_current['force'] = bridge_force_traffic_current + nodal_inter_force_current

    bridge_current_responses, _ = bridge_dynamic_step_forward_opensees(interaction_nodal_force_all_current,
                                                                       bridge_last_step_responses, bridge_props, dt_analysis)

    for vehicle_id in all_vehicle_props_locs.keys():
        vehicle_input_current = vehicle_current_input_calc(all_vehicle_props_locs[vehicle_id], vehicle_axles_roughness_input[vehicle_id],
                                                           bridge_props, bridge_current_responses, current_time, current_index)

        vehicles_reponse_steps_updated[vehicle_id] = vehicle_dynamic_step_forward(vehicle_responses_psteps_all[vehicle_id],
                                                                                  all_vehicle_props_locs[vehicle_id],
                                                                                  vehicle_input_current, dt_analysis)

    return vehicles_reponse_steps_updated, bridge_current_responses, interaction_nodal_force_all_current['force']

def segment_correct_vehicle_axles_resp(bridge_props, all_vehicle_props_locs, bridge_responses_all_steps, vehicle_responses_psteps_all,
                                       base_time_stamps, use_vel_derivative='yes'):
    if use_vel_derivative == 'no':
        for t_step, _ in enumerate(base_time_stamps):
            bridge_current_responses = {key: value[:, t_step].reshape(-1, 1) for key, value in bridge_responses_all_steps.items()}
            for vehicle_id in all_vehicle_props_locs.keys():
                num_axles = all_vehicle_props_locs[vehicle_id]['num_axles']
                axle_local_coords_list = all_vehicle_props_locs[vehicle_id]['axle_local_coords']
                for i in range(num_axles):
                    axle_current_location = all_vehicle_props_locs[vehicle_id]['axle_loc_rear'][t_step] + axle_local_coords_list[i]
                    vehicle_responses_psteps_all[vehicle_id]['acc'][i, t_step] = np.interp(axle_current_location,
                                                                                           bridge_props['nodes_coord'],
                                                                                           bridge_current_responses['vel'][:, -1])
    else:
        for vehicle_id, vehicle_recorded_data in vehicle_responses_psteps_all.items():
            vehicle_responses_psteps_all[vehicle_id]['acc'] = np.gradient(vehicle_recorded_data['vel'], vehicle_recorded_data['time'].flatten(), axis=1)
            vehicle_responses_psteps_all[vehicle_id]['acc'][:, 0] = 0


    for vehicle_id in all_vehicle_props_locs.keys():
        axle_local_coords_list = all_vehicle_props_locs[vehicle_id]['axle_local_coords']
        rear_axle_loc_history = all_vehicle_props_locs[vehicle_id]['axle_loc_rear'] + axle_local_coords_list[0]
        front_axle_loc_history = all_vehicle_props_locs[vehicle_id]['axle_loc_rear'] + axle_local_coords_list[-1]

        enter_index = np.where(rear_axle_loc_history > bridge_props['nodes_coord'][0])[0]
        exit_index = np.where(front_axle_loc_history < bridge_props['nodes_coord'][-1])[0]
        if enter_index.any():
            enter_index = enter_index[0]
            exit_index = exit_index[-1]
            vehicle_responses_psteps_all[vehicle_id]['fully_on_bridge_ranges'] = [int(enter_index), int(exit_index)]

        index_1 = np.where(front_axle_loc_history < bridge_props['nodes_coord'][0])[0]
        if index_1.any():
            vehicle_responses_psteps_all[vehicle_id]['fully_before_bridge_ranges'] = [0, int(index_1[-1])]

        index_2 = np.where(rear_axle_loc_history > bridge_props['nodes_coord'][-1]-1)[0]
        if index_2.any():
            vehicle_responses_psteps_all[vehicle_id]['fully_after_bridge_ranges'] = [int(index_2[0]), -1]

    return vehicle_responses_psteps_all

def main_vehicle_bridge(bridge_props, all_vehicle_props_locs, random_traffic_force_time, road_rough_profile, bridge_responses_initial_cond,
                        vehicle_responses_psteps_all, base_time_stamps, dt_analysis, vehicle_model='moving_mass',
                        plot_mode_shapes=True, print_iterations=True):
    bridge_responses_all_steps, bridge_natural_freqs = bridge_dynamic_step_forward_opensees(random_traffic_force_time, bridge_responses_initial_cond,
                                                                         bridge_props, dt_analysis, base_time_stamps.shape[0], 'yes',
                                                                         plot_mode_shapes=plot_mode_shapes)
    bridge_props['natural_freqs'] = bridge_natural_freqs

    vehicle_axles_roughness_input = determine_vehicle_location_roughness(road_rough_profile, all_vehicle_props_locs)
    vehicle_responses_psteps_all = determine_axles_static_force(all_vehicle_props_locs, vehicle_responses_psteps_all)
    num_iter_each_steps = np.zeros(base_time_stamps.shape)
    interaction_nodal_force_all = copy.deepcopy(random_traffic_force_time)
    for t_step, _ in enumerate(base_time_stamps[:-1]):
        current_time = base_time_stamps[t_step + 1]
        bridge_current_responses = {key: value[:, t_step + 1].reshape(-1, 1) for key, value in bridge_responses_all_steps.items()}
        bridge_last_step_responses = {key: value[:, t_step].reshape(-1, 1) for key, value in bridge_responses_all_steps.items()}
        bridge_force_traffic_current = random_traffic_force_time['force'][:, t_step + 1]

        if vehicle_model == 'moving_mass':

            vehicle_responses_psteps_all, bridge_current_responses, interaction_nodal_force_all['force'][:, t_step + 1], num_iter_each_steps[t_step] = \
                predict_current_vehicle_bridge_response_iterative(current_time, t_step + 1, bridge_force_traffic_current,
                                                                  bridge_current_responses,
                                                                  bridge_last_step_responses, vehicle_axles_roughness_input,
                                                                  vehicle_responses_psteps_all, bridge_props, all_vehicle_props_locs,
                                                                  dt_analysis, convergence_tol=1.0e-12, resp_conve='disp')

            for key, _ in bridge_current_responses.items():
                bridge_responses_all_steps[key][:, t_step + 1] = bridge_current_responses[key][:, -1]  # deepcopy may be required

        else:
            vehicle_responses_psteps_all, bridge_current_responses, interaction_nodal_force_all['force'][:, t_step + 1] = \
                predict_current_vehicle_bridge_response_noniterative(current_time, t_step + 1, bridge_force_traffic_current,
                                                                     bridge_last_step_responses, vehicle_axles_roughness_input,
                                                                     vehicle_responses_psteps_all, bridge_props, all_vehicle_props_locs,
                                                                     dt_analysis)

            for key, _ in bridge_current_responses.items():
                bridge_responses_all_steps[key][:, t_step + 1] = bridge_current_responses[key][:, -1]

            num_iter_each_steps[t_step] = 0
        if print_iterations:
            print(f'Step= {t_step} , Iter= {num_iter_each_steps[t_step]}')

    for key, _ in bridge_responses_all_steps.items():
        bridge_responses_all_steps[key] = bridge_responses_all_steps[key][:, :-1]
    vehicle_responses_psteps_all = segment_correct_vehicle_axles_resp(bridge_props, all_vehicle_props_locs, bridge_responses_all_steps,
                                                                      vehicle_responses_psteps_all, base_time_stamps)

    return bridge_responses_all_steps, vehicle_responses_psteps_all, interaction_nodal_force_all, num_iter_each_steps, vehicle_axles_roughness_input

def save_json_data(foldername, vehicle_responses_psteps_all, all_vehicle_props_locs, bridge_responses_all_steps, bridge_props,
                   vehicle_axles_roughness_input, interaction_nodal_force_all, road_rough_profile):
    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)

    for key, value in vehicle_responses_psteps_all.items():
        value['time'] = value['time'].tolist()
        value['disp'] = value['disp'].tolist()
        value['vel'] = value['vel'].tolist()
        value['acc'] = value['acc'].tolist()
        value['reaction_force'] = value['reaction_force'].tolist()
        value['axles_static_forces'] = value['axles_static_forces'].tolist()
    json_string = json.dumps(vehicle_responses_psteps_all, indent=4)
    with open(foldername + "vehicle_responses_psteps_all.json", "w") as json_file:
        json_file.write(json_string)

    for key, value in all_vehicle_props_locs.items():
        value['axle_loc_rear'] = value['axle_loc_rear'].tolist()
        value['time_loc'] = value['time_loc'].tolist()
    json_string = json.dumps(all_vehicle_props_locs, indent=4)
    with open(foldername + "all_vehicle_props_locs.json", "w") as json_file:
        json_file.write(json_string)

    bridge_responses_all_steps['time'] = bridge_responses_all_steps['time'].tolist()
    bridge_responses_all_steps['disp'] = bridge_responses_all_steps['disp'].tolist()
    bridge_responses_all_steps['vel'] = bridge_responses_all_steps['vel'].tolist()
    bridge_responses_all_steps['acc'] = bridge_responses_all_steps['acc'].tolist()
    json_string = json.dumps(bridge_responses_all_steps, indent=4)
    with open(foldername + "bridge_responses_all_steps.json", "w") as json_file:
        json_file.write(json_string)

    bridge_props['nodes_coord'] = bridge_props['nodes_coord'].tolist()
    json_string = json.dumps(bridge_props, indent=4)
    with open(foldername + "bridge_props.json", "w") as json_file:
        json_file.write(json_string)

    for key, value in vehicle_axles_roughness_input.items():
        value['rough_input'] = value['rough_input'].tolist()
        value['rough_input_derv'] = value['rough_input_derv'].tolist()
    json_string = json.dumps(vehicle_axles_roughness_input, indent=4)
    with open(foldername + "vehicle_axles_roughness_input.json", "w") as json_file:
        json_file.write(json_string)

    interaction_nodal_force_all['time'] = interaction_nodal_force_all['time'].tolist()
    interaction_nodal_force_all['force'] = interaction_nodal_force_all['force'].tolist()
    json_string = json.dumps(interaction_nodal_force_all, indent=4)
    with open(foldername + "interaction_nodal_force_all.json", "w") as json_file:
        json_file.write(json_string)

    road_rough_profile['loc'] = road_rough_profile['loc'].tolist()
    road_rough_profile['value'] = road_rough_profile['value'].tolist()
    json_string = json.dumps(road_rough_profile, indent=4)
    with open(foldername + "road_rough_profile.json", "w") as json_file:
        json_file.write(json_string)


def read_sort_json_data(data_folder_path):
    # read
    for filename in os.listdir(data_folder_path):
        if filename.endswith(".json"):  # Check if the file has a .json extension
            file_path = os.path.join(data_folder_path, filename)
            file_name_without_extension = os.path.splitext(filename)[0]
            with open(file_path, "r") as json_file:
                json_data = json.load(json_file)
            globals()[file_name_without_extension] = json_data
    # sort
    for key, value in vehicle_responses_psteps_all.items():
        value['time'] = np.array(value['time'])
        value['disp'] = np.array(value['disp'])
        value['vel'] = np.array(value['vel'])
        value['acc'] = np.array(value['acc'])
        value['reaction_force'] = np.array(value['reaction_force'])
        value['axles_static_forces'] = np.array(value['axles_static_forces'])

    for key, value in all_vehicle_props_locs.items():
        value['axle_loc_rear'] = np.array(value['axle_loc_rear'])
        value['time_loc'] = np.array(value['time_loc'])

    bridge_responses_all_steps['time'] = np.array(bridge_responses_all_steps['time'])
    bridge_responses_all_steps['disp'] = np.array(bridge_responses_all_steps['disp'])
    bridge_responses_all_steps['vel'] = np.array(bridge_responses_all_steps['vel'])
    bridge_responses_all_steps['acc'] = np.array(bridge_responses_all_steps['acc'])
    bridge_props['nodes_coord'] = np.array(bridge_props['nodes_coord'])

    for key, value in vehicle_axles_roughness_input.items():
        value['rough_input'] = np.array(value['rough_input'])
        value['rough_input_derv'] = np.array(value['rough_input_derv'])

    interaction_nodal_force_all['time'] = np.array(interaction_nodal_force_all['time'])
    interaction_nodal_force_all['force'] = np.array(interaction_nodal_force_all['force'])

    road_rough_profile['loc'] = np.array(road_rough_profile['loc'])
    road_rough_profile['value'] = np.array(road_rough_profile['value'])

def keep_only_measured_responses_bridge(bridge_responses_all_steps, interaction_nodal_force_all, bridge_num_sensors, bridge_props):
    all_recorded_coords = bridge_props['nodes_coord']
    fixed_sensors_locations = np.linspace(0, all_recorded_coords[-1], bridge_num_sensors)

    bridge_responses_all_sensor_steps = {key: np.empty([bridge_num_sensors, bridge_responses_all_steps['disp'].shape[1]])
                                         for key in bridge_responses_all_steps.keys()}

    bridge_responses_all_sensor_steps['time'] = bridge_responses_all_steps['time']
    for step in range(bridge_responses_all_steps['time'].shape[1]):
        bridge_responses_all_sensor_steps['disp'][:, step] = np.interp(fixed_sensors_locations, all_recorded_coords, bridge_responses_all_steps['disp'][:, step])
        bridge_responses_all_sensor_steps['vel'][:, step] = np.interp(fixed_sensors_locations, all_recorded_coords, bridge_responses_all_steps['vel'][:, step])
        bridge_responses_all_sensor_steps['acc'][:, step] = np.interp(fixed_sensors_locations, all_recorded_coords, bridge_responses_all_steps['acc'][:, step])
    bridge_responses_all_sensor_steps['sensors_locs'] = np.array(fixed_sensors_locations)

    interaction_nodal_force_all_sensor = {'force': np.empty([bridge_num_sensors, interaction_nodal_force_all['force'].shape[1]])}
    interaction_nodal_force_all_sensor['time'] = interaction_nodal_force_all['time']

    for step in range(interaction_nodal_force_all['time'].shape[0]):
        interaction_nodal_force_all_sensor['force'][:, step] = np.interp(fixed_sensors_locations, all_recorded_coords, interaction_nodal_force_all['force'][:, step])

    return bridge_responses_all_sensor_steps, interaction_nodal_force_all_sensor
