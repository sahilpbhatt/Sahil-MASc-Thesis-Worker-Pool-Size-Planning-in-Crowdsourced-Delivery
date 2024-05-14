from __future__ import print_function
from gurobipy import * 
from scipy.spatial import distance
from scipy.spatial.distance import euclidean 
import math
import numpy as np 
import copy
import random
from time import time
import matplotlib.pyplot as plt
import ADP_Algorithms_Functions_test_converge as func


# Step 3: run algorithm
def solve(aggregate, stats, data, m, d, v_t, penalty, cfa):
    # Step 3.1:
    driver_attr, demand_attr = {}, {}  # driver attribute a= (sa, ma, oa, time_first_matched, ha, \bar{ha})
    # sa=1 if driver active, ma=1 if driver available, oa: location, ha: active time, \bar{h}_a: fraction
    # demand attribute b = (ob, db, dst, (tbmin, tbmax))
    obj_val, obj_val_list = {}, []
    # t_begin = time()
    for n in range(1, data['N']):
        obj_val[n] = 0
        driver_list = {'act_available': [], 'act_unavailable': [], 'inactive': [], 'exit': []}
        demand_list = {'active': [], 'fulfilled': [], 'expired': []}

    # driver numbering restarts each iteration. Active available drivers are moved forward but re-numbered
        for t in range(data['T']):
            print("n = %d, t = %d" % (n, t))
            m[t, 'exit'] = m['exit'][t]
            if t == 0:
                # Initialize driver attrib.
                for i in range(m[0]):
                    loc_0 = m[t, i]
                    if aggregate:
                        loc_1, loc_2 = func.closest_node(loc_0, data['loc_set_1']), \
                                       func.closest_node(loc_0, data['loc_set_2'])
                    else:
                        loc_1, loc_2 = [], []
                    driver_attr[i] = func.Driver(i, 0, 1, loc_0, loc_1, loc_2, 0, 0, 0)
                    # drivers inactive, available, 0 active time.
                    driver_list['inactive'].append(i)
                    # driver_log[i] = {'all_t': [['t%d' % t, [0, 1, m[t, i], 0, 0, 0]]]}  # , ('t_postdecision', t):

                for i in range(d[0]):  # no. demand from 1 not 0, since 0 means not matched.
                    i_loc_o = d[t, i][0][0]  # d[n][t, i][0][0]
                    i_loc_d = d[t, i][0][1]
                    dst = distance.euclidean(i_loc_o, i_loc_d)
                    demand_attr[i+1] = func.Demand(i+1, i_loc_o, i_loc_d, dst, d[t, i][1], d[t, i][1] + data['delta_t'])
                    demand_list['active'].append(i+1)
                # I'm keeping track of driver no. to give each driver/order a unique identifier at each iter.

                num_drivers = len(driver_list['inactive']) + len(driver_list['act_available'])
                num_demand = len(demand_list['active'])

            # Step 3.3: solve optimization problem
            xab, z, dual_a, num_matches, x_ab_time, dst, tot_dst = func.solve_opt(data, t, n, demand_list, driver_list,
                                                                                  driver_attr, demand_attr, v_t, aggregate, penalty, cfa)
            obj_val[n] += z
            # tot_num_matches += num_matches

            # driver_stats[t, 'dst'] = dst
            # driver_stats[t, 'tot_dst'] = tot_dst
            # Step 3.4 - update attr and vf:
            driver_temp, demand_temp = func.update_attr(driver_list, demand_list, xab, driver_attr, demand_attr, t, data,
                                                        x_ab_time, aggregate)

            # Step 3.5 - compute pre-decision state
            num_drivers, num_demand = func.predecision_state(num_drivers, num_demand, driver_temp, demand_temp, driver_list, demand_list, t, n, m, d,
                                                             driver_attr, demand_attr, data, aggregate, False)
        obj_val_list.append(obj_val[n])

    # t_end = time()
    # time_algorithm = t_end - t_begin
    return obj_val_list


def run(vf_file_name, aggregate, inst, penalty, T, mu, cfa=False):
    # Step 1: read data
    loadname = 'data/wage_guarantee_test_inst_T=192_%d_mu%d.npz' % (inst, mu)  # use same d, m data, regardless of policy
    data_set = np.load(loadname)
    stats = False
    # changed T=144 from T=72 (02/18/21)

    data = {'T': T, 't_interval': 5, 'delta_t': data_set['delta_t'].tolist(), 'N': 1000,
            'eta': 0.8, 'W': 0.8, 'theta': (1 / (2 * math.sqrt(200))), 'theta1': 1,
            'gamma': 0.9, 'Guaranteed_service': 120, 'alpha_penalty': 1.0, 'h_int': 5, 'g_int': 5}

    m = data_set['m'].tolist()
    d = data_set['d'].tolist()
    if aggregate:
        aggregate_str = 'agg'
    else:
        aggregate_str = 'disagg'

    # Step 2: get value function
    v_t = {'n': {}}
    # loop over files to get complete vf
    for ii in range(1, 61):
        filename = vf_file_name[:-4] + '_%d.npz' % ii
        vf_file = np.load(filename)
        v_t_temp = vf_file['v_t'].tolist()
        for key in v_t_temp['n']:
            v_t['n'][key] = v_t_temp['n'][key]

    print('Done reading v_t files')
    if aggregate:
        data['loc_set_1'] = [[0.5 + x, 0.5 + y] for x in range(10) for y in range(10)]  # aggreg. level 1
        data['loc_set_2'] = [[1 + (2 * x), 1 + (2 * y)] for x in range(5) for y in range(5)]

    # Step 3: Solve same inst for n = 1,..., 500
    obj_val_list = solve(aggregate, stats, data, m, d, v_t, penalty, cfa)
    name = loadname[:-4] + '_sol_N%d_%s' % (data['N'], aggregate_str)
    np.savez(name, obj_val_list=obj_val_list)

    plt.plot(range(1, data['N']), obj_val_list, label='Obj val')
    # plt.plot(range(1, N), num_matches_list, label='Num. matches')
    plt.legend()
    # plt.show()
    plt.savefig(name)
    # print(obj_val_list)
    plt.close()

    # name = vf_file_name[:-4] + 'T=%d_test_inst_no%d_' % (T, inst) + inst_string
    # v2: keeps track of dst and active drivers
    # print(obj_val_list)
    # np.savez(name, demand_stats=demand_stats, driver_stats=driver_stats, driver_log=driver_log,
    #          driver_list=driver_list, demand_list=demand_list, tot_num_matches=tot_num_matches,
    #          time_algorithm=time_algorithm)

    # Note: this inst delivery deadline is too high, like we don't have a deadline.
    # Stats to collect:
    # Drivers:
    # No. of drivers that meet the guarantee, No. of drivers below
    # 1 driver, the exact route
    # Demand
    #  % satisfied, % missed