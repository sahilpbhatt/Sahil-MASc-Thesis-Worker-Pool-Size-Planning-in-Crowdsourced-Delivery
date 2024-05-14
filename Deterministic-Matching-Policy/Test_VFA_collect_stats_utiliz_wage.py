from __future__ import print_function
from gurobipy import * 
from scipy.spatial import distance
from scipy.spatial.distance import euclidean 
import math
import numpy as np 
import copy
import random
from time import time
import ADP_Algorithms_Functions_compare_utilization_to_wage_guarantee as func


# Step 3: run algorithm
def solve(aggregate, stats, data, m, d, v_t, penalty, cfa):
    # Step 3.1:
    driver_attr, demand_attr = {}, {}  # driver attribute a= (sa, ma, oa, time_first_matched, ha, \bar{ha})
    # sa=1 if driver active, ma=1 if driver available, oa: location, ha: active time, \bar{h}_a: fraction
    # demand attribute b = (ob, db, dst, (tbmin, tbmax))
    driver_list = {'act_available': [], 'act_unavailable': [], 'inactive': [], 'exit': []}
    demand_list = {'active': [], 'fulfilled': [], 'expired': []}

    # Collect driver stats
    driver_stats = {'guarantee_met': [], 'guarantee_missed': []}  # stores value of bar_ha above/below threshold W
    demand_stats = {'met': [], 'missed': []}
    driver_log, n, tot_num_matches = {}, 999, 0

    # driver numbering restarts each iteration. Active available drivers are moved forward but re-numbered
    t_begin = time()
    for t in range(data['T']):
        print("t = %d" % t)
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
                driver_attr[i] = func.Driver(i, 0, 1, loc_0, loc_1, loc_2, 0, 0, 0, 0)
                # drivers inactive, available, 0 active time.
                driver_list['inactive'].append(i)
                driver_log[i] = {'all_t': [['t%d' % t, [0, 1, m[t, i], 0, 0, 0, 0]]]}  # , ('t_postdecision', t):

            for i in range(d[0]):  # no. demand from 1 not 0, since 0 means not matched.
                i_loc_o = d[t, i][0][0]  # d[n][t, i][0][0]
                i_loc_d = d[t, i][0][1]
                dst = distance.euclidean(i_loc_o, i_loc_d)
                demand_attr[i+1] = func.Demand(i+1, i_loc_o, i_loc_d, dst, d[t, i][1], d[t, i][1] + data['delta_t'],
                                               d[t, i][2])
                demand_list['active'].append(i+1)
            # I'm keeping track of driver no. to give each driver/order a unique identifier at each iter.

            num_drivers = len(driver_list['inactive']) + len(driver_list['act_available'])
            num_demand = len(demand_list['active'])

        # Step 3.3: solve optimization problem
        xab, z, dual_a, num_matches, x_ab_time, x_ab_cost, dst, tot_dst, platform_profit = func.solve_opt(data, t, n, demand_list, driver_list,
                                                                driver_attr, demand_attr, v_t, aggregate, penalty, cfa)
        tot_num_matches += num_matches

        driver_stats[t, 'dst'] = dst
        driver_stats[t, 'tot_dst'] = tot_dst
        driver_stats[t, 'platform_profit'] = platform_profit
        # Step 3.4 - update attr and vf:
        driver_temp, demand_temp = func.update_attr(driver_list, demand_list, xab, driver_attr, demand_attr, t, data,
                                                    x_ab_time, x_ab_cost, aggregate, stats,
                                                    driver_log, driver_stats, demand_stats)

        # Step 3.5 - compute pre-decision state
        num_drivers, num_demand = func.predecision_state(num_drivers, num_demand, driver_temp, demand_temp, driver_list, demand_list, t, n, m, d,
                      driver_attr, demand_attr, data, aggregate, False, stats, driver_log, driver_stats)

        if t == data['T']-1:
            demand_stats['timed_out'] = demand_list['active']
            driver_stats['timed_out'] = []

            for a in driver_list['act_available'] + driver_list['act_unavailable']:
                profit_matching = driver_attr[a].profit_matching
                if (t + 1) * data['t_interval'] - driver_attr[a].time_first_matched >= data['Guaranteed_service'] and \
                        a in driver_list['act_unavailable']:
                    bar_ha = driver_attr[a].bar_ha
                    if bar_ha >= data['W']:
                        driver_stats['guarantee_met'].append([a, bar_ha, [profit_matching],
                                                              driver_attr[a].active_history])
                    else:
                        driver_stats['guarantee_missed'].append([a, bar_ha, [profit_matching],
                                                              driver_attr[a].active_history])
                else:
                    driver_stats['timed_out'].append([a, driver_attr[a].bar_ha, [profit_matching],
                                                              driver_attr[a].active_history])

    t_end = time()
    time_algorithm = t_end - t_begin
    return driver_stats, demand_stats, driver_list, demand_list, driver_log, tot_num_matches, time_algorithm


def run(vf_file_name, aggregate, inst, penalty, T, mu, cfa=False):
    # Step 1: read data
    loadname = 'data/wage_guarantee_test_inst_T=192_%d_mu%d.npz' % (inst, mu)  # use same d, m data, regardless of policy
    data_set = np.load(loadname)
    stats = True
    # changed T=144 from T=72 (02/18/21)

    data = {'T': T, 't_interval': 5, 'delta_t': data_set['delta_t'].tolist(), 'N': 1000,
            'eta': 0.8, 'W': 0.8, 'theta': (1 / (2 * math.sqrt(200))), 'theta1': 1,
            'gamma': 0.9, 'Guaranteed_service': 120, 'alpha_penalty': 1.0, 'h_int': 5, 'g_int': 5}

    m = data_set['m'].tolist()
    d = data_set['d'].tolist()

    # Step 2: get value function
    vf_file = np.load(vf_file_name)
    v_t = vf_file['v_t'].tolist()
    if aggregate:
        data['loc_set_1'] = [[0.5 + x, 0.5 + y] for x in range(10) for y in range(10)]  # aggreg. level 1
        data['loc_set_2'] = [[1 + (2 * x), 1 + (2 * y)] for x in range(5) for y in range(5)]

    # Step 3: Solve same inst for n = 1,..., 500
    driver_stats, demand_stats, driver_list, demand_list, driver_log, tot_num_matches, time_algorithm = \
        solve(aggregate, stats, data, m, d, v_t, penalty, cfa)

    print(driver_stats)
    print(demand_stats)
    print(driver_log[0]['all_t'])
    name = vf_file_name[:-4] + 'T=%d_test_inst_no%d_v2' % (T, inst)
    # v2: keeps track of dst and active drivers
    # print(obj_val_list)
    np.savez(name, demand_stats=demand_stats, driver_stats=driver_stats, driver_log=driver_log,
             driver_list=driver_list, demand_list=demand_list, tot_num_matches=tot_num_matches,
             time_algorithm=time_algorithm)

    # Note: this inst delivery deadline is too high, like we don't have a deadline.
    # Stats to collect:
    # Drivers:
    # No. of drivers that meet the guarantee, No. of drivers below
    # 1 driver, the exact route
    # Demand
    #  % satisfied, % missed