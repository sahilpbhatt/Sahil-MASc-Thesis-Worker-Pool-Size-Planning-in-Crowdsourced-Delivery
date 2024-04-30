from __future__ import print_function
from gurobipy import * 
from scipy.spatial import distance
from scipy.spatial.distance import euclidean 
import math
import numpy as np 
from time import time
import ADP_Algorithms_Functions_compare_utilization_to_wage_guarantee as func
from collections import defaultdict

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# (timestamp: 02/05/21) Not sure why the code is a bit slow as we progress in t
# next: update codes of test instances to get those functions.
# Step 3: run algorithm
def iterate(data, d, v_t, aggregate, penalty, cfa):
    # Step 3.1: while n <= N
    """
    Iterate for n iterations, solving a series of optimization problems to optimize the m
    matching decisions at each time step, to find the best pool sizes 
    Calls the Boltzmann exploration function in the utilities file to set the pool sizes, which is iteratively
    updated at each iteration based on the driver utilization and service level obtained through solving
    the deterministic matching problem in a rolling horizon framework at each time step  
    """
    driver_attr, demand_attr = {}, {} #map the driver and order ids to their attributes stored as Driver and Demand class objects respectively
    obj_val, obj_val_list = {}, [] #store the obj_val at each iteration in obj_val_list
    # driver attribute a= (sa, ma, oa, time_first_matched, ha, \bar{ha})
    # demand attribute b = (ob, db, dst, (tbmin, tbmax))
    t_training_begin = time()
    
    m = {} #We do not store driver data, only demand data, because the number of drivers that enter is subject to endogeous uncertainty

    pool_sizes = [1 for i in range(16)]; 
    #arbitrarily set the pool sizes at 16 periods, each 12 epochs or 12*5 min = 1 h  

    V = defaultdict(lambda:0) #the value function that determines the value of choosing a pool size at a particular period
    #this is outside of for loop to learn across different pool sizes in different iterations 
    alpha = defaultdict(lambda:0) 

    for n in range(1, data['N']): 
        driver_stats = {}
        driver_attr, demand_attr = {}, {}
        obj_val[n] = 0
        driver_list = {'act_available': [], 'act_unavailable': [], 'inactive': [], 'exit': []}
        demand_list = {'active': [], 'fulfilled': [], 'expired': []}
        v_t[n] = {}  # Don't need it {('g%d' % xx) for xx in range(3)}  # value functions for 3 aggreg. levels.
 
        print('\n' + color.BOLD + color.RED + 'N IS EQUAL TO: ', n, color.END); print()
        
        if n > 1:
            pool_sizes = func.boltzmann(V, n) 
 
        print('\npool_sizes: ', pool_sizes)

        obj_val[n] = 0
        driver_list = {'act_available': [], 'act_unavailable': [], 'inactive': [], 'exit': []}
        demand_list = {'active': [], 'fulfilled': [], 'expired': []}
        v_t[n] = {}  # Don't need it {('g%d' % xx) for xx in range(3)}  # value functions for 3 aggreg. levels.
 
        num_drivers_enter = [np.random.binomial(pool_sizes[t//12], data['prob_enter']) for t in range(data['N'])] 

        num_orders_placed = [d[n][t] for t in range(T)] 

        _, m[n] = func.sample_path(data['mu_enter'], data['mu_exit'], data['lambd'],
                                       data['grid_size'], data['t_interval'], data['T'], data['loc_set_0'], num_drivers_enter, num_orders_placed)

        # driver numbering restarts each iteration. Active available drivers are moved forward but re-numbered
        for t in range(data['T']): 
            if t == 0:
                # Initialize driver attrib.
                for i in range(m[n][0]):
                    loc_0 = m[n][t, i]
                    loc_1, loc_2 = func.closest_node(loc_0, data['loc_set_1']), \
                                   func.closest_node(loc_0, data['loc_set_2'])

                    driver_attr[i] = func.Driver(i, 0, 1, loc_0, loc_1, loc_2, 0, 0, 0, 0, 0)
                    # drivers inactive, available, 0 active time.
                    driver_list['inactive'].append(i)

                for i in range(d[n][0]):  # no. demand from 1 not 0, since 0 means not matched.
                    i_loc_o = d[n][t, i][0][0]  # d[n][t, i][0][0]
                    i_loc_d = d[n][t, i][0][1]
                    dst = distance.euclidean(i_loc_o, i_loc_d)
                    demand_attr[i+1] = func.Demand(i+1, i_loc_o, i_loc_d, dst, d[n][t, i][1], 0, 
                                                   d[n][t, i][1] + data['delta_t'], d[n][t, i][2])
                    demand_list['active'].append(i+1)

                # I'm keeping track of driver no. to give each driver/order a unique identifier at each iter.
                num_drivers = len(driver_list['inactive']) + len(driver_list['act_available'])
                num_demand = len(demand_list['active'])

            # Step 3.3: solve optimization problem
            xab, z, dual_a, num_matches, x_ab_time, x_ab_cost, dst, tot_dst, platform_profit = func.solve_opt(data, t, n, demand_list, driver_list,
                                                            driver_attr, demand_attr, v_t, aggregate, penalty, cfa)
            #print("n = %d, t = %d" % (n, t)); print('z: ', z, ' num_matches: ', num_matches); print()
            obj_val[n] += z

            # Step 3.4 - update attr and vf:
            func.update_vf(driver_list, driver_attr, data, t, dual_a, v_t, n, aggregate)

            driver_temp, demand_temp, driver_stats, demand_stats  = func.update_attr(driver_list, demand_list, xab, driver_attr, demand_attr, t,
                                                        data, x_ab_time, x_ab_cost, aggregate, True)
            
            #print('driver_stats: ', driver_stats)

            # Step 3.5 - compute pre-decision state
            if t < data['T'] - 1:
                num_drivers, num_demand = func.predecision_state(num_drivers, num_demand, driver_temp, demand_temp,
                                        driver_list, demand_list, t, n, m, d, driver_attr, demand_attr, data, aggregate,
                                                                 True)

            # Next: (1) collect statistics on key metrics for demand and drivers.
        obj_val_list.append(obj_val[n])

        print('\nRESULTS:\n')
        print('n: ', n, ' obj_val: ', obj_val)

        total_no_drivers = len(driver_list['exit']+driver_list['act_available']+driver_list['inactive']+driver_list['act_unavailable'])
        num_drivers_meeting_guarantee = len(set([x[0] for x in driver_stats['guarantee_met']]))

        frac_drivers_meeting_guarantee = num_drivers_meeting_guarantee/total_no_drivers

        total_no_orders = sum(num_orders_placed)

        no_orders_fulfilled = len(demand_list['fulfilled'])

        frac_demand_met = no_orders_fulfilled/total_no_orders 

        print('total no drivers: ', total_no_drivers)
        print('num drivers meeting guarantee: ', num_drivers_meeting_guarantee)
        print('Fraction meeting guarantee: ', frac_drivers_meeting_guarantee)
        print()
        print('total no orders: ', total_no_orders)
        print('no orders fulfilled: ', no_orders_fulfilled)
        print('Fraction demand met: ', frac_demand_met)
        print()

        num_orders_fulfilled = [0 for i in range(192)]; num_drivers_meeting_guarantee = [0 for i in range(192)]
        for id in set(demand_stats['met']): 
            num_orders_fulfilled[demand_attr[id].placement_time] += 1

        ids_drivers_meeting_guarantee = set([x[0] for x in driver_stats['guarantee_met']])

        for id in driver_attr: 
            if id in ids_drivers_meeting_guarantee: 
                num_drivers_meeting_guarantee[driver_attr[id].time_entered] += 1
 
        num_orders_fulfilled = [sum(num_orders_fulfilled[12*i:12*(i+1)]) for i in range(16)]
        num_drivers_meeting_guarantee = [sum(num_drivers_meeting_guarantee[12*i:12*(i+1)]) for i in range(16)]

        print('\nData for each period:\n')
        print('num_orders_placed: ', [sum(num_orders_placed[12*i:12*(i+1)]) for i in range(16)])
        print('num_orders_fulfilled: ', num_orders_fulfilled)
        print('num_drivers_enter: ', [sum(num_drivers_enter[12*i:12*(i+1)]) for i in range(16)])
        print('num_drivers_meeting_guarantee: ', num_drivers_meeting_guarantee) #should it be when the demand was placed or announced? 
        print()
        
        for p in range(16):

            alpha[p, pool_sizes[p]] += 1

            if alpha[p, pool_sizes[p]] == 0:
                alpha_step_size = 1
            else: 
                alpha_step_size = 1/(alpha[(p, pool_sizes[p])])**0.5 
                
            Qp = sum(num_orders_fulfilled[p:])/sum(num_orders_placed[p:])
            Lp = sum(num_drivers_meeting_guarantee[p:])/sum(num_drivers_enter[p:])
            
            V_new = sum(pool_sizes[p+1:]) + data['w_d']*(n+1)*max(data['Q']-Qp, 0) + data['w_s']*(n+1)*max(data['L']-Lp, 0) 

            V[p, pool_sizes[p]] = (1-alpha_step_size)*V[p, pool_sizes[p]] + alpha_step_size*V_new

        #print('V: ', V)

        print() 

    t_training_end = time()
    time_algorithm = t_training_end - t_training_begin
    return obj_val_list, time_algorithm


# Step 2: Initiate VF
def initiate_VF():
    loc_set_0 = [[0.25+(0.5*x), 0.25+(0.5*y)] for x in range(20) for y in range(20)]  # aggreg. level 0
    loc_set_1 = [[0.5+x, 0.5+y] for x in range(10) for y in range(10)]                # aggreg. level 1
    loc_set_2 = [[1+(2*x), 1+(2*y)] for x in range(5) for y in range(5)]              # aggreg. level 2
    # bar_ha_set = [0.05*x for x in range(20)]
    # remaining_guarantee = [5*x for x in range(int(data['Guaranteed_service']/5) + 1)]
    return loc_set_0, loc_set_1, loc_set_2  #, bar_ha_set, remaining_guarantee
  
def run(penalty, cfa=False): 
    # Step 1: read data + initiate values
    # for penalty in [10, 50, 100, 250]:  # 500, 1000
    loadname = 'data/revised_demand_data_wage_guarantee_test_inst_T=192_1_mu2.npz'
    #loadname = 'data/wage_guarantee_test_inst_T=192_1_mu2.npz'
    data_set = np.load(loadname, allow_pickle=True)
    aggregate = True
    # changed to mu=2 06/28/21
    
    """
    L: the target fraction of drivers meeting the utilization guarantee 
    W: utilization guarantee; W is set to 0.8 to account for the drivers' idle time between matches 
    Q: service level target; set to 1 to maximize service level 
    w_d: weight associated with driver utilization term in objective 
    w_s = 1-w_d #weight associated with service level term in objective
    t_interval is 5 min, the length of a decision epoch; 5*12 = 60 min = 1 h is the length of each period;
    the planning horizon of 16 h consists of 16 periods
    N: the number of iterations; each iteration has the same demand, stored in loadname file, but the
    number of drivers and their locations are determined based on the pool sizes using the sample path function 
    """
    
    data = {'T': 192, 'prob_enter':0.2, 'L': 0.8, 'Q': 1, 'w_d':0.5, 'w_s':0.5, 't_interval': 5, 
            'mu_enter': 2, 'mu_exit': 4, 'lambd': 10, 'delta_t': data_set['delta_t'].tolist(), 'N': 1000, 'grid_size': (10, 10),
            'eta': 0.8, 'W': 0.8, 'theta': (1 / (2 * math.sqrt(200))), 'theta1': 1,
            'gamma': 0.9, 'Guaranteed_service': 120,
            'alpha_penalty': 1.0, 'lambda_initial': 10, 'h_int': 5, 'g_int': 5}
    
    d = {}

    for i in range(data['N']):  
        d[i] = data_set['d'].tolist() #same demand data for consistent comparison between different pool sizes 

    v_t = {0: {}, 'count': {}, 'var': {}, 'last_met': {}} #value function 
    data['loc_set_0'], data['loc_set_1'], data['loc_set_2'] = initiate_VF()
    cfa_str = '_cfa' if cfa else ''
    # Step 3: Run algorithm
    obj_val_list, time_algorithm = iterate(data, d, v_t, aggregate, penalty, cfa)

    print('obj_val_list: ', obj_val_list)

    name = "data/sol_n1000_VFA_T=%d_aggregation%s_penalty%d_mu%d_theta1_no_PCF_utiliz-wage" % (data['T'], cfa_str, penalty, data['mu_enter'])

    np.savez(name, v_t=v_t, obj_val_list=obj_val_list, time_algorithm=time_algorithm)


if __name__ == '__main__':
    penalty, cfa = 250, True
    run(penalty, cfa)
