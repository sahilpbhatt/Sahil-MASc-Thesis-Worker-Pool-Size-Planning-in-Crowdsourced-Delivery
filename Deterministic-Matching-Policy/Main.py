from __future__ import print_function
from gurobipy import * 
from scipy.spatial import distance
from scipy.spatial.distance import euclidean 
import math
import numpy as np 
from time import time
import Functions as func 
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
def iterate(data, d, aggregate, penalty, cfa, pool_sizes, V, alpha):
    # Step 3.1: while n <= N
    driver_attr, demand_attr = {}, {}

    obj_val, obj_val_list = {}, []
    # driver attribute a= (sa, ma, oa, time_first_matched, ha, \bar{ha})
    # demand attribute b = (ob, db, dst, (tbmin, tbmax))
    t_training_begin = time()
    
    m = {} #We do not store driver data, only demand data, because the number of drivers that enter is subject to endogenous uncertainty
    
    av_fraction_demand_met = [0 for i in range(16)]
    av_fraction_meeting_guarantee = [0 for i in range(16)]

    for n in range(1, data['N']):
        driver_stats = {'guarantee_met':[], 'guarantee_missed':[]}; 
        demand_stats = {'met':[], 'missed':[]}; 
        driver_log = {}

        print('\n' + color.BOLD + color.RED + 'N IS EQUAL TO: ', n, color.END); print()

        obj_val[n] = 0
        driver_list = {'available': [], 'unavailable': [],  'exit': []}
        demand_list = {'active': [], 'fulfilled': [], 'expired': []} 

        num_drivers_enter = [np.random.binomial(pool_sizes[t//12], data['prob_enter']) for t in range(data['T'])]
         
        _, m[n] = func.sample_path(data['mu_enter'], data['mu_exit'], data['lambd'],
                                      data['grid_size'], data['t_interval'], data['T'], data['loc_set_0'], num_drivers_enter)

        
        # driver numbering restarts each iteration. Active available drivers are moved forward but re-numbered
        for t in range(data['T']):
            #print("n = %d, t = %d" % (n, t))
            if t == 0:
                # Initialize driver attrib.
                for i in range(m[n][0]):
                    loc_0 = m[n][t, i]
                    loc_1, loc_2 = func.closest_node(loc_0, data['loc_set_1']), \
                                   func.closest_node(loc_0, data['loc_set_2'])

                    driver_attr[i] = func.Driver(i, 1, loc_0, loc_1, loc_2, 0, 0, 30, 0, 0, 0)
                    # drivers inactive, available, 0 active time.
                    driver_list['available'].append(i)

                for i in range(d[n][0]):  # no. demand from 1 not 0, since 0 means not matched.
                    i_loc_o = d[n][t, i][0][0]  # d[n][t, i][0][0]
                    i_loc_d = d[n][t, i][0][1]
                    dst = distance.euclidean(i_loc_o, i_loc_d)
                    demand_attr[i+1] = func.Demand(i+1, i_loc_o, i_loc_d, dst, d[n][t, i][1], 0, 
                                                   d[n][t, i][1] + data['delta_t'], d[n][t, i][2])
                    demand_list['active'].append(i+1)

                # I'm keeping track of driver no. to give each driver/order a unique identifier at each iter.
                num_drivers = len(driver_list['available']+driver_list['unavailable'])
                num_demand = len(demand_list['active'])

            # Step 3.3: solve optimization problem
            xab, z, dual_a, num_matches, x_ab_time, x_ab_cost, dst, tot_dst, platform_profit = func.solve_opt(data, t, n, demand_list, driver_list,
                                                            driver_attr, demand_attr, aggregate, penalty, cfa)
            obj_val[n] += z
 
            driver_temp, demand_temp, driver_stats, demand_stats, driver_attr, demand_attr = func.update_attr(driver_list, demand_list, xab, driver_attr, demand_attr, t,
                                                        data, x_ab_time, x_ab_cost, aggregate, True, driver_log, driver_stats, demand_stats)

            #print('outer t: ', t, ' driver_stats: ', driver_stats)

            # Step 3.5 - compute pre-decision state
        
            num_drivers, num_demand, driver_stats, driver_attr, demand_attr = func.predecision_state(num_drivers, num_demand, driver_temp, demand_temp,
                                    driver_list, demand_list, t, n, m, d, driver_attr, demand_attr, data, aggregate,
                                                                True, True, driver_log, driver_stats)

            # Next: (1) collect statistics on key metrics for demand and drivers.
        obj_val_list.append(obj_val[n])

        print('n: ', n, ' obj_val: ', obj_val)

        #total_no_drivers = len(driver_list['exit']+driver_list['available']+driver_list['unavailable'])
        total_no_drivers = sum([m[n][t] for t in range(192)])
        num_drivers_meeting_guarantee = len(set([x[0] for x in driver_stats['guarantee_met']]))

        frac_drivers_meeting_guarantee = num_drivers_meeting_guarantee/total_no_drivers

        num_orders_placed = [d[n][t] for t in range(192)]
        num_drivers_enter = [m[n][t] for t in range(192)]
        num_orders_placed = [sum(num_orders_placed[12*i:12*(i+1)]) for i in range(16)] 
        num_drivers_enter = [sum(num_drivers_enter[12*i:12*(i+1)]) for i in range(16)] 

        total_no_orders = sum([d[n][t] for t in range(192)])

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

        for id in ids_drivers_meeting_guarantee: 
            num_drivers_meeting_guarantee[driver_attr[id].time_entered] += 1
 
        num_orders_fulfilled = [sum(num_orders_fulfilled[12*i:12*(i+1)]) for i in range(16)]
        num_drivers_meeting_guarantee = [sum(num_drivers_meeting_guarantee[12*i:12*(i+1)]) for i in range(16)]
        frac_demand_met = [sum(num_orders_fulfilled[p:])/sum(num_orders_placed[p:]) for p in range(16)]
        frac_meeting_guarantee = [sum(num_drivers_meeting_guarantee[p:])/sum(num_drivers_enter[p:]) if sum(num_drivers_enter[p:]) != 0 else 1 for p in range(16)]

        print('frac_demand_met: ', frac_demand_met)
        print('frac_meeting_guarantee: ', frac_meeting_guarantee)
        
        for p in range(16): 
            av_fraction_demand_met[p] += frac_demand_met[p]
            av_fraction_meeting_guarantee[p] += frac_meeting_guarantee[p]

        print('\n' + color.BOLD + 'Data for each period:', color.END, '\n')
        print('num_orders_placed: ', num_orders_placed)
        print('num_orders_fulfilled: ', num_orders_fulfilled)
        print('num_drivers_enter: ', num_drivers_enter)
        print('num_drivers_meeting_guarantee: ', num_drivers_meeting_guarantee) #should it be when the demand was placed or announced? 
        print() 
        print("driver_stats['guarantee_met']: ", sorted([id[0] for id in driver_stats['guarantee_met']]), len(driver_stats['guarantee_met'])); print()
        print("driver_stats['guarantee_missed']: ", sorted([id[0] for id in driver_stats['guarantee_missed']]), len(driver_stats['guarantee_missed'])); print()
        
        if sorted([id[0] for id in driver_stats['guarantee_met']]+[id[0] for id in driver_stats['guarantee_missed']]) != [i for i in range(total_no_drivers)]:
            print('ERROR')
            print()

        print()

        #print('av_fraction_demand_met: ', av_fraction_demand_met); print() 
        #print('av_fraction_meeting_guarantee: ', av_fraction_meeting_guarantee); print() 

    for p in range(16): 
        av_fraction_demand_met[p]/=(data['N']-1) 
        av_fraction_meeting_guarantee[p]/=(data['N']-1) 

    t_training_end = time()
    time_algorithm = t_training_end - t_training_begin
    return obj_val_list, time_algorithm, av_fraction_demand_met, av_fraction_meeting_guarantee, V, alpha 


# Step 2: Initiate VF
def initiate_VF():
    loc_set_0 = [[0.25+(0.5*x), 0.25+(0.5*y)] for x in range(20) for y in range(20)]  # aggreg. level 0
    loc_set_1 = [[0.5+x, 0.5+y] for x in range(10) for y in range(10)]                # aggreg. level 1
    loc_set_2 = [[1+(2*x), 1+(2*y)] for x in range(5) for y in range(5)]              # aggreg. level 2
    # bar_ha_set = [0.05*x for x in range(20)]
    # remaining_guarantee = [5*x for x in range(int(data['Guaranteed_service']/5) + 1)]
    return loc_set_0, loc_set_1, loc_set_2  #, bar_ha_set, remaining_guarantee


# if __name__ == '__main__':
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
    
    weight_driver_utilization = 1 

    data = {'T': 192, 'prob_enter':0.2, 'L': 0.8, 'Q': 1, 'w_d':weight_driver_utilization, 'w_s':1-weight_driver_utilization, 
            't_interval': 5, 
            'mu_enter': 2, 'mu_exit': 4, 'lambd': 10, 'delta_t': data_set['delta_t'].tolist(), 'N': 2, 'grid_size': (10, 10),
            'eta': 0.8, 'W': 0.8, 'theta': (1 / (2 * math.sqrt(200))), 'theta1': 1,
            'gamma': 0.9, 'Guaranteed_service': 120,
            'alpha_penalty': 1.0, 'lambda_initial': 10, 'h_int': 5, 'g_int': 5}
    
    d = {}

    for i in range(data['N']):  
        d[i] = data_set['d'].tolist() #same demand data for consistent comparison between different pool sizes 
 
    data['loc_set_0'], data['loc_set_1'], data['loc_set_2'] = initiate_VF()
    cfa_str = '_cfa' if cfa else ''
    # Step 3: Run algorithm
    pool_sizes = [1 for i in range(16)]; #arbitrarily set the pool sizes at 16 periods, each 12 epochs or 12*5 min = 1 h  

    av_pool_sizes = [1 for i in range(16)]; 

    V = defaultdict(lambda:0) #the value function that determines the value of choosing a pool size at a particular period
    #this is outside of for loop to learn across different pool sizes in different iterations 
    alpha = defaultdict(lambda:0) 
 
    I = 100  
    
    for i in range(I):
        
        print('\n' + color.BOLD + color.BLUE + 'I IS EQUAL TO: ', i, color.END); print()
 
        if i > 0:
            pool_sizes = func.boltzmann(V, i)
            for i in range(16): 
                av_pool_sizes[i]+=pool_sizes[i]

        print('pool_sizes: ', pool_sizes)

        obj_val_list, time_algorithm, av_fraction_demand_met, av_fraction_meeting_guarantee, V, alpha  = iterate(data, d, aggregate, penalty, cfa, pool_sizes, V, alpha)

        print('\n' + color.BOLD + color.GREEN + 'OVERALL RESULTS', color.END); print()

        print('obj_val_list: ', obj_val_list); print()  
        print('av_fraction_demand_met: ', av_fraction_demand_met); print() 
        print('av_fraction_meeting_guarantee: ', av_fraction_meeting_guarantee); print() 
    
        for p in range(16):

            alpha[p, pool_sizes[p]] += 1

            if alpha[p, pool_sizes[p]] == 0:
                alpha_step_size = 1
            else: 
                alpha_step_size = 1/(alpha[(p, pool_sizes[p])])**0.5 
                
            Qp = av_fraction_demand_met[p]; Lp = av_fraction_meeting_guarantee[p]
            
            #V_new =  sum(pool_sizes[p+1:]) + data['w_s']*(i+1)*max(data['Q']-Qp, 0) + data['w_d']*(i+1)*max(data['L']-Lp, 0) 
            
            V_new = data['w_s']*(i+1)*max(data['Q']-Qp, 0) + data['w_d']*(i+1)*max(data['L']-Lp, 0) 

            V[p, pool_sizes[p]] = (1-alpha_step_size)*V[p, pool_sizes[p]] + alpha_step_size*V_new
 
        #print('V: ', V); print(); print('alpha: ', alpha); print() 
    
        # name = "data/sol_n1000_VFA_T=%d_aggregation%s_penalty%d_mu%d_theta1_no_PCF_utiliz-wage" % (data['T'], cfa_str, penalty, data['mu_enter'])

        # np.savez(name, obj_val_list=obj_val_list, time_algorithm=time_algorithm)

    for i in range(16): 
        av_pool_sizes[i]/=I

    print('\n' + color.BOLD + color.DARKCYAN + 'OVERALL RESULTS ACROSS ALL ITERATIONS', color.END); print()
    print('av_pool_sizes: ', av_pool_sizes, sum(av_pool_sizes))

if __name__ == '__main__':
    penalty, cfa = 250, True
    run(penalty, cfa)
