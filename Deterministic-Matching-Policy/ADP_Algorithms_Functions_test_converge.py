import numpy as np
import math
from gurobipy import * 
from scipy.spatial import distance
from scipy.spatial.distance import euclidean 
import copy
import random

# temporary file for modifications to functions file to include possibility of aggregation
# data =
# {'t_interval': [], 'h_int': [], 'Guaranteed_service': [], 'gamma': [], 'eta': [], 'W': [], 'T': [], 'delta_t': []}
# Find closest node from discrete set

# Next: 1. make sure to skip complicated portion of update vf for disaggregate instances
# 2. Go through code line by line to check that it works well for aggregate and disaggregate instances.


def closest_node(node, loc_set):
    nodes = np.asarray(loc_set)
    closest_ind = distance.cdist([node], nodes).argmin()
    return loc_set[closest_ind]


# Generate a random sample path for 1 iteration
# Verified for 1. disagg alg, 2.
def sample_path(mu_enter, mu_exit, lambd, grid_size, t_interval, T, loc_set):
    # select sample path
    m_n = {}
    d_n = {}
    for t in range(T):
        m_t = np.random.poisson(mu_enter)  # realization of random driver entrance
        m_n[t] = m_t
        m_n[t, 'exit'] = np.random.poisson(mu_exit)

        # stopped here 11/25/20 7:27 am
        if m_t > 0:
            for i in range(m_t):
                # set random location of drivers on 10x10 grid
                m_n[t, i] = closest_node([np.random.uniform(0, grid_size[0]), np.random.uniform(0, grid_size[1])],
                                         loc_set)

        # Demand info
        d_t = np.random.poisson(lambd)
        d_n[t] = d_t
        if d_t > 0:
            for i in range(d_t):
                # set random origin and destination locations of drivers on 10x10 grid,
                # random announcement time between 2 epochs
                o_discrete = closest_node([np.random.uniform(0, grid_size[0]), np.random.uniform(0, grid_size[1])],
                                          loc_set)
                d_discrete = closest_node([np.random.uniform(0, grid_size[0]), np.random.uniform(0, grid_size[1])],
                                          loc_set)
                while o_discrete == d_discrete:  # to make sure origin != dest
                    d_discrete = closest_node([np.random.uniform(0, grid_size[0]), np.random.uniform(0, grid_size[1])],
                                              loc_set)
                d_n[t, i] = [[o_discrete, d_discrete],
                             np.random.uniform(max(0, (t - 1)) * t_interval, t * t_interval)]

    return d_n, m_n


# Driver and demand classes to generate objects
# Verified for 1. disagg alg, 2.
class Driver:
    def __init__(self, number, sa, ma, loc, loc_1, loc_2, time_first_matched, ha, bar_ha):
        self.number = number
        self.sa = sa    # binary=1 if driver active
        self.ma = ma    # binary=1 if available for matching
        self.loc = loc  # o_a current location
        self.loc_1 = loc_1  # aggregate 1, if disagg, pass []
        self.loc_2 = loc_2
        self.time_first_matched = time_first_matched
        self.ha = ha    # active time spent up to t
        self.bar_ha = bar_ha    # portion of active time driver is utilized
        self.iteration_history = {}

    def __str__(self):
        return 'This is driver no. {self.number}'.format(self=self)


# Verified for 1. disagg alg, 2.
class Demand:
    def __init__(self, number, origin, destination, dist, announce_time, delivery_deadline):
        self.number = number
        self.origin = origin
        self.destination = destination
        self.dst = dist
        self.announce_time = announce_time
        self.delivery_deadline = delivery_deadline
        self.iteration_history = {}

    def __str__(self):
        return 'This is demand no. {self.number}'.format(self=self)


# add compute weights
def compute_weights(key_0, key_1, key_2, v_t):
    key_list = [key_0, key_1, key_2]
    w_vect = []
    for ii in range(3):
        if key_list[ii] in v_t['count']:
            if ii == 0:  # if disaggregate  # Note (01/27/21: weight inverse of variance)
                fetch_var = v_t['var'][key_list[ii]]['sig_2']
            else:
                fetch_var = v_t['var'][key_list[ii]]['sig_2'] + v_t['var'][key_list[ii]]['mu']
            if fetch_var > 1e-6:
                w_vect.append(1 / fetch_var)
            else:  # if var=0, w = 1
                w_vect.append(1.0)  # w may be > 1. # todo: check that this makes sense :)
        else:
            w_vect.append(0)

    tot = sum(w_vect)
    if tot < 1e-5:  # 0
        w = [0, 0, 0]
    else:
        w = [(w_vect[ii] / tot) for ii in range(len(w_vect))]
    return w


# compute rho_a (priority score)
def cal_rho_a(bar_ha):
    # rho_a = [[0, 0.8, 1], [1, 0, 0]]
    # rho_b = [[[0, 0.24999999], [0.5, 0.5]], [[0.25, 0.5, 1], [0.25, 0, 0]]]
    if bar_ha < 0.8:
        rho_a = -1.25 * bar_ha + 1  # eqn of line from above
        # rho_a = -0.625 * bar_ha + 1  # changed line to be 0.5 penalty at 0.8  # changed for rho_a_v2
    else:
        rho_a = 0
    return rho_a


# compute rho_b (priority score)
def cal_rho_b(perc):  # perc away from min time window to fulfill order
    if perc <= 0.25:
        rho_b = 0.5
    elif perc > 0.5:
        rho_b = 0
    else:
        rho_b = -1 * perc + 0.5  # eqn of line from above points
    return rho_b


# Get VF or weighted avg if aggregate
def fetch_vf(aggregate, key_0, v_t, n):
    if key_0 not in v_t['n']:
        vf = 0
    else:
        if n in v_t['n'][key_0][0]:
            ind = v_t['n'][key_0][0].index(n)
            if aggregate:
                vf = v_t['n'][key_0][2][ind]
            else:
                vf = v_t['n'][key_0][1][ind]
        else:
            if min(v_t['n'][key_0][0]) > n:
                vf = 0
            else:
                abs_diff = lambda list_val: n-list_val if list_val < n else 10000
                n_get = min(v_t['n'][key_0][0], key=abs_diff)
                ind = v_t['n'][key_0][0].index(n_get)  # ind of the closest n
                if aggregate:
                    vf = v_t['n'][key_0][2][ind]
                else:
                    vf = v_t['n'][key_0][1][ind]

    return vf


# Setup model
def setup_model(data, t, n, demand_list, driver_list, driver_attr, demand_attr, v_t, aggregate, penalty):
    model_name = 'LP_t%d_n%d' % (t, n)
    mdl = Model(name=model_name)
    # define decision variables: x_tab
    x_ab_feasible = {('b', b): [] for b in demand_list['active']}
    x_ab_cost, x_ab_time, x_ab_dist, keys = {}, {}, {}, []

    drivers = driver_list['act_available'] + driver_list['inactive']
    for a in drivers:
        x_ab_feasible[('a', a)] = [0]
        keys.append((a, 0))
        sa = driver_attr[a].sa

        if sa == 0:
            bar_ha_new = 0
            g = 0
        else:
            if driver_attr[a].bar_ha == 0:
                print('Not sure why bar_ha = %d, ha = %d, sa = %d' % (driver_attr[a].bar_ha,
                                                                      driver_attr[a].ha, sa))
                tot_time = data['t_interval']
            else:
                tot_time = (driver_attr[a].ha / driver_attr[a].bar_ha) + data['t_interval']
            bar_ha_new = round(driver_attr[a].ha / tot_time * 100 / data['h_int']) * data['h_int'] / 100
            g = max(data['Guaranteed_service'] - ((t + 1) * data['t_interval'] - driver_attr[a].time_first_matched), 0)

        # value of having a driver in next dec epoch with such attr
        key_0 = ('t', t+1, 'sa', sa, 'loc', tuple(driver_attr[a].loc), 'bar_ha', bar_ha_new, 'g', g)
        vf = fetch_vf(aggregate, key_0, v_t, n)

        # obj func value when unmatched (matched to 0)
        cost_to_go = (data['gamma'] * vf)
        if sa == 0 or (sa == 1 and g > 0) or (sa == 1 and g <= 0 and bar_ha_new >= data['W']):  # activity time window not over # :
            x_ab_cost[(a, 0)] = 0 + cost_to_go
        else:
            rho_a = cal_rho_a(bar_ha_new)
            x_ab_cost[(a, 0)] = (-penalty * rho_a) + cost_to_go  # penalty = -1000 for high penalty or -1 for regular

        a_loc = tuple(driver_attr[a].loc)

        for b in demand_list['active']:
            b_loc_o = tuple(demand_attr[b].origin)
            dst = distance.euclidean(a_loc, b_loc_o)
            dst_o_d = demand_attr[b].dst
            dst_tot = dst + dst_o_d  # dst from driver to origin + package orig to dest.
            x_ab_dist[(a, b)] = dst
            x_ab_dist[(a, b, 'tot_dst')] = dst_tot
            cont_time = data['eta'] * dst_tot
            disc_time = math.ceil(cont_time / data['t_interval']) * data['t_interval']
            time = disc_time + (t * data['t_interval'])  # time interval + time of t
            t_epoch = int(time/data['t_interval'])

            if time <= demand_attr[b].delivery_deadline:  # if can be delivered before deadline
                # find aggregate expected location:
                loc = demand_attr[b].destination  # loc of demand dest
                sa = 1

                if driver_attr[a].sa == 0:
                    bar_ha_new = 1.0
                    g = max(data['Guaranteed_service'] - disc_time, 0)
                else:
                    if driver_attr[a].bar_ha == 0:
                        print('Not sure why bar_ha = %d, ha = %d, sa = %d' % (driver_attr[a].bar_ha,
                                                                              driver_attr[a].ha, sa))
                        tot_time = disc_time
                    else:
                        tot_time = (driver_attr[a].ha / driver_attr[a].bar_ha) + disc_time
                    bar_ha_new = round((driver_attr[a].ha + disc_time) / tot_time * 100 / data['h_int']) * data['h_int'] / 100
                    g = max(data['Guaranteed_service'] - (time - driver_attr[a].time_first_matched), 0)

                # Fetch VF
                # value of having a driver with such attr in the future (time). # updated 02/08/21
                key_0 = ('t', t_epoch, 'sa', sa, 'loc', tuple(loc), 'bar_ha', bar_ha_new, 'g', g)
                vf = fetch_vf(aggregate, key_0, v_t, n)
                cost_to_go = (data['gamma'] * vf)
                keys.append((a, b))
                x_ab_feasible[('a', a)].append(b)
                x_ab_feasible[('b', b)].append(a)

                # removed since not parametric cost function
                # perc = 1 - float(time / demand_attr[b].delivery_deadline)  # perc of time window remaining
                # rho_b = cal_rho_b(perc)

                # rho_a = cal_rho_a(bar_ha_new)  # get rho_a for each a/b match
                # changed first term of c to dst_origin_dest instead of 1
                if g > 0 or (g <= 0 and bar_ha_new >= data['W']):  # if there's time or no time but guarantee met
                    c = (data['theta1'] * dst_o_d) - (data['theta'] * dst) + cost_to_go
                else:  # no time left but guarantee unmet
                    rho_a = cal_rho_a(bar_ha_new)
                    c = (data['theta1'] * dst_o_d) - (data['theta'] * dst) + cost_to_go + (-penalty * rho_a)

                # Note: right now only accounting for distance from driver to origin
                x_ab_cost[(a, b)] = c
                x_ab_time[(a, b)] = time
    # define decision variables: x_tab
    mdl.xab = mdl.continuous_var_dict(keys, lb=0)

    # define cost function
    mdl.match = mdl.sum(mdl.xab[(a, b)] * x_ab_cost[(a, b)] for a in drivers
                        for b in x_ab_feasible[('a', a)])
    mdl.maximize(mdl.match)

    # Add constraints in a separate function
    # resource constraint
    a_cnst = mdl.add_constraints((mdl.sum(mdl.xab[(a, b)] for b in x_ab_feasible[('a', a)]) == 1, 'Resource_a%d'
                                  % a) for a in drivers)

    # demand constraint
    mdl.add_constraints((mdl.sum(mdl.xab[(a, b)] for a in x_ab_feasible[('b', b)]) <= 1, 'Demand_b%d'
                         % b) for b in demand_list['active'])
    return mdl, x_ab_feasible, x_ab_time, x_ab_dist, a_cnst


def setup_model_cfa(data, t, n, demand_list, driver_list, driver_attr, demand_attr, v_t, aggregate, penalty):
    model_name = 'LP_t%d_n%d' % (t, n)
    mdl = Model(name=model_name)
    # define decision variables: x_tab
    x_ab_feasible = {('b', b): [] for b in demand_list['active']}
    x_ab_cost, x_ab_time, x_ab_dist, keys = {}, {}, {}, []

    drivers = driver_list['act_available'] + driver_list['inactive']
    for a in drivers:
        x_ab_feasible[('a', a)] = [0]
        keys.append((a, 0))
        sa = driver_attr[a].sa

        if sa == 0:
            bar_ha_new = 0
            g = 0
        else:
            if driver_attr[a].bar_ha == 0:
                print('Not sure why bar_ha = %d, ha = %d, sa = %d' % (driver_attr[a].bar_ha,
                                                                      driver_attr[a].ha, sa))
                tot_time = data['t_interval']
            else:
                tot_time = (driver_attr[a].ha / driver_attr[a].bar_ha) + data['t_interval']
            bar_ha_new = round(driver_attr[a].ha / tot_time * 100 / data['h_int']) * data['h_int'] / 100
            g = max(data['Guaranteed_service'] - ((t + 1) * data['t_interval'] - driver_attr[a].time_first_matched), 0)

        # value of having a driver in next dec epoch with such attr
        key_0 = ('t', t+1, 'sa', sa, 'loc', tuple(driver_attr[a].loc), 'bar_ha', bar_ha_new, 'g', g)
        vf = fetch_vf(aggregate, key_0, v_t, n)

        # obj func value when unmatched (matched to 0)
        cost_to_go = (data['gamma'] * vf)
        if sa == 0 or (sa == 1 and bar_ha_new > data['W']):  # activity time window not over # :
            x_ab_cost[(a, 0)] = 0 + cost_to_go
        else:
            rho_a = cal_rho_a(bar_ha_new)
            x_ab_cost[(a, 0)] = (-penalty * rho_a) + cost_to_go  # penalty = -1000 for high penalty or -1 for regular

        a_loc = tuple(driver_attr[a].loc)

        for b in demand_list['active']:
            b_loc_o = tuple(demand_attr[b].origin)
            dst = distance.euclidean(a_loc, b_loc_o)
            dst_o_d = demand_attr[b].dst
            dst_tot = dst + dst_o_d  # dst from driver to origin + package orig to dest.
            x_ab_dist[(a, b)] = dst
            x_ab_dist[(a, b, 'tot_dst')] = dst_tot
            cont_time = data['eta'] * dst_tot
            disc_time = math.ceil(cont_time / data['t_interval']) * data['t_interval']
            time = disc_time + (t * data['t_interval'])  # time interval + time of t
            t_epoch = int(time/data['t_interval'])

            if time <= demand_attr[b].delivery_deadline:  # if can be delivered before deadline
                # find aggregate expected location:
                loc = demand_attr[b].destination  # loc of demand dest
                sa = 1

                if driver_attr[a].sa == 0:
                    bar_ha_new = 1.0
                    g = max(data['Guaranteed_service'] - disc_time, 0)
                else:
                    if driver_attr[a].bar_ha == 0:
                        print('Not sure why bar_ha = %d, ha = %d, sa = %d' % (driver_attr[a].bar_ha,
                                                                              driver_attr[a].ha, sa))
                        tot_time = disc_time
                    else:
                        tot_time = (driver_attr[a].ha / driver_attr[a].bar_ha) + disc_time
                    bar_ha_new = round((driver_attr[a].ha + disc_time) / tot_time * 100 / data['h_int']) * data['h_int'] / 100
                    g = max(data['Guaranteed_service'] - (time - driver_attr[a].time_first_matched), 0)

                # Fetch VF
                # value of having a driver with such attr in the future (time). # updated 02/08/21
                key_0 = ('t', t_epoch, 'sa', sa, 'loc', tuple(loc), 'bar_ha', bar_ha_new, 'g', g)
                vf = fetch_vf(aggregate, key_0, v_t, n)
                cost_to_go = (data['gamma'] * vf)
                keys.append((a, b))
                x_ab_feasible[('a', a)].append(b)
                x_ab_feasible[('b', b)].append(a)

                # removed since not parametric cost function
                perc = 1 - float(time / demand_attr[b].delivery_deadline)  # perc of time window remaining
                rho_b = cal_rho_b(perc)

                rho_a = cal_rho_a(bar_ha_new)  # get rho_a for each a/b match
                # changed first term of c to dst_origin_dest instead of 1
                if g > 0 or (g <= 0 and bar_ha_new >= data['W']):  # if there's time or no time but guarantee met
                    c = (data['theta1'] * dst_o_d) - (data['theta'] * dst) + data['alpha_penalty'] * (rho_a + rho_b) \
                        + cost_to_go
                else:  # no time left but guarantee unmet
                    # rho_a = cal_rho_a(bar_ha_new)
                    c = (data['theta1'] * dst_o_d) - (data['theta'] * dst) + data['alpha_penalty'] * (rho_a + rho_b)\
                        + cost_to_go + (-penalty * rho_a)

                # Note: right now only accounting for distance from driver to origin
                x_ab_cost[(a, b)] = c
                x_ab_time[(a, b)] = time
    # define decision variables: x_tab
    mdl.xab = mdl.continuous_var_dict(keys, lb=0)

    # define cost function
    mdl.match = mdl.sum(mdl.xab[(a, b)] * x_ab_cost[(a, b)] for a in drivers
                        for b in x_ab_feasible[('a', a)])
    mdl.maximize(mdl.match)

    # Add constraints in a separate function
    # resource constraint
    a_cnst = mdl.add_constraints((mdl.sum(mdl.xab[(a, b)] for b in x_ab_feasible[('a', a)]) == 1, 'Resource_a%d'
                                  % a) for a in drivers)

    # demand constraint
    mdl.add_constraints((mdl.sum(mdl.xab[(a, b)] for a in x_ab_feasible[('b', b)]) <= 1, 'Demand_b%d'
                         % b) for b in demand_list['active'])
    return mdl, x_ab_feasible, x_ab_time, x_ab_dist, a_cnst

# solve model
def solve_get_values(mdl, x_ab_feasible, a_cnst, driver_list, x_ab_dist):
    # solve model
    mdl.solve()
    # Get obj and var values
    z = mdl.objective_value

    print('z = ' + str(z))
    xab = {('b', 0): []}
    dual_a = {"all": mdl.dual_values(a_cnst)}
    dst, tot_dst = 0, 0
    # Get solution values
    num_matches = 0
    drivers = driver_list['act_available'] + driver_list['inactive']
    for i in range(len(drivers)):
        a = drivers[i]
        for b in x_ab_feasible[('a', a)]:
            if mdl.xab[(a, b)].solution_value >= 0.99:
                xab[('a', a)] = b
                if b != 0:
                    dst += x_ab_dist[(a, b)]
                    tot_dst += x_ab_dist[(a, b, 'tot_dst')]
                break  # break out of the b loop as soon as you find match
        if xab[('a', a)] != 0:
            xab[('b', b)] = a
            num_matches += 1
        else:
            xab[('b', 0)].append(a)  # driver unmatched
        # Get dual values
        dual_a[a] = dual_a["all"][i]
    print('num of matches = ' + str(num_matches))
    return xab, z, dual_a, num_matches, dst, tot_dst


# setup model then solve it
# Verified for 1. disagg alg, 2. agg alg, 3. Test_VFA_convergence_N
def solve_opt(data, t, n, demand_list, driver_list, driver_attr, demand_attr, v_t, aggregate, penalty, cfa):
    # (02/22/21) note: penalty added to solve high penalty and regular instances automatically
    if not cfa:
        mdl, x_ab_feasible, x_ab_time, x_ab_dist, a_cnst = setup_model(data, t, n, demand_list, driver_list,
                                                        driver_attr, demand_attr, v_t, aggregate, penalty)
    else:
        mdl, x_ab_feasible, x_ab_time, x_ab_dist, a_cnst = setup_model_cfa(data, t, n, demand_list, driver_list,
                                                                       driver_attr, demand_attr, v_t, aggregate,
                                                                       penalty)
    xab, z, dual_a, num_matches, dst, tot_dst = solve_get_values(mdl, x_ab_feasible, a_cnst, driver_list, x_ab_dist)
    return xab, z, dual_a, num_matches, x_ab_time, dst, tot_dst

# update VF
# Verified for 1. disagg alg, 2. agg alg
def update_vf(driver_list, driver_attr, data, t, dual_a, v_t, n, aggregate):
    temp, count, met_before = {}, {}, {}
    drivers = driver_list['act_available'] + driver_list['inactive']  # original list of drivers before matching

    def compute_variance():  # done
        def compute_key_stats(k, k0, level_0):  # check document to verify equations
            num = v_t['count'][k]
            if met_before[k][0]:  # if we have seen this key before
                n_last = met_before[k][1]  # get last n where key is seen before current one
                diff = temp[k0] - v_t[n_last][k]  # new estimate - old
                # print('n_last = ' + str(n_last) + ', met_before= ' + str(met_before[k]))
                # bar_beta  # now over-writing
                b_beta_old = v_t['var'][k]['bar_beta']
                v_t['var'][k]['bar_beta'] = (0.9 * b_beta_old) + (0.1 * diff)
                # disaggregate loc, no need for mu (bias)

                # bar_bar_beta
                bb_beta_old = v_t['var'][k]['bar_bar_beta']
                v_t['var'][k]['bar_bar_beta'] = \
                    (0.9 * bb_beta_old) + (0.1 * math.pow(diff, 2))

                # lambda
                lamb_old = v_t['var'][k]['lamb']  # todo: check this equation
                v_t['var'][k]['lamb'] = math.pow((1 - (1 / (num - 1))), 2) * lamb_old + \
                                        math.pow((1 / (num - 1)), 2)

                # s_{ta}^2
                v_t['var'][k]['s'] = (v_t['var'][k]['bar_bar_beta'] -
                                      math.pow(v_t['var'][k]['bar_beta'], 2)) / (
                                                      1 + v_t['var'][k]['lamb'])

                v_t['var'][k]['sig_2'] = v_t['var'][k]['lamb'] * v_t['var'][k]['s']
                if not level_0:
                    diff_g = v_t[n][k] - v_t[n][k0]
                    v_t['var'][k]['mu'] = math.pow(diff_g, 2)  # store it as mu^2

            else:
                diff = temp[k0]
                v_t['var'][k] = {'bar_beta': {}, 'bar_bar_beta': {}, 'lamb': {}, 's': {}, 'sig_2': {}}
                # diff, diff^2, 1,
                v_t['var'][k]['bar_beta'], v_t['var'][k]['bar_bar_beta'], v_t['var'][k]['lamb'] = \
                    diff, math.pow(diff, 2), 1.00  # lamb = (1/n)^2, n=1

                v_t['var'][k]['s'] = (v_t['var'][k]['bar_bar_beta'] -
                                           math.pow(v_t['var'][k]['bar_beta'], 2)) / (
                                                      1 + v_t['var'][k]['lamb'])

                v_t['var'][k]['sig_2'] = v_t['var'][k]['lamb'] * v_t['var'][k]['s']
                if not level_0:
                    diff_g = v_t[n][k] - v_t[n][k0]
                    v_t['var'][k]['mu'] = math.pow(diff_g, 2)  # store it as mu^2

        for key_0 in temp:
            compute_key_stats(key_0, key_0, True)  # first get states for disaggregate
            for key_x in key_dict[key_0]:  # get states for aggregate
                compute_key_stats(key_x, key_0, False)

    if aggregate:
        key_dict = {}
    for i in range(len(drivers)):
        # 1. Get attribute key. If aggregate get agg. keys as well
        a = drivers[i]  # driver identifier no.
        loc_0 = driver_attr[a].loc
        if aggregate:
            loc_1, loc_2 = driver_attr[a].loc_1, driver_attr[a].loc_2
        sa, bar_ha = driver_attr[a].sa, driver_attr[a].bar_ha
        if sa == 1:  # t since it's start of next epoch.
            g = max(data['Guaranteed_service'] - (t*data['t_interval'] - driver_attr[a].time_first_matched), 0)
        else:
            g = 0  # g is the remaining guarantee time
        if g % data['g_int'] != 0:
            print('Not a discrete value for remaining guaranteed service time')

        key_0 = ('t', t, 'sa', sa, 'loc', tuple(loc_0), 'bar_ha', bar_ha, 'g', g)
        # verified that this is the right way of capturing the VF
        if aggregate:
            key_1, key_2 = list(copy.copy(key_0)), list(copy.copy(key_0))
            key_1[5], key_2[5] = tuple(loc_1), tuple(loc_2)
            key_1, key_2 = tuple(key_1), tuple(key_2)
            key_dict[key_0] = [key_1, key_2]

        # 2. Get value of attr from dual variables
        if key_0 in temp:
            temp[key_0] += dual_a[a]  # todo: check how often this happens
        else:
            temp[key_0] = dual_a[a]
        key_list = [key_0] + key_dict[key_0] if aggregate else [key_0]

        # 3. Keep track of all keys (agg/disagg) without double counting
        for key in key_list:
            if key not in count:
                count[key] = 1
                count[key, 'map'] = [key_0]
            else:
                count[key, 'map'].append(key_0)  # map: key to it's disaggregate

    # 4. update count: to avoid double counting
    for key in count:
        if key in v_t['count']:
            v_t['count'][key] += 1
        else:
            v_t['count'][key] = 1

    # 5. update new values # 5a. Update unique values
    # todo: probably a good idea to review this line by line for agg case
    repeated_keys = []  # keys with more than one occurrence
    # update unrepeated keys
    for key_0 in temp:  # temp stores only disaggregate keys
        key_list = [key_0] + key_dict[key_0] if aggregate else [key_0]
        for key in key_list:
            if len(count[key, 'map']) > 1:  # skip repeated_keys and handle them separately next
                if key not in repeated_keys:
                    repeated_keys.append(key)
            else:
                if key in v_t['last_met']:
                    n_last = v_t['last_met'][key]
                    num_occur = v_t['count'][key]
                    # verify how to update the value of key if met more than once in same n,t
                    met_before[key] = [True, n_last]
                    v_t[n][key] = (1 - (1 / num_occur)) * v_t[n_last][key] + (1 / num_occur) * temp[key_0]

                else:
                    v_t[n][key] = temp[key_0]
                    met_before[key] = [False]
                v_t['last_met'][key] = n  # instead of copying n-1, use v_t['last_met'] to store iteration no.

    # update repeated keys
    for key in repeated_keys:
        value_n = 0  # new value at iteration n of aggregate key
        for key_0 in count[key, 'map']:
            value_n += temp[key_0]  # sum value of disaggregate keys
        if key in v_t['last_met']:  # seen this before
            n_last = v_t['last_met'][key]
            if n_last == n:
                print('Error: Why does n_last equal n?')
            num_occur = v_t['count'][key]
            met_before[key] = [True, n_last]
            v_t[n][key] = (1 - (1 / num_occur)) * v_t[n_last][key] + (1 / num_occur) * value_n  # updated value_n
        else:
            v_t[n][key] = value_n
            met_before[key] = [False]
        v_t['last_met'][key] = n

    if aggregate:
        compute_variance()  # done
    # return v_t


# Step 3.4a: # Done
# This func updates driver and demand attributes after matching.
# Verified for 1. disagg alg, 2. agg alg, 3. Test_VFA_convergence_N
def update_attr(driver_list, demand_list, xab, driver_attr, demand_attr, t, data, x_ab_time, aggregate, stats=False,
                driver_log=[], driver_stats=[], demand_stats=[]):
    # data = {'t_interval': [], 'h_int': [], 'Guaranteed_service': [], 'gamma': [], 'eta': [], 'W': [], 'T': []}
    driver_temp = {'act_av': [], 'act_unav': [], 'inact': []}
    drivers = driver_list['act_available'] + driver_list['inactive']

    for a in drivers:
        # if driver was not matched at t
        if xab[('a', a)] == 0:
            if driver_attr[a].sa == 0:  # if driver is inactive  # driver attr does not change
                driver_temp['inact'].append(a)
                if stats:
                    driver_log[a]['all_t'].append(['t%d_post-decision' % t, 'unmatched', 'inactive'])
            else:  # if driver was previously active
                if stats:
                    driver_log[a]['all_t'].append(['t%d_post-decision' % t, 'unmatched', 'active',
                                                   [driver_attr[a].ha, driver_attr[a].bar_ha]])

                if ((t+1)*data['t_interval'] - driver_attr[a].time_first_matched) >= data['Guaranteed_service']:
                    loc_d = driver_attr[a].loc  # Driver attribute resets, location does not change
                    if aggregate:
                        loc_1, loc_2 = closest_node(loc_d, data['loc_set_1']), closest_node(loc_d, data['loc_set_2'])
                    else:
                        loc_1, loc_2 = [], []
                    if stats:
                        bar_ha = driver_attr[a].bar_ha
                        if bar_ha >= data['W']:
                            driver_stats['guarantee_met'].append([a, bar_ha])
                        else:
                            driver_stats['guarantee_missed'].append([a, bar_ha])

                    driver_attr[a] = Driver(a, 0, 1, loc_d, loc_1, loc_2, 0, 0, 0)
                    driver_temp['inact'].append(a)
                else:  # duration < guaranteed service (& driver was not matched)
                    busy_time = driver_attr[a].ha
                    if driver_attr[a].bar_ha == 0:  # if no prev busy time
                        tot_time = data['t_interval']
                        if busy_time > 1e-6:  # check
                            print('Possible error here! busy_time > tot_time')
                    else:
                        tot_time = (driver_attr[a].ha/driver_attr[a].bar_ha) + data['t_interval']
                    # total prev time(active time/perc active) + t interval
                    bar_ha = round(busy_time/tot_time*100/data['h_int'])*data['h_int']/100
                    if bar_ha > 1.0:
                        print('Stop and debug!')
                    driver_attr[a].bar_ha = bar_ha
                    driver_temp['act_av'].append(a)
                    driver_attr[a].iteration_history[t] = "This driver was not matched at t = " + str(t) \
                                                          + " and is in location " + str(driver_attr[a].loc) + "."
        else:  # Next: continue att update if driver was matched
            b = xab[('a', a)]
            if driver_attr[a].sa == 0:
                driver_attr[a].time_first_matched = t * data['t_interval']
                driver_attr[a].sa = 1

            # check if time it takes to make delivery will be before next t
            if x_ab_time[(a, b)] <= (t+1)*data['t_interval']:  # if fulfill delivery before next period
                driver_temp['act_av'].append(a)
            # if driver will still be delivering by t+1
            else:
                # unavailable, next time available is x_ab_time or possibly change entry 1 to x_ab_time right away
                driver_attr[a].ma = 0
                driver_attr[a].available_time = x_ab_time[(a, b)]
                driver_temp['act_unav'].append(a)

            busy_time = driver_attr[a].ha + (x_ab_time[(a, b)] - t * data['t_interval'])  # additional busy time t
            if driver_attr[a].bar_ha == 0:
                tot_time = (x_ab_time[(a, b)] - t * data['t_interval'])
            else:
                tot_time = (driver_attr[a].ha / driver_attr[a].bar_ha) + (x_ab_time[(a, b)] - t * data['t_interval'])
            if tot_time < 1e-6:
                print('Why is this zero?')  # loc = demand
                perc = 0.0
            else:
                perc = float(busy_time / tot_time)
            driver_attr[a].iteration_history[t] = \
                "This driver was matched with order %d at time t = %d, and is in location %s at time %f." % (b, t,
                                                            str(driver_attr[a].loc),x_ab_time[(a, b)])
            loc = demand_attr[b].destination
            if aggregate:
                loc_1, loc_2 = closest_node(loc, data['loc_set_1']), closest_node(loc, data['loc_set_2'])
            else:
                loc_1, loc_2 = [], []
            driver_attr[a].loc, driver_attr[a].loc_1, driver_attr[a].loc_2 = loc, loc_1, loc_2
            driver_attr[a].ha = busy_time
            bar_ha = round(perc*100/data['h_int'])*data['h_int']/100
            if bar_ha > 1.0:
                print('Stop and debug!')
            driver_attr[a].bar_ha = bar_ha
            if stats:
                driver_log[a]['all_t'].append(['t%d_post_decision' % t,
                    [demand_attr[b].origin, demand_attr[b].destination, x_ab_time[(a, b)], busy_time,
                     driver_attr[a].bar_ha]])

    if t > 0:  # check if previously unavailable drivers are now available
        if len(driver_list['act_unavailable']) > 0:
            for a in driver_list['act_unavailable']:
                avail_time = driver_attr[a].available_time
                if avail_time <= (t+1)*data['t_interval']:  # if driver will be avail at t+1
                    if ((t+1)*data['t_interval'] - driver_attr[a].time_first_matched) >= data['Guaranteed_service']:
                        # driver attr resets if he meets the x hours threshold
                        loc = driver_attr[a].loc
                        if aggregate:
                            loc_1, loc_2 = closest_node(loc, data['loc_set_1']), closest_node(loc, data['loc_set_2'])
                        else:
                            loc_1, loc_2 = [], []

                        if stats:
                            bar_ha = driver_attr[a].bar_ha
                            if bar_ha >= data['W']:
                                driver_stats['guarantee_met'].append([a, bar_ha])
                            else:
                                driver_stats['guarantee_missed'].append([a, bar_ha])
                        driver_attr[a] = Driver(a, 0, 1, loc, loc_1, loc_2, 0, 0, 0)
                        driver_temp['inact'].append(a)
                    else:  # done deliv, within guaranteed service  --> available, active
                        driver_attr[a].ma = 1
                        driver_temp['act_av'].append(a)
                else:  # if not done delivering, move it to unavailable t+1
                    driver_temp['act_unav'].append(a)

    if stats:
        driver_stats[t, 'active'] = len(driver_temp['act_av']) + len(driver_temp['act_unav'])

    demand_temp = {"active": []}  # Now, move on to update demand info :)
    for b in demand_list['active']:
        if ('b', b) not in xab:  # if demand b is not satisfied carry it forward if not missed
            # if deadline is after t+1 + min time
            if demand_attr[b].delivery_deadline >= ((t+1)*data['t_interval']) + (data['eta']*demand_attr[b].dst):
                demand_temp['active'].append(b)
            else:
                demand_list['expired'].append(b)
                if stats:
                    demand_stats['missed'].append(b)
        else:
            demand_list['fulfilled'].append(b)
            if stats:
                demand_stats['met'].append(b)
    return driver_temp, demand_temp


# Verified for 1. disagg alg, 2. agg alg, 3. Test_VFA_convergence_N
def predecision_state(num_drivers, num_demand, driver_temp, demand_temp, driver_list, demand_list, t, n, m, d,
                      driver_attr, demand_attr, data, aggregate, train, stats=False, driver_log=[], driver_stats=[]):
    # (1) check if any drivers are unmatched, if yes, one random leaves if mu_exit>0
    # (2) new arrivals of both drivers and demand from sample path.
    # Note: create complete sample path of n=1 in data generation
    if train:  # train = True if training model
        m_n, d_n = m[n], d[n]
    else:
        m_n, d_n = m, d

    if stats and t + 1 >= data['T']:
        pass
    else:
        if m_n[t, 'exit'] > 0:
            num_inactive = len(driver_temp['inact'])
            samples = min(m_n[t, 'exit'], num_inactive)  # exit at end of t

            # decide on which drivers to exit randomly
            driver_exit = random.sample(driver_temp['inact'], samples)
            for a in driver_exit:
                driver_list['exit'].append(a)
                driver_temp['inact'].remove(a)
                if stats:
                    driver_log[a]['all_t'].append(['t%d' % t, 'exited'])

    def update_driver_attr(num_drivers):
        driver_list['act_available'] = copy.copy(driver_temp['act_av'])
        driver_list['act_unavailable'] = copy.copy(driver_temp['act_unav'])
        driver_list['inactive'] = copy.copy(driver_temp['inact'])

        # Let go of drivers who reached guarantee
        for a in driver_list['act_available']:
            if (t+1)*data["t_interval"] - driver_attr[a].time_first_matched >= data['Guaranteed_service']:
                loc = driver_attr[a].loc
                if aggregate:
                    loc_1, loc_2 = closest_node(loc, data['loc_set_1']), closest_node(loc, data['loc_set_2'])
                else:
                    loc_1, loc_2 = [], []
                if stats:
                    bar_ha = driver_attr[a].bar_ha
                    if bar_ha >= data['W']:
                        driver_stats['guarantee_met'].append([a, bar_ha])
                    else:
                        driver_stats['guarantee_missed'].append([a, bar_ha])

                driver_attr[a] = Driver(a, 0, 1, loc, loc_1, loc_2, 0, 0, 0)
                driver_list['inactive'].append(a)
                driver_list['act_available'].remove(a)
                if stats:
                    driver_log[a]['all_t'].append(['t%d' % (t+1), [0, 1, loc, 0, 0, 0]])
                    # todo: maybe update to also log loc_1 and loc_2
            else:
                if stats:
                    driver_log[a]['all_t'].append(['t%d' % (t + 1), [driver_attr[a].sa, driver_attr[a].ma,
                     driver_attr[a].loc, driver_attr[a].time_first_matched, driver_attr[a].ha, driver_attr[a].bar_ha]])

        # new drivers
        if stats and t+1 >= data['T']:
            pass
        else:
            for j in range(m_n[t+1]):  # create new driver objects range(m[n][t+1])
                a = j+num_drivers
                loc = m_n[t+1, j]
                if aggregate:
                    loc_1, loc_2 = closest_node(loc, data['loc_set_1']), closest_node(loc, data['loc_set_2'])
                else:
                    loc_1, loc_2 = [], []

                if stats:
                    driver_log[a] = {'all_t': ['t%d' % (t + 1), [0, 1, loc, 0, 0, 0]]}
                driver_attr[a] = Driver(a, 0, 1, loc, loc_1, loc_2, 0, 0, 0)

                driver_list['inactive'].append(a)
            num_drivers += m_n[t+1]
        return num_drivers

    def update_demand_attr(num_demand):
        demand_list['active'] = copy.copy(demand_temp['active'])

        if stats and t+1 >= data['T']:
            pass
        else:
            # new demand
            for j in range(d_n[t+1]):  # no. of new demands
                b = num_demand + j + 1  # unique identifier
                i_loc_o = d_n[t+1, j][0][0]
                i_loc_d = d_n[t+1, j][0][1]
                dst = distance.euclidean(i_loc_o, i_loc_d)
                announce_time = d_n[t+1, j][1]
                demand_attr[b] = Demand(b, i_loc_o, i_loc_d, dst, announce_time, announce_time + data['delta_t'])
                demand_list['active'].append(b)
            num_demand += d_n[t+1]
        return num_demand

    num_drivers = update_driver_attr(num_drivers)
    num_demand = update_demand_attr(num_demand)
    return num_drivers, num_demand



