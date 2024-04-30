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

        prob_enter = 0.2; T = 192 
        num_drivers_enter = [np.random.binomial(pool_sizes[t//12], prob_enter) for t in range(T)] 

        #num_drivers_enter = [0, 0, 1, 1, 0, 0, 3, 2, 0, 0, 2, 0, 0, 2, 0, 3, 1, 1, 1, 3, 1, 1, 0, 1, 3, 1, 0, 2, 0, 1, 2, 0, 2, 0, 0, 2, 0, 3, 1, 2, 1, 1, 1, 1, 1, 2, 2, 0, 0, 1, 2, 0, 1, 1, 1, 0, 0, 0, 1, 0, 2, 0, 1, 1, 0, 1, 2, 3, 1, 1, 1, 2, 0, 1, 4, 4, 1, 1, 2, 0, 2, 1, 1, 0, 1, 3, 2, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 3, 0, 1, 2, 2, 1, 2, 2, 1, 2, 1, 0, 0, 1, 1, 0, 1, 3, 0, 1, 4, 2, 0, 1, 1, 0, 2, 2, 2, 3, 1, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0, 2, 3, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 0, 0, 0, 3, 1, 1, 2, 1, 0, 1, 0, 1, 1, 0, 0, 2, 1, 0, 1, 3, 0, 0, 0, 1, 3, 1, 0, 0, 1, 0, 1, 1, 0, 0, 2]

        num_orders_placed = [d[n][t] for t in range(T)] 

        _, m[n] = func.sample_path(data['mu_enter'], data['mu_exit'], data['lambd'],
                                       data['grid_size'], data['t_interval'], data['T'], data['loc_set_0'], num_drivers_enter, num_orders_placed)
        
        #m[n] = {0: 0, (0, 'exit'): 5, 1: 0, (1, 'exit'): 5, 2: 1, (2, 'exit'): 2, (2, 0): [0.75, 9.75], 3: 1, (3, 'exit'): 4, (3, 0): [7.75, 6.25], 4: 0, (4, 'exit'): 7, 5: 0, (5, 'exit'): 2, 6: 3, (6, 'exit'): 3, (6, 0): [7.75, 9.25], (6, 1): [0.75, 5.75], (6, 2): [2.25, 2.25], 7: 2, (7, 'exit'): 2, (7, 0): [7.25, 9.75], (7, 1): [6.75, 4.75], 8: 0, (8, 'exit'): 6, 9: 0, (9, 'exit'): 8, 10: 2, (10, 'exit'): 4, (10, 0): [8.25, 6.75], (10, 1): [0.25, 5.25], 11: 0, (11, 'exit'): 6, 12: 0, (12, 'exit'): 6, 13: 2, (13, 'exit'): 12, (13, 0): [4.25, 5.25], (13, 1): [3.75, 9.25], 14: 0, (14, 'exit'): 8, 15: 3, (15, 'exit'): 3, (15, 0): [6.25, 0.25], (15, 1): [6.75, 0.75], (15, 2): [0.75, 3.25], 16: 1, (16, 'exit'): 8, (16, 0): [6.75, 1.75], 17: 1, (17, 'exit'): 4, (17, 0): [1.25, 2.75], 18: 1, (18, 'exit'): 2, (18, 0): [1.75, 9.25], 19: 3, (19, 'exit'): 7, (19, 0): [3.25, 7.75], (19, 1): [3.75, 6.75], (19, 2): [3.75, 4.25], 20: 1, (20, 'exit'): 5, (20, 0): [5.25, 0.75], 21: 1, (21, 'exit'): 3, (21, 0): [3.25, 5.75], 22: 0, (22, 'exit'): 7, 23: 1, (23, 'exit'): 2, (23, 0): [7.25, 2.25], 24: 3, (24, 'exit'): 2, (24, 0): [5.75, 6.75], (24, 1): [7.25, 0.75], (24, 2): [4.75, 6.25], 25: 1, (25, 'exit'): 6, (25, 0): [2.25, 6.75], 26: 0, (26, 'exit'): 7, 27: 2, (27, 'exit'): 4, (27, 0): [7.75, 3.75], (27, 1): [0.25, 0.75], 28: 0, (28, 'exit'): 4, 29: 1, (29, 'exit'): 6, (29, 0): [0.75, 1.25], 30: 2, (30, 'exit'): 1, (30, 0): [7.25, 1.75], (30, 1): [2.25, 1.25], 31: 0, (31, 'exit'): 2, 32: 2, (32, 'exit'): 5, (32, 0): [1.25, 3.25], (32, 1): [1.75, 1.25], 33: 0, (33, 'exit'): 2, 34: 0, (34, 'exit'): 4, 35: 2, (35, 'exit'): 3, (35, 0): [7.75, 2.75], (35, 1): [1.75, 8.25], 36: 0, (36, 'exit'): 1, 37: 3, (37, 'exit'): 1, (37, 0): [9.25, 0.25], (37, 1): [5.75, 1.25], (37, 2): [1.75, 5.25], 38: 1, (38, 'exit'): 4, (38, 0): [4.75, 7.75], 39: 2, (39, 'exit'): 5, (39, 0): [0.75, 5.75], (39, 1): [8.25, 2.75], 40: 1, (40, 'exit'): 2, (40, 0): [2.75, 4.75], 41: 1, (41, 'exit'): 7, (41, 0): [5.75, 7.75], 42: 1, (42, 'exit'): 5, (42, 0): [9.75, 1.75], 43: 1, (43, 'exit'): 3, (43, 0): [0.75, 6.25], 44: 1, (44, 'exit'): 3, (44, 0): [3.75, 4.75], 45: 2, (45, 'exit'): 3, (45, 0): [7.25, 8.25], (45, 1): [1.25, 9.75], 46: 2, (46, 'exit'): 2, (46, 0): [2.75, 3.75], (46, 1): [6.75, 0.25], 47: 0, (47, 'exit'): 3, 48: 0, (48, 'exit'): 6, 49: 1, (49, 'exit'): 1, (49, 0): [2.75, 5.75], 50: 2, (50, 'exit'): 7, (50, 0): [5.25, 4.25], (50, 1): [9.75, 4.75], 51: 0, (51, 'exit'): 1, 52: 1, (52, 'exit'): 2, (52, 0): [0.25, 8.25], 53: 1, (53, 'exit'): 4, (53, 0): [1.25, 6.25], 54: 1, (54, 'exit'): 5, (54, 0): [9.75, 1.25], 55: 0, (55, 'exit'): 2, 56: 0, (56, 'exit'): 1, 57: 0, (57, 'exit'): 1, 58: 1, (58, 'exit'): 1, (58, 0): [4.75, 7.75], 59: 0, (59, 'exit'): 10, 60: 2, (60, 'exit'): 3, (60, 0): [4.25, 2.75], (60, 1): [0.25, 5.25], 61: 0, (61, 'exit'): 4, 62: 1, (62, 'exit'): 6, (62, 0): [3.25, 8.75], 63: 1, (63, 'exit'): 5, (63, 0): [9.75, 4.75], 64: 0, (64, 'exit'): 4, 65: 1, (65, 'exit'): 6, (65, 0): [1.25, 4.25], 66: 2, (66, 'exit'): 3, (66, 0): [4.75, 1.75], (66, 1): [1.75, 5.75], 67: 3, (67, 'exit'): 2, (67, 0): [5.25, 8.75], (67, 1): [7.25, 2.25], (67, 2): [9.25, 8.25], 68: 1, (68, 'exit'): 10, (68, 0): [6.75, 8.75], 69: 1, (69, 'exit'): 7, (69, 0): [0.25, 9.25], 70: 1, (70, 'exit'): 3, (70, 0): [7.25, 7.25], 71: 2, (71, 'exit'): 4, (71, 0): [0.25, 8.75], (71, 1): [8.25, 3.25], 72: 0, (72, 'exit'): 7, 73: 1, (73, 'exit'): 1, (73, 0): [1.25, 2.25], 74: 4, (74, 'exit'): 4, (74, 0): [1.75, 6.75], (74, 1): [1.75, 9.25], (74, 2): [0.25, 2.75], (74, 3): [6.25, 1.75], 75: 4, (75, 'exit'): 6, (75, 0): [3.75, 0.75], (75, 1): [6.75, 1.75], (75, 2): [0.75, 9.75], (75, 3): [1.25, 8.75], 76: 1, (76, 'exit'): 4, (76, 0): [6.25, 0.75], 77: 1, (77, 'exit'): 3, (77, 0): [1.75, 8.75], 78: 2, (78, 'exit'): 5, (78, 0): [0.25, 5.25], (78, 1): [6.75, 6.75], 79: 0, (79, 'exit'): 3, 80: 2, (80, 'exit'): 5, (80, 0): [8.75, 0.25], (80, 1): [1.75, 2.75], 81: 1, (81, 'exit'): 6, (81, 0): [3.25, 2.25], 82: 1, (82, 'exit'): 1, (82, 0): [2.75, 8.25], 83: 0, (83, 'exit'): 3, 84: 1, (84, 'exit'): 4, (84, 0): [7.75, 0.25], 85: 3, (85, 'exit'): 4, (85, 0): [5.75, 9.75], (85, 1): [3.25, 9.25], (85, 2): [5.25, 0.25], 86: 2, (86, 'exit'): 1, (86, 0): [3.75, 3.75], (86, 1): [7.25, 1.25], 87: 1, (87, 'exit'): 3, (87, 0): [2.75, 7.75], 88: 0, (88, 'exit'): 5, 89: 0, (89, 'exit'): 2, 90: 2, (90, 'exit'): 3, (90, 0): [5.25, 2.75], (90, 1): [8.25, 5.75], 91: 1, (91, 'exit'): 4, (91, 0): [6.25, 3.25], 92: 0, (92, 'exit'): 4, 93: 0, (93, 'exit'): 3, 94: 0, (94, 'exit'): 5, 95: 0, (95, 'exit'): 7, 96: 0, (96, 'exit'): 6, 97: 0, (97, 'exit'): 2, 98: 0, (98, 'exit'): 3, 99: 2, (99, 'exit'): 4, (99, 0): [8.75, 9.25], (99, 1): [0.25, 6.25], 100: 1, (100, 'exit'): 4, (100, 0): [1.75, 7.25], 101: 1, (101, 'exit'): 4, (101, 0): [7.25, 2.25], 102: 1, (102, 'exit'): 3, (102, 0): [5.75, 6.75], 103: 3, (103, 'exit'): 5, (103, 0): [3.75, 2.25], (103, 1): [7.75, 8.75], (103, 2): [2.75, 1.25], 104: 0, (104, 'exit'): 4, 105: 1, (105, 'exit'): 3, (105, 0): [5.75, 5.25], 106: 2, (106, 'exit'): 1, (106, 0): [7.75, 7.25], (106, 1): [5.25, 8.25], 107: 2, (107, 'exit'): 3, (107, 0): [0.75, 2.75], (107, 1): [2.75, 0.75], 108: 1, (108, 'exit'): 6, (108, 0): [2.75, 0.25], 109: 2, (109, 'exit'): 3, (109, 0): [0.25, 9.25], (109, 1): [9.25, 6.25], 110: 2, (110, 'exit'): 4, (110, 0): [1.75, 4.75], (110, 1): [9.75, 1.25], 111: 1, (111, 'exit'): 8, (111, 0): [3.25, 5.25], 112: 2, (112, 'exit'): 4, (112, 0): [7.25, 0.75], (112, 1): [6.75, 7.75], 113: 1, (113, 'exit'): 1, (113, 0): [2.25, 6.75], 114: 0, (114, 'exit'): 3, 115: 0, (115, 'exit'): 5, 116: 1, (116, 'exit'): 6, (116, 0): [6.75, 7.75], 117: 1, (117, 'exit'): 4, (117, 0): [7.75, 4.25], 118: 0, (118, 'exit'): 4, 119: 1, (119, 'exit'): 1, (119, 0): [8.75, 7.25], 120: 3, (120, 'exit'): 3, (120, 0): [7.75, 5.75], (120, 1): [4.25, 1.25], (120, 2): [2.25, 1.75], 121: 0, (121, 'exit'): 3, 122: 1, (122, 'exit'): 5, (122, 0): [1.75, 8.75], 123: 4, (123, 'exit'): 0, (123, 0): [1.75, 9.25], (123, 1): [5.25, 1.75], (123, 2): [8.25, 9.25], (123, 3): [2.75, 2.25], 124: 2, (124, 'exit'): 2, (124, 0): [4.25, 6.75], (124, 1): [4.75, 1.75], 125: 0, (125, 'exit'): 5, 126: 1, (126, 'exit'): 3, (126, 0): [7.25, 0.75], 127: 1, (127, 'exit'): 2, (127, 0): [2.75, 4.75], 128: 0, (128, 'exit'): 7, 129: 2, (129, 'exit'): 5, (129, 0): [2.25, 2.75], (129, 1): [4.75, 5.75], 130: 2, (130, 'exit'): 5, (130, 0): [9.75, 8.75], (130, 1): [6.25, 4.75], 131: 2, (131, 'exit'): 0, (131, 0): [5.75, 3.75], (131, 1): [8.75, 2.75], 132: 3, (132, 'exit'): 2, (132, 0): [9.25, 3.75], (132, 1): [9.75, 6.25], (132, 2): [9.75, 0.75], 133: 1, (133, 'exit'): 2, (133, 0): [7.75, 0.25], 134: 0, (134, 'exit'): 3, 135: 0, (135, 'exit'): 6, 136: 2, (136, 'exit'): 7, (136, 0): [4.25, 2.25], (136, 1): [0.25, 9.75], 137: 0, (137, 'exit'): 4, 138: 1, (138, 'exit'): 6, (138, 0): [1.25, 0.25], 139: 0, (139, 'exit'): 5, 140: 0, (140, 'exit'): 3, 141: 2, (141, 'exit'): 8, (141, 0): [1.75, 4.25], (141, 1): [4.75, 0.75], 142: 0, (142, 'exit'): 4, 143: 0, (143, 'exit'): 2, 144: 2, (144, 'exit'): 1, (144, 0): [5.75, 8.75], (144, 1): [6.75, 6.25], 145: 3, (145, 'exit'): 5, (145, 0): [0.25, 0.75], (145, 1): [8.25, 5.75], (145, 2): [3.75, 8.75], 146: 1, (146, 'exit'): 6, (146, 0): [2.75, 2.25], 147: 2, (147, 'exit'): 1, (147, 0): [1.75, 2.25], (147, 1): [4.75, 9.25], 148: 1, (148, 'exit'): 5, (148, 0): [5.75, 9.75], 149: 2, (149, 'exit'): 5, (149, 0): [3.25, 0.25], (149, 1): [3.75, 7.25], 150: 2, (150, 'exit'): 3, (150, 0): [0.25, 6.75], (150, 1): [4.25, 9.75], 151: 1, (151, 'exit'): 2, (151, 0): [8.75, 2.75], 152: 1, (152, 'exit'): 6, (152, 0): [1.25, 5.25], 153: 1, (153, 'exit'): 5, (153, 0): [8.75, 7.75], 154: 1, (154, 'exit'): 3, (154, 0): [0.75, 1.75], 155: 1, (155, 'exit'): 4, (155, 0): [3.75, 4.25], 156: 2, (156, 'exit'): 3, (156, 0): [2.25, 4.25], (156, 1): [0.75, 9.25], 157: 0, (157, 'exit'): 4, 158: 0, (158, 'exit'): 4, 159: 0, (159, 'exit'): 2, 160: 3, (160, 'exit'): 3, (160, 0): [6.75, 8.25], (160, 1): [4.75, 3.75], (160, 2): [3.75, 5.25], 161: 1, (161, 'exit'): 2, (161, 0): [4.25, 8.25], 162: 1, (162, 'exit'): 4, (162, 0): [6.75, 2.75], 163: 2, (163, 'exit'): 1, (163, 0): [0.25, 3.75], (163, 1): [7.75, 9.25], 164: 1, (164, 'exit'): 3, (164, 0): [2.25, 4.25], 165: 0, (165, 'exit'): 5, 166: 1, (166, 'exit'): 0, (166, 0): [0.25, 5.75], 167: 0, (167, 'exit'): 8, 168: 1, (168, 'exit'): 3, (168, 0): [5.75, 5.75], 169: 1, (169, 'exit'): 3, (169, 0): [6.25, 6.25], 170: 0, (170, 'exit'): 2, 171: 0, (171, 'exit'): 3, 172: 2, (172, 'exit'): 3, (172, 0): [3.25, 5.75], (172, 1): [7.75, 9.25], 173: 1, (173, 'exit'): 2, (173, 0): [1.25, 9.25], 174: 0, (174, 'exit'): 5, 175: 1, (175, 'exit'): 2, (175, 0): [0.25, 8.25], 176: 3, (176, 'exit'): 7, (176, 0): [6.75, 4.25], (176, 1): [9.25, 7.25], (176, 2): [2.25, 9.75], 177: 0, (177, 'exit'): 4, 178: 0, (178, 'exit'): 4, 179: 0, (179, 'exit'): 7, 180: 1, (180, 'exit'): 3, (180, 0): [8.25, 1.75], 181: 3, (181, 'exit'): 4, (181, 0): [3.75, 6.25], (181, 1): [3.75, 0.75], (181, 2): [7.75, 7.25], 182: 1, (182, 'exit'): 11, (182, 0): [1.25, 7.25], 183: 0, (183, 'exit'): 5, 184: 0, (184, 'exit'): 3, 185: 1, (185, 'exit'): 3, (185, 0): [8.25, 2.25], 186: 0, (186, 'exit'): 5, 187: 1, (187, 'exit'): 5, (187, 0): [4.75, 9.25], 188: 1, (188, 'exit'): 3, (188, 0): [7.25, 0.25], 189: 0, (189, 'exit'): 3, 190: 0, (190, 'exit'): 4, 191: 2, (191, 'exit'): 6, (191, 0): [1.25, 8.25], (191, 1): [6.25, 1.75]}

        # d[n], m[n] = d[1], m[1]

        #d[n], m[n] = d, m
        #if n > 1: 
        #    d[n], m[n] = d[1], m[1]

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
        
        L = 0.8; #driver utilization target
        
        Q = 1 #service level target

        w_d = 0.5; w_s = 1-w_d #weights 

        for p in range(16):

            alpha[p, pool_sizes[p]] += 1

            if alpha[p, pool_sizes[p]] == 0:
                alpha_step_size = 1
            else: 
                alpha_step_size = 1/(alpha[(p, pool_sizes[p])])**0.5 
                
            Qp = sum(num_orders_fulfilled[p:])/sum(num_orders_placed[p:])
            Lp = sum(num_drivers_meeting_guarantee[p:])/sum(num_drivers_enter[p:])
            
            V_new = sum(pool_sizes[p+1:]) + w_d*(n+1)*max(Q-Qp, 0) + w_s*(n+1)*max(L-Lp, 0) 

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
 
# if __name__ == '__main__':
def run(penalty, cfa=False):
    # Step 1: read data + initiate values
    # for penalty in [10, 50, 100, 250]:  # 500, 1000
    loadname = 'data/revised_demand_data_wage_guarantee_test_inst_T=192_1_mu2.npz'
    #loadname = 'data/wage_guarantee_test_inst_T=192_1_mu2.npz'
    data_set = np.load(loadname, allow_pickle=True)
    aggregate = True
    # changed to mu=2 06/28/21
    data = {'T': 192, 't_interval': 5, 'mu_enter': 2, 'mu_exit': 4,  # T was saved as 10 in file
            'lambd': 10, 'delta_t': data_set['delta_t'].tolist(), 'N': 1000, 'grid_size': (10, 10),
            'eta': 0.8, 'W': 0.8, 'theta': (1 / (2 * math.sqrt(200))), 'theta1': 1,
            'gamma': 0.9, 'Guaranteed_service': 120,
            'alpha_penalty': 1.0, 'lambda_initial': 10, 'h_int': 5, 'g_int': 5}
    
    d = {}

    for i in range(data['N']):  
        d[i] = data_set['d'].tolist() #same demand data for consistent comparison 

    v_t = {0: {}, 'count': {}, 'var': {}, 'last_met': {}}
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
