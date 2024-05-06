import numpy as np
import Test_VFA_collect_stats as stats
import Test_Myopic_collect_stats as stats_Myopic
import Test_VFA_collect_stats_convergence as stats_con

# Part 1: run test instance for multiple policies
# T72, T144, agg, dis, high penalty, reg
# Test VFA algorithms

'''
FileNames = ['data/sol_n1000_VFA_T=192_disagg_mu2_theta1_no_PCF.npz',
             'data/sol_n1000_VFA_T=192_aggregation_mu2_theta1_no_PCF.npz',
             'data/sol_n1000_VFA_T=192_aggregation_penalty10_mu2_theta1_no_PCF.npz',
             'data/sol_n1000_VFA_T=192_aggregation_penalty50_mu2_theta1_no_PCF.npz',
             'data/sol_n1000_VFA_T=192_aggregation_penalty100_mu2_theta1_no_PCF.npz',
             'data/sol_n1000_VFA_T=192_aggregation_penalty250_mu2_theta1_no_PCF.npz',
             'data/sol_n1000_VFA_T=192_aggregation_cfa_penalty250_mu2_theta1_no_PCF.npz']


# 1. Utilization Policy
penalty = {0: 1, 1: 1, 2: 10, 3: 50, 4: 100, 5: 250, 6: 250}  # dictionary maps file number to penalty
T = 168  # 192-24
mu_val = [2]
# run VFA for T=72
inst_string = 'utilization'
for mu in mu_val:
    for f in range(6, 7):  # range(1, len(FileNames)):
        aggregate = False if f < 1 else True
        val_range = range(1, 11)  # range(6, 11) if f == 1 else range(1, 11)
        for inst in val_range:
            stats.run(FileNames[f], inst_string, aggregate, inst, penalty[f], T, mu, True)
'''

FileNames = ['data/sol_n1000_VFA_T=192_aggregation_cfa_penalty250_mu2_theta1_no_PCF_W80_tau120_convergence.npz']
inst_string, inst, penalty, T, mu, cfa = 'util', 1, 250, 168, 2, True
for aggregate in [True, False]:
    stats_con.run(FileNames[0], aggregate, inst, penalty, T, mu, cfa)
# run VFA for T=144
# for f in range(len(FileNames_2)):
#     aggregate = False if f % 2 == 0 else True
#     penalty = 1 if f < 2 else 1000
#     for inst in range(1, 11):
#         stats.run(FileNames_2[f], aggregate, inst, penalty, 144)

# Myopic policies: T72, T144, high penalty, reg + T72, T144 base case
# for inst in range(1, 11):
#     for T in [72, 144]:
#         for penalty in [1000, 1]:
#             stats_Myopic.run(inst, False, penalty, T)
#         stats_Myopic.run(inst, True, 1, T)  # run base case (penalty is irrelevant)

# Myopic policies: # CFA, myopic, base case
# for mu in mu_val:
#     for inst in range(1, 11):
#         stats_Myopic.run(inst, False, False, 1, T, mu)  # myopic
#         stats_Myopic.run(inst, False, True, 1, T, mu)  # Cost function approximation. cfa = true
#         stats_Myopic.run(inst, True, False, 1, T, mu)  # run base case (penalty is irrelevant)