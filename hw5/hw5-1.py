import gurobipy as gp
from gurobipy import GRB

owls = ['o1', 'o2', 'o3', 'o4', 'o5']
wands = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7']
  
m = gp.Model('ResourceAllocation')
 
x = m.addVars([('s', w) for w in wands] + [(w, o) for o in owls for w in wands] + [(o,'t') for o in owls], name="assign", vtype = GRB.INTEGER) #decision variable

capacity = {'o1':6, 'o2':4, 'o3':5, 'o4':4, 'o5':3}

wanddemand = 3 #each type of wand is to be delivered wanddemand times 

m.addConstrs(x['s', w] <= wanddemand for w in wands) #each wand is delivered at most wanddemand times

m.addConstrs(x[o, 't'] <= capacity[o] for o in owls) #each wand is delivered at most wanddemand times

m.addConstrs(x[w, o] <= 1 for w in wands for o in owls) #each wand is delivered at most once by a particular owl 

m.addConstrs(gp.quicksum(x[w, o] for o in owls) == x['s', w] for w in wands) #flow from each of the wands to an owl equals flow from that owl to the target

m.addConstrs(gp.quicksum(x[w, o] for w in wands) == x[o, 't'] for o in owls) #flow from each of the wands to an owl equals flow from that owl to the target

m.setObjective(gp.quicksum(x['s', w] for w in wands), GRB.MAXIMIZE) #aim is to maximize number of wands delivered subject to the constraints  

m.optimize() 
