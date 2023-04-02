import pandas as pd 
import gurobipy as gp
from gurobipy import GRB

switchingcosts =  pd.read_csv('switchingcosts.txt', header=None).to_numpy().flatten().tolist() 

assignmentcosts = pd.read_csv('assignmentcosts.txt', header=None).to_numpy().flatten().tolist() 
 
capacity = pd.read_csv('capacity.txt', header=None).to_numpy().flatten().tolist() 

flighthours = pd.read_csv('flighthours.txt', header=None).to_numpy().flatten().tolist() 

demand = pd.read_csv('demand.txt', header=None).to_numpy().flatten().tolist() 

def h(i,j,k): #cost of switching i from j to k; i, j and k are 0-based indices 
    return switchingcosts[i*49+j*7+k]

def c(i,j): #cost of assigning i to j 
    return assignmentcosts[i*7+j]

def b(i,j): #Capacity I can carry when doing J 
    return capacity[i*7+j]

def a(i,j): #Flight hours necessary for I to do J 
    return flighthours[i*7+j]

def d(j,s): #demand of j in scenario s 
    return demand[j*20+s]

f = [7231, 6813, 8454, 5956] #capacity of each flight

p = [9.63468793555311, 40.8086824786238, 80.2491638766406, 46.9847514199262, 79.693199749936, 61.5025605680528, 13.1649291954539] #per unit contractor supply cost

littlem = 5; BigM = 10 #littlem = min. no of tons contractor must deliver; BigM = max number of flights that can be switched

m = gp.Model('jobscheduling')

numflight = 4; numloc = 7; numscen = 20

pr = 1/numscen #each scenario has equal probability of occurring 

x = m.addVars([(i,j) for i in range(numflight) for j in range(numloc)], name = 'x', vtype = GRB.INTEGER) #number of type i flights assigned to j

y = m.addVars([(i,j,k,s) for i in range(numflight) for j in range(numloc) for k in range(numloc) for s in range(numscen) if j!=k], name = 'y', vtype = GRB.INTEGER) 
 
t = m.addVars([(j,s) for j in range(numloc) for s in range(numscen)], name = 't', vtype = GRB.BINARY) 

z = m.addVars([(j,s) for j in range(numloc) for s in range(numscen)], name = 'z', vtype = GRB.CONTINUOUS)  
 
m.addConstrs(gp.quicksum(a(i,j)*x[i,j] for j in range(numloc)) <= f[i] for i in range(numflight)) 

m.addConstrs(gp.quicksum(b(i,j)*x[i,j] for i in range(numflight)) + gp.quicksum(b(i,j)*(y[i,k,j,s]-y[i,j,k,s]) for i in range(numflight) for k in range(numloc) if k!=j) + z[j,s] >= d(j,s) for j in range(numloc) for s in range(numscen))  
 
m.addConstrs(gp.quicksum(y[i,j,k,s] for k in range(numloc) if k!=j) <= x[i,j] for i in range(numflight) for j in range(numloc) for s in range(numscen))

m.addConstrs(gp.quicksum(a(i,j)*x[i,j] for j in range(numloc)) +  gp.quicksum(a(i,j)*(y[i,k,j,s]-y[i,j,k,s]) for j in range(numloc) for k in range(numloc) if k!=j) <= f[i] for i in range(numflight) for s in range(numscen))   
                                      
m.addConstrs(z[j,s] <= t[j,s]*d(j,s) for j in range(numloc) for s in range(numscen))

m.addConstrs(z[j,s] >= littlem*t[j,s] for j in range(numloc) for s in range(numscen))

m.addConstrs(gp.quicksum(y[i,k,j,s] for i in range(numflight) for j in range(numloc) for k in range(numloc) if k!=j) <= BigM for s in range(numscen))

m.setObjective(gp.quicksum(c(i,j)*x[i,j] for i in range(numflight) for j in range(numloc)) + pr*gp.quicksum(h(i,j,k)*y[i,j,k,s] for i in range(numflight) for j in range(numloc) for k in range(numloc) for s in range(numscen) if k!=j)+pr*gp.quicksum(p[j]*z[j,s] for j in range(numloc) for s in range(numscen)))     
            
m.optimize()
