import gurobipy as gp
from gurobipy import GRB


def dist(vec1, vec2):
    #Euclidean distance between vec1 and vec2 
    return ((vec1[0]-vec2[0])**2 + (vec1[1]-vec2[1])**2)**0.5 


def c(a, b): 
    #cost for driver a to deliver order b; proportional to Euclidean distance from driver to order origin plus Euclidean distance from order origin to order destination  
    #a is an index and not an attribute vector, as is b
    return dist([driverdata[a][2], driverdata[a][3]], [orderlocation[b][0], orderlocation[b][1]]) + dist([orderlocation[b][2], orderlocation[b][3]], [orderlocation[b][0], orderlocation[b][1]])


if __name__ == '__main__':

    driverdata = [[1,1,1, 1], [1, 1, 2, 1.5], [1, 1, 3, 4]] #[sa, ma, oax, oay] where oa = xy coordinates of driver location; sa is a binary variable that is 1 when the driver is active; ma is a binary variable that is 1 when the driver is available  
    orderlocation = [None, [0, 1.5, 5, 2], [0, 3.5, 4, 4]] #xy coordinates of order origin followed by xy coordinates of order destination
    orderrevenue = [None, 10, 15] #this is q(b) of the objective function q(b) - c(a, b)
    #first entry is None to account for the case of a driver not being assigned to any order 
    
    numdrivers = len(driverdata)
    numorders = len(orderlocation)

    m = gp.Model('driverallocation')

    assign = m.addVars([(i,j) for i in range(numdrivers) for j in range(numorders)], name = 'assign', vtype = GRB.BINARY) 
    #assign[i,j] means driver i (0-based index) is assigned to order j (1-based index; if j = 0 it indicates not being assigned to an order)

    m.addConstrs(gp.quicksum(assign[i,j] for j in range(numorders)) == 1 for i in range(numdrivers)) 
    #constraint (4b); since each driver has a unique index, Rta is 1 
    #each driver is assigned at most once (since a is an index not a vector Ra = 1) or not assigned at all 

    m.addConstrs(gp.quicksum(assign[i,j] for i in range(numdrivers)) <= 1 for j in range(1, numorders)) 
    #constraint (4c); here j ranges from 1 to numorders because the possibility of unassigned orders is not relevant 
    #each order (except None, which represents not being assigned) is fulfilled at most once 

    m.addConstrs(assign[i,j] <= min(driverdata[i][0], driverdata[i][1]) for j in range(1, numorders) for i in range(numdrivers)) 
    #the driver must be both active and available to be assigned; if the driver is inactive or unavailable assign[i,j] = 0  
    #To avoid the 'Infeasible model' error j must range from 1 to numorders not 0, otherwise assign[i,0] can be forced to be 0 if the driver is unavailable/inactive
     
    m.setObjective(gp.quicksum((orderrevenue[j] - c(i, j))*assign[i,j] for i in range(numdrivers) for j in range(1, numorders)), GRB.MAXIMIZE)  
    #j ranges from 1 to numorders because if j = 0 no revenue is earned as a driver was not assigned     
        
    m.optimize() 

