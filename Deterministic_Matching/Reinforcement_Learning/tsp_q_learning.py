import numpy as np
import math
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp 

global scaling_factor; scaling_factor = 1000000 #scaling_factor for OR tools to deal with it only being able to handle integers 
 
#The compute_euclidean_distance_matrix and print_solution functions are used in solve_tsp_with_ortools function 
 
def compute_euclidean_distance_matrix(locations):
    global scaling_factor; 
    """Creates callback to return distance between points."""
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                # Euclidean distance
                distances[from_counter][to_counter] = (scaling_factor*(
                    math.hypot((from_node[0] - to_node[0]),
                               (from_node[1] - to_node[1])))) 
    return distances
 
def print_solution(manager, routing, solution, print_sol = False):
    """Prints solution on console."""
    global scaling_factor
    
    cost = solution.ObjectiveValue()/scaling_factor
    
    #print('Objective: {}'.format(cost))
    index = routing.Start(0)
    plan_output = 'Route:\n'
    route_distance = 0
    array_best_path = []
    
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        array_best_path.append(previous_index)
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    
    if print_sol: 
        plan_output += ' {}\n'.format(manager.IndexToNode(index))
        print(plan_output)
        plan_output += 'Objective: {}m\n'.format(route_distance)
        #print('Best path array: ', array_best_path)
    
    return array_best_path, cost

def solve_tsp_with_ortools(locations):
    global scaling_factor; 
    """Entry point of the program."""
    # Instantiate the data dictionary. 
    data = {'locations': locations, 'num_vehicles': 1, 'depot': 0}

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['locations']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    distance_matrix = compute_euclidean_distance_matrix(data['locations'])

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        array_best_path, cost = print_solution(manager, routing, solution)

        return array_best_path, cost

#two helper functions to solve with TD learning

def euclidean_dist(x,y):
    #x and y are indices in the array locations; this finds the Euclidean distance between the corresponding locations   
    return ((locations[x][0]-locations[y][0])**2+ (locations[x][1]-locations[y][1])**2)**0.5

def path_cost(path):
    #if path is of the form [0, 1, 2] then cost is euclidean_dist(0,1) +euclidean_dist(1,2) +euclidean_dist(2,0) 
    cost = 0 
    for i in range(1, len(path)): 
        cost+=euclidean_dist(path[i], path[i-1])
    cost+=euclidean_dist(path[-1], path[0])
    return cost 

def solve_tsp_with_td_learning(locations): 
    
    #define training parameters
    epsilon = 1 #the percentage of time when we should take a random action 
    discount_factor = 1 #discount factor for future rewards
    learning_rate = 0.1 #the rate at which the AI agent should learn
      
    q_values = np.zeros((len(locations), len(locations)))
 
    depot = 0 
    
    n_training_episodes = 1000 
    
    for episode in range(n_training_episodes): 
        current_state = depot  #starting point at any episode
        
        #A list of the next states that can be visited; depot is 0 so is excluded 
        possible_next_states = [i for i in range(1, len(locations))] #next state to visit; depot must be 0 for this to work  
        
        
        #Add the best feasible next state found by epsilon greedy method at each iteration of the while loop; 
        
        while possible_next_states:  
            
            if np.random.random() < epsilon:
                next_state = np.random.choice(possible_next_states)  #pick random action with probability epsilon  
            else:  
                #otherwise pick action (next state to visit) among those feasible that has maximum reward 
                next_state = possible_next_states[np.argmax(q_values[current_state, possible_next_states])] 
                 
            possible_next_states.remove(next_state) #Remove the state visited as it cannot be visited again 
                 
            reward = -euclidean_dist(current_state, next_state) #aim is to maximize reward so it's negative  
               
            #Update Q value using TD update formula  
            
            q_values[current_state, next_state] += learning_rate*(reward + discount_factor * max(q_values[next_state]) - q_values[current_state, next_state])
            
            #q_values[current_state, next_state] += learning_rate*(reward + discount_factor * max(q_values[next_state], key=lambda x: x if x != 0 else float('-inf')) - q_values[current_state, next_state])
             
            #max(q_values[action, possible_next_states]) instead max(q_values[action]) generates unstable results 
            
            current_state = next_state
         
        q_values[current_state, depot] += learning_rate*(reward -  q_values[current_state, depot]) #terminal state update
            
        #gradual decay of epsilon as the best action is not known in the beginning so it expores randomly 
        #but as there is more certainty the action can be chosen more greedily 
        
        epsilon = 1-episode/n_training_episodes 
        
    print('q_values: ', q_values)
     
    #Determine the optimal path 
    optimal_path = [depot]
    current_state = depot 
    
    possible_next_states = [i for i in range(1, len(locations))]
    
    #at each iteration add the next feasible state that maximizes reward to optimal_path
    #optimal path will be of the form [0, 1, 3, 2] meaning start at 0, then visit 1, then 3, then 2, then return to 0 
    
    while possible_next_states:  
        next_loc = possible_next_states[np.argmax(q_values[current_state, possible_next_states])] 
        current_state = next_loc 
        possible_next_states.remove(next_loc) 
        optimal_path.append(next_loc)
     
    cost = path_cost(optimal_path) #Determine the optimal path's cost 
    
    #print('optimal_path with TD learning: ', optimal_path); print('cost with TD learning: ', cost) 
     
    return optimal_path, cost  
  
if __name__ == '__main__':
    
    # Set the print options to automatically round to 4 decimal places
    np.set_printoptions(precision=4, suppress=True)
     
    # Generate a range of instances and compare the approximate solution with the OR tools solution
    num_instances = 10 
    total_accuracy = 0

    for i in range(num_instances):
        # Generate a random instance with random locations
        num_cities = np.random.randint(15, 25) #random number of locations 
        
        #each location's x and y coordinates are random variables between 0 and 10  
        locations = [(np.random.uniform(0, 10), np.random.uniform(0, 10)) for _ in range(num_cities)]
         
        print(f"\n\nInstance {i+1}:")
        print('locations: ', locations) 
        
        # Solve TSP using OR-Tools 
        or_tools_path, or_tools_path_cost = solve_tsp_with_ortools(locations)
          
        # Solve TSP using TD learning 
        td_learning_path, td_learning_path_cost = solve_tsp_with_td_learning(locations)

        # Calculate the accuracy of the TD learning solution 
        if td_learning_path_cost > or_tools_path_cost: 
            accuracy = (1 - (abs(td_learning_path_cost - or_tools_path_cost) / or_tools_path_cost)) * 100 #percentage 
        else:
            accuracy = 100 #percentage 
        
        #add each percent accuracy to total_accuracy 
        total_accuracy += accuracy
        
        # Print the results for each instance 
        print("Tour found with TD learning:", td_learning_path)
        print("TD learning Tour Cost:", td_learning_path_cost)
        print("Tour found with OR Tools:", or_tools_path)
        print("OR Tools Tour Cost:", or_tools_path_cost)
        print("Accuracy:", accuracy) 

    # Print the average accuracy across all instances
    average_accuracy = total_accuracy / num_instances
    print("\nAverage Percent Accuracy:", average_accuracy)
