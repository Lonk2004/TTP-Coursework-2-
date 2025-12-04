# TODO: TEST HYPERPARAMETERS FOR EVOLUTIONARY ALGORITHM

import os
import random
import math

import numpy as np
import matplotlib.pyplot as plt 

# File used for testing
# Should be dynamic based on user input
fname = "src/resources/a280-n1395.txt" 

def open_file(fname):
    """Reads the knapsack problem data from a file"""
    
    # Check if file exists
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Input file '{fname}' not found.")

    cities = []

    # Read file contents
    with open(fname, "r") as file:
        lines = file.readlines()
        
    # Find the start and end indices for the city coordinates
    start_index = None
    end_index = None
    for i, line in enumerate(lines):
        if "NODE_COORD_SECTION" in line:
            start_index = i + 1
        if "ITEMS SECTION" in line:
            end_index = i
            break

    # Parse city coordinates
    if start_index is not None and end_index is not None:
        for line in lines[start_index:end_index]:
            parts = line.split()
            city = int(parts[0]) - 1
            x = int(parts[1])
            y = int(parts[2])
            cities.append((city, x, y))

    print(cities[0])
            
    return cities

def initialise(cities, population_size=10):
    """Initialises a population for the evolutionary algorithm"""
    chromosomes = []
    # Create initial chromosome excluding the starting city (index 0)
    initial = list(range(1, len(cities)))
    for _ in range(population_size):
        # Generate a random permutation of cities
        chromosome = random.sample(initial, len(initial))
        chromosomes.append(chromosome)
        
    return chromosomes

def fitness(chromosome, cities, velocity=None):
    """Calculates the fitness of a chromosome based on total time taken to complete the tour"""
    
    """I believe that the ant colony optimisation for the knapsack problem should be implemented here
    Based on the chromosome given, get the best packing of items and the resulting velocity to use for time taken"""
    
    total_time = 0
    # Add initial city to start and end
    chromosome = [0] + chromosome + [0]
    
    # For testing purposes, assume constant velocity if none provided
    if velocity is None:
        velocity = [1] * (len(chromosome) - 1)
        
    # Calculate total time taken for the tour
    for i in range(len(chromosome) - 1):
        # Euclidean distance between cities
        city1 = cities[chromosome[i]]
        city2 = cities[chromosome[i + 1]]
        dist = ((city1[1] - city2[1]) ** 2 + (city1[2] - city2[2]) ** 2) ** 0.5
        dist = math.ceil(dist)
        time = dist / velocity[i]
        total_time += time
    return total_time

def selection(t_size, chromosomes, cities):
    """Selects two parents from the population using tournament selection"""
    def tournament(t_size, chromosomes, cities):
        """Tournament selection helper function"""
        indices = []
        for _ in range(t_size):
            indices.append(random.randint(0, len(chromosomes) - 1))
        # Select the chromosome with the best fitness (lowest time)
        parent = min(indices, key=lambda chromosome: fitness(chromosomes[chromosome], cities))
        return parent
    
    parents = []
    # This ensures that the same parent is not selected twice
    chromosomes_selection = chromosomes
    for _ in range(2):
        parent = tournament(t_size, chromosomes_selection, cities)
        parents.append(chromosomes_selection[parent])
        chromosomes_selection = chromosomes_selection[:parent] + chromosomes_selection[parent + 1:]
        
    return parents

def crossover(parents):
    """Performs order crossover on two parents to produce two children"""
    def create_child(subseq, parent):
        """Creates a child chromosome by inserting a subsequence into a parent"""
        # If a city in subseq is already in parent, it should not be added again
        child = [city for city in parent if city not in subseq]
        # Add subsequence at the crossover point
        child = child[:a] + subseq + child[a:]
        return child
    
    # Select crossover points
    a = random.randint(0, len(parents[0]) - 1)
    b = random.randint(a, len(parents[0]) - 1)
    # Get subsequences
    subseq1 = parents[0][a:b]
    subseq2 = parents[1][a:b]
    # Create children
    child1 = create_child(subseq1, parents[1])
    child2 = create_child(subseq2, parents[0])
    return child1, child2

def mutation(chromosome, mutation_rate=0.1):
    """Performs swap mutation on a chromosome"""
    # Go through each gene in the chromosome
    for i in range(len(chromosome)):
        # With a probability of mutation_rate, swap this gene with another random gene
        if random.random() < mutation_rate:
            j = random.randint(0, len(chromosome) - 1)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome

def replacement(chromosomes, children, cities):
    """Replaces the worst chromosomes in the population with new children if they are better"""
    for child in children:
        # Find the chromosome with the worst fitness (longest distance)
        longest_dist = max(range(len(chromosomes)), key=lambda i: fitness(chromosomes[i], cities))
        if fitness(child, cities) < fitness(chromosomes[longest_dist], cities):
            chromosomes[longest_dist] = child
    return chromosomes

def evolutionary_algorithm(fname):
    """Main function to run the evolutionary algorithm"""
    cities = open_file(fname)
    chromosomes = initialise(cities)
    parents = selection(3, chromosomes, cities)
    child1, child2 = crossover(parents)

    # Run the evolutionary algorithm for a set number of generations
    # Number of generations can be adjusted as needed
    for i in range(10000):
        parents = selection(3, chromosomes, cities)
        child1, child2 = crossover(parents)
        mutated_child1 = mutation(child1)
        mutated_child2 = mutation(child2)
        chromosomes = replacement(chromosomes, [mutated_child1, mutated_child2], cities)

    best_chromosome = max(chromosomes, key=lambda chromosome: fitness(chromosome, cities))
    best_value = fitness(best_chromosome, cities)

    print(f"Best chromosome: {best_chromosome}")
    print(f"Best value: {best_value}")
    return best_chromosome

# def open_file_bags(fname):

#     bags = []
#     with open(fname, "r") as file:
#         lines = file.readlines()
        
#     # Find the start and end indices for the city coordinates
#     start_index = None
#     end_index = length(lines)
#     for i, line in enumerate(lines):
#         if "NODE_COORD_SECTION" in line:
#             start_index = i + 1
#         if "ITEMS SECTION" in line:
#             end_index = i
#             break

#     # Parse city coordinates
#     if start_index is not None and end_index is not None:
#         for line in lines[start_index:end_index]:
#             parts = line.split()
#             city = int(parts[0]) - 1
#             x = int(parts[1])
#             y = int(parts[2])
#             bags.append((city, x, y))
            
#     return cities   


##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################    





def open_file_bags(fname):
    """
    parse file to find bags capacity min/max velocity and rent
    """

    bags = []
    with open(fname, "r") as file:
        lines = file.readlines()
        
    # Find the start and end indices for the city coordinates
    start_index = None
    capacity = None
    end_index = len(lines)
    for i, line in enumerate(lines):

        if "CAPACITY OF KNAPSACK" in line:
            capacity = int(line.split(":", 1)[1].strip())
        if "MIN SPEED" in line:
            vmin = float(line.split(":", 1)[1].strip())
        if "MAX SPEED" in line:
            vmax = float(line.split(":", 1)[1].strip())
        if "RENTING RATIO" in line:
            rent = float(line.split(":", 1)[1].strip())
        if "ITEMS SECTION" in line:
            start_index = i + 1
            break

    # Parse city coordinates
    if start_index is not None and end_index is not None:
        for line in lines[start_index:end_index]:
            parts = line.split()
            i = int(parts[0]) - 1
            prof = int(parts[1])
            weight = int(parts[2])
            node = int(parts[3]) - 1
            bags.append((i, prof, weight, node))
            
    return bags, capacity, vmax, vmin, rent   

def find_bags(route, bags):
    """Finds the bags and links them to their location city"""
    possible_bags = {city: [] for city in route}

    for city in route:
       
        for bag in bags:
            #print(bag)
            
            if bag[3] == city:
                possible_bags[city].append(bag[0])

    return possible_bags

def pbag_split(pbags):
    """
    split the bags into their respective information components 
    """
    item = []
    prof = []
    weight = []
    city = []
    for bag in pbags:
        item.append(bag[0])
        prof.append(bag[1])
        weight.append(bag[2])
        city.append(bag[3])
    return item, prof, weight, city
                
def open_file2(fname):
    """Reads the knapsack problem data from a file"""
    
    # Check if file exists
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Input file '{fname}' not found.")

    cities = []

    # Read file contents
    with open(fname, "r") as file:
        lines = file.readlines()
        
    # Find the start and end indices for the city coordinates
    start_index = None
    end_index = None
    for i, line in enumerate(lines):
        if "NODE_COORD_SECTION" in line:
            start_index = i + 1
        if "ITEMS SECTION" in line:
            end_index = i
            break

    # Parse city coordinates
    if start_index is not None and end_index is not None:
        for line in lines[start_index:end_index]:
            parts = line.split()
            city = int(parts[0]) - 1
            x = int(parts[1])
            y = int(parts[2])
            cities.append((city, x, y))

    print(cities[0])
            
    return cities



def distances(cities, route):
    """
    cities: list of (city_index, x, y)
    route: list of city indices in visit order
    returns: list of distances for each leg in the route
    """
    dist = []
    for i in range(len(route) - 1):
        c1 = cities[route[i]]
        c2 = cities[route[i + 1]]
        x1, y1 = c1[1], c1[2]
        x2, y2 = c2[1], c2[2]
        d = math.hypot(x2 - x1, y2 - y1)
        dist.append(d)
    return dist


def initialize_pheromone(n_items):
    return np.ones(n_items, dtype=float)

def compute_heuristic(values, weights):
    """
     Heuristic used will be value density
       value density = value / weight + (small constant in case of 0 weight)
    """
    return values / (weights + 1e-10)



def time_compute(city_weights, vmax, velocity_range, capacity, distance, route):
    counter = 0
    total_weight = 0
    time = 0
    for city in route:
        
        total_weight += city_weights[city][0]
        velocity = vmax - (velocity_range * (total_weight / capacity))
        if (len(distance)-1) > counter:
            time += distance[counter] / velocity
        counter += 1
    return time

def construct_solution3(values, weights, capacity, pheromone, heuristic,
                       alpha, beta, rng, route, bag_index, vmax, vmin, distance, rent, vtw_ratio):
    
    solution = np.zeros(len(values), dtype=int)
    route_shuffle = np.random.permutation(route)
    remaining_capacity = capacity

    total_value = 0.0
    total_weight = 0.0
    time = 0.0
    velocity_range = vmax - vmin
    
    city_weights = {city: [] for city in route}

    
    # iterate items in the actual order you see them in the tour
    for city in route_shuffle:
        city_weight = 0.0
        city_score = 0.0


        for i in bag_index[city]:

            city_tau = pheromone[i] ** alpha
            city_eta = heuristic[i] ** beta
            city_score =+ city_tau * city_eta

        for i in bag_index[city]:  
            
            if weights[i] > remaining_capacity:
                continue

            

            tau = pheromone[i] ** alpha
            eta = heuristic[i] ** beta
            p = tau * eta / ( city_score )
            

            if rng.random() < p:
                solution[i] = 1
                remaining_capacity -= weights[i]
                total_value += values[i]
                total_weight += weights[i]
                city_weight += weights[i]


        city_weights[city].append(city_weight)
    
    time = time_compute(city_weights, vmax, velocity_range, capacity, distance, route)
    post_rent_value = total_value - (rent * time)
    #print("Total time taken:", time)
    #total_value = total_value - final_time * rent

    


    return solution, total_value, total_weight, time, post_rent_value




def evaporate_pheromone(pheromone, rho):
    """
    evaporate pheremone from pheromone matrix
    """
    pheromone *= (1.0 - rho)
    np.maximum(pheromone, 1e-6, out=pheromone)
    return pheromone

def deposit_pheromone(pheromone, solutions, capacity, q, best_value, best_time):
    "deposit pheromone to matrix"
    best_time_value_ratio = best_time/best_value
    for solution, value, weight, time in solutions:
        
        if weight > capacity :
            continue
        time_value_ratio = time/value
        delta = (q/(1+((best_time_value_ratio-time_value_ratio)/best_time_value_ratio)))
        pheromone[solution == 1] += delta

    return pheromone

def compute_pareto_front_time_value(points):
    """
    points: list of (time, value)
    Assumes: time is minimized, value is maximized.
    Returns: list of (time, value) on the Pareto front, sorted by time.
    """
    # sort by time ascending
    pts_sorted = sorted(points, key=lambda x: x[0])
    
    front = []
    best_value = float("-inf")
    
    # since time increases, keep points where value improves
    for t, v in pts_sorted:
        if v > best_value:
            front.append((t, v))
            best_value = v
    
    return front

def aco_knapsack(
    values,
    weights,
    capacity,
    distance,
    route,
    bag_index,
    vmax, vmin,
    rent,
    n_ants=50,
    n_iterations= 500,
    alpha=1,
    beta=1,
    rho=0.1,
    q=1,
    seed=None):

    # convert to numpy arrays
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    times = []
    sol_values = []

    results = []
    vtw_ratio = np.sum(values)/np.sum(weights)

    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")

    rng = np.random.default_rng(seed)

    n_items = len(values)
    pheromone = initialize_pheromone(n_items)
    heuristic = compute_heuristic(values, weights)

    best_solution = np.zeros(n_items, dtype=int)
    best_value = 0.0001
    best_post_rent_value = -1000000000.0
    best_weight = 0.0
    best_time = 10000000000.0

    for _ in range(n_iterations):
        ant_solutions = []

        for _ in range(n_ants):
            solution, value, weight, time, post_rent_value = construct_solution3(
                values, weights, capacity, pheromone, heuristic, alpha, beta, rng, route, bag_index, vmax, vmin, distance, rent, vtw_ratio
            )
            ant_solutions.append((solution, value, weight, time))


            if weight <= capacity and best_value >= best_value and time <= best_time :
                best_solution = solution.copy()
                best_time = time
                best_value = value
                best_weight = weight
                #best_post_rent_value= post_rent_value
            
                #results.append((time, value))

            
            elif weight <= capacity and value >= best_value and random.random() < 0.5:
                best_solution = solution.copy()
                best_time = time
                best_value = value
                best_weight = weight
                #results.append((time, value))


            elif weight <= capacity and time <= best_time and random.random() < 0.5:
                best_solution = solution.copy()
                best_time = time
                best_value = value
                best_weight = weight
                
            results.append((time, value))
            

                

        pheromone = evaporate_pheromone(pheromone, rho)
        pheromone = deposit_pheromone(pheromone, ant_solutions, capacity, q, best_value, best_time)
    
    results = compute_pareto_front_time_value(results)
    #results = sorted(results, key=lambda x: x[0]) 

    with open("Evo_Ant_hybrid-a280-n1395.txt", "w") as f:
        for a, b in results:

            f.write(f"{a} {b}\n")
        f.close
    
    times_plot = [t for t, v in results]
    values_plot = [v for t, v in results]

    plt.figure(figsize=(8, 5))
    plt.plot(times_plot, values_plot, marker='o')
    plt.xlabel("Time")
    plt.ylabel("Solution Value")
    plt.title("Plot of Time vs Value")
    plt.grid(True)
    plt.show()    

    # return Python types for convenience
    return best_solution.tolist(), best_value, best_weight






route = evolutionary_algorithm(fname)

cities = open_file2(fname)

distance = distances(cities, route)

bags, capacity, vmax, vmin, rent  = open_file_bags(fname)

bag_index = find_bags(route, bags)

item, prof, weight, city = pbag_split(bags)




sol, val, wt = aco_knapsack(prof,
    weight,
    capacity,
    distance,
    route,
    bag_index,
    vmax, vmin,
    rent,)

print("Capacity:", capacity)
print("Best solution:", sol)
print("Total value:", val)  
print("Total weight:", wt)
print("sol length:", len(sol))


