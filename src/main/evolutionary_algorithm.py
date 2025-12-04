# TODO: TEST HYPERPARAMETERS FOR EVOLUTIONARY ALGORITHM

import os
import random
import itertools
import math
import csv
import time
import numpy as np
import matplotlib.pyplot as plt

fname = "src/resources/a280-n1395.txt"
fitness_cache = {}

def open_file(fname):
    """Reads the knapsack problem data from a file"""
    
    # Check if file exists
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Input file '{fname}' not found.")

    cities = []
    bags = []

    # Read file contents
    with open(fname, "r") as file:
        lines = file.readlines()
        
    # Find the start and end indices for the city coordinates
    coords_start_index = None
    coords_end_index = None
    for i, line in enumerate(lines):
        if "CAPACITY OF KNAPSACK" in line:
            capacity = int(line.split(":", 1)[1].strip())
        if "MIN SPEED" in line:
            vmin = float(line.split(":", 1)[1].strip())
        if "MAX SPEED" in line:
            vmax = float(line.split(":", 1)[1].strip())
        if "RENTING RATIO" in line:
            rent = float(line.split(":", 1)[1].strip())
        if "NODE_COORD_SECTION" in line:
            coords_start_index = i + 1
        if "ITEMS SECTION" in line:
            coords_end_index = i
            break

    # Parse city coordinates
    if coords_start_index is not None and coords_end_index is not None:
        for line in lines[coords_start_index:coords_end_index]:
            parts = line.split()
            city = int(parts[0]) - 1
            x = int(parts[1])
            y = int(parts[2])
            cities.append((city, x, y))
    
    items_start_index = coords_end_index + 1
    items_end_index = len(lines)
    
    # Parse bag items
    if items_start_index is not None and items_end_index is not None:
        for line in lines[items_start_index:items_end_index]:
            parts = line.split()
            i = int(parts[0]) - 1
            prof = int(parts[1])
            weight = int(parts[2])
            node = int(parts[3]) - 1
            bags.append((i, prof, weight, node))
            
    n = len(cities)
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = cities[i][1], cities[i][2]
        for j in range(n):
            xj, yj = cities[j][1], cities[j][2]
            d = math.hypot(xi - xj, yi - yj)
            dist[i][j] = math.ceil(d)
            
    city_items = {c: [] for c in range(n)}
    for idx, prof, weight, city in bags:
        city_items[city].append(idx)
            
    return cities, dist, bags, capacity, vmax, vmin, rent, city_items

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

def fitness(chromosome, dist, bags, capacity, vmax, vmin, rent, city_items, velocity=None):
    """Calculates the fitness of a chromosome based on total time taken to complete the tour"""
    
    """I believe that the ant colony optimisation for the knapsack problem should be implemented here
    Based on the chromosome given, get the best packing of items and the resulting velocity to use for time taken"""
    key = tuple(chromosome)
    if key in fitness_cache:
        return fitness_cache[key]
    total_time = 0
    # Add initial city to start and end
    chromosome = [0] + chromosome + [0]
    
    # distance = [dist[chromosome[i]][chromosome[i+1]] for i in range(len(chromosome) - 1)]
    
    # bag_index = {city: city_items[city] for city in chromosome}

    # item, prof, weight, city = pbag_split(bags)

    # sol, val, wt, time = aco_knapsack(prof,
    #     weight,
    #     capacity,
    #     distance,
    #     chromosome,
    #     bag_index,
    #     vmax, vmin,
    #     rent,
    #     n_ants=10,
    #     n_iterations=30)
    
    # return time
    
    
    
    # For testing purposes, assume constant velocity if none provided
    if velocity is None:
        # constant velocity of 1
        return sum(dist[chromosome[i]][chromosome[i+1]] for i in range(len(chromosome) - 1))
    else:
        total_time = 0
        for i in range(len(chromosome) - 1):
            total_time += dist[chromosome[i]][chromosome[i+1]] / velocity[i]
        return total_time

def selection(t_size, chromosomes, fitness_values):
    # """Selects two parents from the population using tournament selection"""
    # def tournament(t_size, chromosomes, fitness_values):
    #     """Tournament selection helper function"""
    #     indices = [random.randint(0, len(chromosomes) - 1) for _ in range(t_size)]
    #     best_idx = min(indices, key=lambda i: fitness_values[i])
    #     return chromosomes[best_idx]
    
    # parent1 = tournament(t_size, chromosomes, fitness_values)
    # parent2 = tournament(t_size, chromosomes, fitness_values)
    
    n = len(chromosomes)
    ranks = sorted(range(n), key=lambda i: fitness_values[i])
    selection_probs = [2 * (n - rank) / (n * (n + 1)) for rank in range(n)]
    parent1 = chromosomes[np.random.choice(ranks, p=selection_probs)]
    parent2 = chromosomes[np.random.choice(ranks, p=selection_probs)]
    return [parent1, parent2]

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

def mutation(chromosome, mutation_rate=0.005):
    """Performs swap mutation on a chromosome"""
    # Go through each gene in the chromosome
    for i in range(len(chromosome)):
        # With a probability of mutation_rate, swap this gene with another random gene
        if random.random() < mutation_rate:
            j = random.randint(0, len(chromosome) - 1)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome

def replacement(chromosomes, children, dist, bags, capacity, vmax, vmin, rent, city_items, fitness_values):
    """Replaces the worst chromosomes in the population with new children if they are better"""
    for child in children:
        child_fitness = fitness(child, dist, bags, capacity, vmax, vmin, rent, city_items)
        # Find the chromosome with the worst fitness (longest distance)
        longest_dist = max(range(len(chromosomes)), key=lambda i: fitness_values[i])
        if child_fitness < fitness_values[longest_dist]:
            chromosomes[longest_dist] = child
            fitness_values[longest_dist] = child_fitness
    return chromosomes, fitness_values

def evolutionary_algorithm(fname, population_size=10, t_size=5, mutation_rate=0.005):
    """Main function to run the evolutionary algorithm"""
    cities, dist, bags, capacity, vmax, vmin, rent, city_items = open_file(fname)
    chromosomes = initialise(cities, population_size)
    fitness_values = [fitness(chromosome, dist, bags, capacity, vmax, vmin, rent, city_items) for chromosome in chromosomes]

    # Run the evolutionary algorithm for a set number of generations
    # Number of generations can be adjusted as needed
    for i in range(10000):
        parents = selection(t_size, chromosomes, fitness_values)
        child1, child2 = crossover(parents)
        mutated_child1 = mutation(child1, mutation_rate)
        mutated_child2 = mutation(child2, mutation_rate)
        # replacement returns updated (chromosomes, fitness_values) so unpack both
        chromosomes, fitness_values = replacement(chromosomes, [mutated_child1, mutated_child2], dist, bags, capacity, vmax, vmin, rent, city_items, fitness_values)

    best_idx = min(range(len(chromosomes)), key=lambda i: fitness_values[i])
    best_chromosome = chromosomes[best_idx]
    best_value = fitness_values[best_idx]

    print(f"Best chromosome: {best_chromosome}")
    print(f"Best value: {best_value}")
    return best_value, best_chromosome, dist

########################################################
########################################################
########################################################
########################################################

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
    
    # results = compute_pareto_front_time_value(results)
    # #results = sorted(results, key=lambda x: x[0]) 

    # with open("Evo_Ant_hybrid-a280-n1395.txt", "w") as f:
    #     for a, b in results:

    #         f.write(f"{a} {b}\n")
    #     f.close
    
    # times_plot = [t for t, v in results]
    # values_plot = [v for t, v in results]

    # plt.figure(figsize=(8, 5))
    # plt.plot(times_plot, values_plot, marker='o')
    # plt.xlabel("Time")
    # plt.ylabel("Solution Value")
    # plt.title("Plot of Time vs Value")
    # plt.grid(True)


    # return Python types for convenience
    return best_solution.tolist(), best_value, best_weight, best_time



value, route, distance = evolutionary_algorithm(fname)
plt.show()

# cities = open_file2(fname)

# bags, capacity, vmax, vmin, rent  = open_file_bags(fname)

# bag_index = find_bags(route, bags)

# item, prof, weight, city = pbag_split(bags)




# sol, val, wt = aco_knapsack(prof,
#     weight,
#     capacity,
#     distance,
#     route,
#     bag_index,
#     vmax, vmin,
#     rent,)

# print("Capacity:", capacity)
# print("Best solution:", sol)
# print("Total value:", val)  
# print("Total weight:", wt)
# print("sol length:", len(sol))s

    
fnames= ["src/resources/fnl4461-n4460.txt"]
population_sizes = [20, 50, 100]
tournament_sizes = [5, 10, 15]
mutation_rates = [0.001, 0.005, 0.01]
optimal_parameters = []

# Collect all results for later analysis
all_test_results = []

for fname in fnames:
    best_params = None
    test_results = []
    start_file = time.time()
    for pop_size, t_size, mut_rate in itertools.product(population_sizes, tournament_sizes, mutation_rates):
        print(f"Testing with population size: {pop_size}, tournament size: {t_size}, mutation rate: {mut_rate} for file {fname}")
        best_value, best_chromosome, dist = evolutionary_algorithm(fname, population_size=pop_size, t_size=t_size, mutation_rate=mut_rate)
        record = {
            "file": fname,
            "population_size": pop_size,
            "tournament_size": t_size,
            "mutation_rate": mut_rate,
            "best_value": best_value,
            "best_chromosome": str(best_chromosome)
        }
        test_results.append(record)
        all_test_results.append(record)

        if best_params is None or best_value < best_params[0]:
            best_params = (best_value, pop_size, t_size, mut_rate)
    elapsed = time.time() - start_file
    optimal_parameters.append((fname, best_params))
    # Write per-file CSV with all parameter combinations and results
    out_name = f"results_{os.path.basename(fname)}.csv"
    try:
        with open(out_name, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["file", "population_size", "tournament_size", "mutation_rate", "best_value", "best_chromosome"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in test_results:
                writer.writerow(r)
        print(f"Finished tests for {fname} in {elapsed:.1f}s — wrote {out_name}")
    except Exception as e:
        print(f"Failed to write results for {fname}: {e}")

for params in optimal_parameters:
    print(f"Optimal parameters for {params[0]}: Population Size = {params[1][1]}, Tournament Size = {params[1][2]}, Mutation Rate = {params[1][3]} with Best Value = {params[1][0]}")

# Write aggregated CSV for all files
agg_name = "all_results.csv"
try:
    with open(agg_name, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["file", "population_size", "tournament_size", "mutation_rate", "best_value", "best_chromosome"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_test_results:
            writer.writerow(r)
    print(f"Wrote aggregated results to {agg_name}")
except Exception as e:
    print(f"Failed to write aggregated results: {e}")

