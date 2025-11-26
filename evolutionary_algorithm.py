# TODO: TEST HYPERPARAMETERS FOR EVOLUTIONARY ALGORITHM

import os
import random
import itertools
import math

# File used for testing
# Should be dynamic based on user input
fname = "a280-n1395.txt" 

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
    
evolutionary_algorithm(fname)