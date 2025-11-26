import os
import random

fname = "a280-n1395.txt" 

def open_file(fname):
    """Reads the knapsack problem data from a file"""
    
    # Check if file exists
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Input file '{fname}' not found.")

    capacity = None
    items = []

    # Read file contents
    with open(fname, "r") as file:
        lines = file.readlines()
        
    # Find capacity and items section
    start_index = None    
    for i, line in enumerate(lines):
        if "CAPACITY OF KNAPSACK" in line:
            capacity = int(line.split(":", 1)[1].strip())
        elif "ITEMS SECTION" in line:
            start_index = i + 1
            break

    # Parse items
    for line in lines[start_index:]:
        line = line.strip()
        
        item = line.split()
        item.pop(0)
        items.append([int(item[0]), int(item[1]), int(item[2])])
    
    return capacity, items

def initialise(cities, items, capacity, population_size=10):
    """Initialises a population for the evolutionary algorithm"""
    items = [item for item in items if item[2] in cities]

    chromosomes = []
    chromosome = "0" * len(items)
    for _ in range(population_size):
        new_chromosome = chromosome
        rand_assign = random.randint(0, len(items) - 1)
        while rand_assign > 0:
            item = random.randint(0, len(items) - 1)
            if new_chromosome[item] == "0":
                if fitness(new_chromosome[:item] + "1" + new_chromosome[item + 1:], items, capacity) > 0:
                    new_chromosome = new_chromosome[:item] + "1" + new_chromosome[item + 1:]
                    rand_assign -= 1
        
        chromosomes.append(new_chromosome)
        
    return chromosomes, items

def fitness(chromosome, items, capacity):
    total_value = 0
    total_weight = 0
    for i in range(len(chromosome)):
        if chromosome[i] == "1":
            total_value += items[i][0]
            total_weight += items[i][1]
    return total_value if total_weight <= capacity else 0

def selection(t_size, chromosomes, items, capacity):
    def tournament(t_size, chromosomes, items, capacity):
        indices = []
        for _ in range(t_size):
            indices.append(random.randint(0, len(chromosomes) - 1))
        parent = max(indices, key=lambda chromosome: fitness(chromosomes[chromosome], items, capacity))
        return parent
    
    parents = []
    chromosomes_selection = chromosomes
    for _ in range(2):
        parent = tournament(t_size, chromosomes_selection, items, capacity)
        parents.append(chromosomes_selection[parent])
        chromosomes_selection = chromosomes_selection[:parent] + chromosomes_selection[parent + 1:]
        
    return parents

def crossover(parents, items):
    binary_mask = ""
    for i in range(len(items)):
        binary_mask += str(random.randint(0, 1))
        
    parent1 = parents[0]
    parent2 = parents[1]
    child1 = ""
    child2 = ""
    for i in range(len(binary_mask)):
        if binary_mask[i] == "0":
            child1 += parent1[i]
            child2 += parent2[i]
        else:
            child1 += parent2[i]
            child2 += parent1[i]
    return child1, child2

def mutation(chromosome, mutation_rate=0.01):
    new_chromosome = ""
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            new_chromosome += "1" if chromosome[i] == "0" else "0"
        else:
            new_chromosome += chromosome[i]
    return new_chromosome

def replacement(chromosomes, children, items, capacity):
    for child in children:
        weakest_index = min(range(len(chromosomes)), key=lambda i: fitness(chromosomes[i], items, capacity))
        if fitness(child, items, capacity) > fitness(chromosomes[weakest_index], items, capacity):
            chromosomes[weakest_index] = child
    return chromosomes

def velocity(weight, capacity, vmin=0.1, vmax=1):
    if weight <= capacity:
        return vmax - (weight / capacity) * (vmax - vmin)
    else:
        return vmin

capacity, items = open_file(fname)

cities = [random.randint(2, 280) for _ in range(150)]
cities = list(dict.fromkeys(cities))
chromosomes, items = initialise(cities, items, capacity)

for i in range(10000):
    parents = selection(3, chromosomes, items, capacity)

    child1, child2 = crossover(parents, items)

    mutated_child1 = mutation(child1, mutation_rate=0.05)
    mutated_child2 = mutation(child2, mutation_rate=0.05)
    chromosomes = replacement(chromosomes, [mutated_child1, mutated_child2], items, capacity)

best_chromosome = max(chromosomes, key=lambda chromosome: fitness(chromosome, items, capacity))
best_value = fitness(best_chromosome, items, capacity)

print(f"Best chromosome: {best_chromosome}")
print(f"Best value: {best_value}")