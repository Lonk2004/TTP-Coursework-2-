import os
import math
import csv
import pickle

fnames= ["src/resources/a280-n279.txt", "src/resources/fnl4461-n4460.txt", "src/resources/pla33810-n33809.txt"]

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
            
    return dist

for fname in fnames: 
    dist = open_file(fname)
    print(f"Distance matrix for {fname}:")

    # --- NEW: save distance matrix to CSV ---
    outname = fname.replace(".txt", "_dist.pkl")
    with open(outname, "wb") as f:
        pickle.dump(dist, f)

    print(f"Saved distance matrix to {outname}")