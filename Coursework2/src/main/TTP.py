import numpy as np
import os
from scipy.spatial import KDTree
import seaborn as sns
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm
import time

#Utilities for data loading
def load_ttp_file(filepath):
    cities = []
    items = []
    capacity = 0

    #go through each section
    section = "Header"

    with open(filepath, 'r') as f: 
        for line in f: 
            line = line.strip()
            if not line: 
                continue 
            

            #detect if the section is changed
            if line.startswith("NODE_COORD_SECTION"): 
                section = "Nodes"
                continue
            elif line.startswith("ITEMS SECTION"): 
                section = "Items"
                continue

            #Extract capacity 
            if section == "Header":
                if "CAPACITY OF KNAPSACK" in line:
                    lineparts = line.split(":")
                    capacity = float(lineparts[1].strip())
                elif "MIN SPEED" in line: 
                    lineparts = line.split(":")
                    min_speed = float(lineparts[1].strip())
                elif "MAX SPEED" in line: 
                    lineparts = line.split(":")
                    max_speed = float(lineparts[1].strip())
                elif "RENTING RATIO" in line: 
                    lineparts = line.split(":")
                    renting_ratio = float(lineparts[1].strip())

            #Extract Notes
            elif section == "Nodes": 
                lineparts = line.split()
                cities.append([float(lineparts[1]), float(lineparts[2])])

            #Extract Items
            elif section == "Items":
            #Format: Index, Profit, Weight, Assigned Node
                lineparts = line.split()
                items.append({'id': int(lineparts[0]), 'value': float(lineparts[1]), 
                          'weight': float(lineparts[2]), 'city_id': int(lineparts[3])-1})
    return cities, items, capacity, min_speed, max_speed, renting_ratio

#Plot the route for visualisation
def plot_gaco_route(cities, route, distance, filename="TSP Solution"):
    cities_np = np.array(cities)
    route_np = np.array(route) 

    plt.figure(figsize=(12, 8))
    plt.scatter(cities_np[:, 0], cities_np[:, 1], c='#cccccc', s=10, marker='.', label='Cities')
    
    path_coords = cities_np[route_np]
    plt.plot(path_coords[:, 0], path_coords[:, 1], c='blue', linewidth=1, alpha=0.8, label='GACO Route')
    
    start_city = cities_np[route_np[0]]
    plt.scatter(start_city[0], start_city[1], c='red', s=100, marker='*', zorder=10, label='Start/End')

    plt.title(f"GACO Solution for {os.path.basename(filename)}\nTotal Distance: {distance:.2f}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()
"""
#This algorithm was created using the design from the paper "A two-stage algorithm based on greedy ant colony optimization for travelling thief problem" 
#class TTP: 
    def __init__(self, cities, items, capacity, min_speed, max_speed, renting_ratio):
        self.cities = np.array(cities)
        self.items = items 
        self.capacity = capacity
        self.min_speed = min_speed
        self.max_speed = max_speed 
        self.renting_ratio = renting_ratio
        self.num_cities = len(cities)
        self.dist_matrix = self._compute_distances()


    def _compute_distances(self): 
        dist_matrix = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities): 
            for j in range(self.num_cities):
                if i == j: 
                    continue
                city = self.cities[i]
                city2 = self.cities[j]
                dist_vec = city - city2
                dist = np.linalg.norm(dist_vec)
                dist_matrix[i,j] = np.ceil(dist)
        return dist_matrix
    
#class GACO: 
    def __init__(self, ttp, num_ants=40, alpha =1, beta=5, Q = 0.3): 
        self.ttp = ttp
        self.num_ants = num_ants 
        self.alpha = alpha #pheromone importance
        self.beta = beta #distance importance
        self.rho = 0.9 #evapouration rate
        self.Q = Q #selection coefficent

        self.pheromone_matrix = np.ones((self.ttp.num_cities, self.ttp.num_cities))
        self.selection_matrix = np.ones(self.ttp.num_cities)

        self._init_selection_matrix()

    #item selection algorithm - ensures that the GACO chooses a high value route
    def _run_ISA(self, type1_items): 
        THETA_H = 1/10 
        THETA_G = 1/3
        x_coords = self.ttp.cities[:,0]
        y_coords = self.ttp.cities[:,1]

        X_min, X_max = np.min(x_coords), np.max(x_coords)
        Y_min, Y_max = np.min(y_coords), np.max(y_coords)

        #define a judgement area size 
        g = np.min([X_max - X_min, Y_max-Y_min]) * THETA_G

        #define the step sizes 
        h_x = (X_max - X_min) * THETA_H
        h_y = (Y_max - Y_min) * THETA_H

        areas = []
        x = X_min 

        #iterate through grid 
        while x + h_x < X_max: 
            y = Y_min 
            while y + h_y < Y_max: 
                current_area = {'x_min': x, 'x_max': x+g,
                                'y_min': y, 'y_max': y+g }
                #count high value items in this box 
                count = 0
                for item in type1_items:
                    city_coords = self.ttp.cities[item['city_id']]
                    cx, cy = city_coords[0], city_coords[1]

                    if (current_area['x_min'] <= cx < current_area['x_max'] and 
                        current_area['y_min'] <= cy < current_area['y_max']): 
                        count += 1 
                areas.append({'count': count, 'area': current_area})#save number of items

                y += h_y
            x += h_x
        #prioritise the densest region 
        return sorted(areas, key=lambda a: a['count'], reverse = True)

    def _run_ica(self): 
        #returns final set of selected item IDs from route
        #calculate value density 
        CHI = 0.5
        for item in self.ttp.items: 
            item['density'] = _calculate_value_density(item)

        #sort all items by their value density 
        sorted_items_density = sorted(self.ttp.items, key=lambda x: x['density'], reverse=True)
        current_weight = 0
        selected_items_final = []

        #divide tiems into 2 types based on density
        type1 = []
        type2 = []

        max_type1_weight = self.ttp.capacity * CHI

        #classify items into type 1 and type 2
        for item in sorted_items_density: 
            if current_weight + item['weight'] <= max_type1_weight: 
                type1.append(item)
                current_weight += item['weight']
                selected_items_final.append(item) #add this item to S
            else: 
                type2.append(item)

        #find the dense areas 
        sorted_areas = self._run_ISA(type1)

        unselected_type2 = type2 

        for area_data in sorted_areas: 
            area = area_data['area']

            items_to_select= []

            for item in unselected_type2: 
                #run similar to ISA for type2 items 
                city_coords = self.ttp.cities[item['city_id']]
                cx, cy = city_coords[0], city_coords[1]
                
                # Check if item location Lk is in area [cite: 1147, 1160]
                is_in_area = (area['x_min'] <= cx < area['x_max'] and 
                            area['y_min'] <= cy < area['y_max'])
                
                can_fit = current_weight + item['weight'] <= self.ttp.capacity
                # Check capacity [cite: 1160]
                if is_in_area and can_fit:
                        selected_items_final.append(item)
                        current_weight += item['weight']
                else:
                    items_to_select.append(item)
            #remaining items become list for next aarea 
            unselected_type2 = items_to_select
            #if backpack is full, stop
            if current_weight >= self.ttp.capacity: 
                break 
        selected_item_ids = [int(item['id']) for item in selected_items_final]
        return selected_item_ids

    def _init_selection_matrix(self):

        selected_item_ids = self._run_ica()

        ica_selected_items = [item for item in self.ttp.items if item['id'] in selected_item_ids]
        #Sort items by decending weight
        sorted_items = sorted(ica_selected_items, key=lambda x: x['weight'], reverse=True)

        #Select top Q%
        limit = int(len(sorted_items) * self.Q)

        #modify lambda (selection matrix) for heavy itemed cities
        for i in range(limit): 
            city_id = sorted_items[i]['city_id']
            self.selection_matrix[city_id] = 0.1 #ensure not visited early
    #create mapping from items to city        
    def _map_items_to_cities(self):
        items_by_city = {i: [] for i in range(self.ttp.num_cities)}
        
        for item in self.ttp.items:
            city_index = item['city_id']
            items_by_city[city_index].append(item)
            
        return items_by_city

    def construct_route(self): 
        current_city = 0
        route = [0]
        unvisited = set(range(1, self.ttp.num_cities))

        while unvisited: 
            probs = []
            candidates = list(unvisited)

            denominator = 0
            numerators = []

            for target in candidates: 
                #pheromone input
                tau = self.pheromone_matrix[current_city][target]
                #distance input
                dist = self.ttp.dist_matrix[current_city][target]
                eta = 1.0 / dist if dist > 0 else 0

                #selection of target city
                lam = self.selection_matrix[target]

                numerator = (tau**self.alpha) * (eta ** self.beta) * lam
                numerators.append(numerator)
                denominator += numerator
            #Calculate the probabilites 
            if denominator == 0: 
                probs = [1.0 / len(candidates)] * len(candidates)
            else: 
                probs = [num / denominator for num in numerators]

            next_city = np.random.choice(candidates, p=probs)

            route.append(next_city)
            unvisited.remove(next_city)
            current_city = (next_city)
        return route
    def update_pheromone_trail(self, all_ant_routes, all_ant_distances): 
        #evapouration effects
        self.pheromone_matrix *= (1-self.rho)
        #deposit
        for route, dist in zip(all_ant_routes, all_ant_distances): 
            pheromone_deposit = 1.0/ dist if dist > 0 else 0

            #iterate through route
            for i in range(len(route)-1):
                x = route[i]
                y = route[i+1]
                self.pheromone_matrix[x][y] += pheromone_deposit
                self.pheromone_matrix[y][x] += pheromone_deposit

            end = route[-1]
            beg = route[0]
            self.pheromone_matrix[beg][end] += pheromone_deposit
            self.pheromone_matrix[end][beg] += pheromone_deposit

    #optimise crossing paths
    def two_opt(self, route, dist_matrix):
        best_route = route
        best_distance = self.calculate_total_distance(route, dist_matrix)
        improve = True

        while improve: 
            improve = False 
            for i in range(1, len(route) - 2): 
                for j in range(i+1, len(route)): 
                    if j - i == 1: continue
                    new_route = route[:]
                    new_route[i:j] = route[j-1:i-1:-1] #standard 2-opt swap

                    new_distance = self.calculate_total_distance(new_route, dist_matrix)

                    if new_distance < best_distance: 
                        best_route = new_route
                        best_distance = new_distance
                        improve = True 
                        break 
                if improve: 
                    break
        return best_route, best_distance
    

    def calculate_total_distance(self, route, dist_matrix): 
        total = 0 
        for i in range(len(route) -1): 
            total += dist_matrix[route[i], route[i+1]]
        #return trip
        total += dist_matrix[route[-1], route[0]]
        return total
    def get_available_items_in_route_order(self, route):
        
        #map items 
        if not hasattr(self, 'items_by_city_map'):
            self.items_by_city_map = self._map_items_to_cities()
        
        all_available_items = []
        
        #iterate through the route 
        for city_index in route:
            # Skip the starting city (index 0) as it usually has no items/is only used for return
            if city_index == 0:
                continue

            # retrieve all items located in the current city
            items_in_city = self.items_by_city_map.get(city_index, [])
            
            # add these items to the master list
            all_available_items.extend(items_in_city)
            
        return all_available_items
    
    def run(self, max_iterations =100): 
        best_global_route = None
        best_global_distance = float('inf')

        for iteration in range(max_iterations): 
            all_routes = []
            all_distances = []

            for ant in range(self.num_ants): 
                route = self.construct_route()
                dist = self.calculate_total_distance(route, self.ttp.dist_matrix)

                all_routes.append(route)
                all_distances.append(dist)

                if dist < best_global_distance: 
                    best_global_distance = dist
                    best_global_route = route

            self.update_pheromone_trail(all_routes, all_distances)
            print(f"Iteration {iteration}: Best Distance = {best_global_distance}")

        #finally run 2-opt optimization on best route
        print("Running 2-OPT optimization...")
        final_route, final_dist = self.two_opt(best_global_route, self.ttp.dist_matrix)
        items = self.get_available_items_in_route_order(final_route)
        return final_route, final_dist, items
"""

#Algorithms have been modified by myself with the help of Gemini AI to optimsie for TTPs so large, we cannot store everything as np matricies.
class TTP_Large: 
    def __init__(self, cities, items, capacity, min_speed, max_speed, renting_ratio):
        self.cities = np.array(cities)
        self.items = items 
        self.capacity = capacity
        self.min_speed = min_speed
        self.max_speed = max_speed 
        self.renting_ratio = renting_ratio
        self.num_cities = len(cities)

    #No distance matrix for large problems
    def get_dist(self, i, j):
        """Calculates CEIL_2D distance using scalar math (faster for single items)."""
        c1 = self.cities[i]
        c2 = self.cities[j]
        dx = c1[0] - c2[0]
        dy = c1[1] - c2[1]
        dist = math.sqrt(dx*dx + dy*dy)
        return math.ceil(dist)
    
#Ant Colony Optimisation (GACO)
class GACO_Large: 
    def __init__(self, ttp, num_ants=50, alpha =1, beta=8, rho = 0.1, Q = 0.3, k_neighbors = 100): 
        self.ttp = ttp
        self.num_ants = num_ants 
        self.alpha = alpha #pheromone importance
        self.beta = beta #distance importance
        self.rho = rho #evapouration rate
        self.Q = Q #selection coefficient
        self.k = k_neighbors #only see k neighbours

        self.min_tau = 0.01
        self.max_tau = 6.0
        print("Building KD-Tree for neighbors")
        self.tree = KDTree(self.ttp.cities)

        dists, idxs = self.tree.query(self.ttp.cities, k=self.k + 1)

        self.neighbor_indices = idxs[:, 1:]
        self.neighbor_dists = dists[:, 1:]

        #use sparse matrix representation for memory efficiency
        self.pheromone_matrix = np.full((self.ttp.num_cities, self.k), self.max_tau)

        #Add a small delta to prevent divide by 0 error
        self.heuristic_matrix = 1.0 / (self.neighbor_dists + 1e-10)
        
        self.selection_matrix = np.ones(self.ttp.num_cities)
        print("running ica/isa")
        self._init_selection_matrix()

    #item selection algorithm - ensures that the GACO chooses a high value route
    def _run_ISA(self, type1_items): 
        THETA_H = 1/10 
        THETA_G = 1/3
        x_coords = self.ttp.cities[:,0]
        y_coords = self.ttp.cities[:,1]

        X_min, X_max = np.min(x_coords), np.max(x_coords)
        Y_min, Y_max = np.min(y_coords), np.max(y_coords)

        #define a judgement area size 
        g = np.min([X_max - X_min, Y_max-Y_min]) * THETA_G

        #define the step sizes 
        h_x = (X_max - X_min) * THETA_H
        h_y = (Y_max - Y_min) * THETA_H

        areas = []
        x = X_min 

        #iterate through grid 
        while x + h_x < X_max: 
            y = Y_min 
            while y + h_y < Y_max: 
                current_area = {'x_min': x, 'x_max': x+g,
                                'y_min': y, 'y_max': y+g }
                #count high-value items in this box
                count = 0
                for item in type1_items:
                    city_coords = self.ttp.cities[item['city_id']]
                    cx, cy = city_coords[0], city_coords[1]

                    if (current_area['x_min'] <= cx < current_area['x_max'] and 
                        current_area['y_min'] <= cy < current_area['y_max']): 
                        count += 1 
                areas.append({'count': count, 'area': current_area})#save number of items

                y += h_y
            x += h_x
        #prioritise the densest region 
        return sorted(areas, key=lambda a: a['count'], reverse = True)

    def _run_ica(self): 
        #returns final set of selected item IDs from route
        def _calculate_value_density(item):
            return item['value'] / item['weight'] if item['weight'] > 0 else 0
        
        CHI = 0.5


        for item in self.ttp.items: 
            item['density'] = _calculate_value_density(item)


        #sort all items by their value density 
        sorted_items_density = sorted(self.ttp.items, key=lambda x: x['density'], reverse=True)
        current_weight = 0
        selected_items_final = []

        #divide tiems into 2 types based on density
        type1 = []
        type2 = []

        max_type1_weight = self.ttp.capacity * CHI

        #classify items into type 1 and type 2
        for item in sorted_items_density: 
            if current_weight + item['weight'] <= max_type1_weight: 
                type1.append(item)
                current_weight += item['weight']
                selected_items_final.append(item) #add this item to S
            else: 
                type2.append(item)

        #find the dense areas 
        sorted_areas = self._run_ISA(type1)
        unselected_type2 = type2 

        for area_data in sorted_areas: 
            area = area_data['area']

            items_to_select= []

            for item in unselected_type2: 
                #run similar to ISA for type2 items 
                city_coords = self.ttp.cities[item['city_id']]
                cx, cy = city_coords[0], city_coords[1]
                
                # Check if item location Lk is in area [cite: 1147, 1160]
                is_in_area = (area['x_min'] <= cx < area['x_max'] and 
                            area['y_min'] <= cy < area['y_max'])
                
                can_fit = current_weight + item['weight'] <= self.ttp.capacity
                # Check capacity [cite: 1160]
                if is_in_area and can_fit:
                        selected_items_final.append(item)
                        current_weight += item['weight']
                else:
                    items_to_select.append(item)
            #remaining items become list for next aarea 
            unselected_type2 = items_to_select
            #if backpack is full, stop
            if current_weight >= self.ttp.capacity: 
                break 
        
        return set([item['id'] for item in selected_items_final])

    def _init_selection_matrix(self):
        selected_item_ids = self._run_ica()
        ica_selected_items = [item for item in self.ttp.items if item['id'] in selected_item_ids]
        #Sort items by decending weight
        ica_selected_items.sort(key=lambda x: x['weight'], reverse=True)
        #Select top Q%
        limit = int(len(ica_selected_items) * self.Q)
        #modify lambda (selection matrix) for heavy itemed cities
        for i in range(limit): 
            city_id = ica_selected_items[i]['city_id']
            self.selection_matrix[city_id] = 0.1 #ensure not visited early

    def construct_route(self): 
        n_cities = self.ttp.num_cities
        route = np.zeros(n_cities + 1, dtype=int)
        visited = np.zeros(n_cities, dtype=bool)

        current_city = 0 
        route[0] = 0
        visited[current_city] = True 
        for step in range(1, n_cities): 
            neighbors = self.neighbor_indices[current_city]
            valid_mask = ~visited[neighbors]
            valid_neighbors = neighbors[valid_mask]
            if len(valid_neighbors) > 0:
                k_indices = np.where(valid_mask)[0]
                tau = self.pheromone_matrix[current_city, k_indices]
                eta = self.heuristic_matrix[current_city, k_indices]
                lam = self.selection_matrix[valid_neighbors]
                probs = (tau ** self.alpha) * (eta ** self.beta) * lam
                prob_sum = probs.sum()
                if prob_sum == 0:

                    next_city = np.random.choice(valid_neighbors)
                else:
                    probs /= prob_sum
                    # Roulette Wheel Selection
                    next_city = np.random.choice(valid_neighbors, p=probs)
            else: 
                #pick nearest global unvisited city
                unvisited_indices = np.where(~visited)[0]
                current_pos = self.ttp.cities[current_city]
                dists = np.linalg.norm(self.ttp.cities[unvisited_indices] - current_pos, axis=1)
                nearest_local_idx = np.argmin(dists)
                next_city = unvisited_indices[nearest_local_idx]
            # 5. Move
            route[step] = next_city
            visited[next_city] = True
            current_city = next_city
        # Return to start
        route[-1] = 0
        return route

    def update_pheromone_trail(self, iter_best_route, iter_best_dist, global_best_route, global_best_dist): 
        #evapouration effects
        self.pheromone_matrix *= (1-self.rho)
        #Optimisation: have impact from the best route of the iteration, and the global best route.
        deposit_ib = 1.0/iter_best_dist if iter_best_dist > 0 else 0
        self._deposit_on_route(iter_best_route, deposit_ib)
        deposit_gb = 2.0 / global_best_dist if global_best_dist > 0 else 0
        self._deposit_on_route(global_best_route, deposit_gb)
        np.clip(self.pheromone_matrix, self.min_tau, self.max_tau, out=self.pheromone_matrix)



import numpy as np
from scipy.spatial import KDTree

class GACO_Large: 
    def __init__(self, ttp, num_ants=50, alpha=1.0, beta=8.0, rho=0.1, Q=0.3, k_neighbors=100): 
        """
        Initializes the Ant Colony Optimization for Large instances.
        Uses Candidate Lists (K-Nearest Neighbors) to avoid O(N^2) memory usage.
        """
        self.ttp = ttp
        self.num_ants = num_ants 
        self.alpha = alpha      # Importance of Pheromone (History)
        self.beta = beta        # Importance of Heuristic (Distance/Greediness)
        self.rho = rho          # Evaporation rate (Memory retention)
        self.Q = Q              # Selection coefficient for Item Selection bias
        self.k = k_neighbors    # Size of candidate list (only look at k closest cities)

        # Max-Min Ant System (MMAS) bounds to prevent stagnation
        self.min_tau = 0.01
        self.max_tau = 6.0

        print(f"Building KD-Tree for {self.ttp.num_cities} cities...")
        # OPTIMIZATION: KD-Tree allows for fast spatial queries
        self.tree = KDTree(self.ttp.cities)

        # Pre-calculate only the K-Nearest Neighbors for every city.
        # instead of a full N*N distance matrix.
        dists, idxs = self.tree.query(self.ttp.cities, k=self.k + 1)
        self.neighbor_indices = idxs[:, 1:]   # Indices of neighbors (excluding self)
        self.neighbor_dists = dists[:, 1:]    # Distances to neighbors

        # Sparse Pheromone Matrix: shape (Num_Cities, k). 
        # We only track pheromones on edges connecting to neighbors.
        self.pheromone_matrix = np.full((self.ttp.num_cities, self.k), self.max_tau)
        
        # Heuristic Matrix (Eta): Inverse distance (1/d). Closer cities = higher desirability.
        self.heuristic_matrix = 1.0 / (self.neighbor_dists + 1e-10)
        
        # Selection Matrix: Used to bias the ACO based on item distribution (TTP specific)
        self.selection_matrix = np.ones(self.ttp.num_cities)
        print("Running ICA/ISA initialization...")
        self._init_selection_matrix()

    def _run_ISA(self, type1_items): 
        """
        Item Selection Algorithm (ISA):
        Uses a grid-based spatial hashing approach to find high-density areas 
        of valuable items.
        """
        THETA_H = 1/10; THETA_G = 1/3
        
        # Extract coordinates
        x_coords = self.ttp.cities[:,0]; y_coords = self.ttp.cities[:,1]
        X_min, X_max = np.min(x_coords), np.max(x_coords)
        Y_min, Y_max = np.min(y_coords), np.max(y_coords)
        
        # Define grid cell sizes
        g = np.min([X_max - X_min, Y_max-Y_min]) * THETA_G
        h_x = (X_max - X_min) * THETA_H; h_y = (Y_max - Y_min) * THETA_H
        
        areas = []
        x = X_min 
        # Sliding window over the map to count items
        while x + h_x < X_max: 
            y = Y_min 
            while y + h_y < Y_max: 
                current_area = {'x_min': x, 'x_max': x+g, 'y_min': y, 'y_max': y+g }
                count = 0
                for item in type1_items:
                    city_coords = self.ttp.cities[item['city_id']]
                    cx, cy = city_coords[0], city_coords[1]
                    # Check if item is inside the current grid box
                    if (current_area['x_min'] <= cx < current_area['x_max'] and 
                        current_area['y_min'] <= cy < current_area['y_max']): 
                        count += 1 
                areas.append({'count': count, 'area': current_area})
                y += h_y
            x += h_x
        # Return areas sorted by item count (densest first)
        return sorted(areas, key=lambda a: a['count'], reverse = True)

    def _run_ica(self): 
        """
        Item Clustering Algorithm (ICA):
        Categorizes items by value density and selects a 'knapsack plan' 
        focused on high-density geographical areas.
        """
        def _calculate_value_density(item):
            return item['value'] / item['weight'] if item['weight'] > 0 else 0
        
        CHI = 0.5 # Capacity threshold
        for item in self.ttp.items: 
            item['density'] = _calculate_value_density(item)
            
        # Sort all items by Value/Weight ratio
        sorted_items_density = sorted(self.ttp.items, key=lambda x: x['density'], reverse=True)
        current_weight = 0
        selected_items_final = []
        type1 = []; type2 = []
        max_type1_weight = self.ttp.capacity * CHI
        
        # Split items into Type 1 (High Density) and Type 2 (Remainder)
        for item in sorted_items_density: 
            if current_weight + item['weight'] <= max_type1_weight: 
                type1.append(item)
                current_weight += item['weight']
                selected_items_final.append(item) 
            else: 
                type2.append(item)
        
        # Identify dense areas using ISA
        sorted_areas = self._run_ISA(type1)
        unselected_type2 = type2 
        
        # Attempt to fill remaining capacity with Type 2 items located in those dense areas
        for area_data in sorted_areas: 
            area = area_data['area']
            items_to_select= []
            for item in unselected_type2: 
                city_coords = self.ttp.cities[item['city_id']]
                cx, cy = city_coords[0], city_coords[1]
                
                is_in_area = (area['x_min'] <= cx < area['x_max'] and area['y_min'] <= cy < area['y_max'])
                can_fit = current_weight + item['weight'] <= self.ttp.capacity
                
                if is_in_area and can_fit:
                        selected_items_final.append(item)
                        current_weight += item['weight']
                else:
                    items_to_select.append(item)
            unselected_type2 = items_to_select
            if current_weight >= self.ttp.capacity: break 
            
        return set([item['id'] for item in selected_items_final])

    def _init_selection_matrix(self):
        """
        Uses the result of ICA to bias the ant movement.
        Cities with selected heavy items might be penalized (0.1) to encourage 
        picking them up later in the route (reducing travel cost).
        """
        selected_item_ids = self._run_ica()
        ica_selected_items = [item for item in self.ttp.items if item['id'] in selected_item_ids]
        ica_selected_items.sort(key=lambda x: x['weight'], reverse=True)
        
        # Apply bias to top Q% of heavy items
        limit = int(len(ica_selected_items) * self.Q)
        for i in range(limit): 
            city_id = ica_selected_items[i]['city_id']
            # Modify the selection matrix at this city
            self.selection_matrix[city_id] = 0.1 

    def construct_route(self): 
        """
        Constructs a single route for one ant.
        Includes heuristics for probability and a 'Rescue Mode' for dead ends.
        """
        n_cities = self.ttp.num_cities
        route = np.zeros(n_cities + 1, dtype=int)
        visited = np.zeros(n_cities, dtype=bool)

        # 1. Start at a random city
        current_city = np.random.randint(0, n_cities)
        route[0] = current_city
        visited[current_city] = True 
        
        alpha = self.alpha
        beta = self.beta
        
        # 2. Build the rest of the tour
        for step in range(1, n_cities): 
            neighbors = self.neighbor_indices[current_city]
            valid_mask = ~visited[neighbors]
            
            # CASE A: Neighbors available (Standard ACO)
            if valid_mask.any():
                valid_indices = np.where(valid_mask)[0]
                valid_neighbors = neighbors[valid_indices]
                
                # Retrieve Pheromone (tau) and Distance Heuristic (eta)
                tau = self.pheromone_matrix[current_city, valid_indices]
                eta = self.heuristic_matrix[current_city, valid_indices]
                # Apply TTP-specific Item Bias (lam)
                lam = self.selection_matrix[valid_neighbors]
                
                # Calculate probability formula: (tau^alpha) * (eta^beta) * lam
                probs = (tau ** alpha) * (eta ** beta) * lam
                
                # --- FAST ROULETTE WHEEL SELECTION ---
                cumsum = np.cumsum(probs)
                total = cumsum[-1]
                
                if total == 0:
                    next_city = valid_neighbors[np.random.randint(len(valid_neighbors))]
                else:
                    # Binary search is faster than linear scan for large K
                    r = np.random.random() * total
                    idx = np.searchsorted(cumsum, r)
                    next_city = valid_neighbors[min(idx, len(valid_neighbors)-1)]

            # CASE B: No neighbors available (Rescue Mode)
            else: 
                unvisited_indices = np.where(~visited)[0]
                
                # OPTIMIZATION: Instead of calculating distance to ALL unvisited nodes (slow),
                # sample 50 random unvisited nodes and pick the closest one.
                # This prevents "teleporting" across the map while keeping speed high.
                sample_size = min(50, len(unvisited_indices))
                candidates = np.random.choice(unvisited_indices, size=sample_size, replace=False)
                
                current_pos = self.ttp.cities[current_city]
                
                # Euclidean distance check on sample
                dists = np.sum((self.ttp.cities[candidates] - current_pos)**2, axis=1)
                nearest_sample_idx = np.argmin(dists)
                
                next_city = candidates[nearest_sample_idx]
                
            route[step] = next_city
            visited[next_city] = True
            current_city = next_city
            
        # Return to start to close the loop
        route[-1] = route[0]
        return route

    def update_pheromone_trail(self, iter_best_route, iter_best_dist, global_best_route, global_best_dist): 
        """
        Updates pheromones based on MMAS rules.
        """
        # 1. Global Evaporation
        self.pheromone_matrix *= (1.0 - self.rho)
        
        # 2. Deposit for Iteration Best (Exploitation of current loop)
        deposit_ib = 1.0 / iter_best_dist if iter_best_dist > 0 else 0
        self._deposit_on_route(iter_best_route, deposit_ib)
        
        # 3. Deposit for Global Best (Strong Elitism)
        deposit_gb = 2.0 / global_best_dist if global_best_dist > 0 else 0
        self._deposit_on_route(global_best_route, deposit_gb)
        
        # 4. Clamp values (MMAS Requirement)
        np.clip(self.pheromone_matrix, self.min_tau, self.max_tau, out=self.pheromone_matrix)

    def _deposit_on_route(self, route, amount):
        """Helper to deposit pheromone on edges existing in the neighbor list."""
        for i in range(len(route) -1): 
            x = route[i]
            y = route[i+1]
            
            # Find index of Y in X's neighbor list
            idx_arr = np.where(self.neighbor_indices[x] == y)[0]
            if len(idx_arr) > 0: 
                k_idx = idx_arr[0]
                self.pheromone_matrix[x, k_idx] += amount
            
            # Symmetric update (Undirected graph)
            idx_arr_y = np.where(self.neighbor_indices[y] == x)[0]
            if len(idx_arr_y) > 0: 
                k_idx_y = idx_arr_y[0]
                self.pheromone_matrix[y, k_idx_y] += amount

    def calculate_total_distance(self, route): 
        total = 0.0
        for i in range(len(route) - 1): 
            total += self.ttp.get_dist(route[i], route[i+1])
        return total
    
    def fast_two_opt(self, route, max_passes=1):
        """
        Optimized 2-opt Local Search.
        Only considers swaps if the new node is within the top 20 neighbors.
        O(N * 20) complexity instead of O(N^2).
        """
        best_route = np.array(route)
        best_dist = self.calculate_total_distance(best_route)
        n = len(route) - 1 
        
        # Position array for O(1) index lookup
        pos = np.zeros(self.ttp.num_cities + 1, dtype=int)
        pos[best_route[:-1]] = np.arange(n)
        pos[best_route[-1]] = n 

        for _ in range(max_passes):
            improved = False
            for i in range(n - 1):
                u = best_route[i]
                u_next = best_route[i+1]
                
                # Heuristic Pruning: Only check close neighbors
                neighbors = self.neighbor_indices[u, :20] 
                
                for v in neighbors:
                    if v == u_next: continue
                    j = pos[v]
                    
                    # Ensure valid swap order
                    if j <= i + 1 or j >= n: continue
                    v_next = best_route[j+1]
                    
                    # Calculate gain
                    d_curr = self.ttp.get_dist(u, u_next) + self.ttp.get_dist(v, v_next)
                    d_new = self.ttp.get_dist(u, v) + self.ttp.get_dist(u_next, v_next)
                    
                    if d_new < d_curr:
                        # Perform swap
                        best_route[i+1:j+1] = best_route[i+1:j+1][::-1]
                        best_dist -= (d_curr - d_new)
                        # Update position array
                        pos[best_route[i+1:j+1]] = np.arange(i+1, j+1)
                        improved = True
                        break # First improvement heuristic
                if improved: break
            if not improved: break
        return best_route, best_dist
    
    def full_two_opt(self, route, max_passes=5):
        """
        Full O(N^2) 2-opt. 
        Necessary to fix 'crossing lines' (Global intersections) that fast 2-opt misses.
        """
        # Safety guard for massive datasets
        if self.ttp.num_cities > 5000:
            print(f"Map size ({self.ttp.num_cities}) too large for Global 2-Opt.")
            print("Skipping to prevent infinite runtime. Relying on simpler 2-opt.")
            best_route = np.array(route)
            best_dist = self.calculate_total_distance(best_route)
            n = len(route) - 1 
        
            # Position array for O(1) index lookup
            pos = np.zeros(self.ttp.num_cities + 1, dtype=int)
            pos[best_route[:-1]] = np.arange(n)
            pos[best_route[-1]] = n 
            for pass_num in range(max_passes):
                improved = False
                # Remove the 'break' so we scan the WHOLE route in one pass
                for i in range(n - 1):
                    u = best_route[i]
                    u_next = best_route[i+1]
                    
                    neighbors = self.neighbor_indices[u, :20] 
                    
                    for v in neighbors:
                        if v == u_next: continue
                        j = pos[v]
                        
                        if j <= i + 1 or j >= n: continue
                        v_next = best_route[j+1]
                        
                        d_curr = self.ttp.get_dist(u, u_next) + self.ttp.get_dist(v, v_next)
                        d_new = self.ttp.get_dist(u, v) + self.ttp.get_dist(u_next, v_next)
                        
                        if d_new < d_curr:
                            best_route[i+1:j+1] = best_route[i+1:j+1][::-1]
                            best_dist -= (d_curr - d_new)
                            
                            # Update the pos array for the changed segment
                            # (Costly, but necessary for correctness if we continue)
                            pos[best_route[i+1:j+1]] = np.arange(i+1, j+1)
                            improved = True
                            
                            # DO NOT BREAK HERE. 
                            # Continue to find more swaps in the rest of the route.
                            # We only break the neighbor loop to move to the next 'i'
                            break 
                
                # If we scanned the whole map and found NO improvements, stop early
                if not improved: 
                    break
            return best_route, best_dist
        
        best_route = list(route)
        n = len(best_route) - 1
        improved = True 
        count = 0 
        print(f"Full 2-opt (Max passes: {max_passes})...")
        
        while improved and count<max_passes: 
            improved = False 
            count += 1 
            swaps_in_pass = 0 
            # Iterate every edge (i, i+1) against every other edge (j, j+1)
            for i in range(n-1):           
                A = best_route[i-1]; B = best_route[i] # Warning: Indices slightly offset here, typically A=i, B=i+1
                for j in range(i+2, n):
                    C = best_route[j];   D = best_route[j+1]
                    
                    d_old = self.ttp.get_dist(A, B) + self.ttp.get_dist(C, D)
                    d_new = self.ttp.get_dist(A, C) + self.ttp.get_dist(B, D)
                    
                    if d_new < d_old:
                        best_route[i:j+1] = best_route[i:j+1][::-1]
                        improved = True 
                        swaps_in_pass += 1 
            print(f"Completed Pass {count}: Performed {swaps_in_pass} swaps.")
        final_dist = self.calculate_total_distance(best_route)
        return best_route, final_dist
            
    
    def run(self, max_iterations =100): 
        """
        Main execution loop.
        """
        best_global_route = None
        best_global_distance = float('inf')

        for iteration in range(max_iterations): 
            iter_routes = []
            iter_distances = []

            # 1. Ant Construction Phase
            for ant in range(self.num_ants): 
                route = self.construct_route()
                dist = self.calculate_total_distance(route)

                iter_routes.append(route)
                iter_distances.append(dist)
            
            # 2. Find Best Ant in this iteration
            min_idx = np.argmin(iter_distances)
            iter_best_route = iter_routes[min_idx]
            
            # 3. Educate the Ant (Local Search BEFORE pheromone deposit)
            # This makes the deposited pheromones much stronger
            opt_route, opt_dist = self.fast_two_opt(iter_best_route, max_passes=1)
            
            if opt_dist < best_global_distance: 
                    best_global_distance = opt_dist
                    best_global_route = opt_route

            # 4. Update Pheromones
            self.update_pheromone_trail(opt_route, opt_dist, best_global_route, best_global_distance)
            print(f"finished iteration {iteration}, best distance {best_global_distance}")

        # 5. Final Polish: Run expensive Full 2-Opt on the final result
        print("Running Full 2-OPT optimization...")
        final_route, final_dist = self.full_two_opt(best_global_route)
        print(f"Final Optimization: {best_global_distance} -> {final_dist}")
        return [(final_route, final_dist)]


################## GA ####################


class Solution:
    """
    Class implementation of a solution
    It contains the necessary methods for manipulating individual solutions
    """
    bags = []
    capacity = 0
    mutation_rate = 0.01

    def __init__(self):
        self.chromosome = np.random.randint(0, 2, size=len(Solution.bags))
        self.fitness = None

    def __str__(self):
        return (
            f"Solution(weight={self.get_weight():.2f}, "
            f"value={self.get_value():.2f}, "
            f"fitness={self.fitness:.2f})"
        )

    def get_penalty(self):
        weight = self.get_weight()
        capacity = Solution.capacity

        if weight > capacity:
            return (weight - capacity) ** 3

        return 0

    def get_value(self):
        values = np.array([i for _, i in Solution.bags])
        return np.dot(self.chromosome, values)

    def get_weight(self):
        weights = np.array([i for i, _ in Solution.bags])
        return np.dot(self.chromosome, weights)

    def get_best_ratio_item(self):
        space_remaining = Solution.capacity - self.get_weight()
        best_ratio = None
        index = None

        for i, value in enumerate(self.chromosome):
            if value == 0:
                item_weight = Solution.bags[i][0]
                item_value = Solution.bags[i][1]
                item_ratio = item_value/item_weight
                if item_weight <= space_remaining and (best_ratio is None or item_ratio > best_ratio):
                    index = i
                    best_ratio = item_ratio

        return index

    def get_biggest_fitting(self):
        space_remaining = Solution.capacity - self.get_weight()
        biggest_weight = None
        index = None

        for i, value in enumerate(self.chromosome):
            if value == 0:
                item_weight = Solution.bags[i][0]
                if item_weight <= space_remaining and (biggest_weight is None or item_weight > biggest_weight):
                    index = i
                    biggest_weight = item_weight

        return index

    def calc_fitness(self):
        """
        Fitness evaluation function
        """
        fitness = (self.get_value() - self.get_penalty())
        self.fitness = fitness

    def mutate(self):
        """
        Mutation operator
        """
        for i in range(len(self.chromosome)):
            if random.random() < Solution.mutation_rate:
                self.chromosome[i] = 1 - self.chromosome[i]

    def repair_by_ratio(self):
        """
        Repair operator (DROP + ADD)
        """

        while self.get_weight() > Solution.capacity:
            self.chromosome[self.find_worst_ratio_item()] = 0

        chosen_item = self.get_biggest_fitting()

        while not chosen_item is None:
            self.chromosome[chosen_item] = 1
            chosen_item = self.get_best_ratio_item()

    def find_worst_ratio_item(self):
        index = None
        worst_ratio = None

        for i, value in enumerate(self.chromosome):
            if value == 1:
                item_weight = Solution.bags[i][0]
                item_value = Solution.bags[i][1]

                ratio = item_value/item_weight

                if (worst_ratio is None) or (ratio < worst_ratio):
                    worst_ratio = ratio
                    index = i

        if index is None:
            raise Exception("Empty chromosome")
        else:
            return index


def calc_velocity(weight, w_max, v_max, v_min):
    if weight <= max:
        velocity = v_max - weight/w_max * (v_max - v_min)
    else:
        velocity = v_min
    return velocity


def crossover_weighted(parent1, parent2):
    """
    Crossover operator (weighted crossover)
    """
    fit1 = parent1.fitness
    fit2 = parent2.fitness
    ratio = fit1 / (fit1 + fit2)

    child1 = Solution()
    child2 = Solution()

    for i in range(len(parent1.chromosome)):
        if random.random() < ratio:
            child1.chromosome[i] = parent1.chromosome[i]
        else:
            child1.chromosome[i] = parent2.chromosome[i]

        if random.random() < ratio:
            child2.chromosome[i] = parent1.chromosome[i]
        else:
            child2.chromosome[i] = parent2.chromosome[i]

    return child1, child2


def generate_population(size):
    return [Solution() for _ in range(size)]


def repair_population(population):
    for sol in population:
        sol.repair_by_ratio()


def tournament_selection(population, tournament_size):
    """
    tournament selection
    """
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda s: s.fitness)


def calculate_population_fitness(population):
    for sol in population:
        sol.calc_fitness()


def get_mean_and_best_result(results):
    repair_population(results)
    calculate_population_fitness(results)

    value_total = 0
    best_value = 0
    best_solution = None
    for result in results:
        value = result.get_value()
        value_total += value
        if value > best_value:
            best_value = value
            best_solution = result

    mean_value = value_total / len(results)

    return (best_solution, mean_value)


def paired_test(capacity, bags, seed_list, parameter_tested_values, parameter):
    """
    Main paired test function
    """
    stats = []

    for i, seed in enumerate(seed_list):
        print(i+1, " / ", len(seed_list))
        random.seed(seed)
        stats.append([])

        for par in parameter_tested_values:
            match parameter:
                case "population_size":
                    results = genetic_algorithm(
                        capacity, bags, population_size=par, population_repair=0.8, mutation_rate=0.05, tournament_size=0.8, elite_n=0.2)

                case "population_repair":
                    results = genetic_algorithm(
                        capacity, bags, population_size=50, population_repair=par, mutation_rate=0.05, tournament_size=0.8, elite_n=0.2)

                case "mutation_rate":
                    results = genetic_algorithm(
                        capacity, bags, population_size=50, population_repair=0.8, mutation_rate=par, tournament_size=0.8, elite_n=0.2)

                case "tournament_size":
                    results = genetic_algorithm(
                        capacity, bags, population_size=50, population_repair=0.8, mutation_rate=0.05, tournament_size=par, elite_n=0.2)

                case "elite_n":
                    results = genetic_algorithm(
                        capacity, bags, population_size=50, population_repair=0.8, mutation_rate=0.05, tournament_size=0.8, elite_n=par)

            best, _ = get_mean_and_best_result(results)
            if best:
                best = best.get_value()
                stats[i].append(best)

    per_config_results = [[] for _ in parameter_tested_values]

    for seed in stats:
        for i, val in enumerate(seed):
            per_config_results[i].append(val)

    per_config_means = [float(np.mean(vals)) for vals in per_config_results]
    per_config_deviations = [float(np.std(vals))
                             for vals in per_config_results]

    print("Values: ", parameter_tested_values)
    print("Means: ", per_config_means)
    print("Standard Deviations: ", per_config_deviations)


"""
Functions for comparing configurations
"""


def averaging_test(capacity, bags, seed_list):
    optimals = []
    means = []
    stats = []

    for i, seed in enumerate(seed_list):
        print(i+1, " / ", len(seed_list))
        random.seed(seed)

        results, history = genetic_algorithm(
            capacity, bags, 200, 1.0, 0.05, 0.2, 0.1, generations=100)

        best, mean = get_mean_and_best_result(results)
        if best:
            best = best.get_value()
            optimals.append(best)
            means.append(mean)

        stats.append(history["best"])

    mean = float(np.mean(optimals))
    std_deviation = float(np.std(optimals))

    print("Mean: ", mean)
    print("Standard Deviation: ", std_deviation)

    sns.lineplot(stats, color='skyblue')
    plt.show()

    sns.histplot(means, kde=True, color='skyblue')
    plt.xlabel("Final population mean value")
    plt.ylabel("Density")
    plt.show()

    sns.histplot(optimals, kde=True, color='skyblue')
    plt.xlabel("Fitness Value")
    plt.ylabel("Density")
    plt.show()


def config_comp(capacity, bags, seed_list):
    vars = [[200, 1.0, 0.05, 0.2, 0.1, 10_000], [200, 1.0, 0.08, 0.2, 0.1, 10_000], [
        100, 1.0, 0.05, 0.2, 0.1, 10_000], [1000, 1.0, 0.05, 0.2, 0.1, 10_000]]

    stats = [[] for _ in vars]
    times = [[] for _ in vars]
    for n, i in enumerate(seed_list):
        print(n+1, " / ", len(seed_list))
        random.seed(i)
        for j, values in enumerate(vars):
            start = time.time()
            results, _ = genetic_algorithm(capacity, bags, *values)
            delta = time.time() - start
            times[j].append(delta)
            best, _ = get_mean_and_best_result(results)
            best = float(best.get_value())
            stats[j].append(best)

    for i in range(len(stats)):
        print("\nConfig ", i+1, " mean: ", np.mean(stats[i]))
        print("Config ", i+1, " variance: ", np.std(stats[i]))
        print("Config ", i+1, " average time: ", np.mean(times[i]))

    sns.boxplot(stats)
    plt.xticks(ticks=[i for i in range(len(stats))],
               labels=[i+1 for i in range(len(stats))])
    plt.show()


def run_testing(capacity, bags, seed_list):
    paired_test(capacity, bags, seed_list, [
                10, 50, 100, 200], "population_size")
    paired_test(capacity, bags, seed_list, [
                1.0, 0.8, 0.5], "population_repair")
    paired_test(capacity, bags, seed_list, [
                0.05, 0.01, 0.08, 0.1], "mutation_rate")
    paired_test(capacity, bags, seed_list, [0.8, 0.5, 0.2], "tournament_size")
    paired_test(capacity, bags, seed_list, [0.2, 0.1, 0.3], "elite_n")


def genetic_algorithm(capacity, bags, population_size=200, population_repair=1.0, mutation_rate=0.05, tournament_size=0.2, elite_n=0.1, generations=100):
    """
    Genetic algorithm function
    """

    # Initialisation
    Solution.bags = bags
    Solution.capacity = capacity
    Solution.mutation_rate = mutation_rate
    Solution.fit_counter = 0
    history = {"best": [], "mean": []}

    tournament_n = math.floor(population_size * tournament_size)

    population = generate_population(population_size)
    repair_population(population[:int(population_size * population_repair)])

    # Generation loop
    with tqdm(total=generations) as bar:
        for gen in range(generations):
            calculate_population_fitness(population)
            population = sorted(
                population, key=lambda s: s.fitness, reverse=True)

            # history["best"].append(float(max(s.fitness for s in population)))
            history["best"].append(float(population[0].fitness))
            history["mean"].append(
                float(np.mean([s.fitness for s in population])))

            # elitism operation
            new_population = []
            for i in range(int(population_size*elite_n)):
                new_population.append(population[i])

            while len(new_population) < population_size:
                # tournament selection
                parent1 = tournament_selection(population, tournament_n)
                parent2 = tournament_selection(population, tournament_n)

                # crossover and mutation
                child1, child2 = crossover_weighted(parent1, parent2)
                child1.mutate()
                child2.mutate()

                # ensuring a consistent population size
                if len(new_population) % 2 == 0:
                    new_population.extend([child1, child2])
                else:
                    new_population.append(child1)

            # population replacement and repair
            population = new_population
            repair_population(population[:int(population_size * population_repair)])
            bar.update(1)

    calculate_population_fitness(population)
    population = sorted(population, key=lambda s: s.fitness, reverse=True)

    # history["best"].append(float(max(s.fitness for s in population)))
    history["best"].append(float(population[0].fitness))
    history["mean"].append(float(np.mean([s.fitness for s in population])))

    return population, history





FILENAME = '../resources/a280-n279.txt'

def _calculate_value_density(item):
    """Calculates r_i = p_i / w_i[cite: 1132]."""
    if item['weight'] == 0:
        return float('inf')
    return item['value'] / item['weight']


#GISS Algorithm: Basic implementation 
import numpy as np
import random
from tqdm import tqdm

class GISS_Optimiser: 
    """
    Controller class for the GISS Algorithm.
    """
    def __init__(self, ttp, route):
        self.ttp = ttp
        self.route = route
        self.bags = ttp.items
        self.capacity = ttp.capacity
        
        # Changed to a pre-calculated dictionary (CityID -> Items) to allow for faster lookups, saving time. 
        self.items_map = {}
        for idx, item in enumerate(ttp.items):
            cid = item['city_id']
            if cid not in self.items_map: self.items_map[cid] = []
            self.items_map[cid].append((idx, item['weight']))

        # Pre-calculating route also saves time. 
        self.route_distances = []
        for k in range(len(route) - 1):
            self.route_distances.append(ttp.get_dist(route[k], route[k+1]))

        # Replaced uniform random selection with a position-based probability curve that reflects favour of solutions at the end. 
        city_order_map = {city_id: index for index, city_id in enumerate(route)}
        num_items = len(ttp.items)
        self.item_bias_probs = np.zeros(num_items)
        total_cities = len(route)
        
        for idx, item in enumerate(ttp.items):
            city_id = item['city_id']
            route_pos = city_order_map.get(city_id, 0)
            normalized_pos = route_pos / total_cities
            self.item_bias_probs[idx] = (normalized_pos ** 4)

    def tournament_selection(self, population, tournament_size=5):
        candidates = random.sample(population, tournament_size)
        return max(candidates, key=lambda s: s.fitness)

    def crossover(self, parent1, parent2):
        child = GISS_Solution(self, initialise=False)
        cut_point = random.randint(1, len(parent1.chromosome) - 1)
        child.chromosome = np.concatenate((parent1.chromosome[:cut_point], parent2.chromosome[cut_point:]))
        child.repair()
        return child

    def greedy_pruning(self, solution):
        """
        Added a final local search step. This iterates through selected items 
        and drops them if the rent savings outweigh the item value, 
        fine-tuning the result.
        """
        current_best = solution.fitness
        picked_indices = np.where(solution.chromosome == 1)[0]
        if len(picked_indices) == 0: return solution

        iterator = picked_indices if len(picked_indices) < 1000 else tqdm(picked_indices, desc="Pruning")
        
        for idx in iterator:
            solution.chromosome[idx] = 0
            solution.calc_fitness()
            if solution.fitness > current_best:
                current_best = solution.fitness
            else:
                solution.chromosome[idx] = 1
                solution.fitness = current_best
        return solution

    def run(self, population_size=50, iterations=2000):
        # Switched from a Generational model to a Steady-State model (replace worst), can try both. 
        population = []
        for _ in range(population_size):
            sol = GISS_Solution(self, initialise=True)
            sol.calc_fitness()
            population.append(sol)

        with tqdm(total=iterations, desc="GISS Steady-State") as bar:
            for i in range(iterations):
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
                child = self.crossover(parent1, parent2)
                child.mutate()
                child.calc_fitness()
                
                # replacement of the worst individual
                population.sort(key=lambda s: s.fitness) 
                if child.fitness > population[0].fitness:
                    population[0] = child
                
                bar.update(1)

        population.sort(key=lambda s: s.fitness, reverse=True)
        best_raw = population[0]
        return self.greedy_pruning(best_raw)


class GISS_Solution: 
    def __init__(self, optimiser, initialise=True): 
        self.optimiser = optimiser
        self.fitness = -float('inf')

        if initialise: 
            # Implemented biased initialization using the pre-calculated item_bias_probs, so that chromosomes are more valid to problem.
            rands = np.random.random(len(optimiser.bags))
            thresholds = optimiser.item_bias_probs * 0.05 
            self.chromosome = (rands < thresholds).astype(int)
        else: 
            self.chromosome = np.zeros(len(optimiser.bags), dtype=int)

    def get_weight(self): 
        indices = np.where(self.chromosome==1)[0]
        w = 0
        for i in indices: 
            w+= self.optimiser.bags[i]['weight']
        return w 
    
    def get_value(self):
        indices = np.where(self.chromosome==1)[0]
        v = 0 
        for i in indices: 
            v += self.optimiser.bags[i]['value'] 
        return v 
    
    def repair(self): 
        current_weight = self.get_weight()
        if current_weight > self.optimiser.capacity: 
            ones = np.where(self.chromosome == 1)[0]
            np.random.shuffle(ones)
            for idx in ones: 
                if current_weight <= self.optimiser.capacity: break 
                self.chromosome[idx] = 0 
                current_weight -= self.optimiser.bags[idx]['weight']
    
    def calc_fitness(self): 
        """
        Replaced the standard Knapsack fitness (Value - Weight) with a full TTP simulation.
        This accounts for the renting ratio and velocity degradation over time.
        """
        current_weight = self.get_weight()
        if current_weight > self.optimiser.capacity: 
            self.fitness = -float('inf')
            return 
            
        total_profit = self.get_value()
        total_time = 0 
        current_sack_weight = 0

        vel_span = self.optimiser.ttp.max_speed - self.optimiser.ttp.min_speed
        cap = self.optimiser.capacity

        for i in range(len(self.optimiser.route) - 1):
            # Uses cached distances
            dist = self.optimiser.route_distances[i] 
            
            # Calculates velocity dynamically based on current sack weight
            velocity = self.optimiser.ttp.max_speed - (current_sack_weight / cap) * vel_span
            if velocity < self.optimiser.ttp.min_speed: 
                velocity = self.optimiser.ttp.min_speed
                
            total_time += dist / velocity
            
            # Updates weight only when reaching the specific city
            next_city = self.optimiser.route[i+1]
            if next_city in self.optimiser.items_map:
                for item_idx, item_weight in self.optimiser.items_map[next_city]:
                    if self.chromosome[item_idx] == 1:
                        current_sack_weight += item_weight

        # Final fitness is Profit minus Rent Cost
        self.fitness = total_profit - (total_time * self.optimiser.ttp.renting_ratio)

    def mutate(self):
        """
        Replaced static mutation with dynamic, asymmetric mutation that mytates bested off of the biases from the route.
        """
        n_items = len(self.chromosome)
        base_rate = 1.0 / n_items
        rands = np.random.random(n_items)
        
        # Scale mutation probability by item position
        rate_add_dynamic = base_rate * (self.optimiser.item_bias_probs + 0.1) * 5.0
        rate_drop = 0.5
        
        zeros = (self.chromosome == 0)
        ones = (self.chromosome == 1)
        
        flip_to_one = zeros & (rands < rate_add_dynamic)
        flip_to_zero = ones & (rands < rate_drop)
        
        self.chromosome[flip_to_one] = 1
        self.chromosome[flip_to_zero] = 0
        self.repair()
    
FILENAMES = ['Coursework2/src/resources/a280-n279.txt', 'Coursework2/src/resources/a280-n1395.txt', 'Coursework2/src/resources/a280-n2790.txt', 'Coursework2/src/resources/fnl4461-n4460.txt','Coursework2/src/resources/fnl4461-n22300.txt','Coursework2/src/resources/fnl4461-n44600.txt','Coursework2/src/resources/pla33810-n33809.txt','Coursework2/src/resources/pla33810-n169045.txt','Coursework2/src/resources/pla33810-n338090.txt']
ALT_FILENAMES = ['../resources/a280-n279.txt', '../resources/a280-n1395.txt', '../resources/a280-n2790.txt', '../resources/fnl4461-n4460.txt','../resources/fnl4461-n22300.txt','../resources/fnl4461-n44600.txt','../resources/pla33810-n33809.txt','../resources/pla33810-n169045.txt','../resources/pla33810-n338090.txt']

if __name__ == "__main__":
    for FILENAME in ALT_FILENAMES:
        if os.path.exists(FILENAME):
            print(f"Loading Data: {FILENAME}...")
            cities, items, capacity, min_speed, max_speed, rr = load_ttp_file(FILENAME)

            # 1. Initialize TTP and ACO
            ttp = TTP_Large(cities, items, capacity, min_speed, max_speed, rr)
            # You can tune num_ants and max_iterations here
            gaco = GACO_Large(ttp, num_ants=30, k_neighbors=100) 
                
            # 2. Run ACO
            top_routes = gaco.run(max_iterations=50) 
            best_route = top_routes[0][0]
            best_dist = top_routes[0][1]
            
            print(f"Plotting best route with distance: {best_dist:.2f}")
            plot_gaco_route(cities, best_route, best_dist, FILENAME)

            print("-" * 30)
            print(f"ACO COMPLETE. Optimizing Packing...")
            print("-" * 30)
            
            # 3. Run GISS (Genetic Algorithm)
            # Create the optimizer controller
            giss_opt = GISS_Optimiser(ttp, best_route)
            best_giss_solution = giss_opt.run(population_size=20, iterations=3000)
            
            # 4. Compare vs Empty Baseline
            baseline = GISS_Solution(giss_opt, initialise=False)
            baseline.chromosome[:] = 0
            baseline.calc_fitness()
            
            print("\n" + "="*30)
            print(f"🏆 FINAL RESULTS: {FILENAME} 🏆")
            print("="*30)
            print(f"Empty Bag Score:   {baseline.fitness:.2f}")
            print(f"GISS Best Score:   {best_giss_solution.fitness:.2f}")
            print(f"Profit Gained:     {best_giss_solution.get_value():.2f}")
            print(f"Weight Filled:     {best_giss_solution.get_weight():.2f} / {ttp.capacity}")
            print("="*30)
        else:
            print(f"File not found: {FILENAME}")
                    
#
# if __name__ == "__main__":
#     capacity, bags = read_file()
#
#     seed_list = [i + 30 for i in range(238)]
#     # run_testing(capacity, bags, seed_list)
#     config_comp(capacity, bags, seed_list)
#
#     start = time.time()
#     results, history = genetic_algorithm(
#         capacity, bags, 200, 1.0, 0.05, 0.2, 0.1, generations=100)
#     delta = time.time() - start
#     best, mean = get_mean_and_best_result(results)
#     if best:
#         print(best)
#         print("Best: ", best.get_value(), "\nMean: ", mean, "\nTime: ", delta)
#
#     sns.lineplot(data=history["best"], label='Config 1', color='skyblue')
#
#     plt.xlabel("Generation")
#     plt.ylabel("Best Fitness")
#     plt.legend()
#     plt.show()