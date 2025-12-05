import numpy as np
import os
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import seaborn as sns
import matplotlib.pyplot as plt
import math
import random
from tqdm import tqdm
import time
import sys

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


class GACO_Large: 
    def __init__(self, ttp, num_ants=50, alpha=1.0, beta=8.0, rho=0.1, Q=0.3, k_neighbors=100): 
        """
        Initializes the Ant Colony Optimisation for Large instances.
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
        # KD-Tree allows for fast spatial queries
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
                
                # sample 50 random unvisited nodes and pick the closest one.
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
        Optimised 2-opt Local Search.
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
        print("Running Full 2-OPT optimisation...")
        final_route, final_dist = self.full_two_opt(best_global_route)
        print(f"Final Optimisation: {best_global_distance} -> {final_dist}")
        return [(final_route, final_dist)]
    

import numpy as np
import random
import math
from scipy.spatial.distance import cdist # Required for select_leader

class MOPSO_Particle:
    """
    Represents a single solution in the swarm (a specific set of items picked).
    """
    # 1. Accept a specific capacity fraction (0.0 to 1.0)
    def __init__(self, optimiser, capacity_fraction=1.0):
        self.optimiser = optimiser
        self.dim = len(optimiser.items)
        
        # VIRTUAL LIMIT:
        # Instead of every particle trying to fill the knapsack to 100%,
        # this particle artificially limits itself (e.g., to 50%). 
        # This creates "Sprinter" particles naturally.
        self.capacity_limit = optimiser.capacity * capacity_fraction
        
        # Position is continuous (0.0 to 1.0), converted to binary later
        self.position = np.random.rand(self.dim)
        self.velocity = np.zeros(self.dim)
        
        # Personal Best memory
        self.pbest_position = self.position.copy()
        self.pbest_time = float('inf')   # Minimize Time
        self.pbest_profit = -float('inf') # Maximize Profit
        
        self.current_time = 0
        self.current_profit = 0

    def evaluate(self):
        """
        Converts continuous position to binary plan, enforces constraints, 
        and calculates Time/Profit.
        """
        # Thresholding: > 0.5 means "Attempt to pick item"
        packing_plan = (self.position > 0.5).astype(int)
        current_weight = np.sum(packing_plan * self.optimiser.weights)
        
        # 2. HEURISTIC REPAIR MECHANISM
        # If the particle picked too much (over its virtual or physical limit),
        # we must drop items. We don't drop randomly; we drop the "worst" items.
        if current_weight > self.capacity_limit:
            # Find indices of items currently picked
            selected_indices = np.where(packing_plan == 1)[0]
            
            # Calculate Efficiency Score:
            # Base Ratio = Value / Weight
            # Distance Factor = Cost of carrying it (High at start of tour, Low at end)
            # Efficiency = Base Ratio / Distance Factor
            base_ratios = self.optimiser.values[selected_indices] / (self.optimiser.weights[selected_indices] + 1e-9)
            dist_factors = self.optimiser.item_distance_costs[selected_indices]
            efficiencies = base_ratios / (dist_factors + 1e-9)
            
            # Sort items by efficiency (ascending). We want to drop low efficiency first.
            sorted_args = np.argsort(efficiencies)
            sorted_indices = selected_indices[sorted_args]
            
            # Drop items until we fit inside the limit
            for idx in sorted_indices:
                if current_weight <= self.capacity_limit: 
                    break
                packing_plan[idx] = 0        # Remove from plan
                self.position[idx] = 0.0     # Update continuous position to reflect drop
                current_weight -= self.optimiser.weights[idx]
        
        # 3. Calculate Objectives
        self.current_profit = np.sum(packing_plan * self.optimiser.values)
        # Use the Optimiser's physics engine to calculate time
        self.current_time = self.optimiser.calculate_time(packing_plan, current_weight)

    def update_pbest(self):
        """
        Updates Personal Best using Pareto Dominance logic.
        """
        # Check if current solution is strictly WORSE than pbest
        is_dominated = (self.current_time >= self.pbest_time) and \
                       (self.current_profit <= self.pbest_profit)
        
        # Check if current solution is strictly BETTER than pbest
        dominates_old = (self.current_time <= self.pbest_time) and \
                        (self.current_profit >= self.pbest_profit) and \
                        ((self.current_time < self.pbest_time) or (self.current_profit > self.pbest_profit))

        if dominates_old:
            # If we found a strictly better solution, take it.
            self.pbest_position = self.position.copy()
            self.pbest_time = self.current_time
            self.pbest_profit = self.current_profit
        elif not is_dominated and random.random() < 0.5:
            # If neither dominates (it's a trade-off), 50% chance to update.
            # This helps the particle wander along the Pareto front.
            self.pbest_position = self.position.copy()
            self.pbest_time = self.current_time
            self.pbest_profit = self.current_profit

    def mutate(self, mutation_rate=0.01):
        """
        Randomly flips bits to maintain genetic diversity in the swarm.
        """
        mask = np.random.rand(self.dim) < mutation_rate
        self.position[mask] = 1.0 - self.position[mask]

    def reset_position_with_bias(self, bias_array):
        """
        Used during initialization to seed the particle with a 'smart' starting point
        based on heuristics, rather than pure random noise.
        """
        self.dim = len(bias_array)
        # Add slight noise to the bias so not every particle is identical
        noise = (np.random.rand(self.dim) * 0.2) - 0.1 
        self.position = np.clip(bias_array + noise, 0, 1)
        self.velocity = np.zeros(self.dim)
        
        self.evaluate()
        self.pbest_position = self.position.copy()
        self.pbest_time = self.current_time
        self.pbest_profit = self.current_profit

class MOPSO_Optimiser:
    """
    The Swarm Manager. Handles initialization, physics calculations, 
    and the global archive (Pareto Front).
    """
    def __init__(self, ttp, route):
        self.ttp = ttp
        self.route = route
        self.capacity = ttp.capacity
        self.items = ttp.items
        
        # 1. Standard Vectors for fast numpy math
        self.weights = np.array([i['weight'] for i in ttp.items])
        self.values = np.array([i['value'] for i in ttp.items])
        
        # 2. Pre-calculate Route Distances (Essential for speed in calculate_time)
        self.route_distances = []
        for k in range(len(route) - 1):
            self.route_distances.append(ttp.get_dist(route[k], route[k+1]))
            
        # 3. Map Items to Cities for fast lookup
        self.items_map = {cid: [] for cid in range(ttp.num_cities)}
        for idx, item in enumerate(ttp.items):
            self.items_map[item['city_id']].append((idx, item['weight']))

        # 4. LOCATION-AWARE COST CALCULATION
        # This creates a static penalty array. 
        # An item picked up at the 1st city incurs a high cost (carried for whole tour).
        # An item picked up at the last city incurs low cost.
        city_route_index = {city_id: idx for idx, city_id in enumerate(route)}
        total_cities = len(route)
        
        self.item_distance_costs = np.zeros(len(self.items))
        
        for i, item in enumerate(self.items):
            city_id = item['city_id']
            idx_in_route = city_route_index.get(city_id, 0)
            progress = idx_in_route / total_cities
            
            # Heuristic Adjustment:
            if self.ttp.num_cities < 1000:
                # Small Map: Distance doesn't impact speed enough to worry about "heaviness".
                dist_cost = 1.0
            else:
                # Large Map: Penalize early items heavily using a quadratic curve.
                penalty_curve = (1.0 - progress) ** 2
                dist_cost = 1.0 + (3.0 * penalty_curve)
            
            self.item_distance_costs[i] = dist_cost
            
        # The Archive stores the non-dominated solutions found so far
        self.archive = []

    def get_bias_for_factor(self, tightness_factor):
        """
        Generates a probability array (bias) for initialization.
        tightness_factor: High = Picky (only best items), Low = Greedy (take all).
        """
        valid_weights = self.weights + 1e-9
        ratios = self.values / valid_weights
        mean_ratio = np.mean(ratios)
        
        city_order_map = {city_id: index for index, city_id in enumerate(self.route)}
        total_steps = len(self.route)
        bias_array = np.zeros(len(self.items))

        for idx, item in enumerate(self.items):
            city_id = item['city_id']
            route_pos = city_order_map.get(city_id, 0)
            progression = route_pos / total_steps 
            
            item_ratio = ratios[idx]
            
            # Dynamic Threshold Calculation:
            # We enforce stricter requirements at the start of the tour (progression 0).
            threshold_multiplier = tightness_factor * (1.0 - progression)

            
            required_ratio = mean_ratio * threshold_multiplier

            # Determine initial probability of picking this item
            if item_ratio > required_ratio:
                bias_array[idx] = 0.85 # High chance to pick
            else:
                bias_array[idx] = 0.15 # Low chance to pick
                
        return bias_array

    def initialize_swarm_strategies(self, swarm_size):
        """
        Creates the swarm. Instead of making them all the same, we create a
        spectrum of personalities from 'Sprinters' to 'Heavy Lifters'.
        """
        swarm = []
        for i in range(swarm_size):
            # 1. Distribute Capacity Strategies
            # Particle 0 gets 5% capacity (Sprinter), Particle N gets 100% (Lifter)
            progress = i / swarm_size
            capacity_fraction = 0.05 + (0.95 * progress)
            
            p = MOPSO_Particle(self, capacity_fraction)
            
            # 2. Align Bias with Capacity
            # If you are a Sprinter (low capacity), you must be very PICKY (high factor).
            # If you are a Lifter, you can be GREEDY (low factor).
            factor = (1.0 - progress) * 3.0
            
            bias = self.get_bias_for_factor(factor)
            p.reset_position_with_bias(bias)
            
            self.update_archive(p)
            swarm.append(p)
        return swarm

    def calculate_time(self, packing_plan, total_weight):
        """
        The TTP Physics Engine.
        Calculates total time to traverse the route given the changing weight.
        """
        current_sack_weight = 0
        total_time = 0
        vel_span = self.ttp.max_speed - self.ttp.min_speed
        
        for i in range(len(self.route) - 1):
            dist = self.route_distances[i]
            
            # Standard TTP Formula: Velocity decreases linearly with weight
            velocity = self.ttp.max_speed - (current_sack_weight / self.capacity) * vel_span
            if velocity < self.ttp.min_speed: velocity = self.ttp.min_speed
            
            total_time += dist / velocity
            
            # Add weight of items picked up at the NEXT city
            next_city = self.route[i+1]
            items_at_city = self.items_map[next_city]
            for idx, w in items_at_city:
                if packing_plan[idx] == 1:
                    current_sack_weight += w
        return total_time

    def update_archive(self, particle):
        """
        Maintains the Pareto Front.
        Adds new solution if it's non-dominated. Removes solutions that the new one dominates.
        """
        to_remove = []
        is_dominated = False
        
        p_time = particle.current_time
        p_profit = particle.current_profit
        
        for sol in self.archive:
            s_time, s_profit, _ = sol
            # If existing solution dominates new particle -> Ignore new particle
            if (s_time <= p_time) and (s_profit >= p_profit):
                is_dominated = True
                break
            # If new particle dominates existing solution -> Mark existing for removal
            if (p_time <= s_time) and (p_profit >= s_profit):
                to_remove.append(sol)
        
        if not is_dominated:
            for r in to_remove:
                self.archive.remove(r)
            self.archive.append((p_time, p_profit, particle.position.copy()))
        
        # Keep archive size manageable
        self.prune_archive(max_size=200)

    def prune_archive(self, max_size=200):
        """
        If the archive is too big, remove the most 'crowded' solutions.
        This preserves diversity along the Pareto Front.
        """
        if len(self.archive) <= max_size: return

        # 1. Sort by Time to arrange linearly
        self.archive.sort(key=lambda x: x[0])
        
        # 2. Calculate Crowding Distance
        n = len(self.archive)
        distances = np.zeros(n)
        distances[0] = distances[-1] = float('inf') # Always keep the extremes (Fastest & Richest)
        
        times = [x[0] for x in self.archive]
        profits = [x[1] for x in self.archive]
        time_range = max(times) - min(times)
        profit_range = max(profits) - min(profits)
        
        if time_range == 0 or profit_range == 0: return

        for i in range(1, n-1):
            # Distance is sum of difference to left neighbor and right neighbor
            d_time = (times[i+1] - times[i-1]) / time_range
            d_profit = (profits[i+1] - profits[i-1]) / profit_range 
            distances[i] = d_time + abs(d_profit)

        # 3. Sort by Distance Descending (Keep the most isolated/unique ones)
        enriched = list(zip(self.archive, distances))
        enriched.sort(key=lambda x: x[1], reverse=True)
        
        # 4. Slice to max_size
        self.archive = [x[0] for x in enriched[:max_size]]

    def select_leader(self):
        """
        Selects a Global Best (Leader) for a particle to follow.
        Uses Roulette Wheel Selection based on 'Sparsity'.
        We prefer leaders in empty regions of the objective space to discover new trade-offs.
        """
        if not self.archive: return None
        if len(self.archive) < 3: return random.choice(self.archive)

        costs = np.array([sol[0] for sol in self.archive])
        profits = np.array([sol[1] for sol in self.archive])

        # Normalize objectives
        cost_denom = costs.max() - costs.min()
        if cost_denom == 0: cost_denom = 1.0
        profit_denom = profits.max() - profits.min()
        if profit_denom == 0: profit_denom = 1.0

        norm_costs = (costs - costs.min()) / cost_denom
        norm_profits = (profits - profits.min()) / profit_denom

        points = np.column_stack((norm_costs, norm_profits))

        # Calculate distance matrix between all solutions in archive
        dists = cdist(points, points)
        np.fill_diagonal(dists, float('inf')) 
        
        # Score = distance to nearest neighbor (High score = Isolated = Good Leader)
        sparsity_scores = np.min(dists, axis=1)

        total_sparsity = sparsity_scores.sum()
        if total_sparsity == 0:
            probs = np.ones(len(self.archive)) / len(self.archive)
        else:
            probs = sparsity_scores / total_sparsity

        leader_idx = np.random.choice(len(self.archive), p=probs)
        return self.archive[leader_idx]

    def run(self, swarm_size=100, iterations=100, w=0.5, c1=1.5, c2=1.5, value=0):
        #w is the inerta weight 
        #c1 is the cognative (inidividual) coefficent 
        #c2 is the social coefficent 
        print("Initializing Swarm with Multi-Strategy Heuristics...")
        swarm = self.initialize_swarm_strategies(swarm_size)

        c2 = value
        
        for it in range(iterations):
            for p in swarm:
                leader = self.select_leader()
                if leader is None: continue
                leader_pos = leader[2]
                
                # --- BINARY PSO UPDATE ---
                
                # 1. Update Velocity (Standard PSO formula)
                r1 = np.random.rand(p.dim)
                r2 = np.random.rand(p.dim)
                
                p.velocity = (w * p.velocity) + \
                            (c1 * r1 * (p.pbest_position - p.position)) + \
                            (c2 * r2 * (leader_pos - p.position))
                
                # Clamp velocity so not too extrenuous. 
                p.velocity = np.clip(p.velocity, -6.0, 6.0)

                # 2. Sigmoid Transfer (Velocity -> Probability)
                # S(v) = 1 / (1 + e^-v) gives probability that bit should be 1 as knapsack is binary classification
                prob_is_one = 1 / (1 + np.exp(-p.velocity))

                # 3. Stochastic Decision
                rand_vals = np.random.rand(p.dim)
                p.position = (rand_vals < prob_is_one).astype(float)
                
                # Evaluation and Archiving
                p.evaluate() 
                p.update_pbest()
                self.update_archive(p)
                p.mutate() 
                
            if it % 10 == 0:
                print(f"Iter {it}: Archive Size {len(self.archive)}")
                
        return self.archive

# FILENAMES = ['Coursework2/src/resources/fnl4461-n22300.txt']
FILENAMES = ['../resources/fnl4461-n22300.txt']
ref_points = {
    'a280-n279.txt':      {'min_time': 2613.0,       'max_time': 5444.0,        'min_profit': -42036.0,      'max_profit': -0.0},
    'a280-n1395.txt':     {'min_time': 2613.0,       'max_time': 6573.0,        'min_profit': -489194.0,     'max_profit': -0.0},
    'a280-n2790.txt':     {'min_time': 2613.0,       'max_time': 6646.0,        'min_profit': -1375443.0,    'max_profit': -0.0},
    'fnl4461-n4460.txt':  {'min_time': 185359.0,     'max_time': 442464.0,      'min_profit': -645150.0,     'max_profit': -0.0},
    'fnl4461-n22300.txt': {'min_time': 185359.0,     'max_time': 452454.0,      'min_profit': -7827881.0,    'max_profit': -0.0},
    'fnl4461-n44600.txt': {'min_time': 185359.0,     'max_time': 459901.0,      'min_profit': -22136989.0,   'max_profit': -0.0},
    'pla33810-n33809.txt':{'min_time': 66048945.0,   'max_time': 168432301.0,   'min_profit': -4860715.0,    'max_profit': -0.0},
    'pla33810-n169045.txt':{'min_time': 66048945.0,  'max_time': 169415148.0,   'min_profit': -59472432.0,   'max_profit': -0.0},
    'pla33810-n338090.txt':{'min_time': 66048945.0,  'max_time': 168699977.0,   'min_profit': -168033267.0,  'max_profit': -0.0},
}
        
if __name__ == "__main__":
    all_results = {}

    sizes = [0.5, 0.7, 1.0, 1.5, 2]

    for value in sizes:
        FILENAME = "../resources/fnl4461-n22300.txt"
        if os.path.exists(FILENAME):
            print(f"Loading Data: {FILENAME}...")
            base_name = os.path.basename(FILENAME)
            # Remove extension for clean output naming
            name_only = os.path.splitext(base_name)[0]
            
            cities, items, capacity, min_speed, max_speed, rr = load_ttp_file(FILENAME)

            # 2. Run Optimisation
            # Initialize TTP
            ttp = TTP_Large(cities, items, capacity, min_speed, max_speed, rr)
            
            # Load route
            try:
                best_route = np.loadtxt(f"../../../full_route{base_name}.txt", dtype=int, delimiter=",")
                print(f"Route loaded! Length: {len(best_route)}")
            except OSError:
                print(f"Route file full_route{base_name}.txt not found. Skipping...")
                continue

            print("-" * 30)
            print(f"Optimising Packing (MOPSO) for {base_name}...")
            
            mopso = MOPSO_Optimiser(ttp, best_route)
            # Run MOPSO
            archive = mopso.run(swarm_size=100, iterations=100, value=value)
            
            # 3. Extract Results
            # archive contains tuples: (time, profit, position)
            # We sort by Time for cleaner plotting/file writing
            archive.sort(key=lambda x: x[0])
            
            times = [sol[0] for sol in archive]
            profits = [sol[1] for sol in archive]

            # 4. Save to .f File (Format: Time Profit)
            output_f_file = f"{name_only}-param.f"
            print(f"Saving results to {output_f_file}...")
            with open(output_f_file, "w") as f:
                for t, p in zip(times, profits):
                    # Write Time and Profit separated by space
                    f.write(f"{t:.6f} {int(p)}\n")

            # Store for final plotting
            all_results[base_name+str(value)] = {'times': times, 'profits': profits}

    # plot profit agaisnt time 
    if all_results:

        print(all_results)
        print("Generating Final Comparison Plot...")
        
        num_plots = len(all_results)
        cols = 3
        rows = math.ceil(num_plots / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
        fig.suptitle('Optimisation Results: Negative Profit vs Time', fontsize=16)
        
        # Handle case of single plot (axes is not an array)
        if num_plots == 1:
            axes_flat = [axes]
        else:
            axes_flat = axes.flatten()

        for i, (filename, data) in enumerate(all_results.items()):
            ax = axes_flat[i]
            
            times = data['times']
            profits = data['profits']
            
            # Calculate Negative Profit for plotting
            neg_profits = [-p for p in profits]
            
            # Plot
            ax.scatter(times, neg_profits, c='blue', s=15, alpha=0.7, label='Solutions')
            
            ax.set_title(f"{filename}", fontsize=10, fontweight='bold')
            ax.set_xlabel('Time (Min)', fontsize=8)
            ax.set_ylabel('Negative Profit', fontsize=8)
            ax.grid(True, alpha=0.5)
            
            # Scientific notation if numbers get huge
            ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

        # Hide empty subplots
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    else:
        print("No results to plot.")