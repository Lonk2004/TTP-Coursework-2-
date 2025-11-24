import numpy as np
import os
from scipy.spatial import KDTree

def load_ttp_file(filepath):
    cities = []
    items = []
    capacity = 0

    #go though each section 
    section = "Header"

    with open(filepath, 'r') as f: 
        for line in f: 
            line = line.strip()
            if not line: 
                continue 
            

            #detect if the section is chaned
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
        """Calculates CEIL_2D distance on the fly."""
        c1 = self.cities[i]
        c2 = self.cities[j]
        dist = np.linalg.norm(c1 - c2)
        return np.ceil(dist)
class GACO_Large: 
    def __init__(self, ttp, num_ants=20, alpha =1, beta=5, Q = 0.3, k_neighbors = 50): 
        self.ttp = ttp
        self.num_ants = num_ants 
        self.alpha = alpha #pheromone importance
        self.beta = beta #distance importance
        self.rho = 0.9 #evapouration rate
        self.Q = Q #selection coefficent
        self.k = k_neighbors #only see k neighbords

        print("Building KD-Tree for neighbors")
        self.tree = KDTree(self.ttp.cities)

        dists, idxs = self.tree.query(self.ttp.cities, k=self.k + 1)

        self.neighbor_indices = idxs[:, 1:]
        self.neighbor_dists = dists[:, 1:]

        #use sparse matrix representation for memory efficency
        self.pheromone_matrix = np.ones((self.ttp.num_cities, self.k))

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
        visited[0] = True 
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
                #pick random unvisited city if no neihbors available
                unvisited_indices = np.where(~visited)[0]
                next_city = np.random.choice(unvisited_indices)
            # 5. Move
            route[step] = next_city
            visited[next_city] = True
            current_city = next_city
            
        # Return to start
        route[-1] = 0
        return route

    def update_pheromone_trail(self, all_ant_routes, all_ant_distances): 
        #evapouration effects
        self.pheromone_matrix *= (1-self.rho)
        #deposit
        #optimisation - only update pheamones for edges that exist
        for route, dist in zip(all_ant_routes, all_ant_distances): 
            deposit = 1.0/ dist if dist > 0 else 0
            #optimisation - only update pheamones for edges that exist
            for i in range(len(route) -1): 
                x = route[i]
                y = route[i+1]

                idx_arr = np.where(self.neighbor_indices[x] == y)[0]
                if len(idx_arr) > 0: 
                    k_idx = idx_arr[0]
                    self.pheromone_matrix[x, k_idx] += deposit

                idx_arr_y = np.where(self.neighbor_indices[y] == x)[0]
                if len(idx_arr_y) > 0: 
                    k_idx_y = idx_arr_y[0]
                    self.pheromone_matrix[y, k_idx_y] += deposit



    def calculate_total_distance(self, route): 
        total = 0.0
        for i in range(len(route) - 1): 
            total += self.ttp.get_dist(route[i], route[i+1])
        return total
    
    def two_opt(self, route, max_passes=3):
        best_route = list(route)
        best_distance = self.calculate_total_distance(best_route)
        
        for _ in range(max_passes):
            improved = False
            for i in range(1, len(best_route) - 2):
                
                # FIX: Limit must be len(best_route) - 1 so that j+1 is valid
                limit = int(np.min([i + 500, len(best_route) - 1]))
                
                for j in range(i + 1, limit): 
                    if j - i == 1: continue
                    
                    A = best_route[i-1]; B = best_route[i]
                    C = best_route[j];   D = best_route[j+1]
                    
                    d_old = self.ttp.get_dist(A, B) + self.ttp.get_dist(C, D)
                    d_new = self.ttp.get_dist(A, C) + self.ttp.get_dist(B, D)
                    
                    if d_new < d_old:
                        best_route[i:j+1] = best_route[i:j+1][::-1]
                        best_distance -= (d_old - d_new)
                        improved = True
                        break 
                if improved: break
            if not improved: break
            
        return best_route, best_distance
    
    def run(self, max_iterations =100): 
        best_global_route = None
        best_global_distance = float('inf')

        for iteration in range(max_iterations): 
            all_routes = []
            all_distances = []

            for ant in range(self.num_ants): 
                route = self.construct_route()
                dist = self.calculate_total_distance(route)

                all_routes.append(route)
                all_distances.append(dist)

                if dist < best_global_distance: 
                    best_global_distance = dist
                    best_global_route = route

            self.update_pheromone_trail(all_routes, all_distances)
            print(f"Iteration {iteration}: Best Distance = {best_global_distance}")

        #finally run 2-opt optimization on best route
        print("Running 2-OPT optimization...")
        final_route, final_dist = self.two_opt(best_global_route)
        return final_route, final_dist
    
FILENAME = '../resources/a280-n279.txt'

def _calculate_value_density(item):
    """Calculates r_i = p_i / w_i[cite: 1132]."""
    if item['weight'] == 0:
        return float('inf')
    return item['value'] / item['weight']
    
if os.path.exists(FILENAME):
    # Load
    print("Loading Data...")
    c, i, cap, min, max, rr = load_ttp_file(FILENAME)
        
    # Init
    ttp = TTP_Large(c, i, cap, min, max, rr)
    gaco = GACO_Large(ttp, num_ants=30) # Paper suggests 30-50 ants [cite: 333]
        
    # Run
    best_route, best_dist = gaco.run(max_iterations=50) # Lowered iterations for testing

        
    print("-" * 30)
    print(f"OPTIMIZATION COMPLETE")
    print(f"Final Best Distance: {best_dist}")
    print(f"Route Preview: {best_route[:10]}... -> {best_route[-10:]}")
    print("-" * 30)
else:
    print(f"Error: File {FILENAME} not found.")
                    

        






