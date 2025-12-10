"""
Optimized NSGA-II for Travelling Thief Problem (TTP)
- Parallel fitness evaluation using multiprocessing
- On-demand distance calculation (no precomputed matrix for large instances)
- NumPy-based operations for speed
- Modular class-based architecture
"""

import numpy as np
import random
import json
from typing import List, Tuple, Dict
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from functools import partial


@dataclass
class TTPConfig:
    """Configuration parameters for TTP instance"""
    num_cities: int
    num_items: int
    capacity: float
    max_speed: float
    min_speed: float
    renting_ratio: float


@dataclass
class TTPData:
    """NumPy-based storage for TTP problem data"""
    cities_coords: np.ndarray  # Shape (n_cities, 2) - [x, y] coordinates
    item_profits: np.ndarray   # Shape (n_items,) - profit values
    item_weights: np.ndarray   # Shape (n_items,) - weight values
    item_cities: np.ndarray    # Shape (n_items,) - city assignment (indices)
    config: TTPConfig
    
    # Optional: distance matrix for small instances
    distance_matrix: np.ndarray = None
    use_precomputed_distances: bool = False


class DistanceCalculator:
    """Handles distance calculations - precomputed or on-demand"""
    
    def __init__(self, coords: np.ndarray, precompute_threshold: int = 5000):
        """
        Args:
            coords: NumPy array of shape (n_cities, 2)
            precompute_threshold: If cities < this, precompute matrix
        """
        self.coords = coords
        self.n_cities = len(coords)
        self.use_matrix = self.n_cities < precompute_threshold
        
        if self.use_matrix:
            self.distance_matrix = self._compute_distance_matrix()
            print(f"Precomputed distance matrix ({self.n_cities}x{self.n_cities})")
        else:
            self.distance_matrix = None
            print(f"Using on-demand distance calculation (cities={self.n_cities})")
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """Vectorized distance matrix computation"""
        # Broadcast: (n,1,2) - (1,n,2) = (n,n,2)
        diff = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=2))
    
    def get_distance(self, city_i: int, city_j: int) -> float:
        """
        Get distance between two cities.
        Uses matrix if available, computes on-demand otherwise.
        """
        if self.use_matrix:
            return self.distance_matrix[city_i, city_j]
        else:
            # On-demand calculation for large instances
            diff = self.coords[city_i] - self.coords[city_j]
            return np.linalg.norm(diff)
    
    def get_distances_along_tour(self, tour: List[int]) -> np.ndarray:
        """
        Get all edge distances for a tour.
        Returns array of shape (n_cities,) with distances between consecutive cities.
        """
        if self.use_matrix:
            tour_array = np.array(tour)
            from_cities = tour_array
            to_cities = np.roll(tour_array, -1) 
            return self.distance_matrix[from_cities, to_cities]
        else:
            tour_coords = self.coords[tour]
            next_coords = np.roll(tour_coords, -1, axis=0)
            diffs = tour_coords - next_coords
            return np.linalg.norm(diffs, axis=1)


class TTPEvaluator:
    """Handles fitness evaluation for TTP solutions"""
    
    def __init__(self, data: TTPData, dist_calc: DistanceCalculator):
        self.data = data
        self.dist_calc = dist_calc
        self.config = data.config
        
        # Precompute items per city for faster lookup
        self.items_by_city = self._build_items_by_city_index()
    
    def _build_items_by_city_index(self) -> Dict[int, List[int]]:
        """Build mapping of city_idx -> [item_idx, ...]"""
        items_map = {}
        for item_idx, city_idx in enumerate(self.data.item_cities):
            items_map.setdefault(city_idx, []).append(item_idx)
        return items_map
    
    def evaluate(self, tour: List[int], item_bits: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate TTP solution.
        
        Args:
            tour: List of city indices (permutation)
            item_bits: NumPy array of shape (n_items,) with 0/1 values
        
        Returns:
            (-profit, travel_time) tuple (NSGA-II minimizes both)
        """
        # NEW: NumPy vectorized profit calculation
        total_profit = np.sum(self.data.item_profits * item_bits)
        
        # Travel time with weight-dependent velocity
        travel_time = self._calculate_travel_time(tour, item_bits)
        
        # Return as minimization objectives
        return float(-total_profit), float(travel_time)
    
    def _calculate_travel_time(self, tour: List[int], item_bits: np.ndarray) -> float:
        """
        Calculate travel time considering weight accumulation.
        
        OLD: Loop through tour, update weight at each city
        NEW: Vectorized with NumPy for speed
        """
        n_cities = len(tour)
        tour_array = np.array(tour)
        
        # Get distances along tour
        distances = self.dist_calc.get_distances_along_tour(tour)
        
        # Calculate weight at each city (cumulative pickup)
        weights_at_cities = np.zeros(n_cities)
        
        for pos, city_idx in enumerate(tour):
            # Items picked at this city
            if city_idx in self.items_by_city:
                item_indices = self.items_by_city[city_idx]
                picked_items = item_bits[item_indices]
                picked_weights = self.data.item_weights[item_indices] * picked_items
                weight_picked_here = np.sum(picked_weights)
            else:
                weight_picked_here = 0.0
            
            # Cumulative weight (carry weight from previous cities + pick new items)
            if pos == 0:
                weights_at_cities[pos] = weight_picked_here
            else:
                weights_at_cities[pos] = weights_at_cities[pos-1] + weight_picked_here
        
        # Velocity at each edge (function of weight carried)
        # v = v_max - (current_weight / capacity) * (v_max - v_min)
        velocities = (
            self.config.max_speed - 
            (weights_at_cities / self.config.capacity) * 
            (self.config.max_speed - self.config.min_speed)
        )
        
        # Prevent division by zero or negative velocity
        velocities = np.maximum(velocities, 1e-6)
        
        # Travel time = distance / velocity
        travel_times = distances / velocities
        
        return float(np.sum(travel_times))

class ParallelEvaluator:
    """
    Handles parallel fitness evaluation using multiprocessing.
    
    NEW: Added parallel processing for large populations
    """
    
    def __init__(self, evaluator: TTPEvaluator, n_workers: int = None):
        """
        Args:
            evaluator: TTPEvaluator instance
            n_workers: Number of parallel workers (default: CPU count - 1)
        """
        self.evaluator = evaluator
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        print(f"Parallel evaluator initialized with {self.n_workers} workers")
    
    def evaluate_population(self, population: List[Tuple[List[int], np.ndarray]]) -> List[Tuple[float, float]]:
        """
        Evaluate entire population in parallel.
        
        Args:
            population: List of (tour, item_bits) tuples
        
        Returns:
            List of (obj1, obj2) tuples
        """
        if len(population) < 10:
            # OLD: Serial evaluation for tiny populations
            return [self.evaluator.evaluate(tour, bits) for tour, bits in population]
        
        # NEW: Parallel evaluation for larger populations
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Map evaluation across workers
            results = list(executor.map(
                self._evaluate_single,
                population,
                chunksize=max(1, len(population) // self.n_workers)
            ))
        
        return results
    
    def _evaluate_single(self, individual: Tuple[List[int], np.ndarray]) -> Tuple[float, float]:
        """Wrapper for single evaluation (for multiprocessing)"""
        tour, bits = individual
        return self.evaluator.evaluate(tour, bits)

class GeneticOperators:
    """Genetic operators for TTP (tour + item bitstring)"""
    
    @staticmethod
    def order_crossover(p1_tour: List[int], p2_tour: List[int]) -> Tuple[List[int], List[int]]:
        """
        Order Crossover (OX) for TSP tours.
        Preserves relative order from parents.
        """
        n = len(p1_tour)
        a = random.randint(0, n-2)
        b = random.randint(a+1, n-1)
        
        def ox_single(parent_a, parent_b):
            child = [-1] * n
            # Copy slice from parent_a
            child[a:b+1] = parent_a[a:b+1]
            # Fill remaining by order from parent_b
            pb_remaining = [c for c in parent_b if c not in child[a:b+1]]
            ci = 0
            for i in range(n):
                if child[i] == -1:
                    child[i] = pb_remaining[ci]
                    ci += 1
            return child
        
        return ox_single(p1_tour, p2_tour), ox_single(p2_tour, p1_tour)
    
    
    def two_opt_mutation(self, tour, num_iterations=5):
        """Apply 2-opt local search: try improving random pairs."""
        tour = list(tour)
        
        for _ in range(num_iterations):
            improved = False
            
            # Random 2-opt moves
            for _ in range(10):  # Try 10 random swaps
                i, j = sorted(random.sample(range(len(tour)), 2))
                
                # Check if reversing improves distance
                # (simplified: just apply it, NSGA-II will keep if better)
                tour[i:j] = reversed(tour[i:j])
                improved = True
            
            if not improved:
                break
        
        return tour
    
    @staticmethod
    def uniform_crossover_bits(b1: np.ndarray, b2: np.ndarray, p_swap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uniform crossover for item bitstrings.
        """
        n = len(b1)
        swap_mask = np.random.random(n) < p_swap
        
        c1 = np.where(swap_mask, b2, b1)
        c2 = np.where(swap_mask, b1, b2)
        
        return c1, c2
    
    @staticmethod
    def bitflip_mutation(bits: np.ndarray, p_mut: float = 0.01) -> np.ndarray:
        """
        """
        flip_mask = np.random.random(len(bits)) < p_mut
        return np.where(flip_mask, 1 - bits, bits)


class CapacityRepairer:
    """Repairs infeasible solutions (over capacity)"""
    
    @staticmethod
    def repair_by_ratio(bits: np.ndarray, weights: np.ndarray, 
                       profits: np.ndarray, capacity: float) -> np.ndarray:
        """
        Repair by removing items with worst profit/weight ratio.
        NEW: NumPy vectorized for speed.
        
        Args:
            bits: Item selection bitstring
            weights: Item weights array
            profits: Item profits array
            capacity: Knapsack capacity
        
        Returns:
            Repaired bitstring
        """
        total_weight = np.sum(weights * bits)
        
        if total_weight <= capacity:
            return bits.copy()
        
        # Calculate profit/weight ratios
        ratios = np.where(weights > 0, profits / weights, np.inf)
        
        # Get indices of picked items, sorted by ratio (ascending)
        picked_indices = np.where(bits == 1)[0]
        picked_ratios = ratios[picked_indices]
        sorted_order = np.argsort(picked_ratios)  # ascending
        sorted_picked = picked_indices[sorted_order]
        
        # Remove items with worst ratios until feasible
        bits_repaired = bits.copy()
        for idx in sorted_picked:
            bits_repaired[idx] = 0
            total_weight -= weights[idx]
            if total_weight <= capacity:
                break
        
        return bits_repaired

class NSGA2Core:
    """Core NSGA-II algorithm (non-dominated sorting, crowding, selection)"""
    
    @staticmethod
    def nondominated_sort(objectives: List[Tuple[float, float]]) -> List[List[int]]:
        """
        Fast non-dominated sorting.
        Returns list of fronts (each front is list of individual indices).
        
        Objectives: list of (obj1, obj2) where smaller is better
        """
        N = len(objectives)
        
        # Domination relationships
        S = [set() for _ in range(N)]  # S[p] = set of individuals dominated by p
        n = [0] * N  # n[p] = number of individuals dominating p
        
        # Compute domination
        for p in range(N):
            for q in range(N):
                if p == q:
                    continue
                
                # Check if p dominates q
                p_better_all = all(objectives[p][k] <= objectives[q][k] for k in range(2))
                p_better_some = any(objectives[p][k] < objectives[q][k] for k in range(2))
                
                if p_better_all and p_better_some:
                    S[p].add(q)  # p dominates q
                
                # Check if q dominates p
                q_better_all = all(objectives[q][k] <= objectives[p][k] for k in range(2))
                q_better_some = any(objectives[q][k] < objectives[p][k] for k in range(2))
                
                if q_better_all and q_better_some:
                    n[p] += 1  # q dominates p
        
        # Build fronts
        fronts = []
        F1 = [i for i in range(N) if n[i] == 0]
        fronts.append(F1)
        
        i = 0
        while i < len(fronts) and fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)
        
        return fronts
    
    @staticmethod
    def crowding_distance(front: List[int], objectives: List[Tuple[float, float]]) -> Dict[int, float]:
        """
        Calculate crowding distance for individuals in a front.
        Returns dict mapping index -> crowding distance.
        """
        distance = {i: 0.0 for i in front}
        
        if len(front) <= 2:
            # Boundary solutions get infinite distance
            for i in front:
                distance[i] = float('inf')
            return distance
        
        # For each objective
        for m in range(2):
            # Sort front by objective m
            sorted_front = sorted(front, key=lambda idx: objectives[idx][m])
            
            # Boundary solutions
            distance[sorted_front[0]] = float('inf')
            distance[sorted_front[-1]] = float('inf')
            
            # Objective range
            obj_min = objectives[sorted_front[0]][m]
            obj_max = objectives[sorted_front[-1]][m]
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue  # No variation in this objective
            
            # Crowding distance for interior solutions
            for i in range(1, len(sorted_front) - 1):
                prev_obj = objectives[sorted_front[i-1]][m]
                next_obj = objectives[sorted_front[i+1]][m]
                distance[sorted_front[i]] += (next_obj - prev_obj) / obj_range
        
        return distance
    
    @staticmethod
    def calculate_niche_penalties(objectives: List[Tuple[float, float]], 
                              sigma_share: float = 0.15) -> Dict[int, float]:
        """Calculate niche penalties (crowding in objective space)."""
        n = len(objectives)
        niche_penalties = {i: 1.0 for i in range(n)}
        
        # Normalize objectives
        times = np.array([obj[0] for obj in objectives])      # Extract first element (time)
        profits = np.array([obj[1] for obj in objectives])    # Extract second element (profit)

        time_min, time_max = np.min(times), np.max(times)
        profit_min, profit_max = np.min(profits), np.max(profits)
        
        time_range = time_max - time_min if time_max > time_min else 1.0
        profit_range = profit_max - profit_min if profit_max > profit_min else 1.0
        
        times_norm = (times - time_min) / time_range
        profits_norm = (profits - profit_min) / profit_range
        
        # Niche count
        for i in range(n):
            niche_count = 0.0
            for j in range(n):
                if i != j:
                    dist = np.sqrt((times_norm[i] - times_norm[j])**2 + 
                                (profits_norm[i] - profits_norm[j])**2)
                    if dist < sigma_share:
                        niche_count += 1.0 - (dist / sigma_share)**2
            
            niche_penalties[i] = 1.0 + niche_count
        
        return niche_penalties

    @staticmethod
    def binary_tournament(population_size: int, ranks: List[int], 
                     crowd_dist: Dict[int, float], 
                     niche_penalties: Dict[int, float] = None) -> int:
        """Binary tournament selection with optional niche penalties."""
        a = random.randrange(population_size)
        b = random.randrange(population_size)
        
        # Compare by rank first
        if ranks[a] < ranks[b]:
            return a
        elif ranks[b] < ranks[a]:
            return b
        
        # Rank equal - now include niche penalty
        da = crowd_dist.get(a, 0.0)
        db = crowd_dist.get(b, 0.0)
        
        # Apply niche penalties if provided (lower penalty is better)
        if niche_penalties is not None:
            penalty_a = niche_penalties.get(a, 1.0)
            penalty_b = niche_penalties.get(b, 1.0)
            
            # Prefer solutions with LOWER niche penalties
            if penalty_a < penalty_b:
                return a
            elif penalty_b < penalty_a:
                return b
    
        # Fall back to crowding distance
        if da > db:
            return a
        elif db > da:
            return b
        else:
            return a if random.random() < 0.5 else b


class NSGA2TTP:
    """
    Complete NSGA-II algorithm for TTP.
    Modular design with clear separation of concerns.
    """
    
    def __init__(self, data: TTPData, pop_size: int = 100, 
                 max_gen: int = 200, parallel: bool = True):
        """
        Initialize NSGA-II for TTP.
        
        Args:
            data: TTPData instance
            pop_size: Population size
            max_gen: Maximum generations
            parallel: Use parallel evaluation (recommended for large problems)
        """
        self.data = data
        self.config = data.config
        self.pop_size = pop_size
        self.max_gen = max_gen
        
        # Initialize modules
        self.dist_calc = DistanceCalculator(data.cities_coords)
        self.evaluator = TTPEvaluator(data, self.dist_calc)
        
        # NEW: Parallel or serial evaluation
        if parallel:
            self.parallel_eval = ParallelEvaluator(self.evaluator)
        else:
            self.parallel_eval = None
        
        self.operators = GeneticOperators()
        self.repairer = CapacityRepairer()
        self.nsga2_core = NSGA2Core()
        
        # Algorithm parameters
        self.p_swap_bits = 0.7
        self.p_mut_bit = 0.02
        self.p_mut_tour = 0.3
        self.mutation_ratio = 0.05 
    
    def initialize_population(self):
        population = []
        
        # Mix of random (50%) and heuristic (50%) solutions
        for i in range(self.pop_size):
            if i < self.pop_size // 2:
                # Heuristic: Nearest neighbor tour
                tour = self._nearest_neighbor_tour()
                bits = self._greedy_knapsack()
            else:
                # Random (for diversity)
                tour = list(range(self.config.num_cities))
                random.shuffle(tour)
                bits = np.random.randint(0, 2, self.config.num_items)
                bits = self.repairer.repair_by_ratio(bits, self.data.item_weights, self.data.item_profits, self.config.capacity)#bits, self.data.item_profits, self.config.capacity, tour)
            
            population.append((tour, bits))
        
        return population

    def _nearest_neighbor_tour(self):
        """Build tour using nearest neighbor heuristic."""
        unvisited = set(range(1, self.config.num_cities))
        current = 0
        tour = [current]
        
        while unvisited:
            # Find nearest unvisited city
            nearest = min(unvisited, key=lambda c: self.dist_calc.get_distance(current, c))
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        return tour

    def _greedy_knapsack(self):
        """Fill knapsack greedily by value/weight ratio."""
        # Sort items by value/weight ratio (descending)
        value_weight_ratio = self.data.item_profits / (self.data.item_weights + 1e-6)
        sorted_indices = np.argsort(-value_weight_ratio)  # Descending
        
        bits = np.zeros(self.config.num_items, dtype=int)
        current_weight = 0
        
        for idx in sorted_indices:
            if current_weight + self.data.item_weights[idx] <= self.config.capacity:
                bits[idx] = 1
                current_weight += self.data.item_weights[idx]
            else:
                break  # Capacity exceeded
        
        return bits

    
    def evaluate_population_wrapper(self, population: List[Tuple[List[int], np.ndarray]]) -> List[Tuple[float, float]]:
        """
        Evaluate population (parallel or serial).
        NEW: Automatically chooses parallel vs serial based on initialization.
        """
        if self.parallel_eval is not None:
            return self.parallel_eval.evaluate_population(population)
        else:
            return [self.evaluator.evaluate(tour, bits) for tour, bits in population]
    
    def create_offspring(self, population: List[Tuple[List[int], np.ndarray]], objectives,
                        ranks: List[int], crowd_dist: Dict[int, float]) -> List[Tuple[List[int], np.ndarray]]:
        """
        Create offspring via selection, crossover, mutation.
        """
        offspring = []

        niche_penalties = self.nsga2_core.calculate_niche_penalties(objectives, sigma_share=0.25)
        
        while len(offspring) < self.pop_size:
            # Selection
            p1_idx = self.nsga2_core.binary_tournament(len(population), ranks, crowd_dist, niche_penalties)
            p2_idx = self.nsga2_core.binary_tournament(len(population), ranks, crowd_dist, niche_penalties)
            
            parent1_tour, parent1_bits = population[p1_idx]
            parent2_tour, parent2_bits = population[p2_idx]
            
            # Crossover - Tours
            child1_tour, child2_tour = self.operators.order_crossover(parent1_tour, parent2_tour)
            
            # Crossover - Items
            child1_bits, child2_bits = self.operators.uniform_crossover_bits(
                parent1_bits, parent2_bits, self.p_swap_bits
            )
            
            # Mutation - Tours
            if random.random() < self.p_mut_tour:
                child1_tour = self.operators.two_opt_mutation(child1_tour, num_iterations=3)
            if random.random() < self.p_mut_tour:
                child2_tour = self.operators.two_opt_mutation(child2_tour, num_iterations=3)
            
            # Mutation - Items
            child1_bits = self.operators.bitflip_mutation(child1_bits, self.p_mut_bit)
            child2_bits = self.operators.bitflip_mutation(child2_bits, self.p_mut_bit)
            
            # Repair
            child1_bits = self.repairer.repair_by_ratio(
                child1_bits, self.data.item_weights, 
                self.data.item_profits, self.config.capacity
            )
            child2_bits = self.repairer.repair_by_ratio(
                child2_bits, self.data.item_weights,
                self.data.item_profits, self.config.capacity
            )
            
            offspring.append((child1_tour, child1_bits))
            if len(offspring) < self.pop_size:
                offspring.append((child2_tour, child2_bits))
        
        return offspring
    
    def environmental_selection(self, combined_pop: List[Tuple[List[int], np.ndarray]],
                               combined_objs: List[Tuple[float, float]]) -> Tuple[List, List]:
        """
        Environmental selection: keep best pop_size individuals.
        Uses non-dominated sorting and crowding distance.
        """
        fronts = self.nsga2_core.nondominated_sort(combined_objs)
        
        new_population = []
        new_objectives = []
        
        for front in fronts:
            if len(new_population) + len(front) <= self.pop_size:
                # Add entire front
                for idx in front:
                    new_population.append(combined_pop[idx])
                    new_objectives.append(combined_objs[idx])
            else:
                # Fill remaining slots by crowding distance
                crowd_dist_map = self.nsga2_core.crowding_distance(front, combined_objs)
                sorted_front = sorted(front, key=lambda idx: crowd_dist_map[idx], reverse=True)
                
                remaining = self.pop_size - len(new_population)
                for idx in sorted_front[:remaining]:
                    new_population.append(combined_pop[idx])
                    new_objectives.append(combined_objs[idx])
                break
        
        return new_population, new_objectives
    
    def evolve(self) -> Tuple[List[Tuple[List[int], np.ndarray]], List[Tuple[float, float]]]:
        """
        Main NSGA-II evolution loop.
        
        Returns:
            (final_population, final_objectives)
        """
        print(f"Initializing population (size={self.pop_size})...")
        population = self.initialize_population()
        
        print("Evaluating initial population...")
        objectives = self.evaluate_population_wrapper(population)
        
        print(f"Starting evolution ({self.max_gen} generations)...")
        for gen in range(self.max_gen):
            # Compute ranks and crowding for current population
            fronts = self.nsga2_core.nondominated_sort(objectives)
            
            ranks = [None] * len(population)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            
            crowd_dist = {}
            for front in fronts:
                cd = self.nsga2_core.crowding_distance(front, objectives)
                crowd_dist.update(cd)
            
            # Create offspring
            offspring = self.create_offspring(population, objectives, ranks, crowd_dist)
            
            num_mutants = max(5, int(0.05 * self.pop_size))  # 5% mutants
            for _ in range(num_mutants):
                random_tour = list(range(self.config.num_cities))
                random.shuffle(random_tour)
                random_bits = np.random.randint(0, 2, self.config.num_items)
                random_bits = self.repairer.repair_by_ratio(
                    random_bits, self.data.item_weights, 
                    self.data.item_profits, self.config.capacity
                )
            offspring.append((random_tour, random_bits))

            # Evaluate offspring
            offspring_objs = self.evaluate_population_wrapper(offspring)
            
            # Combine and select
            combined_pop = population + offspring
            combined_objs = objectives + offspring_objs
            
            population, objectives = self.environmental_selection(combined_pop, combined_objs)
            
            # Progress report
            if gen % max(1, self.max_gen // 10) == 0 or gen == self.max_gen - 1:
                fronts_final = self.nsga2_core.nondominated_sort(objectives)
                print(f"Gen {gen+1}/{self.max_gen} | Pop={len(population)} | Front0={len(fronts_final[0])}")
        
        print("Evolution complete!")
        return population, objectives


class TTPFileHandler:
    """Handles loading TTP files and saving results"""
    
    @staticmethod
    def load_ttp_file(filepath: str) -> TTPData:
        """
        Load TTP instance from file.
        Returns TTPData with NumPy arrays.
        """
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        
        # Parse header
        config_dict = {}
        for line in lines:
            if line.startswith("DIMENSION"):
                config_dict['num_cities'] = int(line.split(":")[1])
            elif line.startswith("NUMBER OF ITEMS"):
                config_dict['num_items'] = int(line.split(":")[1])
            elif line.startswith("CAPACITY"):
                config_dict['capacity'] = float(line.split(":")[1])
            elif line.startswith("MIN SPEED"):
                config_dict['min_speed'] = float(line.split(":")[1])
            elif line.startswith("MAX SPEED"):
                config_dict['max_speed'] = float(line.split(":")[1])
            elif line.startswith("RENTING RATIO"):
                config_dict['renting_ratio'] = float(line.split(":")[1])
        
        config = TTPConfig(**config_dict)
        
        # Find city coordinates section
        node_start = None
        for i, line in enumerate(lines):
            if line.startswith("NODE_COORD_SECTION"):
                node_start = i + 1
                break
        
        node_end = node_start + config.num_cities
        
        # Load cities into NumPy array
        cities_coords = np.zeros((config.num_cities, 2))
        for i, line in enumerate(lines[node_start:node_end]):
            parts = line.split()
            cities_coords[i, 0] = float(parts[1])  # x
            cities_coords[i, 1] = float(parts[2])  # y
        
        # Find item section
        item_start = node_end
        while item_start < len(lines):
            first = lines[item_start].split()[0]
            if first.isdigit():
                break
            item_start += 1
        
        # Load items into NumPy arrays
        item_profits = np.zeros(config.num_items)
        item_weights = np.zeros(config.num_items)
        item_cities = np.zeros(config.num_items, dtype=int)
        
        for i, line in enumerate(lines[item_start:item_start + config.num_items]):
            parts = line.split()
            item_profits[i] = float(parts[1])
            item_weights[i] = float(parts[2])
            item_cities[i] = int(parts[3]) - 1  # Convert to 0-indexed
        
        return TTPData(
            cities_coords=cities_coords,
            item_profits=item_profits,
            item_weights=item_weights,
            item_cities=item_cities,
            config=config
        )
    
    @staticmethod
    def save_pareto_front_json(population: List[Tuple[List[int], np.ndarray]],
                               objectives: List[Tuple[float, float]],
                               pareto_indices: List[int],
                               filepath: str):
        """Save Pareto front as JSON"""
        pareto_output = []
        for idx in pareto_indices:
            tour, bits = population[idx]
            pareto_output.append({
                'profit': float(-objectives[idx][0]),  # Convert back to positive
                'time': float(objectives[idx][1]),
                'tour': [int(x) for x in tour],
                'items': [int(i) for i, b in enumerate(bits) if b == 1]
            })
        
        with open(filepath, 'w') as f:
            json.dump(pareto_output, f, indent=2)
        
        print(f"Saved Pareto front to {filepath}")
    
    @staticmethod
    def save_competition_format(population: List[Tuple[List[int], np.ndarray]],
                                objectives: List[Tuple[float, float]],
                                pareto_indices: List[int],
                                instance_name: str):
        """
        Save results in competition format (.x and .f files).
        
        .x file: decision variables (tour + items)
        .f file: objective values (profit, time)
        """
        # .x file (decision variables)
        x_filename = f"{instance_name}.x"
        with open(x_filename, 'w') as f:
            for idx in pareto_indices:
                tour, bits = population[idx]
                tour_str = ' '.join(str(c + 1) for c in tour)
                bits_str = ' '.join(str(int(b)) for b in bits)
                f.write(f"{tour_str}\n{bits_str}\n\n")
        
        # .f file (objectives)
        f_filename = f"{instance_name}.f"
        with open(f_filename, 'w') as f:
            for idx in pareto_indices:
                time = objectives[idx][1]
                profit = -objectives[idx][0]  # Convert to positive
                f.write(f"{time} {profit}\n")
        
        print(f"Saved competition files: {x_filename}, {f_filename}")