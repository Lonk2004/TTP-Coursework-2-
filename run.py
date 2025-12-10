import random
import numpy as np
import sys
from nsga2_ttp import (TTPFileHandler,NSGA2TTP)

def main(POP_SIZE=500, GENERATIONS=100):
    
    # Set random seeds for reproducibility
    random.seed(333)
    np.random.seed(333)
    INSTANCE_FILE = (sys.argv[1:])[0]

    USE_PARALLEL = True  # Toggle parallel evaluation
    print("NSGA-II for Travelling Thief Problem (Optimized)")
    print(f"Instance: {INSTANCE_FILE}")
    print(f"Population: {POP_SIZE}")
    print(f"Generations: {GENERATIONS}")
    print(f"Parallel: {USE_PARALLEL}")
    
    # Load TTP instance
    ttp_data = TTPFileHandler.load_ttp_file(INSTANCE_FILE)
    print(f"TTP Instance: {ttp_data.config.num_cities} cities, {ttp_data.config.num_items} items")
    print(f"Capacity: {ttp_data.config.capacity}, Speed: [{ttp_data.config.min_speed}, {ttp_data.config.max_speed}]")
    
    # Initialize and run NSGA-II
    print("\nInitializing NSGA2")
    nsga2 = NSGA2TTP(
        data=ttp_data,
        pop_size=POP_SIZE,
        max_gen=GENERATIONS,
        parallel=USE_PARALLEL
    )
    
    # Evolve
    final_population, final_objectives = nsga2.evolve()
    
    # Extract Pareto front
    print("\nExtracting Pareto front...")
    fronts = nsga2.nsga2_core.nondominated_sort(final_objectives)
    pareto_front_indices = fronts[0]
    
    print(f"\nFinal Pareto front size: {len(pareto_front_indices)}")
    print("\nTop 5 solutions:")
    for i, idx in enumerate(pareto_front_indices[:5]):
        tour, bits = final_population[idx]
        profit = -final_objectives[idx][0]
        time = final_objectives[idx][1]
        weight = np.sum(ttp_data.item_weights * bits)
        n_items = int(np.sum(bits))
        print(f"  {i+1}. Profit=${profit:.2f} | Time={time:.3f}s | Weight={weight:.2f} | Items={n_items}")
    
    # Save results
    print("\nSaving results...")
    instance_name = INSTANCE_FILE.replace('.txt', '') + "_p" + str(POP_SIZE) + "_g" + str(GENERATIONS)
    
    # JSON format
    TTPFileHandler.save_pareto_front_json(
        final_population, final_objectives, 
        pareto_front_indices, 
        f'pareto_front_{instance_name}.json'
    )
    
    # Competition format
    TTPFileHandler.save_competition_format(
        final_population, final_objectives,
        pareto_front_indices,
        instance_name,
    )

if __name__ == '__main__':
    main()
