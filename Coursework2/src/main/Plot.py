import os
import math
import matplotlib.pyplot as plt

def parse_result_file(filepath):
    """
    Robust parser: Handles spaces, tabs, or commas as delimiters.
    """
    times = []
    profits = []
    
    if not os.path.exists(filepath):
        print(f"!!! Error: File not found: {filepath}")
        return [], []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or headers
            if not line or line[0].isalpha() or "time" in line.lower():
                continue
            
            # Replace commas with spaces to handle CSVs, then split by whitespace
            clean_line = line.replace(',', ' ')
            parts = clean_line.split()
            
            try:
                # We expect at least 2 numbers (Time, Profit)
                if len(parts) >= 2:
                    t = float(parts[0])
                    p = float(parts[1])
                    times.append(t)
                    profits.append(p)
            except ValueError:
                # Skip lines that look like data but aren't numbers
                continue
                
    return times, profits

def plot_team_comparison():
    # Define your folders
    folder_jack = 'DataFiles/Jack'
    folder_tani_madhaven = 'DataFiles/Tani and Madhaven'

    # Check if folders exist
    if not os.path.exists(folder_jack) or not os.path.exists(folder_tani_madhaven):
        print("Error: Ensure folders 'jack' and 'tani' exist in the current directory.")
        return

    # Get common files
    files_jack = set(os.listdir(folder_jack))
    files_tani = set(os.listdir(folder_tani_madhaven))
    common_files = sorted(list(files_jack.intersection(files_tani)))
    
    # Filter for valid data files (ignore hidden files like .DS_Store)
    common_files = [f for f in common_files if not f.startswith('.')]
    
    # Limit to 6
    files_to_plot = common_files[:6]

    if not files_to_plot:
        print("No matching files found between the two folders.")
        return

    print(f"Processing files: {files_to_plot}")

    # Setup Plot
    cols = 3
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
    fig.suptitle('Pareto Front Comparison: J against T & M(Negative Profit vs Time)', fontsize=16)

    axes_flat = axes.flatten()

    for i, filename in enumerate(files_to_plot):
        ax = axes_flat[i]
        
        # 1. Load Data
        path_jack = os.path.join(folder_jack, filename)
        path_tani = os.path.join(folder_tani_madhaven, filename)
        
        times_jack, profits_jack = parse_result_file(path_jack)
        times_tani, profits_tani = parse_result_file(path_tani)

        # DEBUG CHECK: Warn if data is missing
        if not times_jack:
            print(f"WARNING: No data found for JACK in {filename}")
        if not times_tani:
            print(f"WARNING: No data found for TANI in {filename}")

        # 2. Process Data (Flip Profit)
        neg_profits_jack = [-p for p in profits_jack]
        neg_profits_tani = [-p for p in profits_tani]

        # 3. Plot
        if times_jack:
            ax.scatter(times_jack, neg_profits_jack, c='blue', s=20, alpha=0.6, label='Jack', marker='o')
        if times_tani:
            ax.scatter(times_tani, neg_profits_tani, c='red', s=20, alpha=0.6, label='Tani', marker='^')

        # 4. Styling
        ax.set_title(f"{filename}", fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (Min)', fontsize=8)
        ax.set_ylabel('Negative Profit', fontsize=8)
        ax.grid(True, alpha=0.5)
        
        # Only add legend if data exists
        if times_jack or times_tani:
            ax.legend()
        
        # Scientific notation
        ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

    # Hide unused subplots
    for j in range(len(files_to_plot), len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    plot_team_comparison()