import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(patterns=("results_*.csv",)):
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))

    if os.path.exists("all_results.csv") and "all_results.csv" not in files:
        files.append("all_results.csv")

    files = sorted(set(files))

    dfs = {}
    for f in files:
        try:
            df = pd.read_csv(f)
            df["best_value"] = pd.to_numeric(df["best_value"], errors="coerce")
            dfs[f] = df
        except Exception as e:
            print(f"Failed to read {f}: {e}")

    if not dfs:
        raise FileNotFoundError("No CSVs found.")

    return dfs


def plot_file(df, filename, out_dir="analysis_plots"):
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(filename))[0]

    # Rename for plotting (mean fitness)
    df = df.rename(columns={"best_value": "fitness"})

    sns.set(style="whitegrid")

    # --- 1. Mutation Rate vs Fitness ---
    if "mutation_rate" in df.columns:
        plt.figure(figsize=(8,5))
        sns.lineplot(
            data=df,
            x="mutation_rate", y="fitness",
            hue="tournament_size",
            style="population_size",
            markers=True
        )
        plt.title(f"{base} — Mutation Rate vs Fitness")
        plt.xlabel("Mutation Rate")
        plt.ylabel("Fitness (lower is better)")
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.legend(
            title="Tournament Size / Pop Size",
            bbox_to_anchor=(1.05, 1),
            loc="upper left"
        )
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{base}_mutation_rate.png")
        plt.close()

    # --- 2. Tournament Size vs Fitness ---
    if "tournament_size" in df.columns:
        plt.figure(figsize=(8,5))
        sns.lineplot(
            data=df,
            x="tournament_size", y="fitness",
            hue="mutation_rate",
            style="population_size",
            markers=True
        )
        plt.title(f"{base} — Tournament Size vs Fitness")
        plt.xlabel("Tournament Size")
        plt.ylabel("Fitness (lower is better)")
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.legend(
            title="Mutation Rate / Pop Size",
            bbox_to_anchor=(1.05, 1),
            loc="upper left"
        )
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{base}_tournament_size.png")
        plt.close()

    # --- 3. Population Size vs Fitness ---
    if "population_size" in df.columns:
        plt.figure(figsize=(8,5))
        sns.lineplot(
            data=df,
            x="population_size", y="fitness",
            hue="mutation_rate",
            style="tournament_size",
            markers=True
        )
        plt.title(f"{base} — Population Size vs Fitness")
        plt.xlabel("Population Size")
        plt.ylabel("Fitness (lower is better)")
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.legend(
            title="Mutation Rate / Tournament",
            bbox_to_anchor=(1.05, 1),
            loc="upper left"
        )
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{base}_population_size.png")
        plt.close()


def main():
    per_file = load_results()
    for fname, df in per_file.items():
        print(f"Plotting: {fname}")
        plot_file(df, fname)

    print("Saved plots to analysis_plots/")


if __name__ == "__main__":
    main()
