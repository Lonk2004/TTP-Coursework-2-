import re
from pathlib import Path
import matplotlib.pyplot as plt


def extract_best_values(path):
    pattern = re.compile(r"Best Value:\s*([0-9.+\-eE]+)")
    values = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                try:
                    values.append(float(m.group(1)))
                except ValueError:
                    continue
    return values


def main():
    repo_root = Path(__file__).resolve().parents[2]
    file_a = repo_root / "subsequence.txt"
    file_b = repo_root / "oxcrossover.txt"

    vals_a = extract_best_values(file_a) if file_a.exists() else []
    vals_b = extract_best_values(file_b) if file_b.exists() else []

    out_dir = repo_root / "analysis_plots"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "fitness_comparison.png"


    fig, ax = plt.subplots(figsize=(9, 5))

    if vals_a:
        ax.plot(range(1, len(vals_a) + 1), vals_a, marker="o", label="subsequence.txt")
    if vals_b:
        ax.plot(range(1, len(vals_b) + 1), vals_b, marker="s", label="oxcrossover.txt")

    ax.set_xlabel("Run Index")
    ax.set_ylabel("Best Value (lower is better)")
    ax.set_title("Fitness Comparison: subsequence.txt vs oxcrossover.txt")
    ax.legend(loc="best")
    # Invert y-axis so lower fitness values appear higher (lower is better)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
