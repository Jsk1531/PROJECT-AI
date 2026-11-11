import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_plot(filename):
    """Call this BEFORE plt.show()."""
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {path}")


def save_table(df, filename, index=False):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=index)
    print(f"Saved table: {path}")