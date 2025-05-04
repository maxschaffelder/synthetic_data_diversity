import seaborn as sns
import matplotlib as mpl

def set_thesis_style():
    # 1) Base look
    sns.set_style("whitegrid", {
        "grid.color":     "0.90",
        "grid.linestyle": "-",
        "axes.edgecolor": "0.15",
        "axes.linewidth": 1.0
    })
    sns.set_context("paper", font_scale=1.1)

    # 2) Color palette
    palette = [
        "#4C72B0",  # blue
        "#55A868",  # green
        "#C44E52",  # red
        "#8172B2",  # purple
        "#CCB974",  # sand
    ]
    sns.set_palette(palette)

    # 3) Tweak Matplotlib defaults
    mpl.rcParams.update({
        "figure.figsize":     (6, 4),
        "savefig.dpi":        300,
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "xtick.color":        "0.15",
        "ytick.color":        "0.15",
        "text.color":         "0.15",
        "lines.linewidth":    1.8,
        "lines.markeredgewidth": 0.8,
    })