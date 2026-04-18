import seaborn as sns
import matplotlib as mpl

# Module-level constant — importable from other files
WONG_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # pink/magenta
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow (use sparingly on white bg)
    "#000000",  # black
]

def set_thesis_style():
    # 1) Base look
    sns.set_style("whitegrid", {
        "grid.color":     "0.90",
        "grid.linestyle": "-",
        "axes.edgecolor": "0.15",
        "axes.linewidth": 1.0
    })
    sns.set_context("paper", font_scale=1.4)

    # 2) Wong (2011) colorblind-safe palette (Nature Methods, doi:10.1038/nmeth.1618)
    sns.set_palette(WONG_PALETTE)

    # 3) Matplotlib defaults
    mpl.rcParams.update({
        "figure.figsize":        (3.33, 2.5),
        "savefig.dpi":           300,
        "font.family":           "sans-serif",
        "font.sans-serif":       ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "xtick.color":           "0.15",
        "ytick.color":           "0.15",
        "text.color":            "0.15",
        "lines.linewidth":       2.0,
        "lines.markeredgewidth": 0.8,
        "axes.prop_cycle": mpl.cycler(
            color=WONG_PALETTE,
            linestyle=["-", "--", "-.", ":", "-", "--", "-.", ":"],
        ),
    })

# Reference: Wong B. (2011). Color blindness. Nature Methods, 8(6), 441.
# https://doi.org/10.1038/nmeth.1618