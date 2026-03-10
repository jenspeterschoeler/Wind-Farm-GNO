# Article 1 Plotting

This directory contains scripts for generating publication figures for Article 1.

## Directory Structure

```
article_1/
├── cache/                              # Cached data (git-ignored)
│   └── raw_graph_data.pkl              # Raw graph data extracted on Sophia
├── outputs/                            # Generated figures (git-ignored)
├── generate_plot_data.py               # Data extraction for Sophia cluster
├── plot_publication_figures.py         # MASTER script with shared functions + plotting
├── plot_predictions_article.py         # Exploratory plots (imports from master)
├── wt_predictions.py                   # WT error analysis (imports from master)
└── README.md
```

## Workflow

### Step 1: Extract Raw Graph Data (on Sophia)

```bash
# On Sophia cluster where full dataset is available
cd Experiments/article_1
python generate_plot_data.py
```

This extracts the raw graph data for the 4 representative layouts and saves:
- `cache/raw_graph_data.pkl` (~small file, just graph structures)

### Step 2: Copy Cache File Locally

```bash
# From your local machine
scp sophia:<path>/Experiments/article_1/cache/raw_graph_data.pkl ./cache/
```

### Step 3: Generate Publication Figures (locally)

```bash
python plot_publication_figures.py
```

This:
1. Loads the raw graph data
2. Loads the model from `assets/best_model_Vj8/`
3. Generates predictions locally (much faster, no dataset needed)
4. Creates the publication figures

Output in `outputs/`:
- `crossstream_profiles_best_model.pdf` - Cross-stream velocity profiles (improved visibility)
- `wt_spatial_errors_all_layouts_best_model.pdf` - Wind turbine spatial error distribution

## Publication Figures

### Crossstream Profiles (`crossstream_profiles_best_model.pdf`)

Shows velocity deficit profiles at different downstream distances (x/D = 25, 50, 100)
for four layout types (cluster, single string, multiple string, parallel string) at
three wind speeds (6, 12, 18 m/s).

**Visibility improvements**:
- Larger figure size (14" x 20" vs 12" x 16")
- More space for farm layouts (width ratio 2.5 vs 2.0)
- Thicker lines (1.5pt)
- Improved dash pattern (4, 3) vs (2, 2)
- Larger turbine markers
- Legend above figure
- Higher DPI (600)

### WT Spatial Errors (`wt_spatial_errors_all_layouts_best_model.pdf`)

Shows the spatial distribution of prediction errors at wind turbine locations for
all four layout types, aggregated across wind speeds.

## Importing Shared Functions

Other scripts can import shared functions from the master script:

```python
from plot_publication_figures import (
    select_representative_layouts,
    get_max_plot_distance,
    load_article1_model,
    setup_plot_iterator,
    apply_normalizations,
    apply_mask,
)
```

## Legacy Scripts

The original `plot_predictions_article.py` and `wt_predictions.py` scripts have been
cleaned up to:
1. Import shared functions from `plot_publication_figures.py`
2. Remove publication figure generation (now in master script)
3. Keep exploratory/analysis plots for reference

These scripts still require the full dataset and model to run.
