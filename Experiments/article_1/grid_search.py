import json
import os

import numpy as np
import pandas as pd

import wandb

# Get absolute path to assets directory
script_dir = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.abspath(os.path.join(script_dir, "..", "..", "assets"))

# Load scale stats for unscaling metrics
scale_stats_path = os.path.join(assets_path, "best_model_Vj8", "scale_stats.json")
with open(scale_stats_path) as f:
    scale_stats = json.load(f)
velocity_range = scale_stats["velocity"]["range"][0]  # ~29.07 m/s

api = wandb.Api()

# Project is specified by <entity/project-name>
try:
    runs = api.runs("jpsch-dtu-wind-and-energy-systems/GNO_probe_large")
    access_to_wandb = True
except Exception as e:
    access_to_wandb = False
    print(
        "No access to wandb project jpsch-dtu-wind-and-energy-systems/GNO_probe_large; using cached CSV.",
        e,
    )
# Create dataframe from wandb runs
if access_to_wandb:
    summary_list, config_list, name_list = [], [], []
    for run in runs:
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame({"summary": summary_list, "config": config_list, "name": name_list})
    runs_df = runs_df.sort_values(by="name")
    # only sophia dirs
    runs_df = runs_df[runs_df["name"].str.contains("sophia")]
    # save to csv
    runs_df.to_csv("wandb_runs.csv", index=False)
else:
    # load from csv
    runs_df = pd.read_csv("wandb_runs.csv", converters={"summary": eval, "config": eval})


# %% Create df with only things relevant for grid search
grid_search_df = pd.DataFrame()
# extract features of interest

grid_search_df["name"] = runs_df["name"]
grid_search_df["gridsearch_name"] = grid_search_df["name"].str.split("/").str[:-1]
grid_search_df["gridsearch_name"] = grid_search_df["gridsearch_name"].apply(lambda x: "/".join(x))
dirs_sorted = np.sort(grid_search_df["gridsearch_name"].unique())
roman_nums = [
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "XI",
    "XII",
    "XIII",
    "XIV",
]
grid_search_names = [f"{roman_nums[i]}" for i in range(0, len(dirs_sorted))]
# add a gridsearch name column
grid_search_df["gridsearch_name"] = grid_search_df["gridsearch_name"].apply(
    lambda x: grid_search_names[np.where(dirs_sorted == x)[0][0]]
)


grid_search_df["train_mse"] = runs_df["summary"].apply(lambda x: x.get("train/loss"))
grid_search_df["val_mse"] = runs_df["summary"].apply(lambda x: x.get("val/loss(mse)"))
grid_search_df["val_mae"] = runs_df["summary"].apply(lambda x: x.get("val/mae"))


grid_search_df["lr_type"] = runs_df["config"].apply(lambda x: x["optimizer"]["lr_schedule"]["type"])
grid_search_df["lr"] = runs_df["config"].apply(
    lambda x: x["optimizer"]["lr_schedule"].get("init_learning_rate")
)
grid_search_df["lr_2"] = runs_df["config"].apply(
    lambda x: x["optimizer"]["lr_schedule"].get("learning_rate")
)
# combine lr and lr_2 into one column iggnoring nan
grid_search_df["lr"] = grid_search_df["lr"].combine_first(grid_search_df["lr_2"])
grid_search_df = grid_search_df.drop(columns=["lr_2"])

grid_search_df["lr_trigger_steps"] = runs_df["config"].apply(
    lambda x: x["optimizer"]["lr_schedule"].get("boundaries")
)


# if lower boundary is larger than the final step, convert to constant lr
for i in range(len(grid_search_df)):
    if grid_search_df["lr_type"].iloc[i] == "piecewise_constant":
        if grid_search_df["lr_trigger_steps"].iloc[i][0] > runs_df["summary"].iloc[i]["_step"]:
            grid_search_df["lr_type"].iloc[i] = "piecewise_constant_untriggered"

grid_search_df["n_probes"] = runs_df["config"].apply(
    lambda x: x["optimizer"]["batching"].get("n_probe")
)
grid_search_df["max_graphs"] = runs_df["config"].apply(
    lambda x: x["optimizer"]["batching"].get("max_graph_size")
)

# find unique optimizer combinations
grid_search_df["optimizer_config"] = (
    "lr_type: "
    + grid_search_df["lr_type"].astype(str)
    + ", lr: "
    + grid_search_df["lr"].astype(str)
    + ", lr_trigger_steps: "
    + grid_search_df["lr_trigger_steps"].astype(str)
    + ", n_probes: "
    + grid_search_df["n_probes"].astype(str)
    + ", max_graphs: "
    + grid_search_df["max_graphs"].astype(str)
)
grid_search_df["optimizer_config"].unique()
# convert the unique optimizers to letters
optimizer_configs_sorted = grid_search_df["optimizer_config"].unique()
optimizer_config_names = [f"{str(i)}" for i in range(1, len(optimizer_configs_sorted) + 1)]
# add a optimizer config name column
grid_search_df["optimizer_config"] = grid_search_df["optimizer_config"].apply(
    lambda x: optimizer_config_names[np.where(optimizer_configs_sorted == x)[0][0]]
)

grid_search_df["latent_size"] = runs_df["config"].apply(lambda x: x["model"]["latent_size"])
grid_search_df["hidden_layer_size"] = runs_df["config"].apply(
    lambda x: x["model"]["hidden_layer_size"]
)
grid_search_df["num_mlp_layers"] = runs_df["config"].apply(lambda x: x["model"]["num_mlp_layers"])

grid_search_df["num_decoder_layers"] = runs_df["config"].apply(
    lambda x: x["model"].get("num_decoder_layers")
)
grid_search_df["decoder_hidden_layer_size"] = runs_df["config"].apply(
    lambda x: x["model"].get("decoder_hidden_layer_size")
)

# find unique model combinations
grid_search_df["model_config"] = (
    "latent_size: "
    + grid_search_df["latent_size"].astype(str)
    + ", hidden_layer_size: "
    + grid_search_df["hidden_layer_size"].astype(str)
    + ", num_mlp_layers: "
    + grid_search_df["num_mlp_layers"].astype(str)
    + ", num_decoder_layers: "
    + grid_search_df["num_decoder_layers"].astype(str)
    + ", decoder_hidden_layer_size: "
    + grid_search_df["decoder_hidden_layer_size"].astype(str)
)
grid_search_df["model_config"].unique()
# convert the unique models to letters
model_configs_sorted = grid_search_df["model_config"].unique()
model_config_names = [f"{chr(96+i)}" for i in range(1, len(model_configs_sorted) + 1)]
# add a model config name column
grid_search_df["model_config"] = grid_search_df["model_config"].apply(
    lambda x: model_config_names[np.where(model_configs_sorted == x)[0][0]]
)

## Always on
# grid_search_df["drop_out"] = runs_df["config"].apply(
#     lambda x: x["model"]["regularization"].get("encoder_dropout_rate")
# )
# grid_search_df["RBF"] = runs_df["config"].apply(
#     lambda x: x["model"]["RBF_dist_encoder"].get("type")
# )


# sort by grid search name, then by model config, then by optimizer config
grid_search_df = grid_search_df.sort_values(
    by=["gridsearch_name", "model_config", "optimizer_config"]
)
grid_search_df = grid_search_df.reset_index(drop=True)

# %% Create tables of unique optimizer configs and model configs
optimizer_configs = pd.DataFrame()
optimizer_configs["optimizer_config"] = optimizer_configs_sorted
optimizer_configs["optimizer_config_name"] = optimizer_config_names
# recreate columns from the optimizer config string
optimizer_configs["lr_type"] = optimizer_configs["optimizer_config"].apply(
    lambda x: x.split(", ")[0].split(": ")[1]
)
optimizer_configs["lr"] = optimizer_configs["optimizer_config"].apply(
    lambda x: float(x.split(", ")[1].split(": ")[1])
)
optimizer_configs["lr_trigger_steps"] = optimizer_configs["optimizer_config"].apply(
    lambda x: ", ".join(x.split(", ")[2:4]).split(": ")[1]
)

optimizer_configs["n_probes"] = optimizer_configs["optimizer_config"].apply(
    lambda x: int(x.split(", ")[4].split(": ")[1])
)

# remove case for constant lr
for i in range(len(optimizer_configs)):
    if optimizer_configs["lr_type"].iloc[i] == "constant":
        optimizer_configs["n_probes"].iloc[i] = int(
            optimizer_configs["optimizer_config"].iloc[2].split(", ")[3].split(": ")[1]
        )
        optimizer_configs["lr_trigger_steps"].iloc[i] = None


optimizer_configs["max_graphs"] = optimizer_configs["optimizer_config"].apply(
    lambda x: int(x.split(", ")[-1].split(": ")[1])
)

# remove optimizer_configs column
optimizer_configs = optimizer_configs.drop(columns=["optimizer_config"])


# Model configs
model_configs = pd.DataFrame()
model_configs["model_config"] = model_configs_sorted
model_configs["model_config_name"] = model_config_names
# recreate columns from the model config string
model_configs["latent_size"] = model_configs["model_config"].apply(
    lambda x: int(x.split(", ")[0].split(": ")[1])
)
model_configs["hidden_layer_size"] = model_configs["model_config"].apply(
    lambda x: int(x.split(", ")[1].split(": ")[1])
)
model_configs["num_mlp_layers"] = model_configs["model_config"].apply(
    lambda x: int(x.split(", ")[2].split(": ")[1])
)
model_configs["num_decoder_layers"] = model_configs["model_config"].apply(
    lambda x: int(x.split(", ")[3].split(": ")[1])
)
model_configs["decoder_hidden_layer_size"] = model_configs["model_config"].apply(
    lambda x: int(x.split(", ")[4].split(": ")[1])
)
# remove model_configs column
model_configs = model_configs.drop(columns=["model_config"])


## create a latex table for optimizer configs
# rename columns
optimizer_configs_latex = optimizer_configs.rename(
    columns={
        "optimizer_config_name": "ID",
        "lr_type": "LR type",
        "lr": "LR",
        "lr_trigger_steps": "Triggers",
        "n_probes": r"$n_\mathrm{p}$",
        "max_graphs": "$n_G$",
    }
)

# replace the LR types with nicer names
optimizer_configs_latex["LR type"] = optimizer_configs_latex["LR type"].replace(
    {
        "piecewise_constant": "Piecewise constant",
        "piecewise_constant_untriggered": r"Piecewise constant$\dagger$",
        "constant": "Constant",
    }
)
# replace None with -
optimizer_configs_latex["Triggers"] = optimizer_configs_latex["Triggers"].replace({None: "-"})

# export to latex
latex_string = optimizer_configs_latex.to_latex(
    index=False,
    caption="Optimizer configurations used in the grid search.",
    label="tab:optimizer_configs",
    float_format="%.3f",
    column_format="c" * len(optimizer_configs_latex.columns),
    escape=False,
    position="htb",
)

# replace toprule, midrule, bottomrule with hlines
latex_string = latex_string.replace(r"\toprule", r"\hline")
latex_string = latex_string.replace(r"\midrule", r"\hline")
latex_string = latex_string.replace(r"\bottomrule", r"\hline")


# create model configs latex table
model_configs_latex = model_configs.rename(
    columns={
        "model_config_name": "ID",
        "latent_size": r"Latent dim. $Q$",
        "hidden_layer_size": r"$q_\mathrm{int}$",
        "num_mlp_layers": r"$L_\mathrm{int}$",
        "num_decoder_layers": r"$L_\mathrm{dec}$",
        "decoder_hidden_layer_size": r"$q_\mathrm{dec}$",
    }
)

# reorder columns
model_configs_latex = model_configs_latex[
    [
        "ID",
        r"Latent dim. $Q$",
        r"$L_\mathrm{int}$",
        r"$q_\mathrm{int}$",
        r"$L_\mathrm{dec}$",
        r"$q_\mathrm{dec}$",
    ]
]

# export to latex
latex_string_model = model_configs_latex.to_latex(
    index=False,
    caption="Model and optimizer configurations used in the grid search.",
    label="tab:model_and_opt_configs",
    float_format="%.3f",
    column_format="c" * len(model_configs_latex.columns),
    escape=False,
    position="htb",
)
latex_string_model = latex_string_model.replace(r"\toprule", r"\hline")
latex_string_model = latex_string_model.replace(r"\midrule", r"\hline")
latex_string_model = latex_string_model.replace(r"\bottomrule", r"\hline")


# combine both latex tables into one table with multicolumn between
warning_line = (
    r"% DO NOT EDIT THIS TABLE CHANGE IT IN CODE, SEE https://gitlab.windenergy.dtu.dk/superposition-operator/spo-operator-tests/-/blob/main/Experiments/articles_plotting/grid_search.py?ref_type=heads"
    + "\n"
)
table_start = latex_string_model.splitlines()[:3]
tabular_config = "\n" + r"\begin{tabular}{cp{1.3cm}cccc}" + "\n"
multi_col1 = r"\multicolumn{6}{l}{\textbf{Model configurations}} \\" + "\n"
model_content = latex_string_model.splitlines()[5:-2]

multi_col2 = r"\multicolumn{6}{l}{\textbf{Optimizer configurations}} \\" + "\n"
# Get optimizer content without the final \end{table}
optimizer_lines = latex_string.splitlines()[5:]
# Remove the last line (\end{table}) to insert footnote before it
optimizer_content = "\n".join(optimizer_lines[:-1])

# Footnote using minipage
footnote = r"""
\vspace{0.5em}
\begin{minipage}{0.9\linewidth}
\footnotesize
$^{\dagger}$ Learning rate schedule was not triggered during training.
\end{minipage}
\end{table}
"""

combined_latex = (
    warning_line
    + "\n".join(table_start)
    + tabular_config
    + r"\hline"
    + "\n"
    + multi_col1
    + "\n".join(model_content)
    + multi_col2
    + optimizer_content
    + footnote
)


# write to file
with open(os.path.join(assets_path, "combined_configs.tex"), "w") as f:
    f.write(combined_latex)


# %% Resultant models table

resultant_models = grid_search_df[
    [
        "gridsearch_name",
        "model_config",
        "optimizer_config",
        "train_mse",
        "val_mse",
        "val_mae",
    ]
]
resultant_models = resultant_models.sort_values(by="val_mse")
# Unscale metrics to physical units using velocity range
# MSE: multiply by range² to get m²s⁻², MAE: multiply by range to get ms⁻¹
mse_unscale = velocity_range**2  # ~845 for range ~29.07 m/s
mae_unscale = velocity_range  # ~29.07 m/s
round_decimals = 3
resultant_models["train_mse"] = (resultant_models["train_mse"] * mse_unscale).round(round_decimals)
resultant_models["val_mse"] = (resultant_models["val_mse"] * mse_unscale).round(round_decimals)
resultant_models["val_mae"] = (resultant_models["val_mae"] * mae_unscale).round(round_decimals)

# rename columns with physical units
resultant_models = resultant_models.rename(
    columns={
        "gridsearch_name": "Grid search",
        "model_config": "Model ID",
        "optimizer_config": "Opt. ID",
        "train_mse": r"MSE$_\mathrm{trn}$ [$\mathrm{m^2s^{-2}}$]",
        "val_mse": r"MSE$_\mathrm{val}$ [$\mathrm{m^2s^{-2}}$]",
        "val_mae": r"MAE$_\mathrm{val}$ [$\mathrm{ms^{-1}}$]",
    }
)

# mark the best val_mse in bold
best_val_mse = resultant_models[r"MSE$_\mathrm{val}$ [$\mathrm{m^2s^{-2}}$]"].min()
resultant_models[r"MSE$_\mathrm{val}$ [$\mathrm{m^2s^{-2}}$]"] = resultant_models[
    r"MSE$_\mathrm{val}$ [$\mathrm{m^2s^{-2}}$]"
].apply(lambda x: f"\\textbf{{{x}}}" if x == best_val_mse else x)

# mark the best val_mae in bold
best_val_mae = resultant_models[r"MAE$_\mathrm{val}$ [$\mathrm{ms^{-1}}$]"].min()
resultant_models[r"MAE$_\mathrm{val}$ [$\mathrm{ms^{-1}}$]"] = resultant_models[
    r"MAE$_\mathrm{val}$ [$\mathrm{ms^{-1}}$]"
].apply(lambda x: f"\\textbf{{{x}}}" if x == best_val_mae else x)

# mark the best train_mse in bold
best_train_mse = resultant_models[r"MSE$_\mathrm{trn}$ [$\mathrm{m^2s^{-2}}$]"].min()
resultant_models[r"MSE$_\mathrm{trn}$ [$\mathrm{m^2s^{-2}}$]"] = resultant_models[
    r"MSE$_\mathrm{trn}$ [$\mathrm{m^2s^{-2}}$]"
].apply(lambda x: f"\\textbf{{{x}}}" if x == best_train_mse else x)


# export to latex (sized for single column with category separators)
latex_string_results = resultant_models.to_latex(
    index=False,
    caption=r"Trained models from the grid search ranked by validation MSE$_\mathrm{val}$.",
    label="tab:grid_search_results",
    float_format="%.3f",
    column_format=r"p{.55cm}p{.55cm}p{.5cm}|p{1.2cm}p{1.2cm}p{1.2cm}",
    escape=False,
    position="htb",
)
latex_string_results = latex_string_results.replace(r"\toprule", r"\hline")
latex_string_results = latex_string_results.replace(r"\midrule", r"\hline")
latex_string_results = latex_string_results.replace(r"\bottomrule", r"\hline")

# Add multicolumn header row for Configuration | Metrics categories
multicolumn_header = (
    r"\multicolumn{3}{l|}{\textbf{Configuration}} & \multicolumn{3}{l}{\textbf{Metrics}}\\" + "\n"
)
# Insert after first \hline (after tabular declaration)
lines = latex_string_results.split("\n")
for i, line in enumerate(lines):
    if r"\hline" in line:
        lines.insert(i + 1, multicolumn_header.strip())
        break
latex_string_results = "\n".join(lines)


# add warning line
latex_string_results = (
    r"% DO NOT EDIT THIS TABLE CHANGE IT IN CODE, SEE https://gitlab.windenergy.dtu.dk/superposition-operator/spo-operator-tests/-/blob/main/Experiments/articles_plotting/grid_search.py?ref_type=heads"
    + "\n"
    + latex_string_results
)

# write to file
with open(os.path.join(assets_path, "grid_search_results.tex"), "w") as f:
    f.write(latex_string_results)


# %% Best model table for main paper
# Get the best model (first row after sorting by val_mse)
best_model_row = grid_search_df.sort_values(by="val_mse").iloc[0]

# Get the model config details for the best model
best_model_config_id = best_model_row["model_config"]
best_model_config = model_configs[model_configs["model_config_name"] == best_model_config_id].iloc[
    0
]

# Get the optimizer config details for the best model
best_optimizer_config_id = best_model_row["optimizer_config"]
best_optimizer_config = optimizer_configs[
    optimizer_configs["optimizer_config_name"] == best_optimizer_config_id
].iloc[0]

# Create a transposed best model table (parameters as columns)
# Handle learning rate type with dagger for untriggered
lr_type_raw = best_optimizer_config["lr_type"]
if "untriggered" in lr_type_raw.lower():
    lr_type_display = r"\scriptsize{Piecewise const.$^\dagger$}"
else:
    lr_type_display = r"\scriptsize{" + lr_type_raw.replace("_", " ").title() + "}"

# Build LaTeX table manually to support two-row header with category separators
# Categories: Model (5 cols) | Optimizer (4 cols) | Metrics (3 cols)
latex_string_best = rf"""% DO NOT EDIT THIS TABLE CHANGE IT IN CODE, SEE https://gitlab.windenergy.dtu.dk/superposition-operator/spo-operator-tests/-/blob/main/Experiments/articles_plotting/grid_search.py?ref_type=heads
\begin{{table*}}[htb]
\caption{{Best model configuration and performance (grid search {best_model_row['gridsearch_name']}, model {best_model_config_id}, optimizer {best_optimizer_config_id}).}}
\label{{tab:best_model_config}}
\centering
\footnotesize
\begin{{tabular}}{{l|p{{0.5cm}}p{{0.6cm}}p{{0.5cm}}p{{0.6cm}}p{{0.5cm}}|p{{1.2cm}}p{{1.1cm}}p{{0.5cm}}p{{0.7cm}}|p{{1.35cm}}p{{1.35cm}}p{{1.35cm}}}}
\hline
& \multicolumn{{5}}{{c|}}{{\textbf{{Model}}}} & \multicolumn{{4}}{{c|}}{{\textbf{{Optimizer}}}} & \multicolumn{{3}}{{c}}{{\textbf{{Metrics}}}} \\
& $Q$ & $L_\mathrm{{int}}$ & $q_\mathrm{{int}}$ & $L_\mathrm{{dec}}$ & $q_\mathrm{{dec}}$ & LR type & LR & $n_\mathrm{{p}}$ & $n_G$ & MSE$_\mathrm{{trn}}$ [$\mathrm{{m^2s^{{-2}}}}$] & MSE$_\mathrm{{val}}$ [$\mathrm{{m^2s^{{-2}}}}$] & MAE$_\mathrm{{val}}$ [$\mathrm{{ms^{{-1}}}}$] \\
\hline
"""
# Add description row
latex_string_best += (
    r"Description & "
    r"\scriptsize{Latent dim.} & \scriptsize{Int. layers} & \scriptsize{Int. width} & "
    r"\scriptsize{Dec. layers} & \scriptsize{Dec. width} & "
    r"-- & -- & \scriptsize{Probes/ batch} & \scriptsize{Graphs/ batch} & "
    r"-- & -- & -- \\" + "\n"
)
# Add values row with unscaled metrics in physical units
latex_string_best += "Value & "
latex_string_best += (
    f"{int(best_model_config['latent_size'])} & "
    f"{int(best_model_config['num_mlp_layers'])} & "
    f"{int(best_model_config['hidden_layer_size'])} & "
    f"{int(best_model_config['num_decoder_layers'])} & "
    f"{int(best_model_config['decoder_hidden_layer_size'])} & "
    f"{lr_type_display} & "
    f"{best_optimizer_config['lr'] * 1e3:.0f}$\\times 10^{{-3}}$ & "
    f"{int(best_optimizer_config['n_probes'])} & "
    f"{int(best_optimizer_config['max_graphs'])} & "
    f"{best_model_row['train_mse'] * mse_unscale:.3f} & "
    f"{best_model_row['val_mse'] * mse_unscale:.3f} & "
    f"{best_model_row['val_mae'] * mae_unscale:.3f} \\\\\n"
)
# Close table with footnote
latex_string_best += r"""\hline
\end{tabular}
\vspace{0.5em}
\begin{minipage}{0.9\linewidth}
\footnotesize
$^{\dagger}$ Learning rate schedule was not triggered during training.
\end{minipage}
\end{table*}
"""

# Write to file (warning line already included in latex_string_best)
with open(os.path.join(assets_path, "best_model_config.tex"), "w") as f:
    f.write(latex_string_best)

print(
    f"Best model: Grid search {best_model_row['gridsearch_name']}, "
    f"Model config {best_model_config_id}, Optimizer config {best_optimizer_config_id}"
)
print(f"Validation MSE: {best_model_row['val_mse'] * mse_unscale:.3f} m²s⁻²")
print(f"Validation MAE: {best_model_row['val_mae'] * mae_unscale:.3f} ms⁻¹")


# %% Top 5 Models Table with Full Config Details
# Get top 5 from grid_search_df (already sorted by val_mse)
top_five_df = grid_search_df.sort_values(by="val_mse").head(5).reset_index(drop=True)

# Load baseline metrics
baseline_path = os.path.join(assets_path, "no_wake_baseline", "error_metrics.json")
baseline_metrics = None
if os.path.exists(baseline_path):
    with open(baseline_path) as f:
        baseline_metrics = json.load(f)
    # Handle list format
    for key in baseline_metrics:
        if isinstance(baseline_metrics[key], list):
            baseline_metrics[key] = baseline_metrics[key][0]
else:
    print(f"Warning: Baseline metrics not found at {baseline_path}")

# Build the top 5 full config table using train/val metrics from wandb
# Collect full config details for each of the top 5
top_five_full = []
for _i, row in top_five_df.iterrows():
    model_cfg = model_configs[model_configs["model_config_name"] == row["model_config"]].iloc[0]
    opt_cfg = optimizer_configs[
        optimizer_configs["optimizer_config_name"] == row["optimizer_config"]
    ].iloc[0]
    top_five_full.append(
        {
            "row": row,
            "model_cfg": model_cfg,
            "opt_cfg": opt_cfg,
        }
    )

# Find best values for each metric column (for bolding)
train_mse_values = [m["row"]["train_mse"] for m in top_five_full]
val_mse_values = [m["row"]["val_mse"] for m in top_five_full]
val_mae_values = [m["row"]["val_mae"] for m in top_five_full]

best_train_mse = min(train_mse_values)
best_val_mse = min(val_mse_values)
best_val_mae = min(val_mae_values)

# Build LaTeX table (15 columns: 3 IDs | 5 model | 4 optimizer | 3 metrics)
# Using p{} column sizing to fit within page width
latex_top5_full = r"""% DO NOT EDIT THIS TABLE CHANGE IT IN CODE, SEE https://gitlab.windenergy.dtu.dk/superposition-operator/spo-operator-tests/-/blob/main/Experiments/articles_plotting/grid_search.py?ref_type=heads
\begin{table*}[htb]
\caption{Top 5 models from grid search with full configuration parameters and training/validation metrics.}
\label{tab:top_five_full_config}
\centering
\small
\begin{tabular}{p{0.5cm}p{0.5cm}p{0.4cm}|p{0.4cm}p{0.7cm}p{0.4cm}p{0.7cm}p{0.5cm}|p{0.4cm}p{1.25cm}p{0.6cm}p{0.8cm}|p{0.9cm}p{0.9cm}p{0.9cm}}
\hline
\multicolumn{3}{c|}{\textbf{IDs}} & \multicolumn{5}{c|}{\textbf{Model}} & \multicolumn{4}{c|}{\textbf{Optimizer}} & \multicolumn{3}{c}{\textbf{Metrics}} \\
Grid search & Model ID & Opt. ID & $Q$ & $L_\mathrm{int}$ & $q_\mathrm{int}$ & $L_\mathrm{dec}$ & $q_\mathrm{dec}$ & LR type & LR & $n_\mathrm{p}$ & $n_G$ & $\mathrm{MSE_{trn}}$ [$\mathrm{m^2s^{-2}}$] & $\mathrm{MSE_{val}}$ [$\mathrm{m^2s^{-2}}$] & $\mathrm{MAE_{val}}$ [$\mathrm{ms^{-1}}$] \\
\hline
"""
# Add description row
latex_top5_full += (
    r"-- & -- & -- & "
    r"\footnotesize{Latent dim.} & \footnotesize{Int. layers} & \footnotesize{Int. width} & "
    r"\footnotesize{Dec. layers} & \footnotesize{Dec. width} & "
    r"-- & -- & \footnotesize{Probes/ batch} & \footnotesize{Graphs/ batch} & "
    r"-- & -- & -- \\"
    + "\n"
)
latex_top5_full += r"\hline" + "\n"

# Format metrics with bolding for best values
def fmt_metric(val, best_val, decimals=3):
    rounded = round(val, decimals)
    if val == best_val:
        return f"\\textbf{{{rounded}}}"
    return str(rounded)

# Add data rows
for entry in top_five_full:
    row = entry["row"]
    model_cfg = entry["model_cfg"]
    opt_cfg = entry["opt_cfg"]

    # Format learning rate in LaTeX scientific notation
    lr_val = opt_cfg["lr"]
    lr_exp = int(np.floor(np.log10(lr_val)))
    lr_mantissa = lr_val / (10**lr_exp)
    lr_str = f"${lr_mantissa:.0f}\\times 10^{{{lr_exp}}}$"

    # Format LR type with dagger for untriggered (compact)
    lr_type_raw = opt_cfg["lr_type"]
    if "untriggered" in lr_type_raw.lower():
        lr_type_str = r"PC$^\dagger$"
    elif "piecewise" in lr_type_raw.lower():
        lr_type_str = "PC"
    else:
        lr_type_str = "C"

    # Format train/val metrics with unscaling to physical units
    train_mse_str = fmt_metric(row["train_mse"] * mse_unscale, best_train_mse * mse_unscale)
    val_mse_str = fmt_metric(row["val_mse"] * mse_unscale, best_val_mse * mse_unscale)
    val_mae_str = fmt_metric(row["val_mae"] * mae_unscale, best_val_mae * mae_unscale)

    latex_top5_full += (
        f"{row['gridsearch_name']} & {row['model_config']} & {row['optimizer_config']} & "
        f"{int(model_cfg['latent_size'])} & {int(model_cfg['num_mlp_layers'])} & "
        f"{int(model_cfg['hidden_layer_size'])} & {int(model_cfg['num_decoder_layers'])} & "
        f"{int(model_cfg['decoder_hidden_layer_size'])} & "
        f"{lr_type_str} & {lr_str} & {int(opt_cfg['n_probes'])} & {int(opt_cfg['max_graphs'])} & "
        f"{train_mse_str} & {val_mse_str} & {val_mae_str} \\\\\n"
    )

# Close table with footnote
latex_top5_full += r"""\hline
\end{tabular}
\vspace{0.5em}
\begin{minipage}{0.95\linewidth}
\footnotesize
(PC) Piecewise constant\\
$^{\dagger}$ LR schedule not triggered during training
\end{minipage}
\end{table*}
"""

# Write to file
with open(os.path.join(assets_path, "top_five_full_config.tex"), "w") as f:
    f.write(latex_top5_full)

print("\nTop 5 full config table written to assets/top_five_full_config.tex")

# %%
