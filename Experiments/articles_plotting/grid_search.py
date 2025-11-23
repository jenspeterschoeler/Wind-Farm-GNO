import numpy as np
import pandas as pd

import wandb

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
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame(
        {"summary": summary_list, "config": config_list, "name": name_list}
    )
    runs_df = runs_df.sort_values(by="name")
    # only sophia dirs
    runs_df = runs_df[runs_df["name"].str.contains("sophia")]
    # save to csv
    runs_df.to_csv("wandb_runs.csv", index=False)
else:
    # load from csv
    runs_df = pd.read_csv(
        "wandb_runs.csv", converters={"summary": eval, "config": eval}
    )


# %% Create df with only things relevant for grid search
grid_search_df = pd.DataFrame()
# extract features of interest

grid_search_df["name"] = runs_df["name"]
grid_search_df["gridsearch_name"] = grid_search_df["name"].str.split("/").str[:-1]
grid_search_df["gridsearch_name"] = grid_search_df["gridsearch_name"].apply(
    lambda x: "/".join(x)
)
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


grid_search_df["lr_type"] = runs_df["config"].apply(
    lambda x: x["optimizer"]["lr_schedule"]["type"]
)
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
        if (
            grid_search_df["lr_trigger_steps"].iloc[i][0]
            > runs_df["summary"].iloc[i]["_step"]
        ):
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
optimizer_config_names = [
    f"{str(i)}" for i in range(1, len(optimizer_configs_sorted) + 1)
]
# add a optimizer config name column
grid_search_df["optimizer_config"] = grid_search_df["optimizer_config"].apply(
    lambda x: optimizer_config_names[np.where(optimizer_configs_sorted == x)[0][0]]
)

grid_search_df["latent_size"] = runs_df["config"].apply(
    lambda x: x["model"]["latent_size"]
)
grid_search_df["hidden_layer_size"] = runs_df["config"].apply(
    lambda x: x["model"]["hidden_layer_size"]
)
grid_search_df["num_mlp_layers"] = runs_df["config"].apply(
    lambda x: x["model"]["num_mlp_layers"]
)

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
optimizer_configs_latex["Triggers"] = optimizer_configs_latex["Triggers"].replace(
    {None: "-"}
)

# export to latex
latex_string = optimizer_configs_latex.to_latex(
    index=False,
    caption="Optimizer configurations used in the grid search. $\\dagger$ indicates that the learning rate schedule was not triggered during training.",
    label="tab:optimizer_configs",
    float_format="%.3f",
    column_format="c" * len(optimizer_configs_latex.columns),
    escape=False,
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
    caption="Model and optimizer configurations used in the grid search. $\\dagger$ indicates that the learning rate schedule was not triggered during training.",
    label="tab:model_and_opt_configs",
    float_format="%.3f",
    column_format="c" * len(model_configs_latex.columns),
    escape=False,
)
latex_string_model = latex_string_model.replace(r"\toprule", r"\hline")
latex_string_model = latex_string_model.replace(r"\midrule", r"\hline")
latex_string_model = latex_string_model.replace(r"\bottomrule", r"\hline")


# combine both latex tables into one table with multicolumn between
warning_line = (
    rf"% DO NOT EDIT THIS TABLE CHANGE IT IN CODE, SEE https://gitlab.windenergy.dtu.dk/superposition-operator/spo-operator-tests/-/blob/main/Experiments/articles_plotting/grid_search.py?ref_type=heads "
    + "\n"
)
table_start = latex_string_model.splitlines()[:3]
tabular_config = "\n" + r"\begin{tabular}{cp{2cm}cccc}" + "\n"
multi_col1 = r"\multicolumn{6}{l}{\textbf{Model configurations}} \\" + "\n"
model_content = latex_string_model.splitlines()[5:-2]

multi_col2 = r"\multicolumn{6}{l}{\textbf{Optimizer configurations}} \\" + "\n"
optimizer_tail = latex_string.splitlines()[5:]

combined_latex = (
    warning_line
    + "\n".join(table_start)
    + tabular_config
    + r"\hline"
    + "\n"
    + multi_col1
    + "\n".join(model_content)
    + multi_col2
    + "\n".join(optimizer_tail)
)


# write to file
with open("../../assets/combined_configs.tex", "w") as f:
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
# multiply mse and mae columns
multi_factor = 1e3
round_decimals = 3
resultant_models["train_mse"] = (resultant_models["train_mse"] * multi_factor).round(
    round_decimals
)
resultant_models["val_mse"] = (resultant_models["val_mse"] * multi_factor).round(
    round_decimals
)
resultant_models["val_mae"] = (resultant_models["val_mae"] * multi_factor).round(
    round_decimals
)

# rename columns
resultant_models = resultant_models.rename(
    columns={
        "gridsearch_name": "Grid search",
        "model_config": "Model ID",
        "optimizer_config": "Optimizer ID",
        "train_mse": r"MSE$_\mathrm{trn} \times 10^{-3}$",
        "val_mse": r"MSE$_\mathrm{val} \times 10^{-3}$",
        "val_mae": r"MAE$_\mathrm{val} \times 10^{-3}$",
    }
)

# mark the best val_mse in bold
best_val_mse = resultant_models[r"MSE$_\mathrm{val} \times 10^{-3}$"].min()
resultant_models[r"MSE$_\mathrm{val} \times 10^{-3}$"] = resultant_models[
    r"MSE$_\mathrm{val} \times 10^{-3}$"
].apply(lambda x: f"\\textbf{{{x}}}" if x == best_val_mse else x)

# mark the best val_mae in bold
best_val_mae = resultant_models[r"MAE$_\mathrm{val} \times 10^{-3}$"].min()
resultant_models[r"MAE$_\mathrm{val} \times 10^{-3}$"] = resultant_models[
    r"MAE$_\mathrm{val} \times 10^{-3}$"
].apply(lambda x: f"\\textbf{{{x}}}" if x == best_val_mae else x)

# mark the best train_mse in bold
best_train_mse = resultant_models[r"MSE$_\mathrm{trn} \times 10^{-3}$"].min()
resultant_models[r"MSE$_\mathrm{trn} \times 10^{-3}$"] = resultant_models[
    r"MSE$_\mathrm{trn} \times 10^{-3}$"
].apply(lambda x: f"\\textbf{{{x}}}" if x == best_train_mse else x)


# export to latex
latex_string_results = resultant_models.to_latex(
    index=False,
    caption="Trained models from the grid search ranked by validation MSE$_\mathrm{val}$.",
    label="tab:grid_search_results",
    float_format="%.3f",
    column_format=r"p{1cm}" * len(resultant_models.columns),
    escape=False,
)
latex_string_results = latex_string_results.replace(r"\toprule", r"\hline")
latex_string_results = latex_string_results.replace(r"\midrule", r"\hline")
latex_string_results = latex_string_results.replace(r"\bottomrule", r"\hline")


# add warning line
latex_string_results = (
    r"% DO NOT EDIT THIS TABLE CHANGE IT IN CODE, SEE https://gitlab.windenergy.dtu.dk/superposition-operator/spo-operator-tests/-/blob/main/Experiments/articles_plotting/grid_search.py?ref_type=heads "
    + "\n"
    + latex_string_results
)

# write to file
with open("../../assets/grid_search_results.tex", "w") as f:
    f.write(latex_string_results)


# %%
