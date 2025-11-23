# Windfarm Graph Neural Operator (GNO)

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.17671257.svg)](https://doi.org/10.5281/zenodo.17671257)

![GNO configuration](./assets/GNO_probe.svg)
> GNO model conceptual configuration for wind farm flow prediction.

Repository containing the code required to train and evaluate the Graph Neural Operator (GNO) model for wind farm flow prediction, as presented in the publication: INSERT PUBLICATION LINK HERE WHEN AVAILABLE.

The repository includes:

- The Wind farm Graph Neural Operator (GNO) model implemented using Jraph and JAX.
- Jraph based implementation of the GEneralized aggregation Network (GEN), including the Softmax aggregation method by Li et. al.: http://arxiv.org/abs/2006.07739
- PyTorch Geometric data pipeline with on the fly Jraph conversion.
- Hydra-configured training and optional WandB logging
- Plotting, evaluation, and memory/timing scripts
- The "best" model from the publication under `./assets/best_model_Vj8`

![GNO_probe_io](./assets/GNO_probe_IO.svg)
> GNO model input-output configuration for wind farm flow prediction. Demonstrating the concept of wind turbine nodes and probe nodes.


## 🚀 Instructions

### 1) Environment (pixi)

To run the included environment, clone the repository and install the necessary dependencies.

1. Install Pixi

    Follow the instructions for your platform:  
    👉 https://pixi.sh/

2. Set up the environment

    From the project root (where `pixi.toml` is located):

    ```bash
    pixi install
    ```

### 2) Data

#### Use included `mini_graphs`

To ensure the training and test scripts run out of the box, a miniature version of the data is included in this repository.  
The miniature dataset is preconfigured in the test configurations.

#### Publication test data (optional)

Required to run the plotting scripts used in the publication:  
- [plot_predictions_article.py](./Experiments/articles_plotting/plot_predictions_article.py)  
- [wt_predictions.py](./Experiments/articles_plotting/wt_predictions.py)  
- [memory_consumption_calculation.py](./Experiments/articles_plotting/memory_consumption_calculation.py)

1. Download the test data from Zenodo: https://doi.org/10.5281/zenodo.17671257  
2. Unzip the dataset into:  `./data/zenodo_graphs/<contents of the .zip archive>`

#### Re-create the dataset or use custom data

To re-create the dataset or use your own data, follow the instructions in this repository:  
https://github.com/jenspeterschoeler/Wind-farm-Graph-flow-data

To use custom data, ensure it is structured as expected by the data loading scripts in [`torch_loader.py`](./utils/torch_loader.py), and configure a new Hydra config accordingly.

### 3) Train

Train the model only, using the preconfigured test setup:

```bash
pixi shell
python train_GNO_probe.py
```

### 4) Evaluate

#### Evaluate the pretrained model (from the publication)

Run the plotting scripts for the pretrained model (`best_model_Vj8`) included in `assets/`. `best_model_Vj8` corresponds to the model used in the publication.

```bash
pixi shell
python <Experiments/articles_plotting/*>
```

#### Evaluate your own trained model

```bash
pixi shell
python post_process_GNO_probe.py # remeber to adjust `config_path` to your trained model
```

### 5) Full Pipeline

Run the whole training pipeline (including postprocessing and evaluation):

```bash
pixi shell
python main.py
```

Notes:

- CUDA 12 toolchain is configured in [pixi.toml](./pixi.toml).
- If running CPU-only, adjust JAX packages accordingly.