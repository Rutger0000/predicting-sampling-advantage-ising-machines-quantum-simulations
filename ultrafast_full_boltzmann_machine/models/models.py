# %%
import pandas as pd

# %%
flat_map = lambda func, xs: [y for ys in xs for y in func(ys)]

# Creating a list of models
models = [
    *[{
        "L": L,
        "alpha": Alpha,
        "nonsquare": True,
        "preset": "2k_nonsquare",
        "repeat": 5
        } for Alpha, Ls in {
            1: list(range(4, 11, 2)),
            2: list(range(4, 15, 2)),
            3: list(range(4, 15, 2)),
            4: list(range(4, 15, 2)),
            8: list(range(4, 11, 2)),
        }.items() for L in Ls],
    *[{
        "L": L,
        "alpha": Alpha,
        "nonsquare": True,
        "preset": "10k_nonsquare",
        "repeat": 5
        } for Alpha, Ls in {
            2: list(range(16, 23, 2)),
            3: list(range(16, 23, 2)),
            4: list(range(16, 23, 2)),
            8: list(range(12, 21, 2)),
        }.items() for L in Ls]

    ]

standard_models = [
    *[{
        "L": L,
        "alpha": Alpha,
        "nonsquare": False,
        "preset": "2k_standard",
        "repeat": 1
        } for Alpha, Ls in {
            1: list(range(4, 11, 2)),
            2: list(range(4, 15, 2)),
            3: list(range(4, 15, 2)),
            4: list(range(4, 15, 2)),
            8: list(range(4, 11, 2)),
        }.items() for L in Ls],
    *[{
        "L": L,
        "alpha": Alpha,
        "nonsquare": False,
        "preset": "10k_standard",
        "repeat": 1
        } for Alpha, Ls in {
            2: list(range(16, 23, 2)),
            3: list(range(16, 23, 2)),
            4: list(range(16, 23, 2)),
            8: list(range(12, 21, 2)),
        }.items() for L in Ls
    ]
    ]

def expand_model(model):
    return [{
        "L": model["L"],
        "alpha": model["alpha"],
        "nonsquare": model["nonsquare"],
        "model_id": i,
        "preset": model["preset"]
    } for i in range(model["repeat"])]

models = flat_map(expand_model, models)
standard_models = flat_map(expand_model, standard_models)

# %%

###########################
## Model configurations ###
###########################

# The 2k_nonsquare corresponds to low MC preset, while the 10k_nonsquare corresponds to high MC preset
model_presets = {
    "2k_nonsquare": {
        "parallel": True,
        "iterations": 300,
        "directory": lambda model_id: f"modified_RBM_low_model_id={model_id}",
        "version": "v3"
    },
    "10k_nonsquare": {
        "parallel": True,
        "iterations": 300,
        "directory": lambda model_id: f"modified_RBM_high_model_id={model_id}",
        "version": "v3"
    },
    "2k_standard": {
        "parallel": True,
        "iterations": 300,
        "directory": lambda model_id: f"standard_RBM_low_model_id={model_id}",
        "version": "v3"
    },
    "10k_standard": {
        "parallel": True,
        "iterations": 300,
        "directory":  lambda model_id: f"standard_RBM_high_model_id={model_id}",
        "version": "v3"
    }
}

def get_model(row):
    preset = row["preset"]
    model_id = row["model_id"]
    additional_params = {
        **model_presets[preset],
        "nspins": int(row["L"] ** 2),
        "directory": model_presets[preset]["directory"](model_id)
    }
    return pd.Series(additional_params)

# %%

models_df = pd.DataFrame.from_dict(models)
standard_models_df = pd.DataFrame.from_dict(standard_models)

models_df = pd.concat([
    models_df,
    models_df.apply(get_model, axis=1, result_type="expand")
], axis=1)

standard_models_df = pd.concat([
    standard_models_df,
    standard_models_df.apply(get_model, axis=1, result_type="expand")
], axis=1)

# %%

##########
## Autocorrelation configurations
##########

train_models_df = pd.concat([
    models_df,
    standard_models_df
], axis=0)

final_models_df = models_df