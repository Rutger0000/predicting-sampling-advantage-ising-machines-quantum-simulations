# %%

import pandas as pd
import os
import json
from functools import lru_cache
import numpy as np
from ultrafast_full_boltzmann_machine.models.models import final_models_df

#####################
# Read existing converted models
#####################

# Read current models
df = pd.DataFrame.from_dict(final_models_df)

# %%
# Drop columns
df.drop(columns=["parallel", "iterations", "preset", "model_id"], inplace=True)

# %%
df["full"] = False
# %%
# Write to file
df.to_csv("models/all_weights_converted.tsv", sep="\t", index=False)
