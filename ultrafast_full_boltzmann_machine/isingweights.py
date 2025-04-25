from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd

from ultrafast_full_boltzmann_machine.weight_conversion.weight_promotor import promote_weights

from ultrafast_full_boltzmann_machine.config import INTERIM_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    output_path: Path = INTERIM_DATA_DIR / "Eloc_evaluation" / "dataset.parquet",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")

    weights_file = "models/all_weights_converted.tsv"

    df = pd.read_csv(weights_file, sep='\t')

    
    
    # Prepare a list to hold the data for the new DataFrame    
    # Iterate over each row in the DataFrame
    for index, row in tqdm(df.iterrows()):
        # Extract the values from the row
        nspins = row['L']**2
        alpha = row['alpha']
        directory = row['directory']

        try:
            # Call the promote_weights function
            promote_weights(nspins, alpha, directory)
        except Exception as e:
            logger.error(f"Error processing {nspins}, {alpha}, {directory}: {e}")
            continue
    
    #df = load_all("models/all_weights_new.tsv")
    #df.to_parquet(output_path, index=False)
    # for i in tqdm(range(10), total=10):
    #     if i == 5:
    #         logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
