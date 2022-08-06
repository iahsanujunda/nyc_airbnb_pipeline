import json
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv

load_dotenv()

_steps = [
    'download'
]


# This decorator automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):
    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                hydra.utils.get_original_cwd(),
                "get_data",
                parameters={
                    "bucket": config["data"]["bucket"],
                    "object_path": f"{config['data']['object']}",
                    "artifact_name": f"{config['data']['raw_data']}",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw dataset from data store"
                }
            )


if __name__ == "__main__":
    go()
