import json
import mlflow
import tempfile
import os
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv

load_dotenv()

_steps = [
    'download',
    'split',
    'train',
    'evaluate'
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

        if "split" in active_steps:
            _ = mlflow.run(
                hydra.utils.get_original_cwd(),
                "split",
                parameters={
                    "input_artifact": f"{config['data']['raw_data']}:latest",
                    "test_size": config['modeling']['test_size'],
                    "random_seed": config['modeling']['random_seed'],
                    "stratify_by": config['modeling']['stratify_by'],
                }
            )

        if "train" in active_steps:
            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH
            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd()),
                "train",
                parameters={
                    "label": config["data"]["label"],
                    "trainval_artifact": f"{config['data']['training_data']}:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": config["main"]["experiment_name"],
                },
            )

        if "evaluate" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd()),
                "evaluate",
                parameters={
                    "test_model": f'{config["modeling"]["test_model"]}',
                    "test_dataset": f"{config['data']['test_data']}:latest",
                    "label": config['data']['label'],
                },
            )


if __name__ == "__main__":
    go()
