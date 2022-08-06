import wandb
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(levelname)s - %(message)s")
logger = logging.getLogger()


class BaseRunner:
    def __init__(self, wandb_run):
        self.wandb_run = wandb_run

    def log_artifact(self,
                     artifact_name: str,
                     artifact_type: str,
                     artifact_description: str,
                     filename: str) -> wandb.Artifact:
        """Log the provided local filename as an artifact in W&B, and add the artifact path
        to the MLFlow run so it can be retrieved by subsequent steps in a pipeline
        Args:
            artifact_name: name for the artifact
            artifact_type:
                type for the artifact (just a string like "raw_data", "clean_data" and so on)
            artifact_description: a brief description of the artifact
            filename: local filename for the artifact
        Returns:
            Wandb artifact object
        """
        # Log to W&B
        artifact = wandb.Artifact(
            artifact_name,
            type=artifact_type,
            description=artifact_description,
        )
        artifact.add_file(filename)
        self.wandb_run.log_artifact(artifact)
        logger.info(f"Uploading {artifact_name} to Weights & Biases")

        # We need to call .wait() method to ensure that artifact transport has completed
        # before we exit this method execution
        if wandb.run.mode == 'online':
            artifact.wait()

        return artifact

    def log_model(self,
                  artifact_name: str,
                  artifact_type: str,
                  artifact_description: str,
                  model_dir: str) -> wandb.Artifact:
        """Log the provided local filename as an artifact in W&B, and add the artifact path
        to the MLFlow run so it can be retrieved by subsequent steps in a pipeline
        Args:
            artifact_name: name for the artifact
            artifact_type:
                type for the artifact (just a string like "raw_data", "clean_data" and so on)
            artifact_description: a brief description of the artifact
            model_dir: local path for the model directory
        Returns:
            Wandb artifact object
        """
        # Log to W&B
        artifact = wandb.Artifact(
            artifact_name,
            type=artifact_type,
            description=artifact_description,
        )
        artifact.add_dir(model_dir)
        self.wandb_run.log_artifact(artifact)
        # We need to call .wait() method to ensure that artifact transport has completed
        # before we exit this method execution
        if wandb.run.mode == 'online':
            artifact.wait()

        return artifact

    def retrieve_dataset_artifact(self, artifact_name) -> pd.DataFrame:
        """Retrieve wandb artifact as pandas DataFrame, artifact_name should exist in
        the context of current run. This function will only retrieve dataset artifact,
        not model or any other artifact type.
        Args:
            artifact_name: name for the artifact
        Returns:
            DataFrame representation of the artifact
        """
        artifact_local_path = self.wandb_run.use_artifact(artifact_name).file()

        try:
            data = pd.read_parquet(artifact_local_path)
        except FileNotFoundError as err:
            logger.error(f"{artifact_name} is not found")
            raise err

        return data

