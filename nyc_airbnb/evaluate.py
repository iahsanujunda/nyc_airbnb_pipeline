import argparse
import logging
import sys
import tempfile

import mlflow
import pandas as pd
import wandb
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.append(".")
from nyc_airbnb.utils.base_runner import BaseRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(levelname)s - %(message)s")
logger = logging.getLogger()


class TestModelRunner(BaseRunner):
    def __init__(self, wandb_run):
        super().__init__(wandb_run)

    def test_data(self, data: pd.DataFrame, model, label: str):
        # Read test dataset
        X_test = data.copy()
        y_test = X_test.pop(label)

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate
        logger.info("Scoring")
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        logger.info(f"Score: {r2}")
        logger.info(f"MAE: {mae}")

        # Log MAE and r2
        self.wandb_run.summary['Test r2'] = r2
        self.wandb_run.summary['Test mae'] = mae

        return y_pred, r2, mae


if __name__ == "__main__":
    # Process arguments
    parser = argparse.ArgumentParser(description="Clean the training dataset")
    parser.add_argument("test_model",
                        type=str,
                        help="Artifact name of trained model")
    parser.add_argument("test_dataset",
                        type=str,
                        help="Artifact name of test dataset")
    parser.add_argument("label",
                        type=str,
                        help="Label of test dataset")
    args = parser.parse_args()

    wandb_run = wandb.init(job_type="test_model")

    runner = TestModelRunner(wandb_run)
    test_set = runner.retrieve_dataset_artifact(args.test_dataset)
    model = runner.retrieve_model(args.test_model)
    _ = runner.test_data(test_set, model, args.label)

    sys.exit(0)
