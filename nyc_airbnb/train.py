import argparse
import json
import logging
import os
import shutil
import sys
from typing import Dict

import mlflow
import pandas as pd
import wandb
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

sys.path.append(".")
from nyc_airbnb.utils.base_runner import BaseRunner
from nyc_airbnb.utils.pipeline import get_inference_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(levelname)s - %(message)s")
logger = logging.getLogger()


class TrainModelRunner(BaseRunner):
    def __init__(self,
                 wandb_run,
                 label,
                 random_seed,
                 stratify_by,
                 val_size
                 ):
        super().__init__(wandb_run)
        self.label = label
        self.val_size = val_size
        self.stratify_by = stratify_by
        self.random_seed = random_seed

    def train(self,
              data: pd.DataFrame,
              rf_config: Dict[str, float],
              max_tfidf_features):
        """
        Train a model by running fit() method of pipeline.
        Args:
            data: A DataFrame of training dataset
            rf_config:
                A configuration dict to be used as model parameters
            max_tfidf_features:
                hyper-param for tfidf preprocessor
        """
        y = data[self.label]
        X = data.drop(self.label, axis=1)

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=self.val_size,
            stratify=X[self.stratify_by] if self.stratify_by is not None else None,
            random_state=self.random_seed
        )

        logger.info("Preparing pipeline")
        lasso_pipe, features = get_inference_pipeline(
            rf_config,
            max_tfidf_features
        )

        logger.info("Training model")
        trained_model = lasso_pipe.fit(X_train, y_train)

        # training performance metric
        y_pred = lasso_pipe.predict(X)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)

        return r2, mae, trained_model

    def persist_model(self, model, model_artifact_name: str):
        persist_dir = f'{model_artifact_name}_dir'

        # Remove if exists
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

        mlflow.sklearn.save_model(
            model,
            persist_dir,
        )

        self.log_model(
            model_artifact_name,
            "model_export",
            "Pytorch lasso model export",
            persist_dir)


if __name__ == "__main__":
    # Process arguments
    parser = argparse.ArgumentParser(description="Train the model")

    parser.add_argument(
        "label",
        type=str,
        help="Label column name"
    )
    parser.add_argument(
        "trainval_artifact",
        type=str,
        help="Artifact containing the training dataset. It will be split into train and validation"
    )
    parser.add_argument(
        "val_size",
        type=float,
        help="Size of the validation split. Fraction of the dataset, or number of items",
    )
    parser.add_argument(
        "random_seed",
        type=int,
        help="Seed for random number generator",
        default=42
    )
    parser.add_argument(
        "stratify_by",
        type=str,
        help="Column to use for stratification",
        default="none"
    )
    parser.add_argument(
        "rf_config",
        help="Random forest configuration. A JSON dict that will be passed to the "
             "scikit-learn constructor for RandomForestRegressor.",
        default="{}",
    )
    parser.add_argument(
        "max_tfidf_features",
        help="Maximum number of words to consider for the TFIDF",
        default=10,
        type=int
    )
    parser.add_argument(
        "output_artifact",
        type=str,
        help="Name for the output serialized model"
    )
    args = parser.parse_args()

    wandb_run = wandb.init(job_type="training_model")

    with open(args.rf_config) as fp:
        rf_config = json.load(fp)

    # Log model config to wandb runs
    wandb_run.config.update(rf_config)

    # Run training
    runner = TrainModelRunner(
        wandb_run,
        label=args.label,
        random_seed=args.random_seed,
        stratify_by=args.stratify_by if args.stratify_by != "none" else None,
        val_size=args.val_size
    )
    training_set = runner.retrieve_dataset_artifact(args.trainval_artifact)
    r2, mae, TRAINED_MODEL = runner.train(training_set, rf_config, args.max_tfidf_features)

    # Logging to wandb
    logger.info(f'R2 score is {r2}')
    logger.info(f'MAE loss is {mae}')
    wandb_run.summary['Training r2'] = r2
    wandb_run.log({
        "Training mae": mae,
        "Training r2": r2
    })

    # Persist model
    logger.info('Exporting model')
    runner.persist_model(TRAINED_MODEL, args.output_artifact)

    sys.exit(0)
