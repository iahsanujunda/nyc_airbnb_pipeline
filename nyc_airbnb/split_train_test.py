# ./nyc_airbnb/split_train_test.py
import argparse
import logging
import sys
import os
import tempfile

import wandb
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(".")
from nyc_airbnb.utils.base_runner import BaseRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(levelname)s - %(message)s")
logger = logging.getLogger()


class SplitRunner(BaseRunner):
    def __init__(self,
                 wandb_run,
                 test_size,
                 random_seed,
                 stratify_by):
        super().__init__(wandb_run)
        self.test_size = test_size
        self.random_seed = random_seed
        self.stratify_by = stratify_by

    def split_train_test(self,
                         data: pd.DataFrame,
                         dir_name: str):
        trainval, test = train_test_split(
            data,
            test_size=float(self.test_size),
            random_state=int(self.random_seed),
            stratify=data[self.stratify_by] if self.stratify_by != 'none' else None,
        )

        logger.info(f'train proportion contains {trainval.shape[0]}')
        logger.info(f'test proportion contains {test.shape[0]}')

        file_dict = {}
        for data_frame, name in zip([trainval, test], ['trainval', 'test']):
            logger.info(f"Uploading {name}_data.parquet dataset")
            temp_file = os.path.join(dir_name, f'{name}_data.parquet')
            data_frame.to_parquet(
                temp_file,
                index=False,
                engine='pyarrow',
                compression='gzip')
            file_dict[f'{name}_data'] = temp_file

        return file_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split training dataset")
    parser.add_argument("input_artifact",
                        type=str,
                        help="Reference to mlflow artifact of input data")
    parser.add_argument("test_size",
                        type=str,
                        help="The size of test data")
    parser.add_argument("random_seed",
                        type=str,
                        help="Random seed")
    parser.add_argument("stratify_by",
                        type=str,
                        help="Column to use for stratification")
    args = parser.parse_args()

    runner = SplitRunner(
        wandb.init(job_type="split_data"),
        args.test_size,
        args.random_seed,
        args.stratify_by
    )
    dataset = runner.retrieve_dataset_artifact(args.input_artifact)

    with tempfile.TemporaryDirectory() as temp_dir:
        files = runner.split_train_test(dataset, temp_dir)
        for key, file_name in files.items():
            _ = runner.log_artifact(
                f'{key}.parquet',
                key,
                f'{key} split of the dataset',
                file_name
            )

    sys.exit(0)

