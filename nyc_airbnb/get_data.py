import argparse
import logging
import os
import sys
import tempfile
import oss2
import pandas as pd
import wandb

sys.path.append(".")
from nyc_airbnb.utils.base_runner import BaseRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-15s %(levelname)s - %(message)s")
logger = logging.getLogger()
logger.info(sys.path)


class GetOSSDataRunner(BaseRunner):
    def __init__(self,
                 wandb_run,
                 artifact_name,
                 artifact_type,
                 artifact_description):
        super().__init__(wandb_run)
        self.artifact_name = artifact_name
        self.artifact_type = artifact_type
        self.artifact_description = artifact_description

    def get_oss_data(self,
                     bucket,
                     object_path,
                     local_directory):
        self.wandb_run.config.update({
            'bucket': bucket,
            'object-path': object_path
        })

        # Setup OSS Connection
        logger.info("Connecting to Aliyun")
        auth = oss2.Auth(
            os.environ['OSS_ACCESS_KEY_ID'],
            os.environ['OSS_ACCESS_KEY_SECRET']
        )
        bucket = oss2.Bucket(
            auth,
            'https://oss-ap-southeast-5.aliyuncs.com',
            bucket
        )

        object_list = []
        for obj in oss2.ObjectIteratorV2(bucket, prefix=object_path):
            object_list.append(obj.key)

        # Check exported file in OSS
        try:
            assert len(object_list) <= 2
        except AssertionError as err:
            logger.error('Expect OSS path to contain only 1 file')
            raise err

        object_key = object_list[-1]

        logger.info("Downloading object from OSS")
        temp_filename = os.path.join(local_directory, 'csv_file.csv')
        bucket.get_object_to_file(object_key, temp_filename)

        df = pd.read_csv(temp_filename)

        parquet_filename = str(f'{local_directory}/{self.artifact_name}')
        logger.info("Exporting pandas dataframe to %s" % parquet_filename)
        df.to_parquet(
            parquet_filename,
            index=False,
            engine='pyarrow',
            compression='gzip')

        return parquet_filename


if __name__ == "__main__":
    # process arguments
    parser = argparse.ArgumentParser(description="Download URL to a local destination")
    parser.add_argument("bucket", type=str, help="Name of the sample to download")
    parser.add_argument("object_path", type=str, help="Name of the sample to download")
    parser.add_argument("artifact_name", type=str, help="Name for the output artifact")
    parser.add_argument("artifact_type", type=str, help="Output artifact type.")
    parser.add_argument(
        "artifact_description", type=str, help="A brief description of this artifact"
    )
    args = parser.parse_args()

    # apply arguments to run
    runner = GetOSSDataRunner(
        wandb.init(job_type="download_file"),
        args.artifact_name,
        args.artifact_type,
        args.artifact_description
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        LOCAL_FILE = runner.get_oss_data(args.bucket, args.object_path, temp_dir)
        _ = runner.log_artifact(
            args.artifact_name,
            args.artifact_type,
            args.artifact_description,
            LOCAL_FILE
        )

    sys.exit(0)