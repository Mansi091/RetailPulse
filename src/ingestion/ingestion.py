import pandas as pd
import boto3
import io
from src.utils.logger import get_logger
from src.config.settings import (
    USE_S3,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_REGION,
    S3_BUCKET,
    S3_RAW_KEY
)

logger = get_logger()

class DataIngestion:

    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        try:
            if USE_S3 and S3_BUCKET:
                logger.info(f"Starting data ingestion from S3 bucket: {S3_BUCKET}/{S3_RAW_KEY}")
                s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                    region_name=AWS_REGION
                )
                response = s3_client.get_object(Bucket=S3_BUCKET, Key=S3_RAW_KEY)
                df = pd.read_csv(io.BytesIO(response["Body"].read()))
            else:
                logger.info(f"Starting data ingestion from local path: {self.file_path}")
                df = pd.read_csv(self.file_path)

            logger.info(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns")
            return df

        except Exception as e:
            logger.error(f"Ingestion Failed : {e}")
            raise