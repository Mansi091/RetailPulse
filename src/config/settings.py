import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path.cwd()

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "Online Retail.csv"

REPORTS_DIR = BASE_DIR / "data" / "reports"
VALIDATION_REPORT_PATH = REPORTS_DIR / "validation_report.csv"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
CLEANED_DATA_PATH = PROCESSED_DIR / "clean_retail_data.csv"

DB_TYPE = os.getenv("DB_TYPE", "postgresql").lower()
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "retail_warehouse")

# AWS S3 Settings
USE_S3 = os.getenv("USE_S3", "False").lower() in ("true", "1", "yes")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_RAW_KEY = os.getenv("S3_RAW_KEY", "raw/Online Retail.csv")
S3_PROCESSED_KEY = os.getenv("S3_PROCESSED_KEY", "processed/dim_customer.csv")


