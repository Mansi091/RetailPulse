import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from src.database.connection import get_engine
from src.utils.logger import get_logger

logger = get_logger()


class CustomerSegmenter:
    """
    ML Customer Segmentation pipeline using K-Means clustering.
    Calculates Recency, Frequency, and Monetary (RFM) metrics from transaction records,
    scales features to handle skewness, performs K-Means clustering, and maps clusters
    to interpretable customer segments (VIP, Loyal, At-Risk, Hibernating).
    """

    def __init__(self):
        """Initializes database engine connection."""
        self.engine = get_engine()

    def load_data(self):
        """Loads cleaned online retail records from database staging table."""
        logger.info("loading data for customer segmentation")
        query = """
        SELECT
            "CustomerID",
            "InvoiceDate",
            "InvoiceNo",
            "Revenue"
        FROM online_retail_clean
        """
        df = pd.read_sql(query, self.engine)
        logger.info(f"Loaded {df.shape[0]} rows for segmentation")
        return df

    def calculate_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Recency, Frequency, and Monetary values for each customer.
        
        Parameters:
            df (pd.DataFrame): Input transactions.
            
        Returns:
            pd.DataFrame: RFM metrics grouped by CustomerID.
        """
        logger.info("calculating RFM metrics")
        
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        
        max_date = df["InvoiceDate"].max()
        reference_date = max_date + pd.Timedelta(days=1)
        
        rfm = df.groupby("CustomerID").agg(
            LastPurchase=("InvoiceDate", "max"),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum")
        ).reset_index()
        
        rfm["Recency"] = (reference_date - rfm["LastPurchase"]).dt.days
        rfm = rfm.drop(columns=["LastPurchase"])
        
        logger.info(f"Calculated RFM metrics for {len(rfm)} customers")
        return rfm

    def preprocess_and_segment(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses RFM features (handles zeros, applies log scaling, standardizes)
        and clusters customers using K-Means (K=4).
        
        Parameters:
            rfm (pd.DataFrame): Calculated RFM dataframe.
            
        Returns:
            pd.DataFrame: RFM dataframe enriched with cluster indices and segment names.
        """
        logger.info("preprocessing features and running K-Means clustering")
        
        rfm_features = rfm[["Recency", "Frequency", "Monetary"]].copy()
        rfm_features["Monetary"] = rfm_features["Monetary"].clip(lower=0.01)
        
        rfm_log = np.log1p(rfm_features)
        
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_log)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
        
        cluster_summary = rfm.groupby("Cluster").agg(
            Recency_mean=("Recency", "mean"),
            Frequency_mean=("Frequency", "mean"),
            Monetary_mean=("Monetary", "mean")
        ).reset_index()
        
        score_scaler = StandardScaler()
        summary_scaled = score_scaler.fit_transform(
            cluster_summary[["Recency_mean", "Frequency_mean", "Monetary_mean"]]
        )
        
        scores = -summary_scaled[:, 0] + summary_scaled[:, 1] + summary_scaled[:, 2]
        cluster_summary["Score"] = scores
        
        sorted_clusters = cluster_summary.sort_values(by="Score").reset_index(drop=True)
        
        labels = [
            "Hibernating / Lost",
            "At-Risk",
            "Loyal / Active",
            "VIP / Champions"
        ]
        
        cluster_mapping = {}
        for i, row in sorted_clusters.iterrows():
            cluster_mapping[int(row["Cluster"])] = labels[i]
            
        rfm["Segment"] = rfm["Cluster"].map(cluster_mapping)
        
        logger.info(f"Assigned segments: {rfm['Segment'].value_counts().to_dict()}")
        return rfm

    def save_to_db(self, rfm: pd.DataFrame):
        """
        Saves segmented RFM profiles to database dim_customer table, and optionally
        uploads the dimensions file to AWS S3 if enabled.
        """
        logger.info("saving segments back to dim_customer table")
        
        dim_customer = rfm[[
            "CustomerID", "Recency", "Frequency", "Monetary", "Segment"
        ]]
        
        # Database Save
        dim_customer.to_sql(
            "dim_customer",
            self.engine,
            if_exists="replace",
            index=False
        )
        logger.info(f"Successfully saved {len(dim_customer)} records to dim_customer")

        # Optional AWS S3 Upload
        from src.config.settings import (
            USE_S3,
            AWS_ACCESS_KEY_ID,
            AWS_SECRET_ACCESS_KEY,
            AWS_REGION,
            S3_BUCKET,
            S3_PROCESSED_KEY
        )
        import boto3
        import io

        if USE_S3 and S3_BUCKET:
            try:
                logger.info(f"Uploading segmented customer dimensions to S3: {S3_BUCKET}/{S3_PROCESSED_KEY}")
                s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                    region_name=AWS_REGION
                )
                csv_buffer = io.StringIO()
                dim_customer.to_csv(csv_buffer, index=False)
                s3_client.put_object(
                    Bucket=S3_BUCKET,
                    Key=S3_PROCESSED_KEY,
                    Body=csv_buffer.getvalue()
                )
                logger.info("Successfully uploaded dim_customer to S3.")
            except Exception as e:
                logger.error(f"Failed to upload dim_customer to S3: {e}")

    def run_segmentation(self):
        """Runs the complete customer segmentation pipeline end-to-end."""
        try:
            logger.info("starting customer segmentation pipeline")
            df = self.load_data()
            rfm = self.calculate_rfm(df)
            segmented_rfm = self.preprocess_and_segment(rfm)
            self.save_to_db(segmented_rfm)
            print("Customer segmentation pipeline completed successfully!")
        except Exception as e:
            logger.error(f"Customer segmentation failed: {e}")
            raise e