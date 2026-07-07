import pandas as pd
from src.utils.logger import get_logger
from src.config.settings import VALIDATION_REPORT_PATH

logger = get_logger()


class DataValidator:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def generate_report(self):
        try:
            logger.info(
                "Starting validation process..."
            )

            total_count = len(self.df)
            null_customer_id = self.df["CustomerID"].isnull().sum()
            null_description = self.df["Description"].isnull().sum()
            negative_quantity = (self.df["Quantity"] <= 0).sum()
            negative_unit_price = (self.df["UnitPrice"] <= 0).sum()
            duplicate_rows = self.df.duplicated().sum()
            cancelled_orders = self.df["InvoiceNo"].astype(str).str.startswith("C").sum()

            report = {
                "total_rows": total_count,

                "null_customer_id": null_customer_id,
                "null_customer_id_pct": round((null_customer_id / total_count) * 100, 2) if total_count > 0 else 0,
                "decision_null_customer_id": "Excluded - cannot attribute revenue/segment without a customer key",

                "null_description": null_description,
                "null_description_pct": round((null_description / total_count) * 100, 2) if total_count > 0 else 0,
                "decision_null_description": "Excluded - transactions without descriptions are incomplete record anomalies",

                "negative_quantity": negative_quantity,
                "negative_quantity_pct": round((negative_quantity / total_count) * 100, 2) if total_count > 0 else 0,
                "decision_negative_quantity": "Excluded - represents returns, not completed sales",

                "negative_unit_price": negative_unit_price,
                "negative_unit_price_pct": round((negative_unit_price / total_count) * 100, 2) if total_count > 0 else 0,
                "decision_negative_unit_price": "Excluded - items with zero or negative price represent free promotions or administrative adjustments",

                "duplicate_rows": duplicate_rows,
                "duplicate_rows_pct": round((duplicate_rows / total_count) * 100, 2) if total_count > 0 else 0,
                "decision_duplicate_rows": "Excluded - duplicate transactions removed to prevent double-counting",

                "cancelled_orders": cancelled_orders,
                "cancelled_orders_pct": round((cancelled_orders / total_count) * 100, 2) if total_count > 0 else 0,
                "decision_cancelled_orders": "Excluded - cancellations don't reflect realized revenue",
            }

            logger.info(
                "Validation completed successfully"
            )

            return report
        except Exception as e:
            logger.error(f"Validation report generation failed: {e}")
            raise e

    def save_report(self, report):
        try:
            report_df = pd.DataFrame(report.items(), columns=["Metric","Value"])
            
            VALIDATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            report_df.to_csv(VALIDATION_REPORT_PATH, index=False)
            logger.info("Validation report saved")
        except Exception as e:
            logger.error(f"Saving validation report failed: {e}")
            raise e