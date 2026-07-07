import pandas as pd

from src.database.connection import get_engine
from src.utils.logger import get_logger

logger = get_logger()


class DimensionBuilder:
    """
    Builds the dimension tables for the star schema from the cleaned retail database.
    Generates and replaces dim_customer, dim_product, dim_country, and dim_date tables
    to optimize database queries and visual analytics.
    """

    def __init__(self):
        """Initializes database engine and loads cleaned transactions into memory."""
        self.engine = get_engine()
        self.df = pd.read_sql(
            "SELECT * FROM online_retail_clean",
            self.engine
        )

    def create_dim_customer(self):
        """Creates the customer dimension table 'dim_customer' and writes it to the database."""
        logger.info(
            "creating dim customer"
        )

        dim_customer = (
            self.df[
                ["CustomerID"]
            ]
            .dropna()
            .drop_duplicates()
            .reset_index(drop=True)
        )

        dim_customer.to_sql(
            "dim_customer",
            self.engine,
            if_exists="replace",
            index=False
        )

        logger.info(
            f"dim_customer created with {len(dim_customer)} rows"
        )

    def create_dim_product(self):
        """Creates the product dimension table 'dim_product' and writes it to the database."""
        logger.info(
            "creating dim product"
        )

        dim_product = (
            self.df[
                [
                    "StockCode",
                    "Description"
                ]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        dim_product.to_sql(
            "dim_product",
            self.engine,
            if_exists="replace",
            index=False
        )

        logger.info(
            f"dim_product created with {len(dim_product)} rows"
        )

    def create_dim_country(self):
        """Creates the country dimension table 'dim_country' and writes it to the database."""
        logger.info(
            "creating dim country"
        )

        dim_country = (
            self.df[
                ["Country"]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        dim_country.to_sql(
            "dim_country",
            self.engine,
            if_exists="replace",
            index=False
        )

        logger.info(
            f"dim_country created with {len(dim_country)} rows"
        )

    def create_dim_date(self):
        """Creates the date dimension table 'dim_date' and writes it to the database."""
        logger.info(
            "creating dim date"
        )

        dim_date = (
            self.df[
                [
                    "InvoiceDate",
                    "Year",
                    "Month",
                    "Quarter"
                ]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        dim_date.to_sql(
            "dim_date",
            self.engine,
            if_exists="replace",
            index=False
        )

        logger.info(
            f"dim_date created with {len(dim_date)} rows"
        )

    def build_dimensions(self):
        """Orchestrates building all dimension tables sequentially."""
        self.create_dim_customer()

        self.create_dim_product()

        self.create_dim_country()

        self.create_dim_date()

        logger.info(
            "all dimension tables created successfully."
        )