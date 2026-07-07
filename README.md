# End-to-End E-Commerce ETL, ML Customer Segmentation & BI Pipeline

> 💼 **Business Problem & Context**
> E-commerce marketing teams need to prioritize retention spend. This pipeline segments customers by purchasing behavior so marketing can target "At-Risk" customers with win-back campaigns and reward "VIP" customers, instead of treating all customers uniformly.

An end-to-end data engineering and machine learning project that processes e-commerce retail transactions, cleans and validates the dataset, loads it into a PostgreSQL database, segments customers using K-Means clustering, and visualizes business metrics in Power BI.

---

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    %% Extract & Validate
    Raw[Online Retail.csv] -->|Ingest| ETL[main.py Pipeline]
    ETL -->|Quality Report| Val[validation_report.csv]
    
    %% Transform
    ETL -->|Clean & Feature Engineering| Clean[clean_retail_data.csv]
    
    %% Load
    Clean -->|Load via SQLAlchemy| Staging[PostgreSQL: online_retail_clean]
    
    %% Dim Creating
    Staging -->|Populate Dimensions| Dims[(dim_customer, dim_product, dim_country, dim_date)]
    
    %% ML Loop
    Staging -->|Train K-Means| ML[ML Segmentation]
    ML -->|Enrich Customer Data| Dims
    
    %% Power BI
    Dims -->|Import & Relationship| PBI[Power BI Dashboard]
    Staging -->|Query| PBI
```

---

## ✨ Key Features

1. **Robust Ingestion & Validation**: Reads over 540k transaction rows and generates a data-quality audit report tracking duplicates, missing customer IDs, negative values, and cancellations.
2. **Data Cleaning & Feature Engineering**: Filters out returns and null identifiers, creates custom financial features (`Revenue`), and parses temporal fields (`Year`, `Month`, `Quarter`, `Day`, `Weekday`).
3. **Star Schema Dimension Modeling**: Loads data into a PostgreSQL staging table and splits it into structured dimension tables (`dim_customer`, `dim_product`, `dim_country`, `dim_date`).
4. **ML Customer Segmentation**: Extracts RFM (Recency, Frequency, Monetary) metrics, handles features skewness using log-scaling, standardizes values, and trains a K-Means model to segment customers into value-based profiles.
5. **Docker Compose Orchestration**: Simplifies building and executing the pipeline container with a single command, passing credentials and network configurations dynamically.
6. **Power BI Dashboarding**: Pre-designed dashboard mapping KPIs, monthly revenue trends, product performance, and geospatial sales, with support for customer segment slicing.

---

## 🧠 ML Customer Segmentation Insights

The optimal number of clusters was determined to be **$K=4$** by evaluating the elbow curve of inertia and the silhouette scores for $K \in [2, 8]$:

- **$K=2$** yields a high silhouette score ($0.4325$), but represents an oversimplified segmentation (e.g., active vs. inactive) which is not actionable for personalized marketing.
- For **$K \ge 3$**, **$K=4$** achieves the highest silhouette score ($0.3369$) compared to **$K=3$** ($0.3364$) and **$K=5$** ($0.3193$).
- The elbow plot of inertia shows a distinct slow down in the rate of descent at **$K=4$** (bending from $4882.65$ at $K=3$ to $3953.00$ at $K=4$).

The K-Means clustering model partitions customer RFM behavior into 4 distinct value-based profiles:

| Segment | Description | Profile |
| :--- | :--- | :--- |
| **VIP / Champions** | Highest value customers | Lowest Recency (active purchase), highest order count and total spend. |
| **Loyal / Active** | Frequent purchasers | Low Recency, moderate-to-high order count and spend. |
| **At-Risk** | Formerly active customers | Moderate-to-high Recency, below-average frequency and spend. |
| **Hibernating / Lost** | Dormant customers | High Recency, low frequency, and low spend. |

*Visual validation plot can be found in `notebooks/cluster_selection_plot.png`.*

---

## 🚀 Setup & Execution Instructions

### Prerequisites
* Python 3.12
* PostgreSQL Database
* Docker Desktop (Optional, for containerized run)

### Local Development Setup
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <your-repository-url>
   cd DATA_WAREHOUSING
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory:
   ```env
   DB_TYPE=postgresql
   DB_USER=postgres
   DB_PASSWORD=your_database_password
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=retail_dw
   ```
5. Place the raw `Online Retail.csv` in `data/raw/` and execute the pipeline:
   ```bash
   python main.py
   ```

---

### Docker Deployment
You can build and run the pipeline inside a Docker container while connecting to your host PostgreSQL database:

* **Using Docker Compose (Recommended & Simplest)**:
  Run this single command to build the image and run the container:
  ```bash
  docker compose up --build
  ```

* **Using Raw Docker Commands**:
  1. Build the Docker image:
     ```bash
     docker build -t retail-etl .
     ```
  2. Run the pipeline container:
     ```bash
     docker run --env-file .env -e DB_HOST=host.docker.internal --add-host=host.docker.internal:host-gateway retail-etl
     ```
