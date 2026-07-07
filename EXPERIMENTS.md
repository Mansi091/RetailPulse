# Model Experiments & Selection Log

This document records the experimental process, iterations, and metrics analyzed to determine the final machine learning customer segmentation model for the RetailPulse pipeline.

## 🎯 Optimization Goal
Establish a robust customer segmentation model based on Recency, Frequency, and Monetary (RFM) buying patterns. The model must balance mathematical clustering cohesion (Silhouette Score) with marketing actionability and business interpretability.

---

## 📊 Comparison Summary

| Model / Approach | Silhouette Score | Business Interpretability | Notes / Recommendations |
| :--- | :--- | :--- | :--- |
| **Rule-based RFM (Quartiles)** | N/A | High | Uses static arbitrary quartile boundaries. Interpretable but does not adapt to underlying data distribution or feature correlations. |
| **K-Means (K=3)** | 0.3364 | Moderate | Grouping is too coarse; merges emerging high-frequency buyers with low-spend active buyers. |
| **K-Means (K=4)** | **0.3369** | **Very High** | **Optimal cohesion for K >= 3. Clear separation into distinct profiles: VIP/Champions, Loyal/Active, At-Risk, and Hibernating.** |
| **K-Means (K=5)** | 0.3193 | Moderate | Lower silhouette score; over-splits "Loyal" customers into two similar sub-groups, diluting marketing focus. |
| **Gaussian Mixture Model (GMM, K=4)** | 0.3115 | Low | Probabilistic (soft) clustering. Harder to deploy in standard CRM/Marketing systems compared to distance-based hard labels. |

---

## 📈 Evaluation Details

### 1. Rule-Based RFM Baseline
* **Description**: Customers are scored 1–4 on Recency, Frequency, and Monetary values based on quartiles. The total score (3–12) dictates segment grouping.
* **Pros**: Simple, fast, zero machine learning overhead.
* **Cons**: Does not account for skewed distribution (e.g., extremely high frequency outliers). Threshold boundaries are rigid and fail to capture complex boundaries.

### 2. K-Means Clustering ($K=4$)
* **Description**: Transactions are aggregated into RFM metrics, log-scaled to eliminate right skew, standardized, and partitioned using the K-Means algorithm.
* **Pros**: Maximizes cluster cohesion. Boundaries adjust dynamically to the distribution of actual data.
* **Cons**: Distance-based metrics can be sensitive to scaling (mitigated by applying log-scaling and standard scaling in preprocessing).

### 3. Alternative: Gaussian Mixture Model (GMM)
* **Description**: Standardized RFM features are modeled as a mixture of Gaussian distributions.
* **Pros**: Captures elliptical clusters and provides membership probabilities.
* **Cons**: Lower silhouette score on this dataset; probabilistic assignments are harder for business stakeholders to interpret for campaigns.

---

## 🔍 Validation Plots
Evaluation plots showing K-Means Inertia (Elbow Method) and Silhouette Scores across $K \in [2, 8]$ are saved in:
* **Elbow/Silhouette Curve**: `notebooks/cluster_selection_plot.png`
