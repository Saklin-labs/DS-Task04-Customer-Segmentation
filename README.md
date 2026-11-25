(https://github.com/user-attachments/files/35402970/task_04_professional_readme.1.md)
> [!NOTE]
> **IncodeVision Data Science Internship · Task 04 (Track Code: DS)**
>
> End-to-end unsupervised machine learning pipeline that segments **200 mall customers** into **6 actionable marketing personas** using K-Means clustering, PCA projection, and automated z-score profiling (`Silhouette Score: 0.428`).

<div align="center">
  <br />
  <img src="https://img.icons8.com/fluency/96/combo-chart.png" alt="Customer Segmentation Logo" width="84" height="84" />
  <h1>Customer Segmentation</h1>
  <p><strong>Turn raw shopping behavior into 6 actionable customer personas.</strong></p>
  <p>Open-source unsupervised K-Means clustering, PCA visualization, and automated profiling built with <code>scikit-learn</code>, <code>pandas</code>, and <code>seaborn</code>.</p>
  <p>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-6C63FF?style=flat" alt="MIT License" /></a>
    <img src="https://img.shields.io/badge/python-3.9%2B-3776AB?style=flat" alt="Python 3.9+" />
    <img src="https://img.shields.io/badge/scikit--learn-K--Means%20%2B%20PCA-F7931E?style=flat" alt="scikit-learn" />
    <img src="https://img.shields.io/badge/clusters-k%20%3D%206-10B981?style=flat" alt="Clusters k=6" />
    <img src="https://img.shields.io/badge/silhouette-0.428-EC4899?style=flat" alt="Silhouette 0.428" />
  </p>
  <p>
    <a href="#customer-segments--strategy">Segments</a> ·
    <a href="#pipeline-architecture">Pipeline</a> ·
    <a href="#eda--profiling-visuals">Visuals</a> ·
    <a href="#project-structure">Structure</a> ·
    <a href="#quick-start">Quick Start</a>
  </p>
</div>

<br />

<table>
  <tr>
    <td align="center" width="33%">
      <a href="outputs/03_elbow_silhouette.png">
        <img src="outputs/03_elbow_silhouette.png" alt="Elbow and Silhouette Curves" width="100%" />
      </a>
      <br />
      <strong>Optimal k Selection</strong><br />
      <sub>elbow & silhouette · best k = 6 · score 0.428</sub>
    </td>
    <td align="center" width="33%">
      <a href="outputs/04_clusters_income_spending.png">
        <img src="outputs/04_clusters_income_spending.png" alt="Income vs Spending Segments" width="100%" />
      </a>
      <br />
      <strong>Income vs. Spending</strong><br />
      <sub>k-means · 200 customers · 6 clusters</sub>
    </td>
    <td align="center" width="33%">
      <a href="outputs/05_clusters_pca.png">
        <img src="outputs/05_clusters_pca.png" alt="2D PCA Cluster Projection" width="100%" />
      </a>
      <br />
      <strong>2D PCA Projection</strong><br />
      <sub>age, income & spending · 2 components</sub>
    </td>
  </tr>
</table>

<br />

## Customer Segments & Strategy

**Dataset:** Mall Customers (`200 rows × 5 columns`: `CustomerID`, `Gender`, `Age`, `Income`, `SpendingScore`) — audited clean with **0 missing values** and **0 duplicate rows**.

| Cluster | Customers | Avg Age | Avg Income (k$) | Avg Spending | Female % | Segment Persona | Recommended Marketing Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **3** | 39 | 32.7 | 86.5 | 82.1 | 53.8% | **VIP High-Spenders**<br /><sub>High income · High spending · Middle-aged</sub> | Prioritize with VIP loyalty tiers, early access, and premium launches. |
| **4** | 23 | 25.0 | 25.3 | 77.6 | 56.5% | **Trend-Driven Youth**<br /><sub>Low income · High spending · Younger</sub> | Target with flash sales, social media campaigns, and new-arrival drops. |
| **2** | 33 | 41.9 | 88.9 | 17.0 | 42.4% | **Untapped High-Earners**<br /><sub>High income · Low spending · Middle-aged</sub> | Highest growth potential: run personalized offers to convert spending. |
| **0** | 45 | 56.3 | 54.3 | 49.1 | 57.8% | **Steady Mature Base**<br /><sub>Average income · Average spending · Older</sub> | Retain with consistent value bundles and seasonal promotions. |
| **1** | 39 | 26.8 | 57.1 | 48.1 | 64.1% | **Emerging Core**<br /><sub>Average income · Average spending · Younger</sub> | Engage early with membership perks to increase lifetime value. |
| **5** | 21 | 45.5 | 26.3 | 19.4 | 61.9% | **Budget-Conscious**<br /><sub>Low income · Low spending · Middle-aged</sub> | Keep acquisition costs low; focus on clearance and discount alerts. |

> [!TIP]
> **Why `k = 6` instead of `k = 5`?**
>
> Sweeping `k = 2..10` on standardized `Age`, `Income`, and `SpendingScore` yields the peak Silhouette Score at **`k = 6` (`0.428`)** over `k = 5` (`0.417`). Including `Age` splits average spenders into **Older Steady Shoppers (Cluster 0, avg age 56.3)** and **Younger Emerging Shoppers (Cluster 1, avg age 26.8)** for sharper targeting.

---

## Pipeline Architecture

Every stage is fully reproducible across both the interactive notebook and standalone script:

| Step | Stage | Description |
| :---: | :--- | :--- |
| **1** | **Data Audit** | Loads `Mall_Customers.csv` (`200 × 5`), standardizes column names, and verifies `0` nulls and `0` duplicates. |
| **2** | **EDA** | Generates KDE histograms for `Age`, `Income`, and `SpendingScore` alongside a feature correlation heatmap. |
| **3** | **Feature Scaling** | Applies `StandardScaler` to `Age`, `Income`, and `SpendingScore` so Euclidean distances remain unbiased. |
| **4** | **Model Selection** | Evaluates `k = 2..10` via Elbow (Inertia) and Silhouette Score, exporting metrics to `outputs/k_selection.csv`. |
| **5** | **Clustering** | Fits `KMeans(n_clusters=6, n_init=10, random_state=42)` and assigns each customer to a cluster. |
| **6** | **Visualization** | Produces an Income vs. Spending scatter plot and a 2D PCA cluster projection (`PCA(n_components=2)`). |
| **7** | **Profiling & Export** | Aggregates cluster stats, auto-labels personas via z-score thresholds, and saves final CSVs & charts. |

---

## EDA & Profiling Visuals

<details>
<summary><strong>Click to expand additional EDA & segment charts (3 plots)</strong></summary>
<br />

<table>
  <tr>
    <td align="center" width="33%">
      <a href="outputs/01_distributions.png">
        <img src="outputs/01_distributions.png" alt="Feature Distributions" width="100%" />
      </a>
      <br />
      <strong>Feature Distributions</strong><br />
      <sub>age, income & spending KDE</sub>
    </td>
    <td align="center" width="33%">
      <a href="outputs/02_correlation.png">
        <img src="outputs/02_correlation.png" alt="Correlation Heatmap" width="100%" />
      </a>
      <br />
      <strong>Correlation Matrix</strong><br />
      <sub>pairwise feature relationships</sub>
    </td>
    <td align="center" width="33%">
      <a href="outputs/06_segment_spending.png">
        <img src="outputs/06_segment_spending.png" alt="Average Spending per Segment" width="100%" />
      </a>
      <br />
      <strong>Spending by Segment</strong><br />
      <sub>mean spending score per cluster</sub>
    </td>
  </tr>
</table>

</details>

---

## Project Structure

```text
DS-Task04-Customer-Segmentation/
├── data/
│   └── Mall_Customers.csv              # Raw dataset (200 customers)
├── outputs/
│   ├── 01_distributions.png            # Age, Income & Spending histograms
│   ├── 02_correlation.png              # Correlation heatmap
│   ├── 03_elbow_silhouette.png         # Elbow & Silhouette score curves
│   ├── 04_clusters_income_spending.png # Income vs. Spending scatter plot
│   ├── 05_clusters_pca.png             # 2D PCA cluster visualization
│   ├── 06_segment_spending.png         # Average spending score per cluster
│   ├── customers_with_segments.csv     # Original data + assigned Cluster ID
│   ├── segment_profile.csv             # Aggregated stats + auto-generated persona names
│   └── k_selection.csv                 # Inertia & Silhouette metrics for k = 2..10
├── .gitignore
├── customer_segmentation.ipynb         # Interactive notebook with outputs (start here)
├── customer_segmentation.py            # Reusable standalone pipeline script
├── LICENSE                             # MIT License
├── README.md
└── requirements.txt
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Sohan186/DS-Task04-Customer-Segmentation.git
cd DS-Task04-Customer-Segmentation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

Open the full walkthrough with inline outputs in [`customer_segmentation.ipynb`](customer_segmentation.ipynb), or execute the standalone script:

```bash
python customer_segmentation.py
```

---

## Tech Stack & License

- **Core:** Python 3.9+, pandas, NumPy
- **Machine Learning:** scikit-learn (`StandardScaler`, `KMeans`, `PCA`, `silhouette_score`)
- **Visualization:** matplotlib, seaborn
- **Environment:** Jupyter Notebook
- **License:** Distributed under the [MIT License](LICENSE) © 2026 Saklin Mostak Molla
