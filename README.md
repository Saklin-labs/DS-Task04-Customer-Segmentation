[task_04_professional_readme.md](https://github.com/user-attachments/files/35402877/task_04_professional_readme.md)
> [!NOTE]
> **IncodeVision Data Science Internship · Task 04 (Track Code: DS)**
>
> An end-to-end unsupervised machine learning pipeline that segments **200 mall customers** into **6 actionable marketing personas** using K-Means clustering, PCA, and automated z-score profiling.

<div align="center">

<h1>Customer Segmentation (K-Means)</h1>

**Turn raw customer behavior into 6 actionable marketing personas.**

An unsupervised clustering and customer profiling pipeline built on the Mall Customers dataset using `scikit-learn`, `pandas`, and `seaborn`.

![License](https://img.shields.io/badge/license-MIT-5865F2?style=flat-square)
![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-KMeans%20%7C%20PCA-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data%20profiling-150458?style=flat-square&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/notebook-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

<br/>

[Segments](#-customer-segments--strategy) · [Methodology](#-pipeline-architecture) · [Project Structure](#-project-structure) · [Quick Start](#-quick-start) · [Visuals](#-additional-eda--profiling-charts)

</div>

<br/>

<table>
  <tr>
    <td align="center" width="33%">
      <img src="outputs/03_elbow_silhouette.png" alt="Elbow and Silhouette Score" />
      <br/><br/>
      <b>Optimal k Selection</b>
      <br/>
      <sub>elbow + silhouette · best k = 6 (0.428)</sub>
    </td>
    <td align="center" width="33%">
      <img src="outputs/04_clusters_income_spending.png" alt="Income vs Spending Score Clusters" />
      <br/><br/>
      <b>Income vs. Spending</b>
      <br/>
      <sub>6 distinct segments · 200 customers</sub>
    </td>
    <td align="center" width="33%">
      <img src="outputs/05_clusters_pca.png" alt="2D PCA Cluster Projection" />
      <br/><br/>
      <b>2D PCA Projection</b>
      <br/>
      <sub>age, income &amp; spending · 2D view</sub>
    </td>
  </tr>
</table>

---

## Customer Segments & Strategy

**Dataset:** Mall Customers (`200 rows × 5 columns`: `CustomerID`, `Gender`, `Age`, `Income`, `SpendingScore`) — verified clean with **0 missing values** and **0 duplicates**.

| Cluster | Customers | Avg Age | Avg Income | Avg Spending | Female % | Segment Persona | Recommended Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **3** | 39 | 32.7 | $86.5k | 82.1 | 53.8% | **VIP High-Spenders** *(High income, high spending, middle-aged)* | Prioritize with loyalty tiers, VIP perks, and premium launches. |
| **4** | 23 | 25.0 | $25.3k | 77.6 | 56.5% | **Trend-Driven Youth** *(Low income, high spending, younger)* | Target with flash sales, social media drops, and bundle deals. |
| **2** | 33 | 41.9 | $88.9k | 17.0 | 42.4% | **Untapped High-Earners** *(High income, low spending, middle-aged)* | Highest growth opportunity: run personalized re-engagement campaigns. |
| **0** | 45 | 56.3 | $54.3k | 49.1 | 57.8% | **Steady Mature Base** *(Average income, average spending, older)* | Retain with consistent value offers and seasonal promotions. |
| **1** | 39 | 26.8 | $57.1k | 48.1 | 64.1% | **Emerging Core** *(Average income, average spending, younger)* | Nurture early via membership rewards to grow lifetime value. |
| **5** | 21 | 45.5 | $26.3k | 19.4 | 61.9% | **Budget-Conscious** *(Low income, low spending, middle-aged)* | Keep acquisition cost low; focus on clearance and discount alerts. |

> [!TIP]
> **Why `k = 6` instead of `k = 5`?**
>
> Evaluating $k \in [2, 10]$ on standardized `Age`, `Income`, and `SpendingScore` shows the highest Silhouette Score at **`k = 6` (`0.428`)** versus `k = 5` (`0.417`). Adding `Age` cleanly separates average spenders into **Older Steady Shoppers (Cluster 0, avg age 56.3)** and **Younger Growth Shoppers (Cluster 1, avg age 26.8)**.

---

## Pipeline Architecture

Every step is reproducible across both the Jupyter notebook and the standalone Python script:

| Step | Stage | What Happens |
| :---: | :--- | :--- |
| **1** | **Data Audit** | Loads `Mall_Customers.csv` (`200 × 5`), standardizes column names, and confirms `0` nulls and `0` duplicates. |
| **2** | **EDA** | Plots KDE histograms for `Age`, `Income`, and `SpendingScore` plus a feature correlation heatmap. |
| **3** | **Scaling** | Applies `StandardScaler` to `Age`, `Income`, and `SpendingScore` so all features contribute equally to Euclidean distance. |
| **4** | **Model Selection** | Sweeps `k = 2..10` using Elbow (Inertia) and Silhouette Score, exporting metrics to `outputs/k_selection.csv`. |
| **5** | **Clustering** | Fits final `KMeans(n_clusters=6, n_init=10, random_state=42)` and assigns cluster labels. |
| **6** | **Visualization** | Generates Income vs. Spending scatter plot and 2D PCA projection (`PCA(n_components=2)`). |
| **7** | **Profiling & Export** | Computes per-cluster stats, auto-labels personas via z-score thresholds, and saves final CSVs & bar chart. |

---

## Additional EDA & Profiling Charts

<details>
<summary><b>Click to view EDA distributions, correlation heatmap, and segment spending chart</b></summary>
<br/>

<table>
  <tr>
    <td align="center" width="33%">
      <img src="outputs/01_distributions.png" alt="Feature Distributions" />
      <br/><sub><b>Feature Distributions</b></sub>
    </td>
    <td align="center" width="33%">
      <img src="outputs/02_correlation.png" alt="Correlation Heatmap" />
      <br/><sub><b>Correlation Matrix</b></sub>
    </td>
    <td align="center" width="33%">
      <img src="outputs/06_segment_spending.png" alt="Average Spending per Segment" />
      <br/><sub><b>Avg Spending per Cluster</b></sub>
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
Open the full walkthrough with inline outputs in [`customer_segmentation.ipynb`](customer_segmentation.ipynb), or execute the script directly:
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
