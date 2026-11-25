# Customer Segmentation — K-Means Clustering

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />
</p>

<p align="center">
  <b>An unsupervised machine learning project that identifies customer groups based on age, annual income, and spending behaviour.</b>
</p>

---

## Project Overview

Understanding customer behaviour is important for businesses that want to improve marketing strategies, customer engagement, and business decisions.

This project applies K-Means clustering to the Mall Customers dataset to identify groups of customers with similar characteristics.

The analysis uses customer age, annual income, and spending score to discover meaningful customer segments without predefined labels.

## Dataset Information

| Attribute | Details |
|---|---|
| Dataset | Mall Customers |
| Total records | 200 |
| Missing values | 0 |
| Duplicate records | 0 |
| Clustering features | Age, Annual Income, Spending Score |
| Algorithm | K-Means Clustering |
| Selected clusters | 6 |

`CustomerID` was excluded from clustering, while `Gender` was reserved for customer profiling.

## Model Results

<details>
<summary><b>Customer Segment Analysis</b></summary>

| Cluster | Customers | Avg. Age | Avg. Income | Avg. Spending Score |
|---|---:|---:|---:|---:|
| 3 | 39 | 32.7 | 86.5 | 82.1 |
| 4 | 23 | 25.0 | 25.3 | 77.6 |
| 2 | 33 | 41.9 | 88.9 | 17.0 |
| 5 | 21 | 45.5 | 26.3 | 19.4 |
| 0 | 45 | 56.3 | 54.3 | 49.1 |
| 1 | 39 | 26.8 | 57.1 | 48.1 |

</details>

**Best silhouette score:** Approximately 0.428 at k = 6.

The selected model uses six clusters based on the cluster-selection analysis.

## Methodology

### 1. Data Exploration
- Loaded the customer dataset.
- Inspected dimensions, data types, and descriptive statistics.
- Checked missing values and duplicate records.

### 2. Feature Selection
Selected the following features:

- Age
- Annual Income
- Spending Score

Excluded `CustomerID` because it is an identifier rather than a meaningful behavioural feature.

### 3. Feature Scaling

Applied `StandardScaler` to standardize numerical features before clustering.

### 4. Optimal Cluster Selection

Evaluated different values of k from 2 to 10 using:

- Elbow Method
- Silhouette Score

The highest silhouette score was approximately 0.428 at k = 6.

### 5. K-Means Model Training

```python
KMeans(
    n_clusters=6,
    n_init=10,
    random_state=42
)
```

### 6. Customer Profiling

Analysed the resulting clusters using average age, annual income, and spending score to understand customer behaviour.

## Key Insights

- **Cluster 3:** Higher-income customers with high spending scores.
- **Cluster 4:** Younger customers with lower income but high spending scores.
- **Cluster 2:** Higher-income customers with low spending scores.
- **Cluster 5:** Lower-income customers with low spending scores.
- **Cluster 0:** Older customers with moderate income and spending.
- **Cluster 1:** Younger customers with moderate income and spending.

These segments can help businesses think about differentiated marketing and customer engagement strategies.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Repository Structure

```text
DS-Task04-Customer-Segmentation/
│
├── data/
│   └── Mall_Customers.csv
│
├── outputs/
│   ├── 01_distributions.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_elbow_method.png
│   ├── 04_silhouette_scores.png
│   ├── 05_customer_clusters.png
│   ├── 06_segment_spending.png
│   ├── customers_with_segments.csv
│   ├── segment_profile.csv
│   └── k_selection.csv
│
├── customer_segmentation.ipynb
├── customer_segmentation.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Saklin-labs/DS-Task04-Customer-Segmentation.git
cd DS-Task04-Customer-Segmentation
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**macOS / Linux**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python customer_segmentation.py
```

Alternatively, explore the complete workflow in `customer_segmentation.ipynb`.

## Skills Demonstrated

- Unsupervised Machine Learning
- K-Means Clustering
- Feature Selection
- Feature Scaling
- Elbow Method
- Silhouette Analysis
- Customer Segmentation
- Data Visualization
- Cluster Profiling
- Business-Oriented Data Interpretation

## Author

**Saklin Mostak Molla**

B.Sc. Data Science Student | Aspiring Data Scientist

- GitHub: [@Saklin-labs](https://github.com/Saklin-labs)
- LinkedIn: [Saklin Molla](https://www.linkedin.com/in/saklin-molla-824b292b3/)

---

<p align="center">
  <i>Using unsupervised learning to discover meaningful customer patterns.</i>
</p>
