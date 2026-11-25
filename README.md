[README_Task04_Customer_Segmentation.md](https://github.com/user-attachments/files/35402209/README_Task04_Customer_Segmentation.md)
::: {align="center"}
# Customer Segmentation Analysis

**Discovering customer groups with K-Means clustering and data-driven
profiling.**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

**IncodeVision · Data Science Internship · Task 04**
:::

------------------------------------------------------------------------

## Overview

This project applies unsupervised machine learning to segment mall
customers according to their demographic and spending characteristics.
Using the Mall Customers dataset, the workflow explores customer
distributions, scales numerical features, evaluates candidate cluster
counts, and applies K-Means to identify distinct customer groups.

The resulting segments are profiled using average age, annual income,
spending score, and customer count. Visualizations---including an
income-versus-spending plot and a Principal Component Analysis (PCA)
projection---help interpret the clustering results.

## Business Objective

Customer segmentation can help a business understand differences in
customer behaviour and develop more relevant marketing strategies.

This analysis focuses on three numerical features:

-   **Age**
-   **Annual Income (k\$)**
-   **Spending Score (1--100)**

`CustomerID` is retained for identification and reporting, but it is not
used as a clustering feature. Gender is used for segment profiling
rather than model training.

## Dataset

-   **Dataset:** Mall Customers
-   **Records:** 200 customers
-   **Features available:** CustomerID, Gender, Age, Annual Income,
    Spending Score
-   **Input file:** `data/Mall_Customers.csv`

The supplied dataset contains 200 records. The initial inspection found
no missing values or duplicate rows.

## Methodology

### 1. Exploratory Data Analysis

-   Inspected dataset dimensions, data types, and descriptive
    statistics.
-   Examined the distributions of age, annual income, and spending
    score.
-   Reviewed correlations between the numerical features.

### 2. Feature Scaling

Applied `StandardScaler` to age, annual income, and spending score.
Scaling is important for distance-based algorithms such as K-Means
because features with larger numerical ranges can otherwise dominate the
distance calculation.

### 3. Selecting the Number of Clusters

Evaluated candidate values of **k from 2 to 10** using:

-   **Elbow Method:** compared within-cluster inertia.
-   **Silhouette Score:** evaluated how well observations fit within
    their assigned clusters compared with other clusters.

The highest silhouette score in the evaluated range was obtained at **k
= 6** (approximately **0.428**), which was selected for the final
clustering run.

### 4. K-Means Clustering

Trained the final K-Means model with:

-   `n_clusters = 6`
-   `n_init = 10`
-   `random_state = 42`

### 5. Visualization and Segment Profiling

-   Plotted customer clusters by annual income and spending score.
-   Used PCA to project the scaled three-dimensional feature space into
    two dimensions for visualization.
-   Calculated customer counts and average demographic and spending
    characteristics for each cluster.

## Results & Customer Profiles

The final model identifies six customer segments:

  ---------------------------------------------------------------------------------
       Cluster    Customers     Avg. age  Avg. income         Avg. Segment
                                                (k\$)     spending interpretation
                                                             score 
  ------------ ------------ ------------ ------------ ------------ ----------------
             3           39         32.7         86.5         82.1 High income,
                                                                   high spending

             4           23         25.0         25.3         77.6 Low income, high
                                                                   spending

             2           33         41.9         88.9         17.0 High income, low
                                                                   spending

             5           21         45.5         26.3         19.4 Low income, low
                                                                   spending

             0           45         56.3         54.3         49.1 Average income
                                                                   and spending;
                                                                   older customers

             1           39         26.8         57.1         48.1 Average income
                                                                   and spending;
                                                                   younger
                                                                   customers
  ---------------------------------------------------------------------------------

### Potential Business Recommendations

  -----------------------------------------------------------------------
  Segment                             Suggested action
  ----------------------------------- -----------------------------------
  High income, high spending          Prioritize loyalty benefits,
                                      premium experiences, and retention
                                      campaigns.

  Low income, high spending           Promote relevant offers, new
                                      arrivals, and value-oriented
                                      bundles.

  High income, low spending           Test personalized recommendations
                                      and targeted engagement campaigns.

  Low income, low spending            Use cost-efficient communication
                                      and monitor engagement.

  Average income and spending; older  Focus on retention and relevant
  customers                           regular offers.

  Average income and spending;        Encourage repeat visits through
  younger customers                   personalized campaigns and
                                      new-product promotions.
  -----------------------------------------------------------------------

These recommendations are interpretations of the observed segment
profiles, not the results of a controlled marketing experiment.

## Repository Structure

``` text
DS-Task04-Customer-Segmentation/
│
├── data/
│   └── Mall_Customers.csv
│
├── outputs/
│   ├── 01_distributions.png
│   ├── 02_correlation.png
│   ├── 03_elbow_silhouette.png
│   ├── 04_clusters_income_spending.png
│   ├── 05_clusters_pca.png
│   ├── 06_segment_spending.png
│   ├── customers_with_segments.csv
│   ├── segment_profile.csv
│   └── k_selection.csv
│
├── customer_segmentation.ipynb
├── customer_segmentation.py
├── requirements.txt
└── README.md
```

## Visual Results

### Feature distributions

![Feature distributions](outputs/01_distributions.png)

### Correlation between numerical features

![Correlation heatmap](outputs/02_correlation.png)

### Elbow Method and Silhouette Score

![Cluster selection](outputs/03_elbow_silhouette.png)

### Customer segments: Income vs. Spending Score

![Customer clusters](outputs/04_clusters_income_spending.png)

### Two-dimensional PCA visualization

![PCA cluster visualization](outputs/05_clusters_pca.png)

### Average spending score by segment

![Average spending by segment](outputs/06_segment_spending.png)

## Getting Started

### 1. Clone the repository

``` bash
git clone https://github.com/Saklin-labs/DS-Task04-Customer-Segmentation.git
cd DS-Task04-Customer-Segmentation
```

### 2. (Recommended) Create and activate a virtual environment

``` bash
python -m venv .venv
```

**Windows**

``` bash
.venv\Scripts\activate
```

**macOS / Linux**

``` bash
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Run the analysis

``` bash
python customer_segmentation.py
```

The script reads `data/Mall_Customers.csv` and generates the
visualizations, cluster assignments, model-selection table, and segment
profile in the `outputs/` directory.

For the complete step-by-step walkthrough, open
`customer_segmentation.ipynb` in Jupyter Notebook or VS Code.

## Technology Stack

  Tool / Library     Purpose
  ------------------ ----------------------------------------------------
  Python             Core programming language
  Pandas             Data manipulation and profiling
  NumPy              Numerical operations
  Matplotlib         Plotting
  Seaborn            Statistical visualization
  Scikit-learn       StandardScaler, K-Means, silhouette score, and PCA
  Jupyter Notebook   Interactive analysis and documentation

## Skills Demonstrated

-   Exploratory Data Analysis (EDA)
-   Feature selection for clustering
-   Feature standardization
-   Unsupervised machine learning
-   K-Means clustering
-   Elbow Method and Silhouette Score
-   PCA-based dimensionality reduction for visualization
-   Cluster profiling and interpretation
-   Translating analytical findings into business recommendations

## Author

**Saklin Mostak Molla**\
Data Science Student \| Aspiring Data Scientist

-   GitHub: [@Saklin-labs](https://github.com/Saklin-labs)

------------------------------------------------------------------------

::: {align="center"}
*Completed as part of the IncodeVision Data Science Internship --- Task
04.*
:::
