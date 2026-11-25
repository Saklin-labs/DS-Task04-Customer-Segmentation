[README_Task01_Data_Cleaning.md](https://github.com/user-attachments/files/35402169/README_Task01_Data_Cleaning.md)
::: {align="center"}
# Data Cleaning & Preprocessing Pipeline

**Transforming raw Titanic passenger data into clean, consistent,
analysis-ready datasets.**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

**IncodeVision · Data Science Internship · Task 01**
:::

------------------------------------------------------------------------

## Overview

This project demonstrates an end-to-end data cleaning and preprocessing
workflow using the Titanic passenger dataset. The objective is to
identify and address common data quality issues---such as missing
values, duplicate records, inconsistent categorical values, and
numerical outliers---and prepare reliable datasets for further analysis
and machine learning.

The workflow is implemented in both a Jupyter Notebook and a reusable
Python script. Key cleaning decisions are documented, and intermediate
visualizations and summary files are generated for transparency.

## Dataset

-   **Dataset:** Titanic passenger dataset
-   **Initial dimensions:** 891 rows × 15 columns
-   **Source:** [seaborn-data --- Titanic
    dataset](https://github.com/mwaskom/seaborn-data)
-   **Data format:** CSV

The dataset contains passenger attributes such as age, sex, passenger
class, fare, family information, embarkation point, and survival status.

## Key Results

  Data quality measure     Before cleaning   After cleaning
  ---------------------- ----------------- ----------------
  Rows                                 891              784
  Columns                               15               12
  Missing cells                        869                0
  Exact duplicate rows                 107                0

> **Data integrity note:** Seven rows appear identical after
> missing-value imputation. Because the dataset does not provide a
> unique passenger identifier, these records were retained rather than
> assumed to be duplicate passengers.

## Workflow

1.  **Initial inspection**
    -   Reviewed dataset dimensions, data types, missing values, and
        duplicate records.
    -   Generated a visualization of missing-value patterns.
2.  **Duplicate handling**
    -   Removed 107 exact duplicate rows.
    -   Reset the DataFrame index after deduplication.
3.  **Data standardization**
    -   Trimmed whitespace from text columns.
    -   Standardized categorical text casing for consistent values.
4.  **Redundant feature removal**
    -   Removed `alive`, `class`, and `embark_town` because they
        duplicate information represented by other columns.
5.  **Missing-value treatment**
    -   `deck`: filled missing values with `Unknown` because the column
        had a high proportion of missing data.
    -   `embarked`: filled missing values using the mode.
    -   `age`: imputed using the median within each `pclass` and `sex`
        group.
6.  **Data type correction**
    -   Converted suitable columns to integer, boolean, and categorical
        data types.
7.  **Outlier treatment**
    -   Applied the Interquartile Range (IQR) rule to identify outliers
        in `fare` and `age`.
    -   Capped extreme values instead of deleting observations.
8.  **Feature engineering and preprocessing**
    -   Created `family_size` from `sibsp`, `parch`, and the passenger.
    -   Applied one-hot encoding to categorical features.
    -   Standardized selected numerical features for the ML-ready
        output.
9.  **Validation and export**
    -   Checked that the cleaned dataset contains no missing cells.
    -   Exported cleaned data, ML-ready data, a summary table, and a
        processing log.

## Repository Structure

``` text
DS-Task01-Data-Cleaning/
│
├── data/
│   ├── titanic_raw.csv
│   ├── titanic_clean.csv
│   └── titanic_ml_ready.csv
│
├── outputs/
│   ├── 01_missing_before.png
│   ├── 02_outliers_fare.png
│   ├── 03_missing_after.png
│   ├── cleaning_summary.csv
│   └── cleaning_log.txt
│
├── data_cleaning.ipynb
├── data_cleaning.py
├── requirements.txt
└── README.md
```

## Visual Results

### Missing values before cleaning

![Missing values before cleaning](outputs/01_missing_before.png)

### Outlier treatment

![Fare outlier treatment](outputs/02_outliers_fare.png)

### Missing values after cleaning

![Missing values after cleaning](outputs/03_missing_after.png)

## Getting Started

### 1. Clone the repository

``` bash
git clone https://github.com/Saklin-labs/DS-Task01-Data-Cleaning.git
cd DS-Task01-Data-Cleaning
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

### 4. Run the pipeline

``` bash
python data_cleaning.py
```

The script reads `data/titanic_raw.csv` and generates the cleaned
datasets and output files in the project directories.

To explore the full step-by-step analysis, open `data_cleaning.ipynb` in
Jupyter Notebook or VS Code.

## Technology Stack

  Tool / Library     Purpose
  ------------------ ---------------------------------------------------
  Python             Core programming language
  Pandas             Data manipulation and cleaning
  NumPy              Numerical operations
  Matplotlib         Data visualization
  Seaborn            Statistical plots and missing-value visualization
  Jupyter Notebook   Interactive analysis and documentation

## Skills Demonstrated

-   Data quality assessment
-   Missing-value analysis and imputation
-   Duplicate detection and removal
-   Categorical data standardization
-   Data type correction
-   IQR-based outlier detection and capping
-   Feature engineering
-   One-hot encoding and numerical scaling
-   Reproducible Python data workflows

## Author

**Saklin Mostak Molla**\
Data Science Student \| Aspiring Data Scientist

-   GitHub: [@Saklin-labs](https://github.com/Saklin-labs)

------------------------------------------------------------------------

::: {align="center"}
*Completed as part of the IncodeVision Data Science Internship --- Task
01.*
:::
