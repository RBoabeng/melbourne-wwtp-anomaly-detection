# Unsupervised Anomaly Detection in Wastewater SCADA Systems

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange.svg)
![Status](https://img.shields.io/badge/Status-Work_in_Progress-yellow.svg)

## Executive Summary

Modern wastewater treatment plants (WWTPs) generate terabytes of SCADA (Supervisory Control and Data Acquisition) data, yet subtle operational shifts or toxic shock loads often go unnoticed until physical equipment fails or permit violations occur.

This project leverages **Unsupervised Machine Learning (PCA & Isolation Forests)** to build an automated "early warning radar" for a full-scale WWTP in Melbourne, Australia. By analyzing high-frequency energy, climate, and hydraulic data, this model mathematically isolates operational anomalies without relying on pre-labeled target variables.

**Domain Application:** Civil Engineering, Water Infrastructure, Cyber-Physical Systems.
**Technical Focus:** Time-Series Preprocessing, Dimensionality Reduction, Unsupervised Outlier Detection.

---

## Business & Academic Value

* **Operational Resilience:** Detects sudden changes in plant operating regimes (e.g., heavy rainfall events, toxic biological shocks, or sensor drift) before they cascade into system failures.
* **Cost Reduction:** Enables predictive maintenance and reduces the energy footprint by identifying anomalous power consumption patterns.
* **MLOps Readiness:** Demonstrates end-to-end handling of messy, auto-correlated industrial sensor data, moving from raw tabular data to actionable visualizations.

---

## Data Source

The data utilized is the **Full Scale Waste Water Treatment Plant Data (Melbourne)**.

* **Features Include:** Power consumption, hydraulic flow, biological characteristics, and climate data merged via daily timestamps over a 6-year period (2014-2019).
* *Note: Raw datasets are excluded from this repository in accordance with MLOps best practices. Instructions to download the data are provided below.*

---

## Architecture & Tech Stack

1. **Data Engineering:** `pandas`, `numpy` (Time-series imputation, rolling averages, lag features).
2. **Dimensionality Reduction:** `scikit-learn` (Principal Component Analysis to compress high-dimensional sensor data into visualizable 2D/3D spaces).
3. **Anomaly Detection:** `scikit-learn` (Isolation Forest to flag multivariate outliers).
4. **Visualization:** `matplotlib`, `seaborn` (Time-series plotting and anomaly highlighting).

---

## Project Structure

```
melbourne-wwtp-anomaly-detection/
├── data/
│   ├── raw/               <- data from kaggle
│   ├── processed/         <- Cleaned time-series data
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb   <- Data profiling and time-series setup
│   ├── 02_pca_visualization.ipynb  <- Dimensionality reduction
│   ├── 03_isolation_forest.ipynb   <- Unsupervised modeling
├── src/                   <- Refactored Python modules (e.g., preprocessing.py)
├── .gitignore
├── README.md
└── requirements.txt

```
---

## How to Run Locally

**1. Clone the Repository:**

```
git clone [https://github.com/](https://github.com/)[YourUsername]/melbourne-wwtp-anomaly-detection.git
cd melbourne-wwtp-anomaly-detection

```

**2.Install dependencies:**

```
pip install -r requirements.txt

```


**3.Download the Data:**

* Visit Kaggle: Full Scale Waste Water Treatment Plant Data.

* Download `Data-Melbourne_F_fixed.csv` and place it in the `data/raw/` directory.


**4.Run the Notebooks:**

* Launch Jupyter and execute the notebooks in the `notebooks/` directory sequentially.

## Author

**Richard Boabeng**, Civil Engineer & Data Scientist

