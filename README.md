# 🌊 Unsupervised Anomaly Detection in Wastewater SCADA Systems

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_App-red.svg)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)

## 📌 Executive Summary

Modern wastewater treatment plants (WWTPs) generate terabytes of SCADA (Supervisory Control and Data Acquisition) data, yet subtle operational shifts or toxic shock loads often go unnoticed until physical equipment fails or permit violations occur.

This project leverages **Unsupervised Machine Learning (PCA & Isolation Forests)** to build an automated "early warning radar" for a full-scale WWTP in Melbourne, Australia. By analyzing high-frequency energy, climate, and hydraulic data, this model mathematically isolates operational anomalies without relying on pre-labeled target variables.

The project culminates in a live, interactive web dashboard deployed via Streamlit.

**Domain Application:** Civil Engineering, Water Infrastructure, Cyber-Physical Systems.  
**Technical Focus:** Time-Series Preprocessing, Dimensionality Reduction, Unsupervised Outlier Detection, MLOps Configuration.

---

## 🎯 Business & Academic Value

* **Operational Resilience:** Detects sudden changes in plant operating regimes (e.g., heavy rainfall events, toxic biological shocks, or sensor drift) before they cascade into system failures.
* **Cost Reduction:** Enables predictive maintenance and reduces the energy footprint by identifying anomalous power consumption patterns.
* **MLOps Readiness:** Demonstrates end-to-end handling of messy, auto-correlated industrial sensor data using configuration-driven data loaders and modular object-oriented programming.

---

## 📊 Data Source

The data utilized is the **Full Scale Waste Water Treatment Plant Data (Melbourne)**.

* **Features Include:** Power consumption, hydraulic flow, biological characteristics, and climate data merged via daily timestamps over a 6-year period (2014-2019).
* *Note: Raw datasets are excluded from this repository in accordance with best practices. Instructions to download the data are provided below.*

---

## 🧠 Architecture & Tech Stack

1. **Data Engineering:** `pandas`, `numpy` (Time-series imputation, datetime indexing).
2. **Configuration Management:** `pyyaml` (Externalized parameters and feature selection).
3. **Dimensionality Reduction:** `scikit-learn` (Principal Component Analysis).
4. **Anomaly Detection:** `scikit-learn` (Isolation Forest to flag multivariate outliers).
5. **Interactive Dashboard:** `streamlit` (Live UI for dynamic parameter tuning and visualization).

---

## 📂 Project Structure

```
melbourne-wwtp-anomaly-detection/
├── config/
│   └── config.yaml                 <- Centralized parameters and file paths
├── data/
│   ├── raw/                        <- Downloaded Kaggle CSV goes here (gitignored)
│   └── processed/                  <- Cleaned time-series data
├── images/
│   └── app_screenshot.jpg          <- Dashboard screenshots for README
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb   <- Data profiling and time-series setup
│   ├── 02_pca_visualization.ipynb  <- Dimensionality reduction
│   ├── 03_isolation_forest.ipynb   <- Unsupervised modeling
├── src/
│   └── data_loader.py              <- Custom OOP data ingestion module
├── app.py                          <- Streamlit interactive dashboard application
├── .gitignore
├── README.md
└── requirements.txt

```

## How to Run Locally

**1. Clone the Repository:**

```
git clone [https://github.com/](https://github.com/)[YourUsername]/melbourne-wwtp-anomaly-detection.git
cd melbourne-wwtp-anomaly-detection

```

**2. Install dependencies:**

```
pip install -r requirements.txt
```

**3. Download the Data:**

Visit Kaggle: Full Scale Waste Water Treatment Plant Data.

Download `Data-Melbourne_F_fixed.csv` and place it in the `data/raw/` directory.

**4. Run the Interactive Dashboard:**

```
streamlit run app.py
```

**5. Explore the Notebooks:**

* Launch Jupyter and execute the notebooks in the `notebooks/` directory to see the step-by-step mathematical breakdown.

## 6. Results & Visualizations
Below is the interactive Streamlit dashboard built for this project. It allows plant operators to dynamically adjust the Isolation Forest's sensitivity (Expected Anomaly Rate) and instantly visualize the mathematical outliers across 16 different SCADA sensors.

## Author
Richard Boabeng, Civil Engineer & Data Scientist

* [LinkedIn](https://www.linkedin.com/in/richard-boabeng-386992125/)

* [GitHub](https://github.com/RBoabeng)
