# IT5006 Predictive Policing: Crime Hotspot Prediction

This repository contains the coursework deliverables for **Team 21** in **IT5006 Fundamentals of Data Analytics**. The project focuses on **predicting weekly crime hotspots** from historical crime records and presenting the result in a lightweight **Streamlit dashboard** for demonstration.

The repository currently includes:

- a Jupyter notebook for data preparation, feature engineering, modeling, and evaluation
- a deployable Streamlit app for interactive hotspot risk prediction
- the local datasets and model artifact required to reproduce the demo

## Project Overview

The core task is to predict whether a **Chicago police district** is likely to become a **hotspot in the following week**.

The workflow in this repository covers:

- loading and cleaning historical crime data
- aggregating crimes into weekly district-level features
- engineering lag, rolling-window, and category-ratio features
- training and evaluating machine learning models
- serving a demo interface that allows users to input weekly features and receive a hotspot risk score

The Streamlit app is implemented in `deployment/app.py`, and the main notebook is `notebooks/Milestone3_Team21.ipynb`.

## Repository Structure

```text
Team21_IT5006_Predictive_Policing_AY2526Sem2/
├── data/
│   ├── agencies.csv
│   ├── chicago_crime_2015_2025.parquet
│   ├── NIBRS_incident.csv
│   ├── NIBRS_OFFENSE.csv
│   └── NIBRS_OFFENSE_TYPE.csv
├── deployment/
│   ├── app.py
│   └── hotspot_model.joblib
├── docs/
│   └── icons8-detain-50.png
├── notebooks/
│   └── Milestone3_Team21.ipynb
├── requirements.txt
└── README.md
```

## Data Files

The `data/` directory contains the local files used by the notebook and dashboard:

- `chicago_crime_2015_2025.parquet`: main Chicago crime dataset used for training and dashboard defaults
- `NIBRS_incident.csv`
- `NIBRS_OFFENSE.csv`
- `NIBRS_OFFENSE_TYPE.csv`
- `agencies.csv`

No extra download step is required if these files are already present in the repository.

## Modeling Summary

The deployed dashboard uses a **Gradient Boosting** classifier saved as `deployment/hotspot_model.joblib`.

The app uses the following input features:

- `total_crimes`
- `lag_1w`
- `lag_2w`
- `rolling_mean_4w`
- `week_of_year`
- `ratio_violent`
- `ratio_property`
- `ratio_drug`
- `ratio_public_order`

Reference metrics exposed in the dashboard:

| Metric | Value |
| --- | ---: |
| Model | Gradient Boosting |
| Decision threshold | 0.40 |
| Test AUC | 0.9143 |
| Precision | 0.8088 |
| Recall | 0.3595 |
| F1 score | 0.4977 |
| Training period | 2015-2023 |
| Validation period | 2024 |
| Test period | 2025 |

If the saved model artifact is unavailable, the Streamlit app will automatically train a fallback model from the local Chicago dataset.

## Environment Setup

Create and activate a Python environment, then install the project dependencies:

```bash
pip install -r requirements.txt
```

The required packages are listed in `requirements.txt`, including `pandas`, `scikit-learn`, `pyarrow`, `jupyter`, `streamlit`, and `joblib`.

## Running the Notebook

To reproduce the analysis workflow:

```bash
jupyter lab
```

Then open:

```text
notebooks/Milestone3_Team21.ipynb
```

The notebook contains the end-to-end analysis pipeline, including:

- data loading and preprocessing
- weekly aggregation and feature construction
- hotspot label generation
- model training and comparison
- validation and test evaluation

## Running the Streamlit App

From the repository root, start the local dashboard with:

```bash
streamlit run deployment/app.py
```

After Streamlit starts, open the local URL shown in your terminal, typically:

```text
http://localhost:8501
```

The dashboard lets you:

- enter weekly district-level crime features manually
- view the predicted hotspot probability for the next week
- compare the prediction against the model decision threshold
- inspect the model summary and reference metrics in the sidebar

## External Deployment

Because this project uses **Streamlit**, the easiest way to publish a public link is **Streamlit Community Cloud** rather than GitHub Pages.

```text
https://team21it5006predictivepolicingay2526sem2-iptbbnvmkkudnc3j7h9wv.streamlit.app/
```

## Notes and Limitations

- This project is a coursework proof-of-concept.
- Predictions are based on **weekly aggregate features**, not raw incident narratives.
- The dashboard is intended for **analysis and demonstration**, not real-world policing decisions.
- Keep the files in `data/` and `deployment/hotspot_model.joblib` in the repository if you want the deployed app to work without modification.

## Team

- LIN YIHAN
- QI RUIXUAN
- XU QIAOYANG

# AI Declaration

We used GPT-5.4 to genarated some parts of README files. We are responsible for the content and quality of the submitted work.