.. ts documentation master file, created by
   sphinx-quickstart on Wed May 21 11:43:57 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Time Series – RUL Forecasting and Anomaly Detection
===================================================

This documentation provides a detailed walkthrough of a predictive maintenance system developed using the NASA CMAPSS dataset. It demonstrates three main capabilities:

- Predicting Remaining Useful Life (RUL)
- Detecting anomalies
- Forecasting RUL trajectories using sequence models

The system is built using a combination of machine learning and deep learning techniques, including XGBoost and LSTM-based architectures.

.. contents:: Table of Contents
   :depth: 2
   :local:

Project Structure
-----------------

.. code-block:: text

    time-series-rul-anomaly/
    ├── data/
    ├── models/
    ├── src/
    ├── results/
    ├── requirements.txt
    ├── setup.py
    └── README.md

Model Overview
--------------

**XGBoost RUL Prediction**

- Regression model to estimate RUL
- Features: statistical summaries, trend indicators, rate of change
- Hyperparameter optimization via grid search

**LSTM Autoencoder for Anomaly Detection**

- Trained on normal data to detect deviations
- Architecture: Encoder → Bottleneck → Decoder
- Anomalies flagged via reconstruction error + dynamic threshold

**LSTM RUL Forecaster**

- Sequence-to-sequence model
- Bidirectional LSTM + ReLU output layers
- Forecasts full RUL trajectory (not just point prediction)

Dataset
-------

- **Source**: `NASA CMAPSS <https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository>`_
- **Subset**: FD004 (multiple fault modes and conditions)
- **Structure**:
  - 21 sensors, 3 operational settings
  - Training: full engine life
  - Testing: truncated time series

Installation
------------

Clone the repository:

.. code-block:: bash

    git clone https://github.com/mouradboutrid/time-series-rul-anomaly.git
    cd time-series-rul-anomaly

Install Python dependencies:

.. code-block:: bash

    pip install -r requirements.txt

Usage
-----

**Data Preparation**

1. Download CMAPSS dataset
2. Place raw data in ``data/raw/``
3. Run the notebook:

.. code-block:: bash

    jupyter notebook notebooks/01_data_preparation.ipynb

**Run Notebooks in Order**

1. ``01_data_preparation.ipynb``
2. ``02_xgboost_rul_prediction.ipynb``
3. ``03_lstm_autoencoder_anomaly_detection.ipynb``
4. ``04_lstm_forecaster_rul_prediction.ipynb``
5. ``05_model_evaluation.ipynb``

**Using the XGBoost Model**

.. code-block:: python

    import joblib
    from scripts.data_utils import preprocess_data

    model = joblib.load('models/xgboost/xgb_rul_predictor.pkl')
    X_new = preprocess_data(new_data)
    predictions = model.predict(X_new)

Feature Engineering
-------------------

- **Sliding windows**: mean, std, min, max
- **Trends**: slope and curvature
- **Interactions**: sensor-sensor multiplications
- **Normalization**: per operational condition

Optimization
------------

- **XGBoost**: Grid search with CV
- **LSTM**: Bayesian optimization + early stopping

Evaluation Metrics
------------------

- **Regression**: RMSE, MAE, RUL score
- **Anomaly Detection**: Precision, Recall, F1, AUC, TTD

Results
-------

- XGBoost: RMSE ≈ 15–20 cycles
- Autoencoder: Detects failures ~30–50 cycles early
- LSTM Forecaster: Accurate RUL curves up to 50 cycles out

Future Work
-----------

- Add attention/transformer models
- Real-time dashboards & pipelines
- Uncertainty quantification with quantile regression
- Transfer learning for other engine types
- Explainability and RL-based maintenance policies

Dependencies
------------

- Python 3.8+
- NumPy, Pandas, SciPy
- Scikit-learn, XGBoost
- TensorFlow/Keras
- Matplotlib, Seaborn, Plotly


Acknowledgements
----------------

- NASA Prognostics Center for the CMAPSS dataset
- Open-source libraries: XGBoost, TensorFlow, Keras, etc.


















.. toctree::
   :maxdepth: 2
   :caption: Contents:

