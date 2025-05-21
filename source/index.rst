.. ts documentation master file, created by
   sphinx-quickstart on Wed May 21 11:43:57 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ts documentation
================

Time Series Project: RUL Prediction and Anomaly Detection (CMAPSS)
===================================================================

Overview
--------

**Objective:**  
Predict Remaining Useful Life (RUL) of aircraft engines and detect anomalies using multivariate time series data from NASA's CMAPSS dataset.

**Goals:**  

- Develop a regression model for RUL prediction.
- Implement anomaly detection for early failure identification.
- Enable reproducible training and optional deployment pipeline.

Dataset Description
-------------------

**Dataset Source:**  
NASA's Commercial Modular Aero-Propulsion System Simulation (CMAPSS)

**Subset Used:**  
- FD001 (Single operating condition, single fault mode)

**Features:**

- ``engine_id``: Unique engine unit identifier
- ``cycle``: Time cycle (incremental)
- ``op_setting_1, op_setting_2, op_setting_3``: Operational settings
- ``sensor_1`` to ``sensor_21``: Sensor readings
- ``RUL``: Target variable (computed during preprocessing)

Environment and Tools
---------------------

- **Language**: Python 3.10+
- **Libraries**:
  
  - ``pandas``, ``numpy``: Data manipulation
  - ``scikit-learn``: Preprocessing and evaluation
  - ``TensorFlow/Keras`` or ``PyTorch``: Model development
  - ``matplotlib``, ``seaborn``: Visualization
  - ``MLflow`` or ``Weights & Biases``: Experiment tracking
- **Deployment (Optional)**:
  
  - ``FastAPI``, ``Flask``: REST API for inference
  - ``Docker``, ``Kubernetes``: Containerized deployment
  - ``Prometheus``, ``Grafana``: Monitoring and alerting

Pipeline Overview
-----------------

.. code-block:: text

   1. Data Ingestion
   2. Preprocessing
   3. Feature Engineering
   4. RUL Labeling
   5. Model Training
      - RUL Prediction
      - Anomaly Detection
   6. Evaluation
   7. Deployment (Optional)

Detailed Workflow
-----------------

**1. Data Ingestion**

- Load ``train_FD001.txt`` and ``test_FD001.txt`` using ``pandas.read_csv``.
- Parse engine cycle data into structured DataFrames.

**2. Preprocessing**

- Normalize sensor data (MinMaxScaler or StandardScaler).
- Remove sensors with no variation or redundancy.
- Handle missing or outlier values if applicable.

**3. Feature Engineering**

- Rolling statistics (mean, std) on sensor readings.
- Engine health indicators (e.g., deviation from baseline).
- Lag features and cycle-based deltas.

**4. RUL Labeling**

- For training data:  
  ``RUL = max_cycle_per_engine - current_cycle``
- For testing data:  
  Use ``RUL_FD001.txt`` as ground truth.

**5. Model Training**

- **RUL Prediction**:
  
  - Models: LSTM, GRU, CNN, or Transformer-based models
  - Loss: Mean Squared Error (MSE)

- **Anomaly Detection**:
  
  - Approaches: Autoencoders, Isolation Forest, One-Class SVM
  - Use reconstruction error or model confidence scores

**6. Evaluation**

- Metrics: RMSE, MAE, Score function (as in literature)
- Visualizations: RUL curves, prediction intervals, anomaly heatmaps

**7. Deployment (Optional)**

- Export model: ``.h5`` (Keras) or ``.pt`` (PyTorch)
- Create inference endpoint using FastAPI/Flask
- Wrap in Docker container for CI/CD integration

Logging and Monitoring
----------------------

- Track metrics with MLflow or W&B
- Use Prometheus + Grafana dashboards for live monitoring
- Implement alerting for anomaly spikes

Versioning and Reproducibility
------------------------------

- Use Git for source control
- Lock dependencies with ``requirements.txt`` or ``poetry.lock``
- Track experiment metadata and artifacts

Notes
-----

- FD002, FD003, FD004 introduce multiple operating conditions and fault modes for generalization.
- Consider ensemble methods or domain adaptation for production environments.




.. toctree::
   :maxdepth: 2
   :caption: Contents:

