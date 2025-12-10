# AI-powered Predictive Maintenance for Karnataka Wind Farms

## Deep Learning for Wind Turbine SCADA Data

A comprehensive end-to-end system for predicting component failures and estimating Remaining Useful Life (RUL) of wind turbine components using machine learning and deep learning techniques.

**Specifically designed for Karnataka wind farms in districts: Chitradurga, Gadag, and Davangere**

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Documentation](#documentation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Project Overview

### Problem Statement

Wind farms in Karnataka, especially in districts like Chitradurga, Gadag, and Davangere, face:
- **Frequent unplanned downtime** due to undetected component failures
- **High maintenance costs** from expensive emergency repairs
- **Reduced energy yield** when turbines are offline
- **Need for early prediction** to enable timely intervention

### Solution

This project implements a complete predictive maintenance system for Karnataka wind turbines that:

1. **Analyzes SCADA & sensor time-series data** from multiple wind turbines
2. **Predicts component failures** within a future horizon (24-72 hours)
3. **Estimates Remaining Useful Life (RUL)** of key components (gearbox, generator, bearings)
4. **Recommends optimized maintenance schedules** based on predicted failures and RUL

### Key Components

- **Data Pipeline**: Synthetic data generation, loading, preprocessing, and feature engineering
- **Modeling**: Baseline ML models (Logistic Regression, Random Forest, XGBoost) and Deep Learning models (MLP, LSTM/GRU)
- **Evaluation**: Comprehensive metrics and visualizations
- **RUL Estimation**: Time-series deep learning models for RUL prediction
- **Maintenance Scheduling**: Rule-based decision logic and schedule optimization
- **Dashboard**: Interactive Streamlit interface for monitoring and visualization

---

## ✨ Features

- **Synthetic Data Generation**: Realistic SCADA data generator with failure patterns
- **Feature Engineering**: Lag features, rolling statistics, and derived features
- **Multiple Models**: Baseline ML and deep learning models for comparison
- **RUL Prediction**: Sequence models (LSTM/GRU) for time-to-failure estimation
- **Maintenance Optimization**: Cost-aware scheduling with preventive vs corrective maintenance
- **Interactive Dashboard**: Real-time monitoring and visualization
- **Comprehensive Testing**: Unit tests for all major components
- **Well-Documented**: Detailed code comments, docstrings, and documentation

---

## 📁 Project Structure

```
MAJOR PROJECT/
├── data/
│   ├── raw/                    # Raw/synthetic data files
│   └── processed/              # Processed/feature-engineered data
├── notebooks/
│   ├── 01_exploration.ipynb   # Data exploration and visualization
│   ├── 02_modeling_baselines.ipynb  # Baseline model training
│   └── 03_deep_learning_and_RUL.ipynb  # Deep learning and RUL estimation
├── src/
│   ├── config/
│   │   └── config.yaml         # Configuration file
│   ├── data/
│   │   ├── data_loader.py      # Data loading utilities
│   │   ├── preprocessing.py   # Data preprocessing
│   │   ├── feature_engineering.py  # Feature creation
│   │   └── synthetic_data_generator.py  # Synthetic data generation
│   ├── models/
│   │   ├── baseline_models.py  # Baseline ML models
│   │   ├── deep_learning_models.py  # PyTorch models (MLP, LSTM, GRU)
│   │   ├── training.py          # Training utilities
│   │   ├── evaluation.py       # Evaluation metrics and plots
│   │   └── rul_estimation.py   # RUL estimation utilities
│   ├── maintenance/
│   │   ├── maintenance_rules.py  # Rule-based maintenance logic
│   │   └── schedule_optimizer.py  # Maintenance scheduling
│   ├── visualization/
│   │   └── plots.py            # Visualization functions
│   └── utils/
│       ├── config_utils.py     # Configuration utilities
│       └── logging_utils.py    # Logging setup
├── app/
│   └── dashboard.py           # Streamlit dashboard
├── tests/
│   ├── test_data_pipeline.py  # Data pipeline tests
│   ├── test_models.py         # Model tests
│   └── test_rul_logic.py      # RUL and maintenance logic tests
├── reports/
│   ├── project_report_outline.md  # Project report structure
│   └── presentation_outline.md   # Presentation outline
├── run_experiment.py          # Main experiment script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Steps

1. **Clone or download the project**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import torch, pandas, sklearn, streamlit; print('Installation successful!')"
   ```

---

## 🏃 Quick Start

### 1. Generate Synthetic Data

If you don't have real SCADA data, generate synthetic data:

```bash
python src/data/synthetic_data_generator.py
```

This creates `data/raw/wind_turbine_scada.csv` with realistic SCADA data for 10 turbines over 1 year.

### 2. Run the Complete Pipeline

Run the end-to-end experiment:

```bash
python run_experiment.py
```

This will:
- Load and preprocess data
- Create features
- Train baseline and deep learning models
- Evaluate models
- Generate maintenance recommendations
- Save results to `results/` directory

### 3. Launch the Dashboard

Start the interactive dashboard:

```bash
streamlit run app/dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

### 4. Explore Notebooks

Open Jupyter notebooks for detailed exploration:

```bash
jupyter notebook notebooks/
```

---

## 📖 Usage

### Configuration

Edit `src/config/config.yaml` to customize:
- Data paths and parameters
- Model hyperparameters
- Maintenance thresholds
- Training parameters

### Data Pipeline

```python
from src.data.data_loader import load_scada_data
from src.data.preprocessing import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer

# Load data
df = load_scada_data("data/raw/wind_turbine_scada.csv")

# Feature engineering
fe = FeatureEngineer()
df = fe.create_features(df)

# Preprocessing
preprocessor = DataPreprocessor()
df_processed = preprocessor.fit_transform(df)
```

### Model Training

```python
from src.models.baseline_models import BaselineModelTrainer

# Train baseline model
trainer = BaselineModelTrainer()
model = trainer.train_random_forest(X_train, y_train)

# Make predictions
predictions = trainer.predict('random_forest', X_test)
```

### Deep Learning

```python
from src.models.deep_learning_models import MLP, get_device
from src.models.training import train_model

# Create model
device = get_device()
model = MLP(input_size=100, hidden_layers=[128, 64, 32])

# Train
history = train_model(model, train_loader, val_loader, device=device)
```

### Maintenance Recommendations

```python
from src.maintenance.maintenance_rules import MaintenanceRules
from src.maintenance.schedule_optimizer import MaintenanceScheduler

# Evaluate maintenance needs
rules = MaintenanceRules(failure_probability_threshold=0.7, rul_threshold_hours=48)
decision = rules.evaluate_maintenance_need(failure_prob=0.8, rul_hours=30)

# Create schedule
scheduler = MaintenanceScheduler()
schedule = scheduler.create_schedule(turbine_data, planning_horizon_days=7)
```

---

## 📊 Results

### Model Performance

The system evaluates models using:
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression Metrics** (RUL): MSE, RMSE, MAE, MAPE

### Maintenance Optimization

The scheduler optimizes maintenance by:
- Prioritizing high-risk turbines
- Balancing preventive vs corrective maintenance costs
- Minimizing downtime and total cost

---

## 🧪 Testing

Run unit tests:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

---

## 📚 Documentation

### Notebooks

- **01_exploration.ipynb**: Data exploration, statistics, and visualizations
- **02_modeling_baselines.ipynb**: Baseline model training and evaluation
- **03_deep_learning_and_RUL.ipynb**: Deep learning models and RUL estimation

### Reports

See `reports/` directory for:
- **project_report_outline.md**: Detailed project report structure
- **presentation_outline.md**: Presentation slides outline

---

## 🔧 Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're in the project root and `src/` is in Python path
2. **Data not found**: Run `python src/data/synthetic_data_generator.py` first
3. **CUDA errors**: The code works on CPU; GPU is optional
4. **Memory errors**: Reduce batch size or use data subset

### Getting Help

- Check configuration in `src/config/config.yaml`
- Review logs in `logs/project.log`
- Run tests to verify installation: `pytest tests/`

---

## 🎓 Academic Use

This project is designed for:
- Final-year engineering projects (B.E/B.Tech)
- Research in predictive maintenance
- Learning ML/DL for time-series data
- Industrial applications

### Key Contributions

1. End-to-end pipeline from data to deployment
2. Multiple model architectures for comparison
3. Realistic synthetic data generation
4. Maintenance optimization framework
5. Production-ready code structure

---

## 📝 License

This project is provided for educational and research purposes.

---

## 👥 Authors

Engineering Project Team

---

## 🙏 Acknowledgments

- Wind energy industry standards for SCADA data
- Open-source ML/DL libraries (PyTorch, scikit-learn, XGBoost)
- Streamlit for dashboard framework

---

## 📧 Contact

For questions or issues, please refer to the project documentation or create an issue in the repository.

---

**Last Updated**: 2024

**Version**: 1.0.0

