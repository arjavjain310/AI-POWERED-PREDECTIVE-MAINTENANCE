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
