"""
Setup script for the project (optional, for package installation).
"""

from setuptools import setup, find_packages

setup(
    name="wind-turbine-predictive-maintenance",
    version="1.0.0",
    description="AI-powered Predictive Maintenance for Wind Turbines",
    author="Engineering Project Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "torch>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "pyyaml>=6.0",
        "streamlit>=1.25.0",
        "pytest>=7.4.0",
    ],
)

