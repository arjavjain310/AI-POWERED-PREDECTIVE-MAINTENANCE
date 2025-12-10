# Karnataka Wind Farms - Project Setup

## Overview

This project is specifically designed for **Karnataka wind farms** in districts:
- **Chitradurga** - Known for strong wind patterns
- **Gadag** - Moderate to high wind speeds
- **Davangere** - Good wind resource potential

## Problem Statement

Wind farms in Karnataka face:
- **Frequent unplanned downtime** due to undetected component failures
- **High maintenance costs** from emergency repairs
- **Reduced energy yield** when turbines are offline
- **Need for early prediction** to enable timely intervention

## Solution

AI-based predictive maintenance system that:
- Forecasts turbine component health from SCADA data
- Enables timely intervention before failures
- Reduces downtime and maintenance costs
- Improves renewable energy output

## Karnataka-Specific Features

### Climate & Wind Patterns
- **Monsoon Season (June-September)**: Strong winds (8-15 m/s)
- **Pre/Post Monsoon (Mar-May, Oct-Nov)**: Moderate winds (6-12 m/s)
- **Winter (Dec-Feb)**: Lower winds (4-8 m/s)
- **Peak Winds**: Afternoon hours (2-5 PM)

### Temperature Patterns
- **Summer (Mar-May)**: 30-35°C
- **Monsoon (Jun-Sep)**: 25-30°C
- **Post-Monsoon (Oct-Nov)**: 28-32°C
- **Winter (Dec-Feb)**: 22-28°C

### Data Generation

The synthetic data generator creates realistic SCADA data based on:
- Karnataka wind patterns and seasonal variations
- District-specific wind characteristics
- Tropical climate temperature profiles
- Realistic failure patterns for Indian wind farm conditions

## Setup Instructions

### 1. Generate Karnataka-Specific Data

```bash
python src/data/synthetic_data_generator.py
```

This will create data for turbines distributed across:
- Chitradurga district
- Gadag district  
- Davangere district

### 2. Train Models

```bash
python run_experiment.py
```

The models will be trained on Karnataka wind farm data with:
- Region-specific wind patterns
- District-specific characteristics
- Realistic failure scenarios

### 3. Launch Dashboard

```bash
streamlit run app/dashboard.py
```

The dashboard will show:
- Region and district information
- Turbines by district
- Karnataka-specific wind patterns
- Maintenance recommendations

## Model Training for Karnataka

The models are trained to recognize:
1. **Karnataka wind patterns** - Monsoon vs non-monsoon seasons
2. **District-specific characteristics** - Different wind profiles per district
3. **Tropical climate effects** - Higher ambient temperatures
4. **Regional failure patterns** - Common failure modes in Indian wind farms

## Expected Benefits

1. **Reduced Downtime**: Early prediction prevents unplanned shutdowns
2. **Cost Savings**: Preventive maintenance costs less than emergency repairs
3. **Increased Yield**: More uptime = more energy generation
4. **Better Planning**: Optimized maintenance schedules

## Data Characteristics

- **Turbines**: 15+ turbines across 3 districts
- **Sampling**: 10-minute intervals
- **Duration**: 1 year of data
- **Features**: Wind speed, temperature, vibration, power output, etc.
- **Labels**: Failure predictions and RUL estimates

## Next Steps

1. Generate Karnataka-specific data
2. Train models on regional data
3. Evaluate model performance
4. Deploy for real-time monitoring
5. Integrate with actual SCADA systems

---

**Note**: This system is designed to be adaptable to real SCADA data from Karnataka wind farms. Replace synthetic data with actual data when available.

